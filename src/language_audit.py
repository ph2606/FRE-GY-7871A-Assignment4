"""Deterministic language sensitivity audit; never changes primary inclusion.

News input is the headline; social input is own post/comment text. The detector's
English score is an uncalibrated model probability, not a validated accuracy
estimate. Short technical English, names, code and quotations can be mislabeled.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import time

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 787104
DetectorFactory.seed = SEED

ILLUSTRATIVE_REVIEW = {
    'gn_9caedf4ef4b901db02e9': 'English headline; short text is falsely flagged as non-English.',
    'gn_c5dc88248a752c951d91': 'English headline; names and financial figures dominate, but the sentence is English.',
    'gn_a808e6d65d1ca0724ee9': 'English AI-risk headline falsely flagged as Norwegian.',
    'hn_49628021': 'English comment falsely flagged as French.',
    'hn_49635925': 'English comment falsely flagged as French.',
    'hn_49608108': 'English framing around a long French quotation; one document contains multiple languages.',
    'gn_7135d8fb2fab505f0538': 'Non-English headline; Spanish text remains despite an English-oriented retrieval.',
    'gn_b484acd100fcc3690609': 'Non-English headline; detector labels Somali. Not a separately validated language gold label.',
}


def classify(task):
    ident, text, text_hash = task
    DetectorFactory.seed = SEED
    try:
        prediction = detect_langs(text)
        scores = {x.lang: float(x.prob) for x in prediction}
        return {'id': ident, 'text_sha256': text_hash,
                'predicted_language': prediction[0].lang,
                'english_probability': scores.get('en', 0.0),
                'top_language_probability': float(prediction[0].prob),
                'language_probabilities': json.dumps(scores, sort_keys=True),
                'detector_error': ''}
    except LangDetectException as error:
        return {'id': ident, 'text_sha256': text_hash, 'predicted_language': 'undetermined',
                'english_probability': np.nan, 'top_language_probability': np.nan,
                'language_probabilities': '{}', 'detector_error': str(error)}


def write_sensitivity(audit):
    scored_path = ROOT / 'data/processed/scored_documents.csv'
    if not scored_path.exists():
        return
    columns = ['id', 'kind', 'is_comment', 'risk', 'included_analysis', 'vader', 'negative', 'hu_negative']
    scored = pd.read_csv(scored_path, usecols=columns, low_memory=False)
    scored = scored.loc[scored.included_analysis.astype(str).str.lower().eq('true')].copy()
    scored = scored.merge(audit[['id', 'english_probability']], on='id', how='left', validate='many_to_one')
    scored['medium'] = np.where(scored.kind.eq('news'), 'News headlines',
                                np.where(scored.is_comment, 'HN comments', 'HN stories'))
    results = []
    for name, threshold in [('Primary inclusion', None), ('English score >= 0.5', 0.5), ('English score >= 0.9', 0.9)]:
        subset = scored if threshold is None else scored.loc[scored.english_probability >= threshold]
        rows = subset.groupby(['medium', 'risk']).agg(records=('id', 'size'),
                negative_share=('negative', 'mean'), mean_vader=('vader', 'mean'),
                mean_hu_negative=('hu_negative', 'mean')).reset_index()
        rows['language_sensitivity'] = name
        results.append(rows)
    pd.concat(results, ignore_index=True).to_csv(ROOT / 'outputs/language_sentiment_sensitivity.csv', index=False)
    review = audit.loc[audit.id.isin(ILLUSTRATIVE_REVIEW)].copy()
    review['assistant_inspection_note'] = review.id.map(ILLUSTRATIVE_REVIEW)
    review['review_scope'] = 'Purposive illustrative assistant review; not an independent human annotation set or an estimated error rate.'
    review.to_csv(ROOT / 'outputs/language_context_examples.csv', index=False)


def build(max_seconds=270):
    started = time.monotonic()
    frames = []
    for kind in ['news', 'social']:
        data = pd.read_csv(ROOT / f'data/processed/{kind}.csv', keep_default_na=False, low_memory=False)
        if 'exclusion' in data:
            data = data.loc[data.exclusion.eq('')].copy()
        data['kind'] = kind
        data['record_type'] = data.get('record_type', 'headline')
        data['analysis_text'] = data['title' if kind == 'news' else 'text'].fillna('').astype(str)
        frames.append(data[['id', 'kind', 'record_type', 'source', 'date', 'analysis_text', 'url']])
    documents = pd.concat(frames, ignore_index=True)
    input_rows = len(documents)
    assert documents.groupby('id').analysis_text.nunique().max() == 1, 'One document ID has conflicting scored texts.'
    # The cache has one row per ID. Identical-ID/text aliases in source inputs
    # are recorded here; this does not alter their primary inclusion flags.
    documents = documents.drop_duplicates('id', keep='first').copy()
    documents['text_sha256'] = documents.analysis_text.map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    documents['word_count'] = documents.analysis_text.str.findall(r'[A-Za-z]+').str.len()
    out = ROOT / 'data/processed/language_audit.csv'
    results = []
    if out.exists():
        cache = pd.read_csv(out, keep_default_na=False, low_memory=False)
        joined = cache.merge(documents[['id', 'text_sha256']], on=['id', 'text_sha256'], how='inner')
        results = joined[['id', 'text_sha256', 'predicted_language', 'english_probability',
                          'top_language_probability', 'language_probabilities', 'detector_error']].to_dict('records')
    completed = {row['id'] for row in results}
    # Process a fixed 500-per-type diagnostic sample first, guaranteeing a
    # meaningful stratified fallback if the all-document run exceeds its budget.
    sample = documents.sample(frac=1, random_state=SEED).groupby(['kind', 'record_type']).head(500)
    sample_ids = set(sample.id)
    pending = documents.loc[~documents.id.isin(completed)].sample(frac=1, random_state=SEED)
    priority = pending.id.isin(sample_ids).astype(int)
    pending = pending.assign(priority=priority).sort_values('priority', ascending=False, kind='stable')
    tasks = list(pending[['id', 'analysis_text', 'text_sha256']].itertuples(index=False, name=None))
    print(f'Language audit: {len(documents)} documents, {len(completed)} cache hits, {len(tasks)} pending.', flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for left in range(0, len(tasks), 1000):
            batch = tasks[left:left+1000]
            results.extend(pool.map(classify, batch, chunksize=50))
            if (left + len(batch)) % 5000 == 0 or left + len(batch) == len(tasks):
                print(f'Detected {left+len(batch)}/{len(tasks)} pending texts in {time.monotonic()-started:.1f}s.', flush=True)
            if time.monotonic() - started > max_seconds and sample_ids.issubset({r['id'] for r in results}):
                print('Time budget reached; preserving classifications and reporting the fixed stratified diagnostic sample.', flush=True)
                break
    classifications = pd.DataFrame(results)
    audit = documents.merge(classifications, on=['id', 'text_sha256'], how='inner', validate='one_to_one')
    for column in ['english_probability', 'top_language_probability']:
        audit[column] = pd.to_numeric(audit[column], errors='coerce')
    audit['english_at_050'] = audit.english_probability >= 0.5
    audit['english_at_090'] = audit.english_probability >= 0.9
    audit['diagnostic_sample'] = audit.id.isin(sample_ids)
    audit['short_under_10_words'] = audit.word_count < 10
    audit.to_csv(out, index=False)
    complete = len(audit) == len(documents)
    summary_input = audit if complete else audit.loc[audit.diagnostic_sample].copy()
    summary_input['predicted_non_english'] = ~summary_input.predicted_language.isin(['en', 'undetermined'])
    summary_input['detector_failure'] = summary_input.predicted_language.eq('undetermined')
    summary = summary_input.groupby(['kind', 'record_type', 'source'], dropna=False).agg(
        records=('id', 'size'), predicted_non_english=('predicted_non_english', 'sum'),
        english_at_050=('english_at_050', 'sum'), english_at_090=('english_at_090', 'sum'),
        short_under_10_words=('short_under_10_words', 'sum'), detector_failures=('detector_failure', 'sum')).reset_index()
    summary['predicted_non_english_pct'] = 100 * summary.predicted_non_english / summary.records
    summary['scope'] = 'all cleaned records' if complete else 'fixed random diagnostic sample, up to 500 per kind/type'
    summary.to_csv(ROOT / 'outputs/language_summary.csv', index=False)
    totals = summary_input.groupby(['kind', 'record_type'], dropna=False).agg(
        records=('id', 'size'), predicted_non_english=('predicted_non_english', 'sum'),
        english_at_050=('english_at_050', 'sum'), english_at_090=('english_at_090', 'sum'),
        short_under_10_words=('short_under_10_words', 'sum'), detector_failures=('detector_failure', 'sum')).reset_index()
    totals['predicted_non_english_pct'] = 100 * totals.predicted_non_english / totals.records
    totals.to_csv(ROOT / 'outputs/language_summary_by_type.csv', index=False)
    flagged = summary_input.loc[summary_input.predicted_non_english].copy()
    flagged.groupby(['kind', 'record_type'], group_keys=False).head(40).to_csv(
        ROOT / 'outputs/language_flagged_review.csv', index=False)
    metadata = {'detector': f'langdetect {version("langdetect")}', 'deterministic_seed': SEED,
                'detected_utc': datetime.now(timezone.utc).isoformat(),
                'input_rows_before_id_dedup': input_rows,
                'duplicate_id_same_text_aliases': input_rows-len(documents),
                'input_records': len(documents), 'classified_records': len(audit), 'complete': complete,
                'summary_records': len(summary_input), 'elapsed_seconds': time.monotonic()-started,
                'primary_records_dropped': 0,
                'input_scope': 'News headline; social own text (comment or story title plus submission).',
                'english_probability_interpretation': 'Uncalibrated langdetect score; absent English candidate recorded as 0.0. Do not read it as validated probability of correct classification.',
                'caveats': ['Short English headlines and technical text can be mislabeled, especially when dominated by proper nouns, abbreviations, URLs or code.',
                            'Multilingual quotations can trigger a non-English prediction even when the author is writing primarily in English.',
                            'This audit does not validate sentiment accuracy and must not retroactively select the primary corpus.',
                            'A probability threshold sensitivity is an alternative sample, not a corrected English ground truth.'],
                'summary_by_type': totals.to_dict('records'),
                'predicted_language_counts': summary_input.predicted_language.value_counts().to_dict()}
    (ROOT / 'outputs/language_audit.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    write_sensitivity(audit)
    print(json.dumps(metadata, indent=2), flush=True)
    return audit


if __name__ == '__main__':
    build()
