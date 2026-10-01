"""Retrieve auditable broad-AI Hacker News records with an immutable cutoff.

Hacker News is a technology-community sample, not X and not population opinion.
Collection is keyword-based, with adaptive timestamp shards to avoid Algolia's
1,000-result ceiling. All successful API responses and failed access probes are
saved. Engagement is measured at retrieval, not at historical publication time.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import threading
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/social'
NY = ZoneInfo('America/New_York')
START = datetime(2026, 9, 8, tzinfo=NY)
REQUESTED_END = datetime(2026, 10, 1, tzinfo=NY)
REQUESTED_MARKET_END = datetime(2026, 9, 30, 16, tzinfo=NY)
QUERIES = ['AI', 'artificial intelligence', 'LLM', 'ChatGPT', 'OpenAI',
           'Anthropic', 'superintelligence', 'machine learning', 'AGI']
RELEVANCE = re.compile(r'\b(?:ai|artificial intelligence|llms?|large language models?|'
                       r'chatgpt|openai|anthropic|superintelligence|machine learning|agi)\b', re.I)
WORDS = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
URLS = re.compile(r'https?://\S+', re.I)
API = 'https://hn.algolia.com/api/v1/search_by_date'
_local = threading.local()


def session():
    if not hasattr(_local, 'session'):
        s = requests.Session()
        s.headers['User-Agent'] = 'NLP-course-research/1.0 (public academic analysis)'
        retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        s.mount('https://', HTTPAdapter(max_retries=retry))
        _local.session = s
    return _local.session


def clean(value):
    value = value or ''
    # Only HN HTML is decoded; external linked pages are not silently included.
    decoded = BeautifulSoup(value, 'html.parser').get_text(' ', strip=True) if '<' in value else value
    return re.sub(r'\s+', ' ', html.unescape(decoded)).strip()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def get_shard(query, left, right):
    """Save the parent response, then split if even one hit may be truncated."""
    params = {'query': query, 'numericFilters': f'created_at_i>={left},created_at_i<{right}',
              'hitsPerPage': 1000, 'page': 0, 'typoTolerance': 'false', 'queryType': 'prefixNone'}
    key = digest(json.dumps(params, sort_keys=True).encode())[:20]
    path = RAW / 'hn' / f'{key}.json'
    provenance = path.with_suffix('.provenance.json')
    if path.exists() and provenance.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        meta = json.loads(provenance.read_text(encoding='utf-8'))
    else:
        response = session().get(API, params=params, timeout=90)
        response.raise_for_status()
        payload = response.json()
        if 'hits' not in payload:
            raise ValueError(f'Unexpected HN response: {payload}')
        path.write_bytes(response.content)
        meta = {'url': response.url, 'params': params,
                'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                'sha256': digest(response.content), 'http_status': response.status_code}
        provenance.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    meta = {**meta, 'file': str(path.relative_to(ROOT)), 'query': query,
            'start_epoch': left, 'end_epoch': right, 'reported_hits': payload.get('nbHits'),
            'returned_hits': len(payload['hits']), 'exhaustive_nb_hits': payload.get('exhaustiveNbHits')}
    n = payload.get('nbHits', len(payload['hits']))
    if n > len(payload['hits']) or len(payload['hits']) >= 1000:
        if right - left <= 1:
            raise RuntimeError('Unresolved 1,000-hit truncation inside one second.')
        middle = (left + right) // 2
        a, am = get_shard(query, left, middle)
        b, bm = get_shard(query, middle, right)
        return a + b, [{**meta, 'used_as_leaf': False}] + am + bm
    return [(row, query, meta['retrieved_utc']) for row in payload['hits']], [{**meta, 'used_as_leaf': True}]


def probe_access():
    probes = {
        'reddit_search': 'https://www.reddit.com/search.json?q=artificial%20intelligence&sort=new&limit=2',
        'bluesky_search': 'https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q=AI&since=2026-09-08T00:00:00Z&until=2026-09-30T00:00:00Z&limit=2&sort=latest',
        'x_hubinger_event': 'https://x.com/EvanHub/status/2097497037956891126',
    }
    records = []
    for name, url in probes.items():
        path = RAW / f'{name}.response.txt'
        meta_path = RAW / f'{name}.provenance.json'
        if meta_path.exists():
            records.append(json.loads(meta_path.read_text(encoding='utf-8')))
            continue
        try:
            response = session().get(url, timeout=30)
            path.write_bytes(response.content)
            meta = {'name': name, 'requested_url': url, 'final_url': response.url,
                    'status_code': response.status_code, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                    'sha256': digest(response.content), 'file': str(path.relative_to(ROOT)),
                    'usable_corpus': False}
        except requests.RequestException as error:
            meta = {'name': name, 'requested_url': url, 'error': str(error),
                    'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'usable_corpus': False}
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        records.append(meta)
    return records


def collection_config(refresh=False, cutoff_utc=None):
    """Extend only by explicit request, preserving every previous snapshot.

    --refresh requires either an observed --cutoff-utc timestamp or a completed
    ASSIGNMENT4_CUTOFF local date. Future timestamps are always rejected.
    """
    config_path = RAW / 'collection_config.json'
    old = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else None
    requested_day = os.environ.get('ASSIGNMENT4_CUTOFF')
    now = datetime.now(timezone.utc)
    desired_end = REQUESTED_END
    explicit_cutoff = None
    if cutoff_utc:
        explicit_cutoff = datetime.fromisoformat(str(cutoff_utc).replace('Z', '+00:00'))
        if explicit_cutoff.tzinfo is None:
            raise ValueError('--cutoff-utc requires an explicit timezone, for example 2026-09-30T20:00:00Z.')
        explicit_cutoff = explicit_cutoff.astimezone(timezone.utc)
        if not START < explicit_cutoff <= REQUESTED_END:
            raise ValueError('--cutoff-utc must be after the assignment start and no later than 2026-10-01T04:00:00Z.')
        if explicit_cutoff > now:
            raise ValueError('--cutoff-utc cannot be in the future.')
    if requested_day:
        desired_day = datetime.strptime(requested_day, '%Y-%m-%d').replace(tzinfo=NY)
        if not START <= desired_day < REQUESTED_END:
            raise ValueError('ASSIGNMENT4_CUTOFF must be between 2026-09-08 and 2026-09-30 inclusive.')
        desired_end = desired_day + timedelta(days=1)
    if refresh:
        if not requested_day and explicit_cutoff is None:
            raise ValueError('--refresh requires --cutoff-utc or ASSIGNMENT4_CUTOFF=YYYY-MM-DD (inclusive local date).')
        if explicit_cutoff is None and now < desired_end:
            raise ValueError(f'Cannot refresh through {requested_day}: that America/New_York calendar day has not completed.')
        cutoff = explicit_cutoff if explicit_cutoff is not None else desired_end.astimezone(timezone.utc)
        if old and cutoff <= datetime.fromisoformat(old['cutoff_utc']):
            raise ValueError('--refresh must extend the existing publication cutoff.')
        if old:
            old_stamp = datetime.fromisoformat(old['cutoff_utc']).strftime('%Y%m%dT%H%M%SZ')
            archive = RAW / 'snapshots' / old_stamp
            archive.mkdir(parents=True, exist_ok=True)
            preserve = [config_path, ROOT / 'data/processed/social.csv',
                        ROOT / 'data/processed/social_candidates.csv']
            preserve.extend((ROOT / 'outputs').glob('social_*.json'))
            preserve.extend((ROOT / 'outputs').glob('social_*.csv'))
            for path in preserve:
                if path.exists():
                    shutil.copy2(path, archive / path.name)
    elif old:
        if explicit_cutoff is not None and explicit_cutoff != datetime.fromisoformat(old['cutoff_utc']):
            raise ValueError('Changing an existing publication cutoff requires --refresh to preserve its snapshot.')
        return old, datetime.fromisoformat(old['cutoff_utc'])
    else:
        cutoff = explicit_cutoff if explicit_cutoff is not None else min(now, desired_end.astimezone(timezone.utc))
    config = {'cutoff_utc': cutoff.isoformat(), 'cutoff_et': cutoff.astimezone(NY).isoformat(),
              'requested_start_et': START.isoformat(), 'requested_end_exclusive_et': REQUESTED_END.isoformat(),
              'requested_window_complete': cutoff >= REQUESTED_END,
              'requested_calendar_window_complete': cutoff >= REQUESTED_END,
              'requested_market_end_exclusive_et': REQUESTED_MARKET_END.isoformat(),
              'requested_market_window_complete': cutoff >= REQUESTED_MARKET_END,
              'cutoff_scope': 'explicit observed timestamp' if explicit_cutoff is not None else 'calendar-day freeze',
              'queries': QUERIES, 'timezone': 'America/New_York',
              'collection_method': 'Daily intervals; recursively bisect when Algolia returns >=1,000 hits or nbHits exceeds retrieved count.',
              'cutoff_rule': 'Immutable exclusive freeze; publications must precede cutoff. Intentional extension requires --refresh and either an observed --cutoff-utc timestamp or an already completed ASSIGNMENT4_CUTOFF local day.',
              'previous_cutoff_utc': old['cutoff_utc'] if old else None}
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config, cutoff


def build(refresh=False, cutoff_utc=None):
    (RAW / 'hn').mkdir(parents=True, exist_ok=True)
    (ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs').mkdir(parents=True, exist_ok=True)
    config, cutoff = collection_config(refresh=refresh, cutoff_utc=cutoff_utc)
    jobs = []
    day = START
    while day < cutoff:
        end = min(day + timedelta(days=1), cutoff)
        jobs.extend((q, int(day.timestamp()), int(end.timestamp())) for q in QUERIES)
        day += timedelta(days=1)
    responses, request_audit = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(get_shard, *job) for job in jobs]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            rows, audit = future.result()
            responses.extend(rows)
            request_audit.extend(audit)
            if idx % 30 == 0:
                print(f'Completed {idx}/{len(jobs)} initial HN shards; {len(responses)} retrieved hit instances.', flush=True)
    requests_df = pd.DataFrame(request_audit).sort_values(['query', 'start_epoch', 'end_epoch'])
    requests_df.to_csv(ROOT / 'outputs/social_request_audit.csv', index=False)
    by_id = {}
    for record, query, retrieved in responses:
        key = str(record['objectID'])
        if key not in by_id:
            by_id[key] = {'record': record, 'queries': {query}, 'retrieved': retrieved}
        else:
            by_id[key]['queries'].add(query)
            # Deterministic snapshot: use the earliest retrieval for this item.
            if retrieved < by_id[key]['retrieved']:
                by_id[key]['record'], by_id[key]['retrieved'] = record, retrieved
    rows = []
    for ident, item in by_id.items():
        r = item['record']
        tags = r.get('_tags', [])
        kind = 'comment' if 'comment' in tags else ('story' if 'story' in tags else 'other')
        title = clean(r.get('title')) if kind == 'story' else ''
        body = clean(r.get('comment_text') if kind == 'comment' else r.get('story_text'))
        own_text = ' '.join(x for x in (title, body) if x)
        published = datetime.fromisoformat(r['created_at'].replace('Z', '+00:00'))
        author = r.get('author') or ''
        words = WORDS.findall(URLS.sub('', own_text))
        url_count = len(URLS.findall(own_text))
        norm = re.sub(r'\W+', ' ', own_text.lower()).strip()
        relevant = bool(RELEVANCE.search(own_text))
        rows.append({'id': f'hn_{ident}', 'date': published.astimezone(NY).date().isoformat(),
                     'published_at': published.isoformat(), 'published_at_et': published.astimezone(NY).isoformat(),
                     'title': title, 'text': own_text, 'url': f'https://news.ycombinator.com/item?id={ident}',
                     'source': 'Hacker News', 'platform': 'Hacker News', 'author': author,
                     'followers': None, 'views': None, 'score': r.get('points'), 'comments': r.get('num_comments'),
                     'record_type': kind, 'thread_id': r.get('story_id', int(ident) if kind == 'story' else None),
                     'thread_title': clean(r.get('story_title')), 'linked_url': r.get('url') or r.get('story_url'),
                     'timing_precision': 'second', 'retrieved_at': item['retrieved'],
                     'metrics_timing': 'retrieval snapshot; historical publication-time engagement unavailable',
                     'matched_queries': '|'.join(sorted(item['queries'])),
                     'own_text_ai_match': relevant, 'word_count': len(words), 'url_count': url_count,
                     'text_hash': digest(norm.encode()),
                     'dead_or_deleted': bool(r.get('dead') or r.get('deleted') or own_text.lower() in {'[dead]', '[deleted]'}),
                     'author_missing': not bool(author), 'bot_name_flag': bool(re.search(r'(?:^bot|bot$)', author, re.I)),
                     'link_spam_flag': url_count >= 4 and len(words) < 25,
                     'in_window': START <= published < cutoff})
    frame = pd.DataFrame(rows).sort_values(['published_at', 'id']).reset_index(drop=True)
    frame['duplicate_text'] = frame.duplicated(['record_type', 'text_hash'], keep='first')
    frame['author_day_records'] = frame.groupby(['author', 'date'])['id'].transform('size')
    frame['high_activity_flag'] = frame.author_day_records > 100
    frame['low_story_engagement'] = frame.record_type.eq('story') & (pd.to_numeric(frame.score, errors='coerce').fillna(0) < 5)
    frame['exclusion_reason'] = ''
    steps = [('out_of_window', ~frame.in_window),
             ('not_story_or_comment', ~frame.record_type.isin(['story', 'comment'])),
             ('dead_or_deleted', frame.dead_or_deleted), ('missing_author', frame.author_missing),
             ('no_ai_match_in_own_text', ~frame.own_text_ai_match),
             ('fewer_than_three_words', frame.word_count < 3),
             ('link_spam_rule', frame.link_spam_flag), ('duplicate_normalized_text', frame.duplicate_text)]
    audit_steps = []
    remaining = len(frame)
    for name, mask in steps:
        selected = mask & frame.exclusion_reason.eq('')
        count = int(selected.sum())
        frame.loc[selected, 'exclusion_reason'] = name
        remaining -= count
        audit_steps.append({'stage': name, 'removed_at_stage': count, 'remaining': remaining})
    frame['included_main'] = frame.exclusion_reason.eq('')
    frame['included_strict_sensitivity'] = (frame.included_main & ~frame.bot_name_flag &
                                           ~frame.high_activity_flag & ~frame.low_story_engagement & (frame.word_count >= 5))
    frame.to_csv(ROOT / 'data/processed/social_candidates.csv', index=False)
    selected = frame.loc[frame.included_main].copy()
    selected.to_csv(ROOT / 'data/processed/social.csv', index=False)
    pd.DataFrame(audit_steps).to_csv(ROOT / 'outputs/social_filter_flow.csv', index=False)
    selected.groupby(['date', 'record_type']).size().unstack(fill_value=0).to_csv(ROOT / 'outputs/social_daily_counts.csv')
    access = probe_access()
    summary = {**config, 'requested_window_complete': cutoff >= REQUESTED_END,
               'hit_instances_across_leaf_responses': len(responses), 'unique_candidate_ids': len(frame),
               'api_requests_saved': len(request_audit), 'leaf_requests': int(requests_df.used_as_leaf.sum()),
               'unresolved_truncated_leaf_requests': int(((requests_df.reported_hits > requests_df.returned_hits) & requests_df.used_as_leaf).sum()),
               'nonexhaustive_leaf_requests': int((requests_df.used_as_leaf & ~requests_df.exhaustive_nb_hits.fillna(False)).sum()),
               'clean_records': len(selected), 'clean_types': selected.record_type.value_counts().to_dict(),
               'strict_sensitivity_records': int(selected.included_strict_sensitivity.sum()),
               'strict_sensitivity_types': selected.loc[selected.included_strict_sensitivity, 'record_type'].value_counts().to_dict(),
               'unique_authors_clean': selected.author.nunique(), 'unique_threads_clean': selected.thread_id.nunique(),
               'filter_flow': audit_steps,
               'flags_clean': {col: int(selected[col].sum()) for col in ['bot_name_flag', 'high_activity_flag', 'low_story_engagement']},
               'missing_metrics_clean': {col: int(selected[col].isna().sum()) for col in ['followers', 'views', 'score', 'comments']},
               'other_platform_access': access,
               'coverage_limitations': [
                   'HN keyword-index retrieval is a technology-community convenience sample, not all social media or all AI discussion.',
                   'X, Reddit and Bluesky do not contribute records; preserved probes document access limits.',
                   'Followers and views are unavailable in the HN API; no fictitious low-follower or low-view filtering is applied.',
                   'Comment scores are not exposed by Algolia; story points/comment counts are retrieval-time snapshots.',
                   'Bot-name and >100 posts/author/day flags are imperfect heuristics, not proof of automation; they are excluded only in strict sensitivity.',
                   'No language classifier controls primary inclusion; any separate language audit is a sensitivity analysis, and multilingual residuals remain possible.',
                   ('All requested local dates are available at this freeze.' if cutoff >= REQUESTED_END else
                    (f'The requested market window is covered through the September 30 close; later September 30 evening publications are outside the {cutoff.astimezone(NY).isoformat()} freeze.' if cutoff >= REQUESTED_MARKET_END else
                     f'The requested window remains incomplete at {cutoff.astimezone(NY).isoformat()}; later publications are unavailable.')),
                   'For comments title is blank and text is the author comment only; thread_title is context, not sentiment input.',
                   'For stories text contains title plus submitted text, never linked article full text. Do not concatenate title again.',
                   'Strict engagement filtering can produce selection bias and is not the primary corpus.'],
               'social_csv_sha256': digest((ROOT / 'data/processed/social.csv').read_bytes())}
    (ROOT / 'outputs/social_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in {'other_platform_access', 'coverage_limitations'}}, indent=2))
    return selected


def validate_snapshot():
    """Independently reconcile candidate membership, text and dates to raw leaves."""
    request_audit = pd.read_csv(ROOT / 'outputs/social_request_audit.csv')
    expected = {}
    hashes_checked = 0
    for record in request_audit.to_dict('records'):
        path = ROOT / record['file']
        content = path.read_bytes()
        assert digest(content) == record['sha256'], f'Raw response hash mismatch: {path}'
        hashes_checked += 1
        if not record['used_as_leaf']:
            continue
        for item in json.loads(content)['hits']:
            ident = f"hn_{item['objectID']}"
            if ident not in expected or record['retrieved_utc'] < expected[ident][1]:
                expected[ident] = (item, record['retrieved_utc'])
    candidates = pd.read_csv(ROOT / 'data/processed/social_candidates.csv', keep_default_na=False, low_memory=False)
    selected = pd.read_csv(ROOT / 'data/processed/social.csv', keep_default_na=False, low_memory=False)
    assert candidates.id.is_unique and selected.id.is_unique
    assert set(expected) == set(candidates.id)
    assert set(selected.id) == set(candidates.loc[candidates.included_main, 'id'])
    for row in candidates.to_dict('records'):
        original = expected[row['id']][0]
        is_comment = 'comment' in original.get('_tags', [])
        title = clean(original.get('title')) if 'story' in original.get('_tags', []) else ''
        body = clean(original.get('comment_text') if is_comment else original.get('story_text'))
        assert row['text'] == ' '.join(x for x in (title, body) if x), row['id']
        timestamp = datetime.fromisoformat(original['created_at'].replace('Z', '+00:00'))
        assert row['published_at'] == timestamp.isoformat(), row['id']
        assert row['date'] == timestamp.astimezone(NY).date().isoformat(), row['id']
    config = json.loads((RAW / 'collection_config.json').read_text(encoding='utf-8'))
    cutoff = pd.Timestamp(config['cutoff_utc'])
    stamps = pd.to_datetime(selected.published_at, utc=True)
    assert stamps.max() < cutoff and stamps.min() >= pd.Timestamp(START)
    result = {'validated_utc': datetime.now(timezone.utc).isoformat(),
              'raw_response_sha256_checks': hashes_checked,
              'candidate_membership_matches_raw_leaf_union': True,
              'candidate_rows_checked_for_text_and_timestamp_equality': len(candidates),
              'main_membership_matches_documented_inclusion_flag': True,
              'main_unique_ids': len(selected), 'all_published_before_frozen_cutoff': True,
              'earliest_publication_utc': stamps.min().isoformat(),
              'latest_publication_utc': stamps.max().isoformat(),
              'september_30_et_records': int(selected.date.eq('2026-09-30').sum()),
              'status': 'PASS'}
    (ROOT / 'outputs/social_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    build()
