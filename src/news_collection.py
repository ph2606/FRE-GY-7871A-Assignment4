"""Auditable broad-AI headline sample from dated Google News RSS + Guardian.

Searches deliberately contain no risk, fear or negative-sentiment terms. Google
RSS is a ranked, capped discovery source, NOT a census. Guardian AI-tag coverage
is a second, single-outlet sample. Publisher times, retrieval times, text scope,
raw payload hashes and every exclusion are retained. No inferred bot labels.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit
import argparse
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/news'
START = pd.Timestamp('2026-09-08', tz='America/New_York').tz_convert('UTC')
PLANNED_END = pd.Timestamp('2026-10-01', tz='America/New_York').tz_convert('UTC')
QUERIES = {
    'general': '"artificial intelligence"',
    'openai': '(OpenAI OR ChatGPT)',
    'anthropic': '(Anthropic OR "Claude AI")',
    'basket': 'AI (Nvidia OR Meta OR Amazon OR Google OR Microsoft)',
}
AI = re.compile(r'\b(?:AI|artificial intelligence|OpenAI|ChatGPT|Anthropic|Claude|Gemini|'
                r'DeepMind|machine learning|large language model|LLMs?|chatbots?)\b', re.I)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def canonical(url):
    p = urlsplit(url)
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip('/'), '', ''))


def source_type(row):
    """Transparent domain/name flags; unknown outlets are not labeled fake/bots."""
    domain = urlsplit(str(row.get('source_url', ''))).netloc.lower().removeprefix('www.')
    name = str(row.get('source', '')).lower()
    if any(x in name for x in ['pr newswire', 'business wire', 'globenewswire', 'ein presswire', 'accesswire', 'prweb']):
        return 'press_release_wire'
    if domain.endswith('.gov') or '.gov.' in domain or domain.endswith('.mil'):
        return 'government_official'
    if domain.endswith('.edu') or '.ac.' in domain or any(x in name for x in ['university', 'brookings', 'rand corporation', 'council on foreign relations']):
        return 'academic_or_think_tank'
    corporate = ['openai.com', 'anthropic.com', 'microsoft.com', 'blog.google', 'nvidia.com',
                 'aboutamazon.com', 'aws.amazon.com', 'about.fb.com', 'research.google',
                 'huggingface.co', 'deepmind.google', 'ibm.com', 'salesforce.com', 'oracle.com']
    if any(domain == item or domain.endswith('.' + item) for item in corporate):
        return 'corporate_vendor'
    if any(domain == item or domain.endswith('.' + item) for item in ['youtube.com', 'reddit.com', 'facebook.com', 'instagram.com', 'x.com', 'tiktok.com']):
        return 'social_or_video_platform'
    return 'news_or_unclassified'


def fetch(url, path):
    meta_path = path.with_suffix(path.suffix + '.metadata.json')
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != meta['sha256']:
            raise ValueError(f'Cached payload hash mismatch: {path}')
        return payload, meta
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {'url': url, 'retrieved_at': utcnow(), 'errors': []}
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=45, headers={'User-Agent': 'Mozilla/5.0 (academic personal text analysis)'})
            meta.update(status=response.status_code, retrieved_at=utcnow())
            payload = response.content
            meta.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
            path.write_bytes(payload)
            if response.status_code != 200:
                meta['errors'].append(f'HTTP {response.status_code}')
            meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
            return payload, meta
        except requests.RequestException as exc:
            meta['errors'].append(str(exc))
            if attempt == 0:
                time.sleep(2)
    meta.update(status='failed', sha256=None, bytes=0)
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return b'', meta


def rss_job(job):
    day, family = job
    nextday = day + pd.Timedelta(days=1)
    query = f'{QUERIES[family]} after:{day:%Y-%m-%d} before:{nextday:%Y-%m-%d}'
    url = 'https://news.google.com/rss/search?' + urlencode({'q': query, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})
    path = RAW / 'google_rss' / family / f'{day:%Y-%m-%d}.xml'
    payload, meta = fetch(url, path)
    audit = dict(meta, provider='Google News RSS', query_date=f'{day:%Y-%m-%d}', family=family,
                 query=query, raw_path=path.relative_to(ROOT).as_posix(), returned=None, capped=None)
    rows = []
    if meta['status'] == 200:
        try:
            tree = ET.fromstring(payload)
            items = tree.findall('./channel/item')
            audit.update(returned=len(items), capped=len(items) >= 100)
            for item in items:
                source_element = item.find('source')
                source = (source_element.text or '').strip() if source_element is not None else ''
                source_url = source_element.get('url', '') if source_element is not None else ''
                title = (item.findtext('title') or '').strip()
                if source and title.endswith(' - ' + source):
                    title = title[:-len(' - ' + source)]
                original_date = item.findtext('pubDate') or ''
                try:
                    stamp = pd.Timestamp(parsedate_to_datetime(original_date)).tz_convert('UTC').isoformat()
                except (ValueError, TypeError):
                    stamp = ''
                link = item.findtext('link') or ''
                rows.append({'id': 'gn_' + hashlib.sha256(link.encode()).hexdigest()[:20],
                             'published_at': stamp, 'original_timestamp': original_date,
                             'timestamp_kind': 'publisher publication time supplied by Google News RSS',
                             'title': title, 'text': title, 'body': '', 'url': link,
                             'source': source, 'source_url': source_url, 'platform': 'news',
                             'language': 'en', 'language_basis': 'English Google News edition; not language-detector verified',
                             'country': '', 'country_basis': 'not inferred from Google US search edition',
                             'provider': 'Google News RSS', 'text_scope': 'headline', 'section': '',
                             'query_family': family, 'query_date': f'{day:%Y-%m-%d}',
                             'retrieved_at': meta['retrieved_at'], 'raw_path': path.relative_to(ROOT).as_posix(),
                             'sha256': meta['sha256'], 'exclusion': ''})
        except ET.ParseError as exc:
            audit.update(status='parse_failed', error=str(exc))
    time.sleep(.75)
    return rows, audit


def guardian_archive(day):
    url = f'https://www.theguardian.com/technology/artificialintelligenceai/{day:%Y/%b/%d}/all'.lower()
    path = RAW / 'guardian_archives' / f'{day:%Y-%m-%d}.html'
    payload, meta = fetch(url, path)
    rows = []
    if meta['status'] == 200:
        soup = BeautifulSoup(payload, 'html.parser')
        for anchor in soup.select('a[href]'):
            href = anchor['href']
            if href.startswith('/'):
                href = 'https://www.theguardian.com' + href
            if not href.startswith('https://www.theguardian.com/'):
                continue
            match = re.search(r'/2026/sep/(\d{2})/', href)
            if not match:
                continue
            title = anchor.get('aria-label', '') or anchor.get_text(' ', strip=True)
            if len(title.split()) < 4:
                continue
            rows.append({'url': canonical(href), 'archive_title': title,
                         'archive_date': f'{day:%Y-%m-%d}', 'url_date': f'2026-09-{match[1]}',
                         'archive_raw_path': path.relative_to(ROOT).as_posix()})
    audit = dict(meta, provider='Guardian AI tag', query_date=f'{day:%Y-%m-%d}',
                 family='broad_ai_tag', raw_path=path.relative_to(ROOT).as_posix(),
                 returned=len(rows), capped=False)
    return rows, audit


def guardian_article(discovery):
    url = discovery['url']
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    path = RAW / 'guardian_articles' / (key + '.html')
    row = dict(discovery, id='gu_' + key, platform='news', source='The Guardian',
               source_url='https://www.theguardian.com', language='en',
               language_basis='English publisher', country='UK', country_basis='publisher headquarters',
               provider='Guardian AI tag', timestamp_kind='publisher JSON-LD datePublished',
               text_scope='headline', section=urlsplit(url).path.split('/')[1],
               published_at='', original_timestamp='', title=discovery['archive_title'],
               text=discovery['archive_title'], body='', modified_at='', exclusion='')
    # Live blogs and audiovisual pages do not have comparable single-story text.
    if any(marker in url for marker in ['/live/', '/video/', '/audio/', '/gallery/', '/picture/']):
        row['exclusion'] = 'live blog or audiovisual/gallery format'
        return row
    payload, meta = fetch(url, path)
    if meta['status'] in [429, 500, 502, 503, 504]:
        # Preserve the failed response and make one paced, separately cached retry.
        time.sleep(3)
        retry_path = path.with_name(path.stem + '.retry1.html')
        retry_payload, retry_meta = fetch(url, retry_path)
        if retry_meta['status'] == 200:
            payload, meta, path = retry_payload, retry_meta, retry_path
    row.update(retrieved_at=meta['retrieved_at'], raw_path=path.relative_to(ROOT).as_posix(), sha256=meta.get('sha256'))
    if meta['status'] != 200:
        row['exclusion'] = 'article retrieval failure'
        return row
    soup = BeautifulSoup(payload, 'html.parser')
    metadata = {}
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            obj = json.loads(script.string or script.get_text())
            queue = obj if isinstance(obj, list) else [obj]
            while queue:
                item = queue.pop(0)
                if not isinstance(item, dict):
                    continue
                queue.extend(item.get('@graph', []))
                kind = item.get('@type', '')
                if isinstance(kind, str) and ('Article' in kind or kind == 'LiveBlogPosting'):
                    metadata = item
                    break
        except (ValueError, TypeError):
            continue
        if metadata:
            break
    title = metadata.get('headline') or (soup.select_one('h1').get_text(' ', strip=True) if soup.select_one('h1') else row['title'])
    stamp = metadata.get('datePublished', '')
    paragraphs = soup.select('[data-gu-name="body"] p') or soup.select('div.article-body-commercial-selector p')
    body = '\n'.join(p.get_text(' ', strip=True) for p in paragraphs) or metadata.get('articleBody', '')
    row.update(title=title, text=title, body=body, body_words=len(re.findall(r'\b\w+\b', body)),
               published_at=stamp, original_timestamp=stamp, modified_at=metadata.get('dateModified', ''),
               content_vintage='Body/headline as retrieved; historical unedited vintage is unavailable')
    return row


def resolve_cutoff(requested=None, *, saved=None, refresh=False, now=None):
    """Interpret YYYY-MM-DD as an inclusive New York day, never future data."""
    current = pd.Timestamp(now or utcnow()).tz_convert('UTC')
    if saved is not None and not refresh:
        return pd.Timestamp(saved).tz_convert('UTC')
    if refresh and not requested:
        raise ValueError('--refresh requires ASSIGNMENT4_CUTOFF=YYYY-MM-DD or an explicit --cutoff.')
    if requested and re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(requested)):
        day = pd.Timestamp(requested, tz='America/New_York')
        if not pd.Timestamp('2026-09-08', tz='America/New_York') <= day <= pd.Timestamp('2026-09-30', tz='America/New_York'):
            raise ValueError('Inclusive cutoff date must be 2026-09-08 through 2026-09-30.')
        desired = (day + pd.Timedelta(days=1)).tz_convert('UTC')
    else:
        desired = pd.Timestamp(requested or current)
        if desired.tzinfo is None:
            raise ValueError('Explicit timestamp cutoff must include timezone; a date alone means an inclusive New York day.')
        desired = desired.tz_convert('UTC')
    if desired > PLANNED_END:
        raise ValueError('Cutoff exceeds the requested assignment window.')
    if refresh and desired > current:
        raise ValueError('Cannot refresh through the requested cutoff: that America/New_York calendar day or timestamp has not completed.')
    if refresh and saved is not None and desired <= pd.Timestamp(saved):
        raise ValueError('--refresh must extend the saved publication cutoff.')
    return min(desired, PLANNED_END, current)


def run(cutoff=None, workers=3, refresh=False):
    global RAW
    base_raw = ROOT / 'data/raw/news'
    if (base_raw / 'active_collection.json').exists():
        active = json.loads((base_raw / 'active_collection.json').read_text(encoding='utf-8'))
        RAW = ROOT / active['raw_directory']
    else:
        RAW = base_raw
    old_freeze = RAW / 'collection_freeze.json'
    saved = json.loads(old_freeze.read_text())['cutoff_utc'] if old_freeze.exists() else None
    freeze = resolve_cutoff(cutoff or os.environ.get('ASSIGNMENT4_CUTOFF'), saved=saved, refresh=refresh)
    if refresh:
        # Validate before writing anything; preserve every prior raw vintage.
        RAW = base_raw / 'vintages' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    RAW.mkdir(parents=True, exist_ok=True)
    for folder in ['outputs', 'data/processed']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    freeze_path = RAW / 'collection_freeze.json'
    freeze_path.write_text(json.dumps({'cutoff_utc': freeze.isoformat(), 'start_utc': START.isoformat(),
                                    'planned_end_exclusive_utc': PLANNED_END.isoformat()}, indent=2), encoding='utf-8')
    (base_raw / 'active_collection.json').write_text(json.dumps({
        'raw_directory': RAW.relative_to(ROOT).as_posix(), 'cutoff_utc': freeze.isoformat(),
        'refresh': refresh, 'previous_vintages': 'Preserved on disk; --refresh never deletes historical raw files.'
    }, indent=2), encoding='utf-8')
    # Date queries can return adjacent-day publication dates. Buffer discovery by
    # one day, then enforce inclusion using the returned publication clock.
    days = pd.date_range((START.normalize() - pd.Timedelta(days=1)).tz_localize(None), freeze.normalize().tz_localize(None))
    records, acquisition, discoveries = [], [], []
    jobs = [(day, family) for day in days for family in QUERIES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(rss_job, job): job for job in jobs}
        for number, task in enumerate(as_completed(futures), 1):
            rows, audit = task.result()
            records.extend(rows)
            acquisition.append(audit)
            if number % 8 == 0:
                print(f'Google RSS {number}/{len(jobs)} requests; {len(records)} rows', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rows, audit in pool.map(guardian_archive, days):
            discoveries.extend(rows)
            acquisition.append(audit)
    discovered = pd.DataFrame(discoveries)
    discovered.to_csv(RAW / 'guardian_discovery_all.csv', index=False)
    if len(discovered):
        discovered = discovered.drop_duplicates('url')
        discovered = discovered.loc[discovered.url_date.between('2026-09-08', freeze.tz_convert('America/New_York').strftime('%Y-%m-%d'))]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for number, row in enumerate(pool.map(guardian_article, discovered.to_dict('records')), 1):
                records.append(row)
                if number % 30 == 0:
                    print(f'Guardian articles {number}/{len(discovered)}', flush=True)
    # The failed access probe is retained; service failure is never zero coverage.
    probe = 'https://api.gdeltproject.org/api/v2/doc/doc?' + urlencode({
        'query': '"artificial intelligence" sourcelang:english', 'mode': 'artlist', 'format': 'json',
        'maxrecords': 250, 'startdatetime': '20260908000000', 'enddatetime': '20260909000000', 'sort': 'datedesc'})
    payload, meta = fetch(probe, RAW / 'gdelt_access_probe.txt')
    acquisition.append(dict(meta, provider='GDELT DOC access probe', query_date='2026-09-08', family='access_probe', returned=None, capped=None))
    frame = pd.DataFrame(records).fillna('')
    frame.to_csv(RAW / 'all_discovered_records.csv', index=False)
    if frame.empty:
        raise RuntimeError('No records acquired; cannot fabricate a corpus')
    stamp = pd.to_datetime(frame.published_at, utc=True, errors='coerce', format='mixed')
    frame['published_at'] = stamp.map(lambda value: value.isoformat() if pd.notna(value) else '')
    frame['date'] = stamp.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d').fillna('')
    guardian = frame.source.str.lower().eq('the guardian')
    frame.loc[guardian, 'title'] = frame.loc[guardian, 'title'].str.replace(r'\s+\|\s+(?:AI \(artificial intelligence\)|Technology|Science|US news|Business).*$', '', regex=True)
    frame['text'] = frame.title
    is_rss = frame.provider.eq('Google News RSS')
    frame['timestamp_precision'] = is_rss.map({True: 'date_only_conservative', False: 'publisher_datetime'})
    available = stamp.copy()
    next_midnight = (stamp.dt.tz_convert('America/New_York').dt.normalize() + pd.Timedelta(days=1)).dt.tz_convert('UTC')
    available.loc[is_rss] = next_midnight.loc[is_rss]
    modified = pd.to_datetime(frame.modified_at, utc=True, errors='coerce', format='mixed')
    use_modified = ~is_rss & modified.notna() & modified.gt(stamp)
    available.loc[use_modified] = modified.loc[use_modified]
    frame['conservative_available_at'] = available.map(lambda value: value.isoformat() if pd.notna(value) else '')
    frame['causal_time_eligible'] = available.notna() & available.le(freeze)
    frame['has_explicit_ai_headline'] = frame.title.map(lambda text: bool(AI.search(text)))
    frame['source_type'] = frame.apply(source_type, axis=1)
    frame['media_eligible'] = frame.source_type.eq('news_or_unclassified')
    for mask, reason in [(stamp.isna(), 'missing or invalid publication timestamp'),
                         ((stamp < START) | (stamp > freeze), 'outside frozen information window'),
                         (~is_rss & modified.gt(freeze), 'publisher modification after frozen cutoff'),
                         (frame.title.str.split().str.len().lt(4), 'headline shorter than four words')]:
        frame.loc[frame.exclusion.eq('') & mask, 'exclusion'] = reason
    # Search can match article bodies. Keep all such items in audit, but require
    # explicit AI headline relevance for a consistent headline-only analysis.
    frame.loc[frame.exclusion.eq('') & ~frame.has_explicit_ai_headline, 'exclusion'] = 'no explicit AI term in headline'
    frame['headline_key'] = frame.title.str.lower().str.replace(r'[^\w]+', ' ', regex=True).str.strip()
    frame['source_key'] = frame.source.str.lower().str.replace(r'^the\s+', '', regex=True).str.strip()
    # Do not let an RSS copy bypass a publisher's known post-cutoff modification.
    late_keys = set(frame.loc[~is_rss & modified.gt(freeze), ['source_key', 'headline_key']].itertuples(index=False, name=None))
    if late_keys:
        known_late = pd.Series([key in late_keys for key in frame[['source_key', 'headline_key']].itertuples(index=False, name=None)], index=frame.index)
        frame.loc[frame.exclusion.eq('') & known_late, 'exclusion'] = 'publisher modification after frozen cutoff'
    # Prefer publisher-hosted Guardian records over Google copies when equivalent.
    frame['provider_priority'] = frame.provider.map({'Guardian AI tag': 0, 'Google News RSS': 1}).fillna(2)
    frame = frame.sort_values(['provider_priority', 'published_at', 'id', 'query_family'], kind='stable')
    eligible = frame.exclusion.eq('')
    duplicates = frame.loc[eligible].duplicated(['source_key', 'headline_key'], keep='first')
    frame.loc[duplicates[duplicates].index, 'exclusion'] = 'duplicate normalized headline within publisher'
    frame.to_csv(ROOT / 'outputs/news_filter_audit.csv', index=False)
    included = frame.loc[frame.exclusion.eq('')].sort_values(['published_at', 'id'])
    included.to_csv(ROOT / 'data/processed/news.csv', index=False)
    source_audit = included.groupby(['source', 'source_url', 'source_type'], dropna=False).size().reset_index(name='headlines')
    source_audit.sort_values('headlines', ascending=False).to_csv(ROOT / 'outputs/news_source_audit.csv', index=False)
    pd.DataFrame(acquisition).to_csv(ROOT / 'outputs/news_acquisition_audit.csv', index=False)
    coverage = included.groupby(['date', 'provider']).size().unstack(fill_value=0)
    coverage.to_csv(ROOT / 'outputs/news_daily_coverage.csv')
    article_response_metadata = [json.loads(path.read_text(encoding='utf-8')) for path in (RAW / 'guardian_articles').glob('*.metadata.json')]
    summary = {
        'cutoff_utc': freeze.isoformat(), 'cutoff_new_york': freeze.tz_convert('America/New_York').isoformat(),
        'start_inclusive': START.isoformat(), 'planned_end_exclusive': PLANNED_END.isoformat(),
        'window_complete': freeze >= PLANNED_END, 'broad_queries': QUERIES,
        'raw_discovered_rows': len(frame), 'included_headlines': len(included),
        'unique_publishers': int(included.source.nunique()), 'per_provider': included.provider.value_counts().to_dict(),
        'source_type_counts': included.source_type.value_counts().to_dict(),
        'media_eligible_headlines': int(included.media_eligible.sum()),
        'conservative_causal_time_eligible': int(included.causal_time_eligible.sum()),
        'exclusions': frame.loc[frame.exclusion.ne(''), 'exclusion'].value_counts().to_dict(),
        'requests': len(acquisition), 'request_statuses': pd.Series([str(a['status']) for a in acquisition]).value_counts().to_dict(),
        'requests_scope': 'Discovery requests (RSS and Guardian archives) plus GDELT probe; article responses counted separately below.',
        'article_responses_saved': len(article_response_metadata),
        'article_response_statuses': pd.Series([str(a['status']) for a in article_response_metadata]).value_counts().to_dict(),
        'google_rss_capped_requests': sum(a.get('capped') is True for a in acquisition if a['provider'] == 'Google News RSS'),
        'google_rss_requests': len(jobs), 'google_rss_practical_cap': 100,
        'selection': 'Four fixed broad-AI query families across dates. English US Google News search edition plus Guardian AI tag. No negative/risk search terms.',
        'main_text': 'Headline only for comparable scope; Guardian bodies retained separately, never mislabeled as historical real-time text.',
        'limits': ['Ranked/capped RSS results are not a census; retrieval may change on rerun.',
                   'Publisher times supplied by RSS are not independently verified; no precision inferred when absent.',
                   'Conservative availability: Google RSS at next New York midnight; Guardian at max(datePublished,dateModified). Raw supplied timestamps retained.',
                   'Google US edition is not publisher country. Nationality remains unknown unless publisher known.',
                   'Exact headline/publisher duplicates removed; related event reporting and syndication can remain.',
                   'Search matched-body stories without explicit AI headline terms are excluded, a relevance/coverage tradeoff.',
                   f'Frozen through {freeze.tz_convert("America/New_York").isoformat()}; later publications are unobserved, not zero-sentiment observations.',
                   'No follower/view/bot filters apply to newsroom headlines; these fields are unavailable.'],
    }
    (ROOT / 'outputs/news_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return included


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', default=None, help='Timezone-aware frozen timestamp; defaults to saved freeze or now')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--refresh', action='store_true', help='Requery into a new raw vintage; preserve all prior payloads')
    args = parser.parse_args()
    run(args.cutoff, args.workers, args.refresh)
