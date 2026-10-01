"""Pin project dependency closure and record hashes without distributing data."""
from pathlib import Path
from importlib import metadata
import hashlib
import json
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]
DIRECT=['numpy','pandas','scipy','statsmodels','matplotlib','requests','beautifulsoup4',
        'vaderSentiment','nbformat','nbclient','ipykernel','pytest','PyMuPDF','langdetect']
seen={}
def add(name):
    key=canonicalize_name(name)
    if key in seen:return
    dist=metadata.distribution(name);seen[key]=dist.version
    for value in dist.requires or []:
        req=Requirement(value)
        if req.marker is None or req.marker.evaluate({'extra':''}):add(req.name)

for name in DIRECT:add(name)
(ROOT/'requirements.txt').write_text('\n'.join(f'{x}=={metadata.version(x)}' for x in DIRECT)+'\n')
(ROOT/'requirements-lock.txt').write_text('# Python 3.12; installed dependency closure for this project.\n'+
    '\n'.join(f'{k}=={v}' for k,v in sorted(seen.items()))+'\n')
s=json.loads((ROOT/'outputs/summary.json').read_text())
lines=['# Submitted snapshot','',f"Analysis dates: 2026-09-08 through {s['cutoff']}; text stops at 16:00 New York on the final day.",
       f"Market sessions: {s['sessions']}. Corpus counts: "+'; '.join(f'{k}: {v:,}' for k,v in s['documents'].items())+'.','',
       'Dataset hashes are identifiers, not distributed datasets. Public-source reruns can differ because texts, rankings, engagement and price adjustments change.',
       '', '| Local input | SHA256 |','| --- | --- |']
for name in ['news.csv','social.csv','market_levels.csv','scored_documents.csv','language_audit.csv']:
    p=ROOT/'data/processed'/name
    lines.append(f'| {name} | {hashlib.sha256(p.read_bytes()).hexdigest()} |')
lines += ['', 'The raw/derived inputs remain local. The executed notebook preserves aggregate exhibits only.']
(ROOT/'SNAPSHOT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'Pinned {len(seen)} packages and recorded snapshot hashes.')
