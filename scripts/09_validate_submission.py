"""Validate actual submission artifacts, counts and frozen observation clocks."""
from pathlib import Path
import hashlib
import json
import re
import fitz
import nbformat
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'outputs/summary.json').read_text())
assert summary['cutoff']=='2026-09-30'
assert summary['sessions']==17
pdf=fitz.open(ROOT/'Panagiotis_Housos_Assignment4.pdf')
assert len(pdf)==6,f'Expected six pages, found {len(pdf)}.'
text='\n'.join(p.get_text() for p in pdf)
# Cambria's PDF extraction maps text hyphens to nonbreaking hyphens.
numeric_text=text.translate(str.maketrans({'\u2011':'-','\u2010':'-','\u2212':'-'}))
assert 'September 30 pending' not in text
assert 'provisional submission' not in text.lower()
for p in pdf:
    assert len(p.get_text().split())>200,'Unexpected empty/overflow page.'
    assert not any(r.x0<0 or r.y0<0 or r.x1>p.rect.width+1 or r.y1>p.rect.height+1
                   for r in [fitz.Rect(b[:4]) for b in p.get_text('blocks')]),'Clipped content.'
nb=nbformat.read(ROOT/'assignment4.ipynb',as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code),'Unexecuted notebook cells.'
assert not [o for c in code for o in c.get('outputs',[]) if o.output_type=='error']
docs=pd.read_csv(ROOT/'data/processed/scored_documents.csv',low_memory=False)
included=docs.loc[docs.included_analysis]
assert not included.id.duplicated().any()
assert len(included)==sum(summary['documents'].values())
published=pd.to_datetime(included.published_at,utc=True,format='mixed')
assert published.max()<=pd.Timestamp('2026-09-30T20:00:00Z')
market=pd.read_csv(ROOT/'data/processed/market_daily.csv',index_col='date',parse_dates=True)
assert market.index[-1]==pd.Timestamp('2026-09-30')
assert market[['basket','spy','qqq']].notna().all().all()
core=pd.read_csv(ROOT/'outputs/granger_core.csv')
main=core.loc[core.outcome.eq('spread')&core.lag.eq(1)].iloc[0]
assert f'{main.effect_per_01:.3f}' in numeric_text,'Primary effect missing from PDF.'
assert f'{main.p_holm:.3f}' in numeric_text,'Primary adjusted p missing from PDF.'
news=json.loads((ROOT/'outputs/news_manifest.json').read_text())
social=json.loads((ROOT/'outputs/social_audit.json').read_text())
assert news['raw_discovered_rows']==news['included_headlines']+sum(news['exclusions'].values())
assert social['unique_candidate_ids']==social['clean_records']+sum(s['removed_at_stage'] for s in social['filter_flow'])
assert social['unresolved_truncated_leaf_requests']==0
result={'pdf_pages':len(pdf),'executed_code_cells':len(code),'notebook_errors':0,
        'included_documents':len(included),'market_sessions':len(market),
        'primary_regression_observations':int(main.n),'all_checks_passed':True,
        'report_sha256':hashlib.sha256((ROOT/'Panagiotis_Housos_Assignment4.pdf').read_bytes()).hexdigest()}
(ROOT/'outputs/submission_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
