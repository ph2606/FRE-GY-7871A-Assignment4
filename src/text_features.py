"""Frozen lexical sentiment and explicit, auditable AI-risk topic rules.

Scores measure wording, not factual risk probabilities or the writer's stance.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile
import numpy as np
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .config import ROOT, START, CUTOFF

TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
AI = re.compile(r'\b(?:AI|AGI|LLMs?|GPT(?:[- ]?\d\w*)?|ChatGPT|artificial intelligence|machine learning|large language models?|generative (?:AI|models?)|superintelligen\w*|Anthropic|OpenAI|Claude|Gemini|DeepSeek|Hubinger|Coxon)\b', re.I)
TOPICS = {
 'existential': r'\b(?:extinct\w*|existential|superintelligen\w*|alignment|kill (?:us|everyone|all|human\w*)|end of humanity|human extinction|doom\w*|catastroph\w*|loss of control|self.improv\w*)\b',
 'misuse': r'\b(?:cyber\w*|hack\w*|deepfake\w*|disinform\w*|misinform\w*|weapon\w*|biosecurity|bioweapon\w*|fraud\w*|scam\w*|misuse|malicious|blackmail\w*|surveill\w*|rogue|decept\w*|deceiv\w*)\b',
 'societal': r'\b(?:layoffs?|job losses?|job displacement|unemploy\w*|copyright|discriminat\w*|privacy|suicid\w*|psychosis|mental health|safety|bias(?:ed|es)?|addict\w*|harm\w*|unsafe|danger\w*|threat\w*|AI risks?)\b',
}
TOPIC_PATTERNS = {k:re.compile(v,re.I) for k,v in TOPICS.items()}
MITIGATION = re.compile(r'\b(?:safeguard\w*|oversight|guardrail\w*|regulat\w*|safety tests?|safety evaluat\w*|alignment|risk reduc\w*|mitigat\w*|audits?|monitor\w*)\b',re.I)

def lexicon():
    folder = ROOT/'data/raw/lexicons'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'opinion_lexicon.zip'
    url='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/opinion_lexicon.zip'
    if not path.exists():
        r=requests.get(url,timeout=45);r.raise_for_status();path.write_bytes(r.content)
        (folder/'hu_liu_metadata.json').write_text(json.dumps({'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),
            'sha256':hashlib.sha256(r.content).hexdigest(),'citation':'Hu and Liu (2004), Mining and Summarizing Customer Reviews'},indent=2))
    meta=json.loads((folder/'hu_liu_metadata.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as z:
        lists={}
        for name in ['positive','negative']:
            txt=z.read(f'opinion_lexicon/{name}-words.txt').decode('latin-1')
            lists[name]={s.strip() for s in txt.splitlines() if s.strip() and not s.startswith(';')}
    return lists

def score_text(text, analyzer, lists):
    text=str(text)
    words=TOKEN.findall(text.lower())
    n=len(words)
    v=analyzer.polarity_scores(text)['compound'] if n else np.nan
    return {'vader':v,'negative':float(v<=-0.05) if n else np.nan,
            'positive':float(v>=0.05) if n else np.nan, 'n_words':n,
            'hu_negative':sum(w in lists['negative'] for w in words)/n if n else np.nan,
            'hu_positive':sum(w in lists['positive'] for w in words)/n if n else np.nan,
            **{k:bool(p.search(text)) for k,p in TOPIC_PATTERNS.items()},
            'mitigation':bool(MITIGATION.search(text))}

def build():
    out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for kind in ['news','social']:
        f=pd.read_csv(out/f'{kind}.csv',low_memory=False).fillna('')
        f['kind']=kind
        # Collector outputs retain their own exclusions, if any.
        if 'included' in f:
            f=f.loc[f.included.astype(str).str.lower().isin(['true','1'])].copy()
        if 'exclusion' in f:
            f=f.loc[f.exclusion.eq('')].copy()
        frames.append(f)
    data=pd.concat(frames,ignore_index=True)
    data['published_at']=pd.to_datetime(data.published_at,utc=True,errors='coerce',format='mixed')
    data['date']=data.published_at.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    # Article bodies are used only for a separately labeled robustness analysis.
    data['analysis_text']=np.where(data.kind.eq('news'),data.title,data.text)
    data['analysis_text']=data.analysis_text.fillna('').astype(str)
    data['canonical_text']=data.analysis_text.str.lower().str.replace(r'https?://\S+','',regex=True).str.replace(r'[^a-z0-9]+',' ',regex=True).str.strip()
    available=pd.to_datetime(data.get('conservative_available_at',pd.Series(pd.NaT,index=data.index)),utc=True,format='mixed',errors='coerce')
    data['dedup_clock']=available.fillna(data.published_at)
    data=data.sort_values(['dedup_clock','id'],kind='stable').reset_index(drop=True)
    data['analysis_exclusion']=''
    def exclude(mask,label):
        data.loc[mask & data.analysis_exclusion.eq(''),'analysis_exclusion']=label
    exclude(data.published_at.isna(),'missing_publication_time')
    exclude(data.date.lt(START)|data.date.gt(CUTOFF),'outside_local_window')
    if 'media_eligible' in data:
        exclude(data.kind.eq('news') & data.media_eligible.astype(str).str.lower().isin(['false','0']), 'known_nonmedia_source')
    exclude(data.analysis_text.str.findall(TOKEN).str.len().lt(5),'under_5_words')
    exclude(~data.analysis_text.str.contains(AI),'no_AI_anchor_in_scored_text')
    # Exact copies count once within medium; cross-medium overlap remains visible.
    eligible=data.analysis_exclusion.eq('')
    duplicate=pd.Series(False,index=data.index)
    duplicate.loc[eligible]=data.loc[eligible].duplicated(['kind','canonical_text'])
    exclude(duplicate,'exact_text_duplicate_within_medium')
    data['is_comment']=data.get('item_type',pd.Series('',index=data.index)).eq('comment')
    # Support alternate collector label without guessing from text.
    if 'type' in data:
        data['is_comment'] |= data['type'].eq('comment')
    if 'record_type' in data:
        data['is_comment'] |= data['record_type'].eq('comment')
    lists=lexicon();analyzer=SentimentIntensityAnalyzer()
    scores=pd.DataFrame([score_text(x,analyzer,lists) for x in data.analysis_text],index=data.index)
    data=pd.concat([data,scores],axis=1)
    data['risk']=data[list(TOPICS)].any(axis=1)
    data['topic']=np.select([data.existential,data.misuse,data.societal],['existential','misuse','societal'],default='other_AI')
    data['included_analysis']=data.analysis_exclusion.eq('')
    data.to_csv(out/'scored_documents.csv',index=False)
    data.groupby(['kind','analysis_exclusion'],dropna=False).size().rename('n').reset_index().to_csv(out/'analysis_filter_audit.csv',index=False)
    sample=data.loc[data.included_analysis].sample(frac=1,random_state=787104).groupby(['kind','risk']).head(15)
    sample[['id','kind','analysis_text','risk','topic','vader','negative','hu_negative','url']].to_csv(out/'text_review_sample.csv',index=False)
    spec={'tokenizer':TOKEN.pattern,'ai_anchor':AI.pattern,'risk_topics':TOPICS,'mitigation':MITIGATION.pattern,
          'sentiment':'VADER 3.3.2 compound; negative <= -0.05, positive >= 0.05; untrained on this corpus',
          'lexical':'Hu-Liu negative word count / all alphabetic tokens; no stopword deletion or stemming',
          'warning':'Topic regexes retrieve risk language, not endorsement, factual probability, or independently validated stance.'}
    (out/'text_method.json').write_text(json.dumps(spec,indent=2))
    print(data.groupby(['kind','included_analysis']).size().to_string())
    return data

if __name__=='__main__':build()
