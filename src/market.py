"""Cached Yahoo adjusted closes; basket weights fixed before return inspection."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import requests
import numpy as np
import pandas as pd
from .config import ROOT, CUTOFF, START, TICKERS, BENCHMARKS, validate_cutoff

def collect(refresh=False):
    now = validate_cutoff()
    raw = ROOT / 'data/raw/market'
    out = ROOT / 'data/processed'
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    # September 4 is the prior trading close; September 7 is Labor Day.
    begin = int(pd.Timestamp('2026-09-01', tz='America/New_York').timestamp())
    end = int((pd.Timestamp(CUTOFF, tz='America/New_York') + pd.Timedelta(days=1)).timestamp())
    def one(symbol):
        path = raw / f'{symbol}_{CUTOFF}.json'
        meta_path = path.with_suffix('.metadata.json')
        if refresh or not path.exists():
            response = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
                params={'period1': begin, 'period2': end, 'interval':'1d', 'events':'div,splits'},
                headers={'User-Agent':'Mozilla/5.0'}, timeout=45)
            response.raise_for_status()
            data = response.content
            if path.exists():
                history = raw / 'versions'
                history.mkdir(exist_ok=True)
                suffix = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
                (history / f'{path.stem}_{suffix}.json').write_bytes(path.read_bytes())
                if meta_path.exists():
                    (history / f'{path.stem}_{suffix}.metadata.json').write_bytes(meta_path.read_bytes())
            path.write_bytes(data)
            meta_path.write_text(json.dumps({'symbol':symbol, 'url':response.url,
                'retrieved_at':datetime.now(timezone.utc).isoformat(),
                'sha256':hashlib.sha256(data).hexdigest()}, indent=2), encoding='utf-8')
        data = path.read_bytes()
        meta = json.loads(meta_path.read_text())
        assert hashlib.sha256(data).hexdigest() == meta['sha256']
        result = json.loads(data)['chart']['result'][0]
        dates = pd.to_datetime(result['timestamp'], unit='s', utc=True).tz_convert('America/New_York').normalize().tz_localize(None)
        adjusted = result['indicators']['adjclose'][0]['adjclose']
        close = result['indicators']['quote'][0]['close']
        frame = pd.DataFrame({'date':dates, 'ticker':symbol, 'adjusted_close':adjusted, 'close':close})
        closes = frame.date.dt.tz_localize('America/New_York') + pd.Timedelta(hours=16)
        frame = frame.loc[closes.le(now) & frame.date.le(CUTOFF)].copy()
        assert not frame.date.duplicated().any()
        assert frame.adjusted_close.dropna().gt(0).all()
        meta.update(first=str(frame.date.min().date()), last=str(frame.date.max().date()), n=len(frame))
        return frame, meta
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, TICKERS + BENCHMARKS))
    levels = pd.concat([x[0] for x in results]).sort_values(['date','ticker'])
    levels.to_csv(out/'market_levels.csv', index=False)
    wide = levels.pivot(index='date', columns='ticker', values='adjusted_close')
    # This assignment's September 2026 cash calendar has one weekday holiday.
    # Restore a session missing from every feed so pct_change cannot bridge it.
    expected = pd.bdate_range('2026-09-01', CUTOFF)
    expected = expected[expected != pd.Timestamp('2026-09-07')]
    expected_closes = expected.tz_localize('America/New_York') + pd.Timedelta(hours=16)
    expected = expected[expected_closes <= now]
    unexpected = wide.index.difference(expected)
    if len(unexpected):
        raise ValueError(f'Market dates outside the completed September cash calendar: {unexpected.tolist()}')
    wide = wide.reindex(expected)
    wide.index.name = 'date'
    returns = 100 * wide.pct_change(fill_method=None)
    daily = pd.DataFrame(index=wide.index)
    daily['basket'] = returns[TICKERS].mean(axis=1).where(returns[TICKERS].notna().all(axis=1))
    daily['spy'] = returns.SPY
    daily['qqq'] = returns.QQQ
    daily['spread'] = daily.basket - daily.spy
    daily['spread_qqq'] = daily.basket - daily.qqq
    # A fixed-share basket initialized at the actual last pre-window close.
    baseline = wide.loc[wide.index < START].iloc[-1]
    bh = wide[TICKERS].div(baseline[TICKERS]).mean(axis=1,skipna=False)
    daily['buy_hold'] = 100 * bh.pct_change(fill_method=None)
    daily = daily.loc[START:CUTOFF]
    for col in ['basket','spy','qqq','buy_hold']:
        daily[col+'_wealth'] = (1+daily[col]/100).cumprod(skipna=False)
    daily.to_csv(out/'market_daily.csv', index_label='date')
    constituents = []
    for symbol in TICKERS+BENCHMARKS:
        last = wide.loc[:CUTOFF,symbol].dropna().iloc[-1]
        constituents.append({'ticker':symbol, 'weight':1/len(TICKERS) if symbol in TICKERS else 0,
                             'baseline_date':str(baseline.name.date()), 'baseline_adjusted':baseline[symbol],
                             'last_date':str(wide[symbol].dropna().index[-1].date()),
                             'last_adjusted':last, 'cumulative_return_pct':100*(last/baseline[symbol]-1)})
    pd.DataFrame(constituents).to_csv(out/'constituents.csv', index=False)
    manifest={'cutoff':CUTOFF, 'requested_end':'2026-09-30', 'completed_requested_window':CUTOFF=='2026-09-30',
              'weights':dict.fromkeys(TICKERS,1/len(TICKERS)), 'rebalancing':'daily at previous close',
              'units':'simple percent adjusted-close return; benchmark spread is not risk-adjusted alpha',
              'vintage':'latest adjusted prices at retrieval; no historical vintage claim', 'sources':[x[1] for x in results]}
    (out/'market_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(daily.round(3).to_string())
    return daily
