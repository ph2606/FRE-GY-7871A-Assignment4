from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = '2026-09-08'
REQUESTED_END = '2026-09-30'
# Analysis ends at the completed final cash-market close, not a future price.
CUTOFF = os.environ.get('ASSIGNMENT4_CUTOFF', '2026-09-30')
TICKERS = ['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AVGO']
BENCHMARKS = ['SPY', 'QQQ']
SEED = 787104

def validate_cutoff():
    now = pd.Timestamp.now(tz='America/New_York')
    if pd.Timestamp(CUTOFF).date() > now.date():
        raise ValueError('Cannot acquire observations for a future local date.')
    if CUTOFF > REQUESTED_END:
        raise ValueError('Cutoff exceeds assignment window.')
    return now
