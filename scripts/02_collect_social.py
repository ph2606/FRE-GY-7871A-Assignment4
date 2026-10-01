"""Collect frozen, auditable HN social corpus; reuse cached raw responses."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.social_collection import build, validate_snapshot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true',
                        help='Extend cutoff to observed timestamp or completed ASSIGNMENT4_CUTOFF local day; archive previous snapshot.')
    parser.add_argument('--cutoff-utc',
                        help='Exclusive observed UTC cutoff, e.g. 2026-09-30T20:00:00Z; future timestamps rejected.')
    args = parser.parse_args()
    build(refresh=args.refresh, cutoff_utc=args.cutoff_utc)
    validate_snapshot()
