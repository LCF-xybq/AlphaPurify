#!/usr/bin/env python3
"""
Read stock codes from results/stocks_from_index_all.csv and fetch minute-level K-line data for them.

Usage:
    python tools/fetch_data/get_stocks_from_index_min.py                    # 5-minute bars (default)
    python tools/fetch_data/get_stocks_from_index_min.py --force            # overwrite existing data
    python tools/fetch_data/get_stocks_from_index_min.py --minutes 10 15    # also fetch 10- and 15-minute bars
    python tools/fetch_data/get_stocks_from_index_min.py --codes path/to/codes.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.data.stock import sync_stock_from_codes

INDEX_STOCKS_CSV = PROJECT_ROOT / "results" / "stocks_from_index_all.csv"


def main():
    parser = argparse.ArgumentParser(description="Sync index constituent stocks minute K-line data")
    parser.add_argument("--force", action="store_true", help="Overwrite existing data (default: skip)")
    parser.add_argument(
        "--minutes",
        type=int,
        nargs="+",
        default=[],
        choices=[5, 10, 15, 30, 60],
        help="Additional minute frequencies to fetch (5-minute is always synced)",
    )
    parser.add_argument(
        "--codes",
        default=str(INDEX_STOCKS_CSV),
        help=f"Stock codes CSV file (default: {INDEX_STOCKS_CSV})",
    )
    args = parser.parse_args()

    csv_path = Path(args.codes)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        print("Run first: python tools/fetch_data/get_index_stocks.py all")
        return

    df = pd.read_csv(csv_path)
    codes = df["code"].tolist()
    print(f"Loaded {len(codes)} stocks from {csv_path}")

    # Always sync 5min (default frequency)
    print("\n--- Syncing 5min ---")
    count = sync_stock_from_codes(codes, frequency="5", force=args.force)
    print(f"Stock sync (5min) completed: {count} rows")

    # Sync additional minute frequencies if specified
    for freq in args.minutes:
        print(f"\n--- Syncing {freq}min ---")
        count = sync_stock_from_codes(codes, frequency=str(freq), force=args.force)
        print(f"Stock sync ({freq}min) completed: {count} rows")


if __name__ == "__main__":
    main()
