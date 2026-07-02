#!/usr/bin/env python3
"""
Read stock codes from results/stocks_from_index_all.csv and fetch K-line data for them.

Usage:
    python tools/fetch_data/get_stocks_from_index.py                      # daily only (default)
    python tools/fetch_data/get_stocks_from_index.py --force              # overwrite existing data
    python tools/fetch_data/get_stocks_from_index.py --freq w m           # weekly + monthly
    python tools/fetch_data/get_stocks_from_index.py --freq d w m         # daily + weekly + monthly
    python tools/fetch_data/get_stocks_from_index.py --codes path/to/codes.csv
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
    parser = argparse.ArgumentParser(description="Sync index constituent stocks K-line data")
    parser.add_argument("--force", action="store_true", help="Overwrite existing data (default: skip)")
    parser.add_argument(
        "--freq",
        type=str,
        nargs="+",
        default=[],
        choices=["d", "w", "m"],
        help="Additional frequencies to fetch (daily is always synced)",
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

    # Always sync daily (default frequency)
    print("\n--- Syncing d ---")
    count = sync_stock_from_codes(codes, frequency="d", force=args.force)
    print(f"Stock sync (d) completed: {count} rows")

    # Sync additional frequencies if specified
    for freq in args.freq:
        print(f"\n--- Syncing {freq} ---")
        count = sync_stock_from_codes(codes, frequency=freq, force=args.force)
        print(f"Stock sync ({freq}) completed: {count} rows")


if __name__ == "__main__":
    main()
