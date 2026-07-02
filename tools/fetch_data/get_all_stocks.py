#!/usr/bin/env python3
"""
Fetch all A-share stock codes for a given date and write them to results/all_stock_{date}.csv.

Usage:
    python get_all_stocks.py --date 2026-04-15
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.baostock_client import BaostockClient

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_all_stock(date: str) -> pd.DataFrame:
    client = BaostockClient()
    rs = client.get_stock_list(date)
    if rs is None:
        raise RuntimeError(f"Failed to fetch stock list for {date}")

    data_list = []
    while rs.error_code == "0" and rs.next():
        data_list.append(rs.get_row_data())

    df = pd.DataFrame(data_list, columns=rs.fields)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch all A-share stock codes for a date")
    parser.add_argument("--date", required=True, help="Query date, format YYYY-MM-DD")
    args = parser.parse_args()

    df = fetch_all_stock(args.date)
    # Keep Shanghai 6xx/688 and Shenzhen 0xx/002/300 codes
    mask = df["code"].str.startswith(("sh.6", "sz.0"))
    df = df[mask].copy()
    output_path = RESULTS_DIR / f"all_stock_{args.date}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} stocks to {output_path}")


if __name__ == "__main__":
    main()
