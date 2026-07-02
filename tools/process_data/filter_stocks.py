#!/usr/bin/env python3
"""
Filter stock universe for factor analysis and ML training.

Removes low-price, ST, newly listed, and illiquid stocks.

Usage:
    python tools/process_data/filter_stocks.py
    python tools/process_data/filter_stocks.py --close 30
    python tools/process_data/filter_stocks.py --close 60 --min_amount 1000 --min_days 120
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger

from core.data.db import get_duckdb_connection

DATA_DIR = PROJECT_ROOT / "data"


def filter_stocks(
    max_close: float = 30,
    min_amount: float = 20000,
    min_days: int = 250,
) -> pd.DataFrame:
    """Filter stocks from DuckDB, removing high-price, ST, newly listed, and illiquid stocks.

    Args:
        max_close: Maximum latest close price (remove stocks above this).
        min_amount: Minimum 20-day average daily amount, in units of 10000 CNY (wan).
        min_days: Minimum number of trading days since listing.

    Returns:
        DataFrame with [code, latest_close, avg_amount_20d, trade_days]
    """
    conn = get_duckdb_connection()

    # Get latest date and per-stock stats
    df = conn.execute("""
        SELECT
            code,
            LAST(close ORDER BY date) AS latest_close,
            AVG(amount) FILTER (
                WHERE date >= (SELECT MAX(date) - INTERVAL '20 DAY' FROM stock_kline_d)
            ) / 10000 AS avg_amount_20d,
            COUNT(*) AS trade_days,
            -- Latest ST status
            LAST("isST" ORDER BY date) AS latest_is_st
        FROM stock_kline_d
        GROUP BY code
    """).df()

    total = len(df)
    logger.info(f"Total stocks in DB: {total}")

    # Filter 1: high price
    df = df[df["latest_close"] <= max_close]
    logger.info(f"After close <= {max_close}: {len(df)}")

    # Filter 2: ST
    df = df[df["latest_is_st"] != 1]
    logger.info(f"After removing ST: {len(df)}")

    # Filter 3: newly listed
    df = df[df["trade_days"] >= min_days]
    logger.info(f"After removing < {min_days} days: {len(df)}")

    # Filter 4: illiquid
    df = df[df["avg_amount_20d"] >= min_amount]
    logger.info(f"After avg_amount >= {min_amount}wan: {len(df)}")

    result = df[["code", "latest_close", "avg_amount_20d", "trade_days"]].copy()
    result = result.sort_values("code").reset_index(drop=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Filter stock universe")
    parser.add_argument("--close", type=float, default=30, help="Max close price, remove stocks above this (default: 30)")
    parser.add_argument("--min_amount", type=float, default=20000, help="Min 20-day avg daily amount, in wan (10000 CNY) units (default: 20000)")
    parser.add_argument("--min_days", type=int, default=250, help="Min trading days since listing (default: 250)")
    args = parser.parse_args()

    result = filter_stocks(
        max_close=args.close,
        min_amount=args.min_amount,
        min_days=args.min_days,
    )

    # Get latest date from DB
    conn = get_duckdb_connection()
    latest_date = conn.execute("SELECT MAX(date) FROM stock_kline_d").fetchone()[0]
    date_str = latest_date.strftime("%Y%m%d")

    output_path = DATA_DIR / f"filtered_stocks_{date_str}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"\n=== Filter result ===")
    print(f"Criteria: close<={args.close}, amount>={args.min_amount}wan, days>={args.min_days}, ST excluded")
    print(f"Stocks kept: {len(result)}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
