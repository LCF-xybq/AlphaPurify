#!/usr/bin/env python3
"""
Split a DuckDB table by code into per-code CSV or Parquet files.

Usage:
    python tools/process_data/to_csv_parquet.py --type index_d
    python tools/process_data/to_csv_parquet.py --type stock_d --save_type csv --save_path ./data
"""
import argparse
import sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.data.db import get_duckdb_connection

DATA_DIR = PROJECT_ROOT / "data"

TABLE_MAP = {
    "index_d": "index_kline_d",
    "index_w": "index_kline_w",
    "index_m": "index_kline_m",
    "stock_d": "stock_kline_d",
    "stock_w": "stock_kline_w",
    "stock_m": "stock_kline_m",
    "stock_5": "stock_kline_5min",
    "stock_15": "stock_kline_15min",
    "stock_30": "stock_kline_30min",
    "stock_60": "stock_kline_60min",
}


def _get_codes(table: str) -> list[str]:
    conn = get_duckdb_connection()
    return [row[0] for row in conn.execute(f"SELECT DISTINCT code FROM {table} ORDER BY code").fetchall()]


def _type_to_freq(type_key: str) -> str:
    """index_d -> d, stock_5 -> 5, etc."""
    return type_key.split("_", 1)[1]


def _export_code(args: tuple[str, str, str, str, str]) -> tuple[str, int]:
    """Export one code's data; return (code, row_count)."""
    code, table, save_type, base_path, type_key = args
    conn = get_duckdb_connection()

    df = conn.execute(
        f"SELECT * FROM {table} WHERE code = '{code}' ORDER BY date"
    ).df()

    if df.empty:
        return code, 0

    # Parse start_date and end_date (the date column may be a date type or string)
    dates = df["date"].astype(str)
    start_date = dates.min()
    end_date = dates.max()

    freq = _type_to_freq(type_key)
    filename = f"{code}_{start_date}_{end_date}_{freq}.{save_type}"
    out_dir = Path(base_path) / table / save_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if save_type == "csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    return code, len(df)


def main():
    parser = argparse.ArgumentParser(description="Export table to CSV/Parquet files by code")
    parser.add_argument("--type", required=True, help=f"Table type, one of: {', '.join(TABLE_MAP.keys())}")
    parser.add_argument("--save_type", default="parquet", choices=["csv", "parquet"], help="Output format (default: parquet)")
    parser.add_argument("--save_path", default=str(DATA_DIR), help=f"Base save path (default: {DATA_DIR})")
    args = parser.parse_args()

    if args.type not in TABLE_MAP:
        print(f"Unsupported --type: {args.type}. Supported: {', '.join(TABLE_MAP.keys())}")
        return

    table = TABLE_MAP[args.type]
    codes = _get_codes(table)
    if not codes:
        print(f"No data in table {table}")
        return

    print(f"Exporting {len(codes)} codes from {table} to {args.save_path} ({args.save_type})")

    # Build work items: (code, table, save_type, save_path, type)
    work_items = [
        (code, table, args.save_type, args.save_path, args.type)
        for code in codes
    ]

    total_rows = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_export_code, item): item[0] for item in work_items}
        for future in as_completed(futures):
            code = futures[future]
            try:
                _, count = future.result()
                total_rows += count
                print(f"  {code}: {count} rows")
            except Exception as e:
                print(f"  {code}: ERROR {e}")

    print(f"\nDone. Total {total_rows} rows exported to {args.save_path}/{TABLE_MAP[args.type]}/{args.save_type}/")


if __name__ == "__main__":
    main()
