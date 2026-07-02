#!/usr/bin/env python3
"""
Export DuckDB tables to parquet files for backup and offline loading.

Usage:
    python tools/process_data/export_tables.py                    # export all
    python tools/process_data/export_tables.py --only fundamental  # export specific
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger

from core.data.db import get_duckdb_connection

OUTPUT_DIR = PROJECT_ROOT / "data"

TABLES = {
    "kline_d": {
        "sql": "SELECT * FROM stock_kline_d ORDER BY code, date",
        "file": "stock_kline_d.parquet",
    },
    "kline_w": {
        "sql": "SELECT * FROM stock_kline_w ORDER BY code, date",
        "file": "stock_kline_w.parquet",
    },
    "kline_m": {
        "sql": "SELECT * FROM stock_kline_m ORDER BY code, date",
        "file": "stock_kline_m.parquet",
    },
    "index_d": {
        "sql": "SELECT * FROM index_kline_d ORDER BY code, date",
        "file": "index_kline_d.parquet",
    },
    "index_w": {
        "sql": "SELECT * FROM index_kline_w ORDER BY code, date",
        "file": "index_kline_w.parquet",
    },
    "index_m": {
        "sql": "SELECT * FROM index_kline_m ORDER BY code, date",
        "file": "index_kline_m.parquet",
    },
    "industry": {
        "sql": "SELECT * FROM stock_industry ORDER BY code",
        "file": "stock_industry.parquet",
    },
    "fundamental": {
        "sql": "SELECT * FROM stock_fundamental ORDER BY code, report_date",
        "file": "stock_fundamental.parquet",
    },
    "factor_cs": {
        "sql": None,  # already saved by pipeline.py
        "file": "factor_cross_section.parquet",
    },
}


def export_table(conn, name: str, sql: str, path: Path) -> int:
    df = conn.execute(sql).df()
    if df.empty:
        logger.warning(f"{name}: empty, skipping")
        return 0
    df.to_parquet(path, index=False)
    logger.info(f"{name}: {len(df)} rows -> {path}")
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Export DuckDB tables to parquet")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=list(TABLES.keys()),
                        help="Export only specified tables")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_duckdb_connection()
    targets = args.only or list(TABLES.keys())
    total = 0

    for name in targets:
        info = TABLES[name]
        path = output_dir / info["file"]

        if info["sql"] is None:
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                logger.info(f"{name}: already exists ({size_mb:.1f} MB), skipping")
            else:
                logger.warning(f"{name}: file not found, run pipeline.py first")
            continue

        count = export_table(conn, name, info["sql"], path)
        total += count

    logger.info(f"Done. Total {total} rows exported to {output_dir}/")


if __name__ == "__main__":
    main()
