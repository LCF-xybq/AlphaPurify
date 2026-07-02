#!/usr/bin/env python3
"""
Fetch the CSRC industry classification for all A-shares and write to the
DuckDB stock_industry table.

All baostock calls go through BaostockClient for unified rate limiting (avoids blacklisting).

Usage:
    python tools/fetch_data/get_industry.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger

from utils.baostock_client import BaostockClient
from core.data.db import get_duckdb_connection

# A-share stock code prefixes we care about
STOCK_PREFIXES = ("sh.600", "sh.601", "sh.603", "sh.605", "sh.688", "sz.000", "sz.002", "sz.003", "sz.300")


def fetch_industry_batch(client: BaostockClient) -> pd.DataFrame:
    """Fetch all industry data via batch query (code='')."""
    rs = client.query_stock_industry(code="")
    if rs is None:
        return pd.DataFrame()
    data = []
    while rs.error_code == "0" and rs.next():
        data.append(rs.get_row_data())

    df = pd.DataFrame(data, columns=rs.fields)
    mask = df["code"].str.startswith(STOCK_PREFIXES)
    df = df[mask].copy()
    logger.info(f"Batch query: {len(df)} stocks")
    return df


def fetch_industry_per_code(client: BaostockClient, codes: list[str]) -> pd.DataFrame:
    """Fetch industry data by querying each code individually."""
    data = []
    for code in codes:
        rs = client.query_stock_industry(code=code)
        if rs is None:
            continue
        while rs.error_code == "0" and rs.next():
            data.append(rs.get_row_data())
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["updateDate", "code", "code_name", "industry", "industryClassification"])
    return df


def get_missing_codes() -> list[str]:
    """Find stocks in stock_kline_d that have no valid industry data."""
    conn = get_duckdb_connection()
    rows = conn.execute("""
        SELECT DISTINCT a.code
        FROM stock_kline_d a
        LEFT JOIN stock_industry b ON a.code = b.code AND b.industry != ''
        WHERE b.code IS NULL
        ORDER BY a.code
    """).fetchall()
    return [r[0] for r in rows]


def save_to_duckdb(df: pd.DataFrame) -> int:
    conn = get_duckdb_connection()
    conn.execute(
        """INSERT OR REPLACE INTO stock_industry (code, code_name, industry, industry_classification, update_date)
        SELECT code, code_name, industry, industryClassification,
               CAST(updateDate AS DATE) FROM df"""
    )
    count = conn.execute("SELECT COUNT(*) FROM stock_industry").fetchone()[0]
    logger.info(f"Saved {len(df)} rows (total: {count})")
    return count


def print_stats() -> None:
    conn = get_duckdb_connection()
    total = conn.execute("SELECT COUNT(*) FROM stock_industry").fetchone()[0]
    with_industry = conn.execute("SELECT COUNT(*) FROM stock_industry WHERE industry != ''").fetchone()[0]
    n_industry = conn.execute("SELECT COUNT(DISTINCT industry) FROM stock_industry WHERE industry != ''").fetchone()[0]

    # Coverage against stock_kline_d
    coverage = conn.execute("""
        SELECT COUNT(DISTINCT a.code)
        FROM stock_kline_d a
        JOIN stock_industry b ON a.code = b.code AND b.industry != ''
    """).fetchone()[0]
    total_kline = conn.execute("SELECT COUNT(DISTINCT code) FROM stock_kline_d").fetchone()[0]

    print("\n=== Industry classification stats ===")
    print(f"stock_industry total stocks: {total} (with industry: {with_industry})")
    print(f"Industries: {n_industry}")
    coverage_pct = coverage / total_kline * 100 if total_kline else 0.0
    print(f"stock_kline_d coverage: {coverage}/{total_kline} ({coverage_pct:.1f}%)")
    print("\nTop 10 industries:")
    rows = conn.execute("""
        SELECT industry, COUNT(*) as cnt
        FROM stock_industry
        WHERE industry != ''
        GROUP BY industry
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
    for industry, cnt in rows:
        print(f"  {industry}: {cnt}")


def main():
    client = BaostockClient()

    # Step 1: Batch fetch
    df_batch = fetch_industry_batch(client)

    conn = get_duckdb_connection()
    conn.execute("DELETE FROM stock_industry")
    save_to_duckdb(df_batch)

    # Step 2: Fill gaps for stocks in stock_kline_d missing industry
    missing = get_missing_codes()
    if missing:
        logger.info(f"Filling industry data for {len(missing)} missing stocks...")
        df_fill = fetch_industry_per_code(client, missing)
        if not df_fill.empty:
            mask = df_fill["code"].str.startswith(STOCK_PREFIXES)
            df_fill = df_fill[mask].copy()
            save_to_duckdb(df_fill)

        # Check remaining
        still_missing = get_missing_codes()
        if still_missing:
            logger.warning(f"{len(still_missing)} stocks still without industry data")

    client.logout()
    print_stats()


if __name__ == "__main__":
    main()
