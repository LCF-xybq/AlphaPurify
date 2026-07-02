#!/usr/bin/env python3
"""
Fetch quarterly fundamental data (profit/growth/balance/cash_flow/operation) for all
A-shares and store it in DuckDB.

Baostock limits: 50k requests/day, no concurrency. This script goes through BaostockClient
for unified rate limiting (0.1s spacing + 45k/day circuit breaker), runs serially, and
resumes per (code, year, quarter).

Usage:
    python tools/fetch_data/get_fundamental.py
    python tools/fetch_data/get_fundamental.py --year 2024 --quarter 3
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from loguru import logger

from utils.baostock_client import BaostockClient, BaostockDailyLimitError
from core.data.db import get_duckdb_connection

YEAR_RANGE = range(2024, 2027)
QUARTERS = [1, 2, 3, 4]


def _safe_float(val) -> float:
    if val is None or val == "":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _fetch_fundamental_one(client: BaostockClient, code: str, year: int, quarter: int) -> dict | None:
    """Fetch all 5 fundamental APIs for one (code, year, quarter) via the rate-limited client."""
    row = {
        "code": code, "year": year, "quarter": quarter,
        "pub_date": "", "report_date": "",
        "roe": np.nan, "net_margin": np.nan, "gross_margin": np.nan,
        "eps_growth": np.nan, "rev_growth_yoy": np.nan, "profit_growth_yoy": np.nan,
        "debt_ratio": np.nan, "current_ratio": np.nan,
        "ocf_to_net_profit": np.nan, "asset_turnover": np.nan,
        "eps_ttm": np.nan,
    }

    # profit: roe, net_margin, gross_margin, pubDate, statDate
    rs = client.query_profit_data(code=code, year=year, quarter=quarter)
    if rs is not None and rs.error_code == "0" and rs.next():
        data = rs.get_row_data()
        fields = rs.fields
        row["pub_date"] = data[fields.index("pubDate")] if "pubDate" in fields else ""
        row["report_date"] = data[fields.index("statDate")] if "statDate" in fields else ""
        row["roe"] = _safe_float(data[fields.index("roeAvg")]) if "roeAvg" in fields else np.nan
        row["net_margin"] = _safe_float(data[fields.index("npMargin")]) if "npMargin" in fields else np.nan
        row["gross_margin"] = _safe_float(data[fields.index("gpMargin")]) if "gpMargin" in fields else np.nan
        row["eps_ttm"] = _safe_float(data[fields.index("epsTTM")]) if "epsTTM" in fields else np.nan

    # growth: eps_growth, rev_growth, profit_growth
    rs = client.query_growth_data(code=code, year=year, quarter=quarter)
    if rs is not None and rs.error_code == "0" and rs.next():
        data = rs.get_row_data()
        fields = rs.fields
        row["eps_growth"] = _safe_float(data[fields.index("YOYEPSBasic")]) if "YOYEPSBasic" in fields else np.nan
        row["rev_growth_yoy"] = _safe_float(data[fields.index("YOYPNI")]) if "YOYPNI" in fields else np.nan
        row["profit_growth_yoy"] = _safe_float(data[fields.index("YOYNI")]) if "YOYNI" in fields else np.nan

    # balance: debt_ratio, current_ratio
    rs = client.query_balance_data(code=code, year=year, quarter=quarter)
    if rs is not None and rs.error_code == "0" and rs.next():
        data = rs.get_row_data()
        fields = rs.fields
        row["debt_ratio"] = _safe_float(data[fields.index("liabilityToAsset")]) if "liabilityToAsset" in fields else np.nan
        row["current_ratio"] = _safe_float(data[fields.index("currentRatio")]) if "currentRatio" in fields else np.nan

    # cash_flow: ocf_to_net_profit
    rs = client.query_cash_flow_data(code=code, year=year, quarter=quarter)
    if rs is not None and rs.error_code == "0" and rs.next():
        data = rs.get_row_data()
        fields = rs.fields
        row["ocf_to_net_profit"] = _safe_float(data[fields.index("CFOToNP")]) if "CFOToNP" in fields else np.nan

    # operation: asset_turnover
    rs = client.query_operation_data(code=code, year=year, quarter=quarter)
    if rs is not None and rs.error_code == "0" and rs.next():
        data = rs.get_row_data()
        fields = rs.fields
        row["asset_turnover"] = _safe_float(data[fields.index("AssetTurnRatio")]) if "AssetTurnRatio" in fields else np.nan

    if not row.get("report_date"):
        return None
    return row


def _save_one(row: dict) -> None:
    """Save a single row to DuckDB immediately (resume-safe)."""
    df = pd.DataFrame([row])
    if "pub_date" in df.columns:
        df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.date
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date

    conn = get_duckdb_connection()
    conn.execute("CREATE OR REPLACE TEMP TABLE temp_fund AS SELECT * FROM df")
    conn.execute("""
        DELETE FROM stock_fundamental
        WHERE EXISTS (SELECT 1 FROM temp_fund WHERE stock_fundamental.code = temp_fund.code AND stock_fundamental.report_date = temp_fund.report_date)
    """)
    conn.execute("""
        INSERT INTO stock_fundamental (code, report_date, pub_date, year, quarter,
            roe, net_margin, gross_margin, eps_growth, rev_growth_yoy, profit_growth_yoy,
            debt_ratio, current_ratio, ocf_to_net_profit, asset_turnover, eps_ttm)
        SELECT code, report_date, pub_date, year, quarter,
            roe, net_margin, gross_margin, eps_growth, rev_growth_yoy, profit_growth_yoy,
            debt_ratio, current_ratio, ocf_to_net_profit, asset_turnover, eps_ttm
        FROM temp_fund
    """)
    conn.execute("DROP TABLE temp_fund")


def _get_existing_keys() -> set[tuple[str, int, int]]:
    """Resume checkpoint: which (code, year, quarter) combos are already in DB."""
    conn = get_duckdb_connection()
    rows = conn.execute("SELECT DISTINCT code, year, quarter FROM stock_fundamental").fetchall()
    return {(r[0], r[1], r[2]) for r in rows}


def fetch_fundamental(
    codes: list[str],
    years: list[int] | None = None,
    quarters: list[int] | None = None,
    max_workers: int = 1,
) -> pd.DataFrame:
    # NOTE: baostock forbids concurrent connections. max_workers is kept for
    # backward compatibility but ignored; >1 logs a warning.
    if max_workers and max_workers > 1:
        logger.warning(
            f"max_workers={max_workers} ignored — baostock forbids concurrent connections; running serially"
        )

    if years is None:
        years = list(YEAR_RANGE)
    if quarters is None:
        quarters = QUARTERS

    existing = _get_existing_keys()
    # Build the full task list, then drop already-fetched keys
    tasks = [
        (code, year, quarter)
        for year in years
        for quarter in quarters
        for code in codes
        if (code, year, quarter) not in existing
    ]
    skipped = len(codes) * len(years) * len(quarters) - len(tasks)
    logger.info(
        f"Fetching fundamental: {len(tasks)} pending ({skipped} already fetched, "
        f"{len(codes)} codes × {len(years)} years × {len(quarters)} quarters); "
        f"~{len(tasks) * 5} API calls needed"
    )
    logger.info(f"Today's baostock usage so far: {BaostockClient().get_daily_usage()}/45000")

    if not tasks:
        logger.info("Nothing to fetch — all combinations already in DB")
        return pd.DataFrame()

    client = BaostockClient()
    all_rows = []
    done = 0
    saved = 0

    try:
        for code, year, quarter in tasks:
            row = _fetch_fundamental_one(client, code, year, quarter)
            done += 1
            if row is not None:
                all_rows.append(row)
                _save_one(row)
                saved += 1

            if done % 200 == 0:
                usage = client.get_daily_usage()
                logger.info(
                    f"Progress: {done}/{len(tasks)} (saved {saved}); "
                    f"today's API usage: {usage}/45000"
                )
    except BaostockDailyLimitError as e:
        logger.error(
            f"Daily API quota exhausted at {done}/{len(tasks)} — stopping gracefully. "
            f"Re-run tomorrow to continue. Detail: {e}"
        )

    df = pd.DataFrame(all_rows)
    logger.info(f"Fetched {len(df)} new records (processed {done}/{len(tasks)} pending tasks)")
    return df


def print_stats():
    conn = get_duckdb_connection()
    total = conn.execute("SELECT COUNT(*) FROM stock_fundamental").fetchone()[0]
    stocks = conn.execute("SELECT COUNT(DISTINCT code) FROM stock_fundamental").fetchone()[0]
    date_range = conn.execute("SELECT MIN(report_date), MAX(report_date) FROM stock_fundamental").fetchone()
    print("\n=== Fundamental data stats ===")
    print(f"Total records: {total}")
    print(f"Stocks: {stocks}")
    print(f"Report date range: {date_range[0]} ~ {date_range[1]}")


def main():
    parser = argparse.ArgumentParser(description="Fetch fundamental data from baostock")
    parser.add_argument("--year", type=int, nargs="+", default=None, help="Years to fetch")
    parser.add_argument("--quarter", type=int, nargs="+", default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=1, help="(deprecated) ignored — baostock forbids concurrency")
    args = parser.parse_args()

    conn = get_duckdb_connection()
    codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM stock_kline_d ORDER BY code").fetchall()]
    logger.info(f"Fetching fundamental data for {len(codes)} stocks")

    fetch_fundamental(codes, years=args.year, quarters=args.quarter, max_workers=args.workers)
    print_stats()


if __name__ == "__main__":
    main()
