#!/usr/bin/env python3
"""
Fetch quarterly fundamental data (profit/growth/balance/cash_flow/operation) AND daily
valuation data (peTTM/pb/psTTM/isST) for A-shares and store in DuckDB.

Baostock limits: 50k requests/day hard cap, no concurrency. All calls go through
BaostockClient (0.1s spacing + 45k/day circuit breaker, retries with backoff).
Resume is per (code, year, quarter) for fundamentals and per code for valuation;
empty results are recorded in fetch_log_* tables so they are not retried.

Usage:
    # Default: fundamental only (backward compat)
    python tools/fetch_data/get_fundamental.py
    python tools/fetch_data/get_fundamental.py --year 2024 --quarter 3

    # Valuation only (1 API call per stock)
    python tools/fetch_data/get_fundamental.py --valuation

    # Both
    python tools/fetch_data/get_fundamental.py --fundamental --valuation
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from loguru import logger

from utils.baostock_client import BaostockClient, BaostockDailyLimitError, ADJUSTFLAG
from core.data.db import get_duckdb_connection

YEAR_RANGE = range(2024, 2027)
QUARTERS = [1, 2, 3, 4]

VALUATION_FIELDS = "date,code,peTTM,pbMRQ,psTTM,isST"


def _safe_float(val) -> float:
    if val is None or val == "":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# =============================================================================
# Attempt-log helpers — tracks every (code, year, quarter) we've queried,
# including empty results, so resume does not retry IPO-before-listing quarters
# forever.
# =============================================================================

def _mark_fundamental_attempted(code: str, year: int, quarter: int) -> None:
    conn = get_duckdb_connection()
    conn.execute(
        "INSERT OR IGNORE INTO fetch_log_fundamental (code, year, quarter) VALUES (?, ?, ?)",
        [code, year, quarter],
    )


def _mark_valuation_attempted(code: str) -> None:
    conn = get_duckdb_connection()
    conn.execute(
        "INSERT OR IGNORE INTO fetch_log_valuation (code) VALUES (?)",
        [code],
    )


# =============================================================================
# Quarterly fundamentals
# =============================================================================

def _fetch_fundamental_one(client: BaostockClient, code: str, year: int, quarter: int) -> dict | None:
    """Fetch fundamental APIs for one (code, year, quarter).

    Early-returns None if the profit API has no row — profit_data is the most
    reliable existence check; the other 4 APIs will also be empty. This saves
    4 calls per empty task (e.g. quarters before a stock's IPO).
    """
    row = {
        "code": code, "year": year, "quarter": quarter,
        "pub_date": "", "report_date": "",
        "roe": np.nan, "net_margin": np.nan, "gross_margin": np.nan,
        "eps_growth": np.nan, "rev_growth_yoy": np.nan, "profit_growth_yoy": np.nan,
        "debt_ratio": np.nan, "current_ratio": np.nan,
        "ocf_to_net_profit": np.nan, "asset_turnover": np.nan,
        "eps_ttm": np.nan,
    }

    # Profit: existence check + ROE/margins
    rs = client.query_profit_data(code=code, year=year, quarter=quarter)
    if rs is None or rs.error_code != "0" or not rs.next():
        return None
    data = rs.get_row_data()
    fields = rs.fields
    row["pub_date"] = data[fields.index("pubDate")] if "pubDate" in fields else ""
    row["report_date"] = data[fields.index("statDate")] if "statDate" in fields else ""
    row["roe"] = _safe_float(data[fields.index("roeAvg")]) if "roeAvg" in fields else np.nan
    row["net_margin"] = _safe_float(data[fields.index("npMargin")]) if "npMargin" in fields else np.nan
    row["gross_margin"] = _safe_float(data[fields.index("gpMargin")]) if "gpMargin" in fields else np.nan
    row["eps_ttm"] = _safe_float(data[fields.index("epsTTM")]) if "epsTTM" in fields else np.nan

    # Growth
    rs = client.query_growth_data(code=code, year=year, quarter=quarter)
    if rs is not None and rs.error_code == "0" and rs.next():
        data = rs.get_row_data()
        fields = rs.fields
        row["eps_growth"] = _safe_float(data[fields.index("YOYEPSBasic")]) if "YOYEPSBasic" in fields else np.nan
        row["rev_growth_yoy"] = _safe_float(data[fields.index("YOYPNI")]) if "YOYPNI" in fields else np.nan
        row["profit_growth_yoy"] = _safe_float(data[fields.index("YOYNI")]) if "YOYNI" in fields else np.nan

    # Balance
    rs = client.query_balance_data(code=code, year=year, quarter=quarter)
    if rs is not None and rs.error_code == "0" and rs.next():
        data = rs.get_row_data()
        fields = rs.fields
        row["debt_ratio"] = _safe_float(data[fields.index("liabilityToAsset")]) if "liabilityToAsset" in fields else np.nan
        row["current_ratio"] = _safe_float(data[fields.index("currentRatio")]) if "currentRatio" in fields else np.nan

    # Cash flow
    rs = client.query_cash_flow_data(code=code, year=year, quarter=quarter)
    if rs is not None and rs.error_code == "0" and rs.next():
        data = rs.get_row_data()
        fields = rs.fields
        row["ocf_to_net_profit"] = _safe_float(data[fields.index("CFOToNP")]) if "CFOToNP" in fields else np.nan

    # Operation
    rs = client.query_operation_data(code=code, year=year, quarter=quarter)
    if rs is not None and rs.error_code == "0" and rs.next():
        data = rs.get_row_data()
        fields = rs.fields
        row["asset_turnover"] = _safe_float(data[fields.index("AssetTurnRatio")]) if "AssetTurnRatio" in fields else np.nan

    if not row.get("report_date"):
        return None
    return row


def _save_one(row: dict) -> None:
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
    """Resume checkpoint: (code, year, quarter) keys already saved OR attempted-empty."""
    conn = get_duckdb_connection()
    rows = conn.execute("""
        SELECT code, year, quarter FROM stock_fundamental
        UNION
        SELECT code, year, quarter FROM fetch_log_fundamental
    """).fetchall()
    return {(r[0], r[1], r[2]) for r in rows}


def fetch_fundamental(
    codes: list[str],
    years: list[int] | None = None,
    quarters: list[int] | None = None,
    max_workers: int = 1,
) -> pd.DataFrame:
    if max_workers and max_workers > 1:
        logger.warning(
            f"max_workers={max_workers} ignored — baostock forbids concurrent connections; running serially"
        )

    if years is None:
        years = list(YEAR_RANGE)
    if quarters is None:
        quarters = QUARTERS

    existing = _get_existing_keys()
    tasks = [
        (code, year, quarter)
        for year in years
        for quarter in quarters
        for code in codes
        if (code, year, quarter) not in existing
    ]
    skipped = len(codes) * len(years) * len(quarters) - len(tasks)
    logger.info(
        f"Fetching fundamental: {len(tasks)} pending ({skipped} already done or attempted-empty, "
        f"{len(codes)} codes × {len(years)} years × {len(quarters)} quarters)"
    )
    logger.info(f"Today's baostock usage so far: {BaostockClient().get_daily_usage()}/45000")

    if not tasks:
        logger.info("Nothing to fetch — all combinations already processed")
        return pd.DataFrame()

    client = BaostockClient()
    all_rows = []
    done = 0
    saved = 0

    try:
        for code, year, quarter in tasks:
            row = _fetch_fundamental_one(client, code, year, quarter)
            done += 1
            # Always mark attempted so empty quarters are not retried next run
            _mark_fundamental_attempted(code, year, quarter)
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
            f"Re-run tomorrow; resume will skip already-attempted keys. Detail: {e}"
        )
    except KeyboardInterrupt:
        logger.warning(
            f"Interrupted at task {done}/{len(tasks)} (saved {saved} new rows). "
            f"Resume checkpoint is up to date — re-run to continue from task {done + 1}."
        )

    df = pd.DataFrame(all_rows)
    logger.info(f"Fetched {len(df)} new records (processed {done}/{len(tasks)} pending tasks)")
    return df


# =============================================================================
# Daily valuation (peTTM / pb / psTTM / isST)
# =============================================================================

def _get_pending_valuation_tasks(
    codes: list[str], start_date: str, end_date: str,
) -> list[tuple[str, str, str]]:
    """Build per-code fetch tasks, skipping codes already attempted (success or empty).

    For codes with prior data, resume from max(date)+1.
    For codes never attempted, fetch from start_date.
    For codes attempted but empty, skip.
    """
    if not codes:
        return []
    conn = get_duckdb_connection()
    codes_literal = ", ".join(f"'{c}'" for c in codes)
    saved = dict(conn.execute(
        f"SELECT code, MAX(date) FROM stock_valuation_d "
        f"WHERE code IN ({codes_literal}) GROUP BY code"
    ).fetchall())
    attempted = {
        r[0] for r in conn.execute(
            f"SELECT code FROM fetch_log_valuation WHERE code IN ({codes_literal})"
        ).fetchall()
    }

    pending = []
    for code in codes:
        if code in saved:
            max_d = str(saved[code])
            next_day = (pd.Timestamp(max_d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if next_day > end_date:
                continue
            pending.append((code, next_day, end_date))
        elif code not in attempted:
            pending.append((code, start_date, end_date))
    return pending


def _fetch_valuation_one(
    client: BaostockClient,
    code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """One baostock call per (code, date_range) — returns daily valuation rows.

    Returns None on API failure (caller must NOT mark attempted — retry next run).
    Returns empty DataFrame on success-but-no-rows (caller should mark attempted).
    Renames baostock's `pbMRQ` to `pb` for downstream compatibility.
    """
    rs = client.query_history_k_data_plus(
        code=code,
        fields=VALUATION_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=ADJUSTFLAG,
    )
    if rs is None or rs.error_code != "0":
        logger.warning(f"valuation fetch failed for {code}: {getattr(rs, 'error_msg', 'n/a')}")
        return None

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)
    # baostock uses pbMRQ; downstream code expects pb
    if "pbMRQ" in df.columns:
        df = df.rename(columns={"pbMRQ": "pb"})
    return df


def _save_valuation_chunk(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for col in ("peTTM", "pb", "psTTM"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce").astype("Int64")

    df = df.replace({pd.NA: None})
    df = df.dropna(subset=["code", "date"])
    df = df.drop_duplicates(subset=["code", "date"], keep="last")
    if df.empty:
        return 0

    conn = get_duckdb_connection()
    conn.execute("CREATE OR REPLACE TEMP TABLE temp_val AS SELECT * FROM df")
    conn.execute("""
        DELETE FROM stock_valuation_d
        WHERE EXISTS (SELECT 1 FROM temp_val
                      WHERE stock_valuation_d.code = temp_val.code
                        AND stock_valuation_d.date = temp_val.date)
    """)
    count = conn.execute("""
        INSERT INTO stock_valuation_d (code, date, peTTM, pb, psTTM, isST)
        SELECT code, date, peTTM, pb, psTTM, isST FROM temp_val
    """).fetchone()[0]
    conn.execute("DROP TABLE temp_val")
    return count


def fetch_valuation(
    codes: list[str],
    start_date: str = "2024-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch daily peTTM/pb/psTTM/isST for each code. 1 API call per code, resume-safe."""
    if not codes:
        logger.warning("No codes provided")
        return pd.DataFrame()
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    pending = _get_pending_valuation_tasks(codes, start_date, end_date)
    logger.info(
        f"Fetching valuation: {len(pending)} codes pending out of {len(codes)} "
        f"(rest already saved or attempted-empty); ~{len(pending)} API calls needed"
    )
    logger.info(f"Today's baostock usage so far: {BaostockClient().get_daily_usage()}/45000")

    if not pending:
        logger.info("Nothing to fetch — all codes already processed")
        return pd.DataFrame()

    client = BaostockClient()
    all_dfs = []
    done = 0
    saved_rows = 0

    try:
        for code, s, e in pending:
            df = _fetch_valuation_one(client, code, s, e)
            done += 1
            # Mark attempted only on successful call (incl. empty result).
            # Skip marking on None (API failure) so the next run retries.
            if df is None:
                continue
            _mark_valuation_attempted(code)
            if not df.empty:
                rows = _save_valuation_chunk(df)
                saved_rows += rows
                all_dfs.append(df)
            if done % 50 == 0:
                usage = client.get_daily_usage()
                logger.info(
                    f"Progress: {done}/{len(pending)} codes (saved {saved_rows} rows); "
                    f"today's API usage: {usage}/45000"
                )
    except BaostockDailyLimitError as e:
        logger.error(
            f"Daily API quota exhausted at {done}/{len(pending)} — stopping gracefully. "
            f"Re-run tomorrow; resume will skip already-attempted codes. Detail: {e}"
        )
    except KeyboardInterrupt:
        logger.warning(
            f"Interrupted at code {done}/{len(pending)} (saved {saved_rows} new rows). "
            f"Resume checkpoint is up to date — re-run to continue from code {done + 1}."
        )

    out = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    logger.info(f"Fetched {len(out)} new valuation rows from {done}/{len(pending)} codes")
    return out


# =============================================================================
# Stats / CLI
# =============================================================================

def print_stats():
    conn = get_duckdb_connection()
    total = conn.execute("SELECT COUNT(*) FROM stock_fundamental").fetchone()[0]
    stocks = conn.execute("SELECT COUNT(DISTINCT code) FROM stock_fundamental").fetchone()[0]
    date_range = conn.execute("SELECT MIN(report_date), MAX(report_date) FROM stock_fundamental").fetchone()
    print("\n=== Fundamental data stats ===")
    print(f"Total records: {total}")
    print(f"Stocks: {stocks}")
    print(f"Report date range: {date_range[0]} ~ {date_range[1]}")
    fund_attempted = conn.execute("SELECT COUNT(*) FROM fetch_log_fundamental").fetchone()[0]

    val_total = conn.execute("SELECT COUNT(*) FROM stock_valuation_d").fetchone()[0]
    val_stocks = conn.execute("SELECT COUNT(DISTINCT code) FROM stock_valuation_d").fetchone()[0]
    val_range = conn.execute("SELECT MIN(date), MAX(date) FROM stock_valuation_d").fetchone()
    val_attempted = conn.execute("SELECT COUNT(*) FROM fetch_log_valuation").fetchone()[0]

    print("\n=== Valuation data stats ===")
    print(f"Total records: {val_total}")
    print(f"Stocks: {val_stocks}")
    print(f"Date range: {val_range[0]} ~ {val_range[1]}")

    print("\n=== Fetch logs (resume checkpoints) ===")
    print(f"fundamental attempted: {fund_attempted} (code, year, quarter) keys")
    print(f"valuation attempted:   {val_attempted} codes")


def main():
    parser = argparse.ArgumentParser(description="Fetch fundamental + valuation data from baostock")
    parser.add_argument("--year", type=int, nargs="+", default=None, help="Years to fetch (fundamental only)")
    parser.add_argument("--quarter", type=int, nargs="+", default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=1, help="(deprecated) ignored — baostock forbids concurrency")
    # Mutually meaningful flags — default behavior unchanged when none specified
    parser.add_argument("--fundamental", action="store_true", default=None,
                        help="Fetch quarterly fundamentals (default if neither flag given)")
    parser.add_argument("--valuation", action="store_true", default=None,
                        help="Fetch daily valuation (peTTM/pb/psTTM/isST)")
    parser.add_argument("--start", default="2024-01-01", help="Start date for valuation fetch")
    parser.add_argument("--end", default=None, help="End date for valuation fetch (default: today)")
    args = parser.parse_args()

    # Resolve which fetches to run
    if args.fundamental is None and args.valuation is None:
        # Default: fetch both — most users want a complete fundamental+valuation dataset
        do_fund, do_val = True, True
    else:
        do_fund = bool(args.fundamental)
        do_val = bool(args.valuation)

    conn = get_duckdb_connection()
    codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM stock_kline_d ORDER BY code").fetchall()]
    logger.info(f"Loaded {len(codes)} stock codes from stock_kline_d")

    if do_fund:
        logger.info("=== Fetching quarterly fundamentals ===")
        fetch_fundamental(codes, years=args.year, quarters=args.quarter, max_workers=args.workers)

    if do_val:
        logger.info("=== Fetching daily valuation ===")
        fetch_valuation(codes, start_date=args.start, end_date=args.end)

    print_stats()


if __name__ == "__main__":
    main()
