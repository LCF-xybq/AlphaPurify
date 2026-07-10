"""
Fetch OHLCV panel data from the local DuckDB store, attach daily valuation,
quarterly fundamentals (point-in-time), industry classification, and compute
factors according to a YAML combination config.

Two factor sources are supported:
  - Alpha191 formulas (factors/alpha191.py): triggered when `alpha:` contains
    "all" or 3-digit ids like "001", "189". Needs `amount` in addition to
    OHLCV (auto-fetched).
  - TA-Lib indicators (legacy): triggered when `alpha:` contains TA-Lib names
    like "sma", "rsi", "macd". Same behavior as before.

Usage:
    python tools/research/fetch_data.py --config data/all_factors.yaml \\
        --save data/all_factors.parquet --warmup 400

    # Optional external factors for Alpha030/075/149/181/182:
    --ff3 data/ff3.parquet --benchmark data/benchmark.parquet

Config layout (see data/all_factors.yaml):
    codes:                     [sh.601991, sz.002491, ...]
    date:                      [{start: "2024-01-01", end: "2026-07-03"}]
    alpha:                     [all]              # Alpha191 mode
                                 # OR [sma, macd]  # TA-Lib mode
    exposure:                  [...]              # optional, same logic as alpha
    include_fundamentals: true
    include_valuation:     true
    drop_st:               true

Output columns:
    datetime, symbol, open, high, low, close, volume, amount,
    <factor cols>, <fundamental cols>, <valuation cols>, industry
"""

import argparse
import os
import re
import sys

import duckdb
import pandas as pd
import yaml

# Reuse the TA-Lib factor registry already maintained in tools/process_data/pre_factor.py
_PROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "process_data")
sys.path.insert(0, os.path.abspath(_PROCESS_DIR))
from pre_factor import ALL_FACTOR_DEFS, compute_talib_factors  # noqa: E402

# Alpha191 factor library (project-local)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from factors import compute_alpha_factors, ALPHA191_REGISTRY  # noqa: E402

NAME_TO_IDX = {name: i for i, (name, _, _) in enumerate(ALL_FACTOR_DEFS)}

# Composite TA-Lib names that expand to multiple output columns
_COMPOSITE = {
    "bb":        ["bb_upper", "bb_middle", "bb_lower"],
    "bollinger": ["bb_upper", "bb_middle", "bb_lower"],
    "stoch":     ["stoch_slowk", "stoch_slowd"],
    "macd":      ["macd", "macd_signal", "macd_hist"],
}

_DEFAULT_DB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "quant.db")
)
_TABLE = "stock_kline_d"

# Fundamental columns pulled from stock_fundamental when include_fundamentals=true
_FUNDAMENTAL_COLS = [
    "roe", "net_margin", "gross_margin", "eps_growth",
    "rev_growth_yoy", "profit_growth_yoy", "debt_ratio",
    "current_ratio", "ocf_to_net_profit", "asset_turnover", "eps_ttm",
]
_VALUATION_COLS = ["peTTM", "pb", "psTTM"]


def load_config(path: str):
    """Parse combination yaml → (codes, start, end, alpha_names, exposure_names, opts)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    codes = cfg.get("codes") or []
    if isinstance(codes, dict):
        codes = [c for grp in codes.values() for c in (grp or [])]

    start = end = None
    date_section = cfg.get("date") or []
    if isinstance(date_section, dict):
        start = date_section.get("start")
        end = date_section.get("end")
    else:
        for item in date_section:
            if isinstance(item, dict):
                start = start or item.get("start")
                end = end or item.get("end")
            elif isinstance(item, str):
                if start is None:
                    start = item
                else:
                    end = item

    def _as_list(key):
        v = cfg.get(key) or []
        if isinstance(v, dict):
            return [f for grp in v.values() for f in (grp or [])]
        return list(v)

    alpha_list = _as_list("factor") if cfg.get("factor") else _as_list("alpha")
    exposure_list = _as_list("exposure") or _as_list("exposure_cols")

    opts = {
        "include_fundamentals": bool(cfg.get("include_fundamentals", False)),
        "include_valuation": bool(cfg.get("include_valuation", False)),
        "drop_st": bool(cfg.get("drop_st", True)),
    }
    return codes, start, end, alpha_list, exposure_list, opts


def resolve_factor_indices(factor_names):
    """Map user-facing TA-Lib names to indices in ALL_FACTOR_DEFS."""
    indices = []
    for raw in factor_names:
        name = str(raw).strip().lower()
        if name in _COMPOSITE:
            for sub in _COMPOSITE[name]:
                if sub in NAME_TO_IDX:
                    indices.append(NAME_TO_IDX[sub])
        elif name in NAME_TO_IDX:
            indices.append(NAME_TO_IDX[name])
        else:
            raise ValueError(
                f"Unknown TA-Lib factor '{raw}'. Available: {sorted(NAME_TO_IDX)}"
            )
    return sorted(set(indices))


_ALPHA191_ID_RE = re.compile(r"^\d{1,3}$")


def is_alpha191_mode(factor_names) -> bool:
    """True if the alpha list looks like Alpha191 ids ("all" or 3-digit numbers)."""
    if not factor_names:
        return False
    for raw in factor_names:
        s = str(raw).strip().lower()
        if s == "all":
            return True
        # Strip optional "alpha191_" prefix
        if s.startswith("alpha191_"):
            s = s[len("alpha191_"):]
        if _ALPHA191_ID_RE.match(s):
            return True
    return False


def resolve_alpha191_names(factor_names) -> list[str] | None:
    """Return explicit alpha191 ids, or None for "all".
    Raises ValueError on unknown ids."""
    if any(str(s).strip().lower() == "all" for s in factor_names):
        return None
    out = []
    for raw in factor_names:
        s = str(raw).strip().lower()
        if s.startswith("alpha191_"):
            s = s[len("alpha191_"):]
        s = s.zfill(3)
        if s not in ALPHA191_REGISTRY:
            raise ValueError(f"Unknown alpha191 id '{raw}'. Valid: 001..191")
        out.append(s)
    return sorted(set(out))


def _table_exists(con, table_name: str) -> bool:
    return con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()[0] > 0


def _pit_merge_fundamentals(daily_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time merge: for each (code, date), attach the latest quarterly
    fundamental row whose pub_date <= date. Forward-fills per stock.

    Uses pub_date (announcement) rather than report_date (period-end) to avoid
    leaking future information into the panel.
    """
    if fund_df.empty:
        return daily_df

    daily_df = daily_df.sort_values(["code", "date"])
    fund_df = fund_df.sort_values(["code", "pub_date"])
    fund_df = fund_df.drop_duplicates(subset=["code", "pub_date"], keep="last")

    out_chunks = []
    for code, g in daily_df.groupby("code", sort=False):
        f = fund_df[fund_df["code"] == code].drop(columns=["code"])
        if f.empty:
            continue
        merged = pd.merge_asof(
            g.sort_values("date"),
            f,
            left_on="date",
            right_on="pub_date",
            direction="backward",
        )
        out_chunks.append(merged)
    if not out_chunks:
        return daily_df
    return pd.concat(out_chunks, ignore_index=True)


def fetch_panel(
    db_path: str,
    codes,
    start,
    end,
    warmup_days: int = 0,
    include_fundamentals: bool = False,
    include_valuation: bool = False,
    drop_st: bool = True,
) -> pd.DataFrame:
    """Query DuckDB for OHLCV plus optional fundamentals/valuation/industry."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    if not codes:
        raise ValueError("'codes' is empty in config.")
    if not (start and end):
        raise ValueError("Config needs 'date.start' and 'date.end'.")

    fetch_start = start
    if warmup_days > 0:
        fetch_start = (
            pd.to_datetime(start) - pd.Timedelta(days=warmup_days)
        ).strftime("%Y-%m-%d")

    codes_literal = ", ".join(f"'{c}'" for c in codes)

    con = duckdb.connect(db_path, read_only=True)
    try:
        # 1. OHLCV+amount (stock_kline_d already excludes isST=1 rows — filtering
        # happens at fetch time in core/data/stock.py, so no explicit ST filter
        # needed here. `amount` is required by Alpha191 VWAP derivation.)
        df = con.execute(f"""
            SELECT code, date, open, high, low, close, volume, amount
            FROM {_TABLE}
            WHERE code IN ({codes_literal})
              AND date >= DATE '{fetch_start}'
              AND date <= DATE '{end}'
            ORDER BY code, date
        """).fetchdf()

        if df.empty:
            raise ValueError(
                f"No rows returned. Check codes (e.g. {codes[:3]}) and date range "
                f"[{fetch_start}, {end}] against table '{_TABLE}'."
            )

        df["date"] = pd.to_datetime(df["date"])

        # 2. Daily valuation (peTTM / pb / psTTM)
        if include_valuation and _table_exists(con, "stock_valuation_d"):
            val = con.execute(f"""
                SELECT code, date, peTTM, pb, psTTM
                FROM stock_valuation_d
                WHERE code IN ({codes_literal})
                  AND date >= DATE '{fetch_start}'
                  AND date <= DATE '{end}'
            """).fetchdf()
            if not val.empty:
                val["date"] = pd.to_datetime(val["date"])
                df = df.merge(val, on=["code", "date"], how="left")

        # 3. PIT-merge quarterly fundamentals via pub_date
        if include_fundamentals and _table_exists(con, "stock_fundamental"):
            fund = con.execute(f"""
                SELECT code, pub_date, {", ".join(_FUNDAMENTAL_COLS)}
                FROM stock_fundamental
                WHERE pub_date IS NOT NULL
                  AND code IN ({codes_literal})
                ORDER BY code, pub_date
            """).fetchdf()
            if not fund.empty:
                fund["pub_date"] = pd.to_datetime(fund["pub_date"])
                df = _pit_merge_fundamentals(df, fund)
                df = df.drop(columns=["pub_date"], errors="ignore")

        # 4. Industry classification (always attach if available — needed for
        # industry neutralization in ml/dataset.py)
        if _table_exists(con, "stock_industry"):
            ind = con.execute(f"""
                SELECT code, industry FROM stock_industry
                WHERE code IN ({codes_literal})
            """).fetchdf()
            if not ind.empty:
                df = df.merge(ind, on="code", how="left")
                df["industry"] = df["industry"].fillna("UNKNOWN").replace("", "UNKNOWN")
    finally:
        con.close()

    df = df.rename(columns={"code": "symbol", "date": "datetime"})
    return df


def compute_factors(df: pd.DataFrame, factor_indices):
    """Apply TA-Lib factors per symbol, return (df, produced_cols)."""
    parts = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("datetime")
        parts.append(compute_talib_factors(g, factor_indices))
    out = pd.concat(parts, ignore_index=True)
    produced = [ALL_FACTOR_DEFS[i][0] for i in factor_indices]
    return out, produced


def compute_alpha191(df: pd.DataFrame, names: list[str] | None, ctx: dict | None = None):
    """Apply Alpha191 factors to the full panel (cross-sectional ops need it).
    Returns (df_aug, produced_cols)."""
    out = compute_alpha_factors(df, names=names, ctx=ctx, skip_on_error=True)
    prefix = "alpha191_"
    produced = [c for c in out.columns if c.startswith(prefix)]
    return out, produced


def save_panel(df: pd.DataFrame, path: str):
    ext = os.path.splitext(path)[1].lower()
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"--save extension must be .parquet or .csv, got '{ext}'")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch stock panel data and compute factors (Alpha191 or TA-Lib) from a YAML config."
    )
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--save", required=True, help="Output file (.parquet or .csv).")
    parser.add_argument(
        "--db",
        default=None,
        help=f"DuckDB path (default: {_DEFAULT_DB}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=400,
        help=(
            "Calendar days to pre-fetch before `start` for factor warmup "
            "(default 400, covers Alpha191's longest window SUM(RET,250) + buffer). "
            "60 is enough for TA-Lib-only configs."
        ),
    )
    parser.add_argument("--ff3", default=None,
                        help="Optional parquet with mkt/smb/hml columns for Alpha030/147/181/182.")
    parser.add_argument("--benchmark", default=None,
                        help="Optional parquet with open/close columns for benchmark-index alphas.")
    args = parser.parse_args()

    codes, start, end, alpha_list, exposure_list, opts = load_config(args.config)
    factor_list = list(alpha_list) + list(exposure_list)
    print(f"[fetch] symbols: {len(codes)}  range: [{start}, {end}]")
    print(f"[fetch] alpha:    {alpha_list}")
    print(f"[fetch] exposure: {exposure_list}")
    print(f"[fetch] include_fundamentals={opts['include_fundamentals']}  "
          f"include_valuation={opts['include_valuation']}  drop_st={opts['drop_st']}")

    # Branch on factor source: Alpha191 vs TA-Lib
    alpha191_mode = is_alpha191_mode(factor_list)
    talib_indices = None
    a191_names = None
    if alpha191_mode:
        a191_names = resolve_alpha191_names(factor_list)
        print(f"[fetch] factor source: Alpha191 ({'all 191' if a191_names is None else len(a191_names)} selected)")
    else:
        talib_indices = resolve_factor_indices(factor_list)
        print(f"[fetch] factor source: TA-Lib ({len(talib_indices)} factors)")

    db_path = os.path.abspath(args.db) if args.db else _DEFAULT_DB
    df = fetch_panel(
        db_path, codes, start, end,
        warmup_days=args.warmup,
        include_fundamentals=opts["include_fundamentals"],
        include_valuation=opts["include_valuation"],
        drop_st=opts["drop_st"],
    )
    print(f"[fetch] panel rows={len(df)}  symbols={df['symbol'].nunique()}  (incl. {args.warmup}d warmup)")

    if alpha191_mode:
        ctx = _load_ctx(args.ff3, args.benchmark, df)
        df, produced = compute_alpha191(df, a191_names, ctx)
    else:
        df, produced = compute_factors(df, talib_indices)

    # Trim warmup rows — output only the requested [start, end] window
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].reset_index(drop=True)

    # Build final column list: meta + factors + fundamental + valuation + industry
    keep = ["datetime", "symbol", "open", "high", "low", "close", "volume", "amount"]
    keep = [c for c in keep if c in df.columns]
    keep += [c for c in produced if c in df.columns]
    if opts["include_fundamentals"]:
        keep += [c for c in _FUNDAMENTAL_COLS if c in df.columns]
    if opts["include_valuation"]:
        keep += [c for c in _VALUATION_COLS if c in df.columns]
    if "industry" in df.columns:
        keep.append("industry")
    # Deduplicate while preserving order
    seen = set()
    keep = [c for c in keep if not (c in seen or seen.add(c))]
    df = df[keep]

    save_panel(df, args.save)
    print(f"[fetch] saved -> {args.save}  ({len(df)} rows, {len(df.columns)} cols)")


def _load_ctx(ff3_path: str | None, benchmark_path: str | None, panel: pd.DataFrame) -> dict | None:
    """Load optional external factors and align to the panel's index."""
    if not ff3_path and not benchmark_path:
        return None
    from factors.data import load_external
    # Build MultiIndex reference
    tmp = panel[["symbol", "datetime"]].copy()
    tmp["symbol"] = tmp["symbol"].astype(str)
    tmp["datetime"] = pd.to_datetime(tmp["datetime"])
    ref_index = pd.MultiIndex.from_arrays([tmp["symbol"].values, tmp["datetime"].values],
                                          names=["symbol", "datetime"])
    return load_external(ff3_path, benchmark_path, ref_index)


if __name__ == "__main__":
    main()
