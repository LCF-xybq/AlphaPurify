"""
Fetch OHLCV panel data from the local DuckDB store and compute TA-Lib factors
according to a YAML combination config.

Usage:
    python tools/research/fetch_data.py --config data/combination1.yaml --save out.parquet

Config layout (see data/combination1.yaml):
    codes:    [sh.601991, sz.002491, ...]
    date:     [{start: "2026-06-01", end: "2026-07-01"}]
    alpha:    [sma, macd]        # primary factor(s) — also accepts key 'factor'
    exposure: [rsi, cmo, mom]    # other risk factors for attribution (optional)

Both `alpha` and `exposure` columns are computed and written to the output.
Output columns: datetime, symbol, close, volume, <alpha_cols...>, <exposure_cols...>
This script does NOT winsorize / standardize — that is left to backtest.py.
"""

import argparse
import os
import sys

import duckdb
import pandas as pd
import yaml

# Reuse the TA-Lib factor registry already maintained in tools/process_data/pre_factor.py
_PROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "process_data")
sys.path.insert(0, os.path.abspath(_PROCESS_DIR))
from pre_factor import ALL_FACTOR_DEFS, compute_talib_factors  # noqa: E402

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


def load_config(path: str):
    """Parse combination yaml → (codes, start, end, alpha_names, exposure_names)."""
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

    return codes, start, end, alpha_list, exposure_list


def resolve_factor_indices(factor_names):
    """Map user-facing factor names to indices in ALL_FACTOR_DEFS."""
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
                f"Unknown factor '{raw}'. Available: {sorted(NAME_TO_IDX)}"
            )
    return sorted(set(indices))


def fetch_panel(db_path: str, codes, start, end, warmup_days: int = 0) -> pd.DataFrame:
    """Query the DuckDB kline table for the requested symbols and date range.

    When ``warmup_days > 0``, the query extends ``start`` backwards by that many
    calendar days so TA-Lib factors have enough history to be defined at the
    very first row of the requested range. The warmup rows are returned to the
    caller (which computes factors on them) and trimmed afterwards.
    """
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
    query = f"""
        SELECT code, date, open, high, low, close, volume
        FROM {_TABLE}
        WHERE code IN ({codes_literal})
          AND date >= DATE '{fetch_start}'
          AND date <= DATE '{end}'
        ORDER BY code, date
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise ValueError(
            f"No rows returned. Check codes (e.g. {codes[:3]}) and date range "
            f"[{fetch_start}, {end}] against table '{_TABLE}'."
        )

    df = df.rename(columns={"code": "symbol", "date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
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
        description="Fetch stock panel data and compute TA-Lib factors from a YAML combination config."
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
        default=60,
        help=(
            "Calendar days to pre-fetch before `start` for factor warmup "
            "(default 60, covering SMA(20), MACD(26), etc.). "
            "These rows are used to seed factor values and then trimmed."
        ),
    )
    args = parser.parse_args()

    codes, start, end, alpha_list, exposure_list = load_config(args.config)
    factor_list = list(alpha_list) + list(exposure_list)
    factor_indices = resolve_factor_indices(factor_list)
    print(f"[fetch] symbols: {len(codes)}  range: [{start}, {end}]")
    print(f"[fetch] alpha:    {alpha_list}")
    print(f"[fetch] exposure: {exposure_list}")

    db_path = os.path.abspath(args.db) if args.db else _DEFAULT_DB
    df = fetch_panel(db_path, codes, start, end, warmup_days=args.warmup)
    print(f"[fetch] panel rows={len(df)}  symbols={df['symbol'].nunique()}  (incl. {args.warmup}d warmup)")

    df, produced = compute_factors(df, factor_indices)

    # Trim warmup rows — output only the requested [start, end] window
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].reset_index(drop=True)

    keep = ["datetime", "symbol", "close", "volume"] + [
        c for c in produced if c in df.columns
    ]
    df = df[keep]

    save_panel(df, args.save)
    print(f"[fetch] saved -> {args.save}  ({len(df)} rows, cols={df.columns.tolist()})")


if __name__ == "__main__":
    main()
