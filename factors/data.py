"""VWAP derivation and external factor (FF3, benchmark) injection.

`ensure_vwap(df)` returns the VWAP column, derived in this order:
    1. existing `vwap` column
    2. `amount / volume` (baostock amount/volume match the standard VWAP definition)
    3. typical price `(high + low + close) / 3` (last-resort fallback)

`load_external(ff3_path, benchmark_path)` reads optional parquet files and
returns a `ctx` dict (`{mkt, smb, hml, bench_open, bench_close}`) for alphas
that need them. Missing alphas are skipped by `compute_alpha_factors`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYMBOL_LEVEL = "symbol"
DATE_LEVEL = "datetime"


def ensure_vwap(df: pd.DataFrame) -> pd.Series:
    if "vwap" in df.columns:
        return df["vwap"].astype(float)
    if "amount" in df.columns and "volume" in df.columns:
        vol = df["volume"].astype(float)
        amt = df["amount"].astype(float)
        return amt / vol.replace(0, np.nan)
    return (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0


def _load_panel(path: str) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[factors.data] failed to load {path}: {e}")
        return None


def _align_to(panel: pd.DataFrame, ref_index: pd.MultiIndex,
              name: str, symbol_col: str = "symbol", date_col: str = "datetime") -> pd.Series:
    """Align a (symbol, datetime)-keyed panel column to ref_index."""
    if name not in panel.columns:
        return None
    sub = panel[[symbol_col, date_col, name]].copy()
    sub[symbol_col] = sub[symbol_col].astype(str)
    sub = sub.set_index([symbol_col, date_col]).sort_index()
    s = sub[name].astype(float)
    return s.reindex(ref_index)


def load_external(ff3_path: str | None,
                  benchmark_path: str | None,
                  ref_index: pd.MultiIndex) -> dict:
    """Build a ctx dict for alphas that need MKT/SMB/HML/BENCH_OPEN/BENCH_CLOSE.

    Each value is a Series aligned to `ref_index`, or absent if unavailable.
    """
    ctx: dict = {}

    ff3 = _load_panel(ff3_path) if ff3_path else None
    if ff3 is not None:
        for col, key in [("mkt", "mkt"), ("smb", "smb"), ("hml", "hml")]:
            s = _align_to(ff3, ref_index, col)
            if s is not None:
                ctx[key] = s

    bench = _load_panel(benchmark_path) if benchmark_path else None
    if bench is not None:
        for col, key in [("open", "bench_open"), ("close", "bench_close")]:
            s = _align_to(bench, ref_index, col)
            if s is not None:
                ctx[key] = s

    return ctx
