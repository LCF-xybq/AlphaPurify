"""Alpha191 factor library.

Public API:
    compute_alpha_factors(df, names=None, ctx=None, prefix="alpha191_") -> pd.DataFrame
    process_panel_dir(input_dir, factor_names=None, ctx=None, ...) -> dict
    ALPHA191_REGISTRY : dict[str, callable]
"""

from __future__ import annotations

import pandas as pd

from .alpha191 import ALPHA191_REGISTRY, MissingExternal


def _align_back(vals: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Realign a (symbol, datetime)-indexed Series to df's flat row order.
    df must contain `symbol` and `datetime` columns."""
    if not isinstance(vals.index, pd.MultiIndex):
        return pd.Series(vals.values, index=df.index)
    tmp = vals.rename("_v").reset_index()
    tmp["symbol"] = tmp["symbol"].astype(str)
    df_sym = df["symbol"].astype(str)
    df_dt = pd.to_datetime(df["datetime"])
    keys = pd.DataFrame({"symbol": df_sym.values, "datetime": df_dt.values})
    merged = keys.merge(tmp, on=["symbol", "datetime"], how="left")["_v"]
    return pd.Series(merged.values, index=df.index)


def compute_alpha_factors(
    df: pd.DataFrame,
    names: list[str] | None = None,
    ctx: dict | None = None,
    prefix: str = "alpha191_",
    skip_on_error: bool = True,
) -> pd.DataFrame:
    """Append Alpha191 factor columns to df, returning a new DataFrame.

    Args:
        df: panel DataFrame with columns datetime, symbol, open, high, low,
            close, volume, and optionally amount/vwap.
        names: alpha numbers to compute (e.g. ["001", "002"]). None = all 191.
        ctx: optional dict of external factors (mkt, smb, hml, bench_open,
            bench_close), each a Series aligned to df's index.
        prefix: output column name prefix.
        skip_on_error: if True, alphas that raise MissingExternal or other
            errors are skipped (logged); if False, the exception propagates.

    Returns:
        A new DataFrame with original columns plus alpha191_NNN columns.
    """
    out = df.copy().reset_index(drop=True)
    selected = sorted(ALPHA191_REGISTRY.keys()) if names is None else list(names)
    skipped: list[tuple[str, str]] = []
    new_cols: dict[str, pd.Series] = {}
    for num in selected:
        fn = ALPHA191_REGISTRY.get(num)
        if fn is None:
            skipped.append((num, "not in registry"))
            continue
        try:
            vals = fn(df, ctx)
            if isinstance(vals, pd.Series):
                new_cols[f"{prefix}{num}"] = _align_back(vals, df)
            else:
                new_cols[f"{prefix}{num}"] = pd.Series(vals, index=df.index)
        except MissingExternal as e:
            skipped.append((num, f"missing external: {e}"))
        except Exception as e:
            if skip_on_error:
                skipped.append((num, f"{type(e).__name__}: {e}"))
            else:
                raise
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    if skipped:
        print(f"[factors] skipped {len(skipped)} alpha(s):")
        for num, reason in skipped:
            print(f"  alpha191_{num}: {reason}")
    return out


from .io import process_panel_dir  # noqa: E402

__all__ = [
    "compute_alpha_factors",
    "process_panel_dir",
    "ALPHA191_REGISTRY",
    "MissingExternal",
]
