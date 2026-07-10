"""Pre-walk-forward factor screening.

Goal: drop factors that have no business being in the model.
  - |single-factor IC| too small (signal weaker than noise floor)
  - finite-ratio too low (long-window formulas producing mostly NaN — e.g.
    alpha_138 / alpha_064 in our 520-symbol panel)

The screening IC is computed on the FULL panel (not per walk-forward window).
That is fine for screening — we are not training on it, just deciding which
factors to keep. Per-window IC used for OOS evaluation is unchanged.

Usage:
    keep = screen_factors(panel_with_label, horizon=5, factor_cols=factor_cols)
    panel_kept = panel[meta + keep + [label]]
    run_walkforward(panel_kept, ...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def screen_factors(
    panel: pd.DataFrame,
    horizon: int = 5,
    factor_cols: list[str] | None = None,
    min_abs_ic: float = 0.005,
    min_finite_ratio: float = 0.5,
    date_col: str = "datetime",
    label_col: str | None = None,
    verbose: bool = True,
) -> list[str]:
    """Return the list of factor columns that pass both IC and finite-ratio gates.

    Args:
        panel: must already have `fut_ret_{horizon}` attached. Caller does that
               via `attach_forward_return` (keeps label leakage inside the
               screen; we are only ranking factors, not training).
        factor_cols: candidates. None = auto-detect via `factor_columns()`.
        min_abs_ic: drop factors whose mean per-date spearman IC is in
                    [-min_abs_ic, +min_abs_ic].
        min_finite_ratio: drop factors whose non-NaN ratio is below this.
    """
    if factor_cols is None:
        from .dataset import factor_columns
        factor_cols = factor_columns(panel)
    if label_col is None:
        label_col = f"fut_ret_{horizon}"
    if label_col not in panel.columns:
        raise ValueError(f"panel must have {label_col} attached; call attach_forward_return first")

    # Per-date spearman IC, then mean across dates. Per-date is the right
    # aggregation: cross-sectional IC day-by-day, averaged — matches the
    # metric used in evaluate_oos.
    def _ic_per_day(g: pd.DataFrame, col: str) -> float:
        s = g[[col, label_col]].dropna()
        if len(s) < 5:
            return np.nan
        if s[col].nunique() < 2 or s[label_col].nunique() < 2:
            return 0.0
        return s[col].corr(s[label_col], method="spearman")

    grouped = panel.groupby(date_col)

    keep: list[str] = []
    stats: list[tuple[str, float, float]] = []
    for f in factor_cols:
        ic_series = grouped.apply(_ic_per_day, f, include_groups=False).dropna()
        mean_ic = float(ic_series.mean()) if len(ic_series) else float("nan")
        finite_ratio = float(panel[f].replace([np.inf, -np.inf], np.nan).notna().mean())
        if abs(mean_ic) >= min_abs_ic and finite_ratio >= min_finite_ratio:
            keep.append(f)
        stats.append((f, mean_ic, finite_ratio))

    if verbose:
        stats.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0.0, reverse=True)
        print(f"[screen] {len(keep)}/{len(factor_cols)} factors passed "
              f"(|IC|≥{min_abs_ic}, finite≥{min_finite_ratio:.0%})")
        print(f"[screen] top 10 by |IC|:")
        for f, ic, fr in stats[:10]:
            mark = "✓" if f in keep else "✗"
            print(f"  {mark} {f:20s} IC={ic:+.4f}  finite={fr:.2f}")
        print(f"[screen] bottom 10:")
        for f, ic, fr in stats[-10:]:
            mark = "✓" if f in keep else "✗"
            print(f"  {mark} {f:20s} IC={ic:+.4f}  finite={fr:.2f}")

    return keep
