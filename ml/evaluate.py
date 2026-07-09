"""OOS evaluation: IC, t-stat, quantile spread, and parquet export."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.stats as stats


def evaluate_oos(oos_preds: pd.DataFrame) -> dict:
    """Compute pooled IC stats on the concatenated OOS predictions.

    Expected columns: datetime, symbol, pred, fut_ret
    """
    if oos_preds.empty:
        return {}

    # Per-date cross-sectional spearman IC
    ic_per_date = oos_preds.groupby("datetime").apply(
        lambda g: g["pred"].corr(g["fut_ret"], method="spearman")
        if len(g) >= 5 else np.nan,
        include_groups=False,
    ).dropna()

    mean_ic = float(ic_per_date.mean())
    std_ic = float(ic_per_date.std(ddof=1))
    ir = mean_ic / std_ic if std_ic > 0 else float("nan")
    t_stat, p_value = stats.ttest_1samp(ic_per_date.values, 0.0)

    # Pooled spearman
    pooled = oos_preds["pred"].corr(oos_preds["fut_ret"], method="spearman")

    # Quantile means (5 bins, by pred within each date)
    def _qbin(s):
        if len(s) < 5:
            return pd.Series(np.nan, index=s.index)
        try:
            return pd.qcut(s, 5, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=s.index)

    oos_preds = oos_preds.copy()
    oos_preds["q_bin"] = oos_preds.groupby("datetime")["pred"].transform(_qbin)
    q_means = oos_preds.dropna(subset=["q_bin"]).groupby("q_bin")["fut_ret"].mean()

    return {
        "n_obs": len(oos_preds),
        "n_dates": int(ic_per_date.shape[0]),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ir": ir,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "pooled_spearman": float(pooled),
        "quantile_means": {f"Q{i+1}": float(q_means.loc[i]) if i in q_means.index else float("nan")
                          for i in range(5)},
    }


def save_oas_parquet(oos_preds: pd.DataFrame, path: str,
                    pred_col: str = "pred"):
    """Save OOS predictions in the panel format FactorAnalyzer expects:
        datetime, symbol, close, alpha_composite
    """
    out = oos_preds[["datetime", "symbol", "close", pred_col]].rename(
        columns={pred_col: "alpha_composite"}
    )
    out = out.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def print_summary(stats: dict, month_log: pd.DataFrame):
    print("\n" + "=" * 70)
    print("OOS Summary (walk-forward, expanding window, purge+embargo)")
    print("=" * 70)
    if not stats:
        print("(no OOS predictions)")
        return
    print(f"  observations:   {stats['n_obs']:>8}")
    print(f"  test dates:     {stats['n_dates']:>8}")
    print(f"  mean IC:        {stats['mean_ic']:+.4f}")
    print(f"  std IC:         {stats['std_ic']:.4f}")
    print(f"  IR:             {stats['ir']:+.4f}")
    print(f"  t-stat:         {stats['t_stat']:+.4f}")
    print(f"  p-value:        {stats['p_value']:.4f}")
    print(f"  pooled spearman:{stats['pooled_spearman']:+.4f}")
    print("\n  quantile means (avg fut_ret per bin):")
    for q, v in stats["quantile_means"].items():
        print(f"     {q}: {v:+.5f}")

    print("\nPer-month IC:")
    print(month_log[["test_month", "n_train", "n_test", "mean_ic"]].to_string(index=False))
