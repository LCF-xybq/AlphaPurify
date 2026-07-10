"""Walk-forward main loop: monthly expanding-window LightGBM.

For each test month:
  1. Train on all rows strictly before the test month (minus purge_days tail).
  2. Preprocess (winsorize + zscore) using parameters fitted on TRAIN only.
  3. Train LightGBM on the FULL train slice — no holdout, no early stopping.
     `n_estimators` is fixed in `default_lgbm_params`; sweep it via OOS IC.
  4. Predict on the test month (after embargo), record per-date IC.

Returns a DataFrame with all OOS predictions concatenated.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from .dataset import (
    attach_forward_return,
    build_splits,
    factor_columns,
)
from .model import train_lgbm, predict_lgbm


def _ic_per_date(group: pd.DataFrame, pred_col: str, label_col: str) -> float:
    if len(group) < 5:
        return np.nan
    return group[pred_col].corr(group[label_col], method="spearman")


def _count_test_months(panel: pd.DataFrame, min_train_months: int) -> int:
    """Count test months without preprocessing — just calendar arithmetic."""
    trading_days = pd.DatetimeIndex(sorted(panel["datetime"].unique()))
    if len(trading_days) == 0:
        return 0
    months = sorted({(d.year, d.month) for d in trading_days})
    return max(0, len(months) - min_train_months)


def run_walkforward(
    panel: pd.DataFrame,
    horizon: int = 5,
    min_train_months: int = 6,
    purge_days: int = 5,
    embargo_days: int = 5,
    rank_transform: bool = False,
    train_window_months: int | None = None,
    rank_normalize_pred: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward loop.

    Returns:
      oos_preds: long DataFrame [datetime, symbol, close, pred, fut_ret] for
                 all OOS months concatenated. `pred` is the composite alpha
                 (per-date rank-normalized to (0, 1) when rank_normalize_pred
                 is True — eliminates scale drift across months).
      month_log: per-month stats [test_month, n_train, n_test, mean_ic].

    Memory: iterates `build_splits` as a generator and forces gc.collect()
    after each window so peak RAM is bounded by ONE window's preprocessed
    train+test (not all 21 at once). Materializing all splits with list()
    was the OOM killer on the full 520-symbol panel.
    """
    panel = attach_forward_return(panel, horizon=horizon)
    factor_cols = factor_columns(panel)

    split_kwargs = dict(
        horizon=horizon,
        min_train_months=min_train_months,
        purge_days=purge_days,
        embargo_days=embargo_days,
        rank_transform=rank_transform,
        train_window_months=train_window_months,
    )
    if verbose:
        n_splits = _count_test_months(panel, min_train_months)
        win_desc = "expanding" if train_window_months is None else f"rolling {train_window_months}M"
        feat_desc = "rank" if rank_transform else "zscore"
        pred_desc = "rank-normalized" if rank_normalize_pred else "raw"
        print(f"[wf] up to {n_splits} test months; {len(factor_cols)} features; "
              f"train={win_desc}; features={feat_desc}; pred={pred_desc}")

    oos_chunks: list[pd.DataFrame] = []
    month_log: list[dict] = []

    for i, split in enumerate(build_splits(panel, **split_kwargs), 1):
        train = split.train.sort_values(["datetime", "symbol"])
        test = split.test.sort_values(["datetime", "symbol"])
        label_col = f"fut_ret_{horizon}"

        # Skip windows where train or test has no usable rows after label-NaN drop
        # (typically the last partial month where forward-return labels are NaN).
        if len(train) == 0 or len(test) == 0 or test[label_col].notna().sum() == 0:
            if verbose:
                print(f"[wf] ({i}) {split.test_month}  SKIPPED "
                      f"(n_train={len(train)}, n_test={len(test)}, "
                      f"test_finite_label={int(test[label_col].notna().sum())})")
            del train, test
            gc.collect()
            continue

        # Cast to float32 to halve memory before handing to LightGBM
        X_train = train[factor_cols].to_numpy(dtype=np.float32, copy=False, na_value=np.nan)
        y_train = train[label_col].to_numpy(dtype=np.float32, copy=False)
        X_test = test[factor_cols].to_numpy(dtype=np.float32, copy=False, na_value=np.nan)

        model = train_lgbm(X_train, y_train, feature_names=factor_cols)

        preds = predict_lgbm(model, X_test)
        # Per-date cross-sectional rank normalize to (0, 1). This eliminates
        # the LightGBM output scale drift across walk-forward windows — a
        # window trained on noisier data produces predictions with std ~0.02
        # while a cleaner window has std ~0.005, and pooled-spearman picks
        # that up as negative correlation even though per-date IC is positive.
        if rank_normalize_pred:
            tmp = pd.DataFrame({"datetime": test["datetime"].values, "pred": preds})
            tmp["pred"] = tmp.groupby("datetime")["pred"].rank(pct=True, method="average")
            preds = tmp["pred"].values
        oos_chunk = pd.DataFrame({
            "datetime": test["datetime"].values,
            "symbol": test["symbol"].values,
            "close": test["close"].values,
            "pred": preds,
            "fut_ret": test[label_col].values,
        })
        oos_chunks.append(oos_chunk)

        # Per-date OOS IC on the slim chunk (avoids re-materializing test)
        ic_series = oos_chunk.groupby("datetime").apply(
            _ic_per_date, "pred", "fut_ret", include_groups=False
        ).dropna()
        mean_ic = float(ic_series.mean()) if len(ic_series) else float("nan")

        month_log.append({
            "test_month": split.test_month,
            "n_train": split.n_train,
            "n_test": split.n_test,
            "mean_ic": mean_ic,
        })

        if verbose:
            print(f"[wf] ({i}) {split.test_month}  "
                  f"n_train={split.n_train:>6}  n_test={split.n_test:>5}  "
                  f"OOS_IC={mean_ic:+.4f}")

        # Free per-window memory before the next iteration materializes new splits
        del train, test, X_train, y_train, X_test, model
        gc.collect()

    oos_preds = pd.concat(oos_chunks, ignore_index=True) if oos_chunks else pd.DataFrame()
    month_log_df = pd.DataFrame(month_log)
    return oos_preds, month_log_df
