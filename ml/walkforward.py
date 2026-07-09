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


def run_walkforward(
    panel: pd.DataFrame,
    horizon: int = 5,
    min_train_months: int = 6,
    purge_days: int = 5,
    embargo_days: int = 5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward loop.

    Returns:
      oos_preds: long DataFrame [datetime, symbol, close, pred, fut_ret] for
                 all OOS months concatenated. `pred` is the composite alpha.
      month_log: per-month stats [test_month, n_train, n_test, mean_ic].
    """
    panel = attach_forward_return(panel, horizon=horizon)
    factor_cols = factor_columns(panel)

    oos_chunks: list[pd.DataFrame] = []
    month_log: list[dict] = []

    splits = list(build_splits(
        panel,
        horizon=horizon,
        min_train_months=min_train_months,
        purge_days=purge_days,
        embargo_days=embargo_days,
    ))
    if verbose:
        print(f"[wf] {len(splits)} test months to process; "
              f"{len(factor_cols)} features each")

    for i, split in enumerate(splits, 1):
        train = split.train.sort_values(["datetime", "symbol"])
        test = split.test.sort_values(["datetime", "symbol"])
        label_col = f"fut_ret_{horizon}"

        X_train = train[factor_cols].values
        y_train = train[label_col].values
        X_test = test[factor_cols].values

        model = train_lgbm(X_train, y_train, feature_names=factor_cols)

        # Predict test month
        preds = predict_lgbm(model, X_test)
        test = test.assign(pred=preds)
        oos_chunk = test[["datetime", "symbol", "close", "pred", label_col]].rename(
            columns={label_col: "fut_ret"}
        )
        oos_chunks.append(oos_chunk)

        # Per-month OOS IC
        ic_series = test.groupby("datetime").apply(
            _ic_per_date, "pred", label_col, include_groups=False
        ).dropna()
        mean_ic = float(ic_series.mean()) if len(ic_series) else float("nan")

        month_log.append({
            "test_month": split.test_month,
            "n_train": split.n_train,
            "n_test": split.n_test,
            "mean_ic": mean_ic,
        })

        if verbose:
            print(f"[wf] ({i}/{len(splits)}) {split.test_month}  "
                  f"n_train={split.n_train:>6}  n_test={split.n_test:>5}  "
                  f"OOS_IC={mean_ic:+.4f}")

    oos_preds = pd.concat(oos_chunks, ignore_index=True) if oos_chunks else pd.DataFrame()
    month_log_df = pd.DataFrame(month_log)
    return oos_preds, month_log_df
