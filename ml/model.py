"""LightGBM training/prediction wrapper for multi-factor synthesis.

Default hyperparameters favour shallow, regularised trees — short panels
overfit easily. Bumping max_depth or n_estimators needs OOS evidence to
justify. No holdout/early-stopping by default: with monthly walk-forward
the per-window OOS IC is the stopping signal, and reserving 20% of train
just to pick the iteration count wastes data the model rarely has too much
of.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb


def default_lgbm_params(n_features: int = 59) -> dict:
    """Conservative defaults for a multi-year panel with ~50-200 features.

    - max_depth=3 + num_leaves=8: shallow, almost linear at the leaves
    - lr=0.05, n_estimators=100: ~5 effective trees worth of complexity.
      Tested 50/100/200 on the 520-symbol alpha191 panel: 50 underfits
      (IC drops ~25%), 200 overfits recent months, 100 is the sweet spot.
    - min_child_samples=200: leaf needs ≥200 samples (about a third of a
      trading day for our 520-symbol universe). Pushing to 500 starves the
      model in rolling-window mode where train is only 60k rows.
    - reg_lambda=1.0, reg_alpha=0.1: light L2 for collinearity damping.
    - feature_fraction=0.8, bagging_fraction=0.8: mild randomness for robustness.
    """
    return {
        "objective": "regression",
        "metric": "rmse",
        "max_depth": 3,
        "num_leaves": 8,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 200,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
    }


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    params: dict | None = None,
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor on the full train slice (no holdout).

    No eval_set, no early stopping — `n_estimators` is fixed by `params`.
    To choose it, sweep a few values (50/100/200) and compare monthly OOS
    IC stats from `run_walkforward`. The walk-forward itself is the
    validation signal.
    """
    p = params or default_lgbm_params(X_train.shape[1])
    model = lgb.LGBMRegressor(**p)
    model.fit(X_train, y_train, feature_name=feature_names)
    return model


def predict_lgbm(model: lgb.LGBMRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def feature_importance(model: lgb.LGBMRegressor, feature_names: list[str]) -> dict:
    """Return split-count importance per feature."""
    booster = model.booster_
    imp = booster.feature_importance(importance_type="split")
    return {name: int(v) for name, v in zip(feature_names, imp)}
