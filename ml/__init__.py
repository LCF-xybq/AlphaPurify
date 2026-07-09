"""Multi-factor synthesis via ML.

Walk-forward LightGBM that combines 59 TA-Lib factors into a single composite
alpha. Strict out-of-sample: every month retrained on expanding history with
purged training tail and embargoed test head to prevent label leakage.
"""

from .dataset import load_panel, generate_walkforward_splits, fit_preprocessor, transform
from .model import train_lgbm, predict_lgbm, default_lgbm_params
from .walkforward import run_walkforward
from .evaluate import evaluate_oos, save_oas_parquet

__all__ = [
    "load_panel",
    "generate_walkforward_splits",
    "fit_preprocessor",
    "transform",
    "train_lgbm",
    "predict_lgbm",
    "default_lgbm_params",
    "run_walkforward",
    "evaluate_oos",
    "save_oas_parquet",
]
