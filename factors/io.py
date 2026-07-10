"""Panel-mode IO: read all per-symbol files in a directory, merge into a single
panel, compute alpha factors, then split the result back per-symbol to the
original files (in-place).

This panel path is required because most Alpha191 formulas contain
cross-sectional RANK — those cannot be computed in per-file mode.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from . import compute_alpha_factors


def _read_one(path: str) -> pd.DataFrame | None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    return None


def _write_one(df: pd.DataFrame, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)


def process_panel_dir(
    input_dir: str,
    factor_names: list[str] | None = None,
    ctx: dict | None = None,
    file_format: str = "auto",
    prefix: str = "alpha191_",
) -> dict:
    """Read all CSV/parquet files in input_dir, merge into a panel DataFrame,
    compute alpha factors, then split back per symbol and overwrite each file.

    Each input file must contain columns: datetime, symbol, open, high, low,
    close, volume, [amount]. After processing, alpha191_NNN columns are
    appended to each file.
    """
    if file_format == "auto":
        files = sorted(glob.glob(os.path.join(input_dir, "*.csv")) +
                       glob.glob(os.path.join(input_dir, "*.parquet")))
    elif file_format == "csv":
        files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    elif file_format == "parquet":
        files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    else:
        raise ValueError(f"unsupported file_format: {file_format}")

    if not files:
        return {"success": 0, "skipped": [], "errors": ["no input files"], "alpha_skipped": []}

    pieces: list[pd.DataFrame] = []
    skipped: list[str] = []
    for fp in files:
        df = _read_one(fp)
        if df is None or "symbol" not in df.columns or "datetime" not in df.columns:
            skipped.append(fp)
            continue
        # Drop any existing alpha191_* columns so re-runs don't duplicate
        df = df[[c for c in df.columns if not c.startswith(prefix)]]
        pieces.append(df)

    if not pieces:
        return {"success": 0, "skipped": skipped, "errors": ["no valid panel files"], "alpha_skipped": []}

    panel = pd.concat(pieces, ignore_index=True)
    # Capture original column order to preserve it on write
    orig_cols = list(panel.columns)
    panel_aug = compute_alpha_factors(panel, names=factor_names, ctx=ctx, prefix=prefix)
    alpha_cols = [c for c in panel_aug.columns if c not in orig_cols]
    new_cols = orig_cols + alpha_cols
    panel_aug = panel_aug[new_cols]

    # Split per symbol and overwrite each file
    success = 0
    errors: list[str] = []
    for fp, orig_df in zip(files, pieces):
        sym = orig_df["symbol"].iloc[0] if len(orig_df) else None
        if sym is None:
            skipped.append(fp)
            continue
        sub = panel_aug[panel_aug["symbol"].astype(str) == str(sym)]
        # Restore original column order for non-alpha cols; append alpha cols at end
        try:
            _write_one(sub, fp)
            success += 1
        except Exception as e:
            errors.append(f"{fp}: {e}")

    return {
        "success": success,
        "skipped": skipped,
        "errors": errors,
        "alpha_skipped": [c for c in alpha_cols if panel_aug[c].isna().all()],
    }
