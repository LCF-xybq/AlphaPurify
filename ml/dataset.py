"""Panel loading, walk-forward splitting, and per-window preprocessing.

Design constraints (anti-leakage):
  - Walk-forward is monthly. Train is expanding (all rows before test month).
  - Min train months gates the first test month (default 6).
  - Each window computes its own winsorize thresholds, industry means, and
    zscore mean/std on the train slice, then applies them to both train and
    test. The global panel statistics never enter a window's preprocessing.
  - Industry neutralization is applied when an `industry` column is present:
    subtract per-industry means (date-averaged on train) from each factor.
    This prevents the model from learning sector-wide level differences and
    forces it to learn within-industry ranking signal.
  - Label = forward 5-day return. Because the label observes future bars,
    the train tail within `purge_days` of the test month is dropped (purge),
    and the test head within `purge_days` of train is dropped (embargo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

# Columns considered metadata, not factors
_META_COLS = {"datetime", "symbol", "open", "high", "low", "close", "volume", "amount", "industry"}


@dataclass
class WindowSplit:
    """One walk-forward window."""
    train: pd.DataFrame          # preprocessed train (with label)
    test: pd.DataFrame           # preprocessed test (with label)
    test_month: str              # "YYYY-MM"
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # last train date BEFORE purge
    n_train: int
    n_test: int


def load_panel(parquet_path: str) -> pd.DataFrame:
    """Load the all-factors parquet produced by fetch_data.py.

    Expected columns: datetime, symbol, close, volume, <59 factor cols>.
    """
    df = pd.read_parquet(parquet_path)
    if "datetime" not in df.columns or "symbol" not in df.columns or "close" not in df.columns:
        raise ValueError(
            f"Panel must have datetime/symbol/close columns, got {df.columns.tolist()}"
        )
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return df


def factor_columns(df: pd.DataFrame) -> list[str]:
    """All non-meta columns, excluding forward-return labels added by
    attach_forward_return. A common leakage trap is to forget this filter
    and end up training on the label itself."""
    return [
        c for c in df.columns
        if c not in _META_COLS and not c.startswith("fut_ret_")
    ]


def attach_forward_return(df: pd.DataFrame, horizon: int,
                          price_col: str = "close",
                          symbol_col: str = "symbol",
                          date_col: str = "datetime") -> pd.DataFrame:
    """Add `fut_ret_{horizon}` column = price(T+h)/price(T) - 1, per symbol."""
    out = df.sort_values([symbol_col, date_col]).copy()
    out[f"fut_ret_{horizon}"] = (
        out.groupby(symbol_col)[price_col].shift(-horizon) / out[price_col] - 1
    )
    return out


def generate_walkforward_splits(
    panel: pd.DataFrame,
    min_train_months: int = 6,
    purge_days: int = 5,
    embargo_days: int = 5,
    train_window_months: int | None = None,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yield (train_start, train_end_inclusive, test_start, test_end_inclusive)
    tuples in calendar dates.

    - First test month begins after `min_train_months` of training data.
    - purge_days and embargo_days are in TRADING DAYS (not calendar days),
      because forward-return labels look `horizon` trading days ahead —
      using calendar days leaks future bars when the gap spans weekends.
    - train_end_inclusive = the trading day that is `purge_days` trading days
      before the first trading day of the test month. This guarantees that a
      horizon-`purge_days` forward-return label on the last train row resolves
      strictly before the test month.
    - test_start = first trading day of test month + `embargo_days` trading days.
    - test_end_inclusive = last trading day of test month.
    - train_window_months: None = expanding (train_start fixed at panel start,
      default). If int (e.g. 6), train_start rolls forward — only the trailing
      N months before the test month are used as training data. Rolling adapts
      faster to regime changes at the cost of fewer training rows per window.
    """
    trading_days = pd.DatetimeIndex(sorted(panel["datetime"].unique()))
    if len(trading_days) == 0:
        return

    td_list = list(trading_days)
    panel_start = trading_days[0]
    months = sorted({(d.year, d.month) for d in trading_days})

    for i in range(min_train_months, len(months)):
        test_year, test_month = months[i]
        month_start = pd.Timestamp(year=test_year, month=test_month, day=1)
        if test_month == 12:
            next_month_first = pd.Timestamp(year=test_year + 1, month=1, day=1)
        else:
            next_month_first = pd.Timestamp(year=test_year, month=test_month + 1, day=1)

        test_month_days = trading_days[
            (trading_days >= month_start) & (trading_days < next_month_first)
        ]
        if len(test_month_days) <= embargo_days:
            continue
        first_test_day = test_month_days[0]
        test_start = test_month_days[embargo_days]
        test_end = test_month_days[-1]

        # Train end: purge_days trading days before first_test_day
        first_test_pos = td_list.index(first_test_day)
        if first_test_pos <= purge_days:
            continue
        train_end = td_list[first_test_pos - purge_days - 1]

        # Train start: expanding (panel_start) OR rolling N months back
        if train_window_months is None:
            train_start = panel_start
            # In expanding mode, gate the first test month on min_train_months
            if train_end < panel_start + pd.Timedelta(days=30 * min_train_months - 5):
                continue
        else:
            # In rolling mode, train_window_months itself is the gate. We just
            # need to make sure the rolling-start month index is valid; if it
            # would go negative, clamp to 0 (effectively expanding for the
            # first few windows until enough months accumulate).
            win_start_idx = max(0, i - train_window_months)
            win_year, win_month = months[win_start_idx]
            win_start = pd.Timestamp(year=win_year, month=win_month, day=1)
            win_start_days = trading_days[trading_days >= win_start]
            train_start = win_start_days[0] if len(win_start_days) else panel_start
            # Skip only if we haven't accumulated min_train_months of panel
            # data yet (so the very first eligible test month is still gated).
            if i < min_train_months:
                continue

        yield train_start, train_end, test_start, test_end


# =============================================================================
# Per-window preprocessing — parameters fitted on TRAIN only
# =============================================================================

@dataclass
class Preprocessor:
    """Cross-sectional winsorize (mad) + industry-neutralize + zscore, fitted per factor."""
    median: pd.Series                      # factor -> median
    mad: pd.Series                         # factor -> mad
    industry_means: pd.DataFrame | None    # index=industry, cols=factor_cols (date-averaged train means)
    mean: pd.Series                        # factor -> mean (after winsorize + neutralize)
    std: pd.Series                         # factor -> std (after winsorize + neutralize)
    factor_cols: list[str]
    winsorize_k: float = 5.0               # MAD multiplier for winsorize


def fit_preprocessor(train_df: pd.DataFrame, factor_cols: list[str],
                     date_col: str = "datetime",
                     industry_col: str = "industry",
                     winsorize_k: float = 5.0) -> Preprocessor:
    """Fit winsorize + industry-neutralize + zscore parameters on the train slice.

    Pipeline (computed on train only):
      1. Per-date cross-sectional median + MAD → averaged across dates → winsorize thresholds.
      2. Winsorize train values.
      3. If `industry_col` exists: compute per-(date, industry) means on winsorized
         values, then average across dates to get a single mean per (industry, factor).
      4. Subtract industry means (per-row) from winsorized values.
      5. Compute final mean/std on winsorized+neutralized values for zscore.

    Averaging per-date stats into single per-factor numbers (rather than keeping
    per-date arrays) prevents train-specific date patterns from leaking into
    test preprocessing.
    """
    sub = train_df[[date_col] + factor_cols]
    grouped = sub.groupby(date_col)
    medians = grouped.median(numeric_only=True)
    mads = grouped.apply(
        lambda g: (g[factor_cols] - g[factor_cols].median()).abs().median(),
        include_groups=False,
    )
    medians_mean = medians.mean(axis=0)
    mad_mean = mads.mean(axis=0).fillna(1e-9)

    clip_lo = medians_mean - winsorize_k * mad_mean
    clip_hi = medians_mean + winsorize_k * mad_mean
    clipped = train_df[factor_cols].clip(clip_lo, clip_hi, axis=1)

    # Industry means (per industry, date-averaged) on winsorized values
    industry_means: pd.DataFrame | None = None
    if industry_col in train_df.columns:
        ind_series = train_df[industry_col].fillna("UNKNOWN").replace("", "UNKNOWN")
        tmp = clipped.copy()
        tmp[date_col] = train_df[date_col].values
        tmp[industry_col] = ind_series.values
        per_date_ind = tmp.groupby([date_col, industry_col])[factor_cols].mean()
        industry_means = per_date_ind.groupby(level=industry_col).mean()

        # Subtract industry means from clipped values
        for f in factor_cols:
            mapped = ind_series.map(industry_means[f]).fillna(0.0)
            clipped[f] = clipped[f].values - mapped.values

    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0).replace(0, 1e-9)

    return Preprocessor(
        median=medians_mean, mad=mad_mean,
        industry_means=industry_means,
        mean=mean, std=std,
        factor_cols=list(factor_cols),
        winsorize_k=winsorize_k,
    )


def transform(df: pd.DataFrame, prep: Preprocessor,
              industry_col: str = "industry",
              date_col: str = "datetime",
              rank_transform: bool = False) -> pd.DataFrame:
    """Apply winsorize + industry-neutralize + (zscore OR per-date rank).

    When rank_transform=True, the final step replaces zscore with per-date
    cross-sectional percentile rank in (0, 1). Rank features have stable
    scale across dates but lose magnitude information — empirically this
    HURTS LightGBM IC on the alpha191 panel (zscore wins by ~25% on t-stat)
    because LightGBM trains on continuous labels and benefits from the
    magnitude signal. Default is zscore (False); rank_transform is kept as
    an option for experimentation.
    """
    clip_lo = prep.median - prep.winsorize_k * prep.mad
    clip_hi = prep.median + prep.winsorize_k * prep.mad
    out = df.copy()

    has_industry = prep.industry_means is not None and industry_col in out.columns
    if has_industry:
        ind_series = out[industry_col].fillna("UNKNOWN").replace("", "UNKNOWN")

    for f in prep.factor_cols:
        # 1. Winsorize
        out[f] = df[f].clip(clip_lo[f], clip_hi[f])
        # 2. Industry neutralize
        if has_industry:
            mapped = ind_series.map(prep.industry_means[f]).fillna(0.0)
            out[f] = out[f].values - mapped.values
        # 3. Standardize: zscore (when rank_transform=False) or per-date rank
        if rank_transform:
            out[f] = out.groupby(date_col)[f].rank(pct=True, method="average")
        else:
            out[f] = (out[f] - prep.mean[f]) / prep.std[f]
    return out


# =============================================================================
# Convenience: build full window splits with preprocessing applied
# =============================================================================

def build_splits(
    panel: pd.DataFrame,
    horizon: int = 5,
    min_train_months: int = 6,
    purge_days: int = 5,
    embargo_days: int = 5,
    winsorize_k: float = 5.0,
    rank_transform: bool = False,
    train_window_months: int | None = None,
) -> Iterator[WindowSplit]:
    """Yield fully preprocessed WindowSplit objects.

    `panel` must already have `fut_ret_{horizon}` attached. Caller does that
    once on the full panel (label leakage is bounded because purge+embargo
    are applied at split time)."""
    label_col = f"fut_ret_{horizon}"
    factor_cols = factor_columns(panel)
    if label_col not in panel.columns:
        raise ValueError(f"panel must have {label_col} attached; call attach_forward_return first")

    for train_start, train_end, test_start, test_end in generate_walkforward_splits(
        panel, min_train_months=min_train_months,
        purge_days=purge_days, embargo_days=embargo_days,
        train_window_months=train_window_months,
    ):
        train_mask = (panel["datetime"] >= train_start) & (panel["datetime"] <= train_end)
        test_mask = (panel["datetime"] >= test_start) & (panel["datetime"] <= test_end)
        train_raw = panel.loc[train_mask].copy()
        test_raw = panel.loc[test_mask].copy()
        if len(train_raw) == 0 or len(test_raw) == 0:
            continue

        prep = fit_preprocessor(train_raw, factor_cols, winsorize_k=winsorize_k)
        train_proc = transform(train_raw, prep, rank_transform=rank_transform)
        test_proc = transform(test_raw, prep, rank_transform=rank_transform)

        # Drop rows where LABEL is NaN (last `horizon` days of panel).
        # Feature NaN is left in place — LightGBM handles it natively, and
        # dropping rows on any-feature-NaN would discard most of the panel
        # once alpha191's long-window formulas (SUM(RET,250), CORR(..,230))
        # are in the feature set.
        train_proc = train_proc.dropna(subset=[label_col])
        test_proc = test_proc.dropna(subset=[label_col])

        yield WindowSplit(
            train=train_proc,
            test=test_proc,
            test_month=f"{test_start.year:04d}-{test_start.month:02d}",
            train_start=train_start,
            train_end=train_end,
            n_train=len(train_proc),
            n_test=len(test_proc),
        )
