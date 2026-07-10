"""Smoke test: construct a small synthetic panel and run every alpha to make
sure nothing crashes. Numeric correctness for specific alphas is checked by
hand-computed expectations where feasible.
"""

import numpy as np
import pandas as pd
import pytest

from factors import compute_alpha_factors, ALPHA191_REGISTRY
from factors import operators as op


@pytest.fixture
def small_panel():
    """Synthetic panel where all symbols start from the same base, so cross-sectional
    ranks fluctuate day to day (realistic for short-term factor behavior)."""
    rng = np.random.default_rng(42)
    n_dates = 300
    n_syms = 20
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    syms = [f"s{i:02d}" for i in range(n_syms)]
    rows = []
    for sym in syms:
        # Independent random walks all starting from 100 — ranks will fluctuate
        rets = rng.standard_normal(n_dates) * 0.02
        close = 100 * np.exp(np.cumsum(rets))
        open_ = close * (1 + rng.standard_normal(n_dates) * 0.005)
        high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n_dates)) * 0.005)
        low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n_dates)) * 0.005)
        volume = 1e6 * (1 + 0.5 * rng.standard_normal(n_dates)).clip(min=0.1)
        amount = volume * (high + low + close) / 3
        for i in range(n_dates):
            rows.append((dates[i], sym, open_[i], high[i], low[i], close[i], volume[i], amount[i]))
    df = pd.DataFrame(rows, columns=["datetime", "symbol", "open", "high", "low", "close", "volume", "amount"])
    return df


def test_registry_complete():
    assert len(ALPHA191_REGISTRY) == 191
    missing = set(f"{i:03d}" for i in range(1, 192)) - set(ALPHA191_REGISTRY.keys())
    assert not missing


def test_all_alphas_run(small_panel):
    """Every alpha must execute without raising. The smoke test just makes sure
    no alpha crashes and produces at least one finite value in the warmed-up tail.

    We do NOT assert a high finite ratio here because synthetic pct-rank data
    has many ties, and rolling CORR is legitimately undefined when one input
    has zero variance in a window. Real numeric validation happens against
    market data in `test_alpha191.py::test_real_panel_smoke`.

    5 alphas (030/075/149/181/182) need external data and are skipped."""
    df = small_panel
    out = compute_alpha_factors(df, names=None, skip_on_error=False)
    alpha_cols = [c for c in out.columns if c.startswith("alpha191_")]
    assert len(alpha_cols) == 186  # 191 - 5 external
    last_idx = out.groupby("symbol").tail(100)
    failures = []
    for col in alpha_cols:
        s = last_idx[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            failures.append(col)
    assert not failures, f"all-NaN alphas in tail: {failures}"


def test_external_data_alphas_skipped(small_panel):
    """Alphas requiring MKT/SMB/HML/BENCHMARK should be skipped without ctx."""
    df = small_panel
    out = compute_alpha_factors(df, names=["030", "075", "149", "181", "182"], skip_on_error=True)
    for num in ["030", "075", "149", "181", "182"]:
        col = f"alpha191_{num}"
        assert col not in out.columns or out[col].isna().all()


# Hand-checked operator correctness ------------------------------------------

def test_delay_delta():
    s = pd.Series([1.0, 2, 3, 4, 5], index=pd.MultiIndex.from_tuples(
        [("A", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)) for i in range(5)],
        names=["symbol", "datetime"]))
    assert op.delay(s, 1).iloc[1] == 1.0
    assert op.delta(s, 1).iloc[1] == 1.0


def test_sma_recursive():
    """SMA(X, N, M): Y_t = (X_t*M + Y_{t-1}*(N-M)) / N
    For N=3, M=1, alpha=1/3, with X=[3,5,7,9]:
      Y_0 = 3
      Y_1 = (5 + 2*3) / 3 = 11/3
      Y_2 = (7 + 2*11/3) / 3 = (21+22)/9 = 43/9
    """
    s = pd.Series([3.0, 5, 7, 9], index=pd.MultiIndex.from_tuples(
        [("A", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)) for i in range(4)],
        names=["symbol", "datetime"]))
    got = op.sma(s, 3, 1).iloc
    assert abs(got[0] - 3.0) < 1e-10
    assert abs(got[1] - 11/3) < 1e-10
    assert abs(got[2] - 43/9) < 1e-10


def test_wma():
    """WMA(X, 3) with weights [1,2,3]/6.
      idx 2 window [1,2,3]: (1*1+2*2+3*3)/6 = 14/6
      idx 3 window [2,3,4]: (2*1+3*2+4*3)/6 = 20/6
      idx 4 window [3,4,5]: (3*1+4*2+5*3)/6 = 26/6"""
    s = pd.Series([1.0, 2, 3, 4, 5], index=pd.MultiIndex.from_tuples(
        [("A", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)) for i in range(5)],
        names=["symbol", "datetime"]))
    got = op.wma(s, 3).iloc
    assert np.isnan(got[0])
    assert np.isnan(got[1])
    assert abs(got[2] - 14/6) < 1e-10
    assert abs(got[3] - 20/6) < 1e-10
    assert abs(got[4] - 26/6) < 1e-10


def test_rank_cross_sectional():
    """Rank must be within-date percentile rank."""
    idx = pd.MultiIndex.from_tuples([
        ("A", pd.Timestamp("2024-01-01")),
        ("B", pd.Timestamp("2024-01-01")),
        ("C", pd.Timestamp("2024-01-01")),
    ], names=["symbol", "datetime"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    got = op.rank(s)
    assert abs(got.iloc[0] - 1/3) < 1e-10  # smallest -> 1/3
    assert abs(got.iloc[2] - 1.0) < 1e-10  # largest -> 1.0


def test_corr_rolling():
    """CORR(x, y, n) with perfect positive correlation should be 1.0."""
    idx = pd.MultiIndex.from_tuples([
        ("A", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)) for i in range(5)
    ], names=["symbol", "datetime"])
    x = pd.Series([1.0, 2, 3, 4, 5], index=idx)
    y = pd.Series([2.0, 4, 6, 8, 10], index=idx)
    got = op.corr(x, y, 3).iloc
    assert abs(got[2] - 1.0) < 1e-10
    assert abs(got[4] - 1.0) < 1e-10
