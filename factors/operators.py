"""WorldQuant-style operators for Alpha191 formulas.

All operators act on `pd.Series` indexed by a 2-level MultiIndex `(symbol, datetime)`,
sorted within each symbol by datetime. The DataFrame helper `attach_index(df, symbol_col,
date_col)` converts a flat panel DataFrame to this convention.

Time-series operators group by `symbol` (rolling within one stock).
Cross-sectional operators group by `datetime` (rank across stocks on the same day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYMBOL_LEVEL = "symbol"
DATE_LEVEL = "datetime"


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def attach_index(df: pd.DataFrame,
                 symbol_col: str = "symbol",
                 date_col: str = "datetime") -> pd.DataFrame:
    """Return a copy with a (symbol, datetime) MultiIndex, sorted within symbol."""
    out = df.copy()
    out[symbol_col] = out[symbol_col].astype(str)
    out = out.set_index([symbol_col, date_col]).sort_index()
    return out


def _ensure_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    raise TypeError(f"expected Series, got {type(x)}")


# ---------------------------------------------------------------------------
# Time-series primitives (group by symbol)
# ---------------------------------------------------------------------------

def delay(s: pd.Series, n: int) -> pd.Series:
    """DELAY(x, n): value of x n periods ago within each symbol."""
    return s.groupby(level=SYMBOL_LEVEL, group_keys=False).shift(n)


def delta(s: pd.Series, n: int) -> pd.Series:
    """DELTA(x, n): x - DELAY(x, n)."""
    return s - delay(s, n)


def tsmax(s: pd.Series, n: int) -> pd.Series:
    return s.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).max().reset_index(level=0, drop=True)


def tsmin(s: pd.Series, n: int) -> pd.Series:
    return s.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).min().reset_index(level=0, drop=True)


def sum_(s: pd.Series, n: int) -> pd.Series:
    """SUM(x, n): rolling sum over n bars."""
    return s.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)


def mean(s: pd.Series, n: int) -> pd.Series:
    """MEAN(x, n): rolling mean."""
    return s.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).mean().reset_index(level=0, drop=True)


def std(s: pd.Series, n: int) -> pd.Series:
    """STD(x, n): rolling population std (ddof=0, matches most platforms)."""
    return s.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).std(ddof=0).reset_index(level=0, drop=True)


def prod(s: pd.Series, n: int) -> pd.Series:
    """PROD(x, n): rolling product."""
    return s.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).apply(np.prod, raw=True).reset_index(level=0, drop=True)


def sumac(s: pd.Series, n: int) -> pd.Series:
    """SUMAC(x, n): accumulative sum over n periods. Equivalent to SUM(x, n) —
    the Alpha191 reference treats SUMAC and SUM identically."""
    return sum_(s, n)


def count(cond: pd.Series, n: int) -> pd.Series:
    """COUNT(cond, n): number of True in trailing n bars."""
    return cond.astype("float64").groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)


def sumif(s: pd.Series, n: int, cond: pd.Series) -> pd.Series:
    """SUMIF(x, n, cond): rolling sum of x where cond holds."""
    masked = s.where(cond.astype(bool), 0.0)
    return sum_(masked, n)


def tsrank(s: pd.Series, n: int) -> pd.Series:
    """TSRANK(x, n): percentile rank of current value within trailing n bars, per symbol."""
    def _fn(g: pd.Series) -> pd.Series:
        return g.rolling(n).apply(lambda w: pd.Series(w).rank().iloc[-1] / len(w), raw=False)
    return _by_symbol(s, _fn)


def highday(s: pd.Series, n: int) -> pd.Series:
    """HIGHDAY(x, n): position (1..n, oldest=1) of the max within the trailing n bars."""
    def _fn(g: pd.Series) -> pd.Series:
        return g.rolling(n).apply(lambda w: int(np.argmax(w)) + 1, raw=True)
    return _by_symbol(s, _fn)


def lowday(s: pd.Series, n: int) -> pd.Series:
    """LOWDAY(x, n): position (1..n, oldest=1) of the min within the trailing n bars."""
    def _fn(g: pd.Series) -> pd.Series:
        return g.rolling(n).apply(lambda w: int(np.argmin(w)) + 1, raw=True)
    return _by_symbol(s, _fn)


def _by_symbol(s: pd.Series, fn) -> pd.Series:
    """Apply fn to each per-symbol group and concat. Robust to single-group case."""
    parts = []
    for sym, g in s.groupby(level=SYMBOL_LEVEL, sort=False):
        parts.append(fn(g))
    if not parts:
        return pd.Series(dtype=float, index=s.index[:0])
    return pd.concat(parts).reindex(s.index)


def corr(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    """CORR(x, y, n): rolling Pearson correlation per symbol.
    Returns NaN when either series has zero variance in the window (avoids inf)."""
    df = pd.DataFrame({"x": x, "y": y})
    out = _by_symbol_pair(df, lambda g: g["x"].rolling(n).corr(g["y"]))
    return out.replace([np.inf, -np.inf], np.nan)


def _by_symbol_pair(df: pd.DataFrame, fn) -> pd.Series:
    parts = []
    for sym, g in df.groupby(level=SYMBOL_LEVEL, sort=False):
        parts.append(fn(g))
    if not parts:
        return pd.Series(dtype=float, index=df.index[:0])
    return pd.concat(parts).reindex(df.index)


def covariance(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    """COVARIANCE(x, y, n): rolling covariance per symbol (ddof=0)."""
    df = pd.DataFrame({"x": x, "y": y})
    out = _by_symbol_pair(df, lambda g: g["x"].rolling(n).cov(g["y"], ddof=0))
    return out.replace([np.inf, -np.inf], np.nan)


def sma(s: pd.Series, n: int, m: int) -> pd.Series:
    """SMA(X, N, M): recursive EMA, Y_t = (X_t * M + Y_{t-1} * (N-M)) / N.
    Equivalent to ewm(alpha=M/N, adjust=False)."""
    alpha = m / n
    return _by_symbol(s, lambda g: g.ewm(alpha=alpha, adjust=False).mean())


def _linear_weights(n: int) -> np.ndarray:
    w = np.arange(1, n + 1, dtype=float)
    return w / w.sum()


def wma(s: pd.Series, n: int) -> pd.Series:
    """WMA(X, N): weighted MA, weights 1, 2, ..., n normalized (newer = larger)."""
    w = _linear_weights(n)
    return _by_symbol(s, lambda g: g.rolling(n).apply(lambda v: (v * w).sum(), raw=True))


def decaylinear(s: pd.Series, n: int) -> pd.Series:
    """DECAYLINEAR(X, N): weighted MA with weights 1..n (newer = larger).
    Semantically identical to WMA in this implementation; both follow the
    standard 'linear decay' definition used in most Alpha191 references."""
    return wma(s, n)


# ---------------------------------------------------------------------------
# Regression operators
# ---------------------------------------------------------------------------

def _rolling_beta_seq(y: pd.Series, n: int) -> pd.Series:
    """REGBETA(y, SEQUENCE(n), n): rolling slope of y against 1..n."""
    i = np.arange(1, n + 1, dtype=float)
    sum_i = i.sum()
    sum_i2 = (i * i).sum()
    denom = n * sum_i2 - sum_i * sum_i
    def _fn(g: pd.Series) -> pd.Series:
        sum_y = g.rolling(n).sum()
        sum_iy = g.rolling(n).apply(lambda v: (v * i).sum(), raw=True)
        return (n * sum_iy - sum_i * sum_y) / denom
    return _by_symbol(y, _fn)


def _rolling_beta_xy(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
    sum_x = x.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)
    sum_y = y.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)
    xy = (x * y)
    sum_xy = xy.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)
    x2 = (x * x)
    sum_x2 = x2.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)
    return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)


def regbeta(y: pd.Series, x, n: int) -> pd.Series:
    """REGBETA(y, x, n): rolling OLS slope of y on x over the trailing n bars.
    Pass x="sequence" to regress against 1..n."""
    if isinstance(x, str) and x.lower() == "sequence":
        return _rolling_beta_seq(y, n)
    return _rolling_beta_xy(y, x, n)


def _rolling_resid_seq(y: pd.Series, n: int) -> pd.Series:
    """REGRESI(y, SEQUENCE(n), n): residual at the latest bar."""
    i = np.arange(1, n + 1, dtype=float)
    sum_i = i.sum()
    sum_i2 = (i * i).sum()
    denom = n * sum_i2 - sum_i * sum_i
    last_i = float(n)
    def _fn(g: pd.Series) -> pd.Series:
        sum_y = g.rolling(n).sum()
        sum_iy = g.rolling(n).apply(lambda v: (v * i).sum(), raw=True)
        beta = (n * sum_iy - sum_i * sum_y) / denom
        alpha = (sum_y - beta * sum_i) / n
        y_hat = alpha + beta * last_i
        return g - y_hat
    return _by_symbol(y, _fn)


def _rolling_resid_xy(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
    sum_x = x.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)
    sum_y = y.groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)
    sum_xy = (x * y).groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)
    sum_x2 = (x * x).groupby(level=SYMBOL_LEVEL, group_keys=False).rolling(n).sum().reset_index(level=0, drop=True)
    beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    alpha = (sum_y - beta * sum_x) / n
    y_hat = alpha + beta * x
    return y - y_hat


def regresi(y: pd.Series, x, n: int) -> pd.Series:
    """REGRESI(y, x, n): residual of the latest bar in the rolling OLS of y on x."""
    if isinstance(x, str) and x.lower() == "sequence":
        return _rolling_resid_seq(y, n)
    return _rolling_resid_xy(y, x, n)


def regresi_multi(y: pd.Series, xs: list[pd.Series], n: int) -> pd.Series:
    """Multi-variate rolling OLS residual. Used by Alpha030 (y ~ MKT + SMB + HML).
    Computed per (symbol, datetime) via numpy lstsq on the trailing n rows of design matrix."""
    if not xs:
        raise ValueError("regresi_multi requires at least one regressor")
    df = pd.concat([y.rename("y")] + [x.rename(f"x{i}") for i, x in enumerate(xs)], axis=1)
    n_x = len(xs)
    def _fn(g: pd.DataFrame) -> pd.Series:
        yv = g["y"].values
        Xv = np.column_stack([np.ones(len(g))] + [g[f"x{i}"].values for i in range(n_x)])
        out = np.full(len(g), np.nan)
        for t in range(n - 1, len(g)):
            window_y = yv[t - n + 1: t + 1]
            window_X = Xv[t - n + 1: t + 1]
            try:
                coef, *_ = np.linalg.lstsq(window_X, window_y, rcond=None)
                out[t] = window_y[-1] - window_X[-1] @ coef
            except np.linalg.LinAlgError:
                out[t] = np.nan
        return pd.Series(out, index=g.index)
    parts = []
    for sym, g in df.groupby(level=SYMBOL_LEVEL, sort=False):
        parts.append(_fn(g))
    if not parts:
        return pd.Series(dtype=float, index=df.index[:0])
    return pd.concat(parts).reindex(df.index)


# ---------------------------------------------------------------------------
# Element-wise utilities
# ---------------------------------------------------------------------------

def sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


def abs_(s: pd.Series) -> pd.Series:
    return s.abs()


def log_(s: pd.Series) -> pd.Series:
    return np.log(s)


def max_(a, b):
    """MAX(a, b): element-wise max. Overloaded: MAX(x, n) with int n == TSMAX(x, n)."""
    if isinstance(b, int):
        return tsmax(a, b)
    return np.maximum(a, b)


def min_(a, b):
    """MIN(a, b): element-wise min. Overloaded: MIN(x, n) with int n == TSMIN(x, n)."""
    if isinstance(b, int):
        return tsmin(a, b)
    return np.minimum(a, b)


def filter_(s: pd.Series, cond: pd.Series) -> pd.Series:
    """FILTER(x, cond): x where cond else 0."""
    return s.where(cond.astype(bool), 0.0)


def safe_divide(num, den) -> pd.Series:
    """Division that yields NaN where den is 0 or NaN (no warning)."""
    den = den.replace(0, np.nan)
    return num / den


# ---------------------------------------------------------------------------
# Cross-sectional
# ---------------------------------------------------------------------------

def rank(s: pd.Series) -> pd.Series:
    """RANK(x): cross-sectional percentile rank within each datetime."""
    return s.groupby(level=DATE_LEVEL, group_keys=False).rank(pct=True, method="average")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def sequence(n: int) -> np.ndarray:
    """SEQUENCE(n): 1..n. Used as a regressor in REGBETA — pass as the magic string 'sequence'."""
    return np.arange(1, n + 1, dtype=float)


def retn(close: pd.Series) -> pd.Series:
    """RET: daily pct return = close.pct_change() per symbol."""
    return close.groupby(level=SYMBOL_LEVEL, group_keys=False).pct_change()
