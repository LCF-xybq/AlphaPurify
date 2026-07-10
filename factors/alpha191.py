"""WorldQuant Alpha191 factor formulas (191 factors).

Each `alpha_NNN(df, ctx=None) -> pd.Series` translates one formula from
`docs/alpha191.md`. The input `df` is a flat panel DataFrame with columns
`datetime, symbol, open, high, low, close, volume, [amount], [vwap]`.
The returned Series is aligned to df's index.

External-data alphas (Alpha030/147/149/181/182 need MKT/SMB/HML;
Alpha110/112/155/etc. need BENCHMARK) raise `MissingExternal` if the
required key is absent from `ctx`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import operators as op
from .data import ensure_vwap

DATE_LEVEL = op.DATE_LEVEL
SYMBOL_LEVEL = op.SYMBOL_LEVEL


class MissingExternal(KeyError):
    """Raised when an alpha needs an external factor (MKT/SMB/HML/BENCHMARK) not in ctx."""


def _indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with (symbol, datetime) MultiIndex, sorted."""
    out = df.copy()
    out[SYMBOL_LEVEL] = out[SYMBOL_LEVEL].astype(str)
    out[DATE_LEVEL] = pd.to_datetime(out[DATE_LEVEL])
    return out.set_index([SYMBOL_LEVEL, DATE_LEVEL]).sort_index()


def _cols(df_or_indexed: pd.DataFrame):
    """Return (close, open, high, low, volume, vwap, amount) as Series on the
    (symbol, datetime) MultiIndex. Accepts either flat df or pre-indexed df."""
    if not isinstance(df_or_indexed.index, pd.MultiIndex):
        df = _indexed(df_or_indexed)
    else:
        df = df_or_indexed
    close = df["close"].astype(float)
    open_ = df["open"].astype(float) if "open" in df.columns else close
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    vwap = ensure_vwap(df.reset_index())
    vwap.index = df.index
    amount = df["amount"].astype(float) if "amount" in df.columns else pd.Series(np.nan, index=df.index)
    return close, open_, high, low, vol, vwap, amount


# ===========================================================================
# Alpha001 - Alpha050
# ===========================================================================

def alpha_001(df, ctx=None):
    """(-1*CORR(RANK(DELTA(LOG(VOLUME),1)), RANK(((CLOSE-OPEN)/OPEN)), 6))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.corr(op.rank(op.delta(np.log(v), 1)), op.rank((c - o) / o), 6)


def alpha_002(df, ctx=None):
    """(-1*DELTA((((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)), 1))"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = ((c - l) - (h - c)) / (h - l)
    return -1 * op.delta(inner, 1)


def alpha_003(df, ctx=None):
    """SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))), 6)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_c = op.delay(c, 1)
    cond_up = c > prev_c
    branch = np.where(cond_up, op.min_(l, prev_c), op.max_(h, prev_c))
    inner = np.where(c == prev_c, 0.0, c - branch)
    return op.sum_(pd.Series(inner, index=c.index), 6)


def alpha_004(df, ctx=None):
    """((((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(((SUM(CLOSE,2)/2)<((SUM(CLOSE,8)/8)-STD(CLOSE,8)))?1:(((1<(VOLUME/MEAN(VOLUME,20)))||((VOLUME/MEAN(VOLUME,20))==1))?1:(-1))))"""
    c, o, h, l, v, vw, amt = _cols(df)
    s8 = op.sum_(c, 8) / 8
    sd8 = op.std(c, 8)
    s2 = op.sum_(c, 2) / 2
    vm = v / op.mean(v, 20)
    cond1 = s8 + sd8 < s2
    cond2 = s2 < s8 - sd8
    cond3 = (vm > 1) | (vm == 1)
    inner = np.where(cond3, 1.0, -1.0)
    inner = np.where(cond2, 1.0, inner)
    inner = np.where(cond1, -1.0, inner)
    return pd.Series(inner, index=c.index)


def alpha_005(df, ctx=None):
    """(-1*TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.tsmax(op.corr(op.tsrank(v, 5), op.tsrank(h, 5), 5), 3)


def alpha_006(df, ctx=None):
    """(RANK(SIGN(DELTA((((OPEN*0.85)+(HIGH*0.15))), 4)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = o * 0.85 + h * 0.15
    return -1 * op.rank(op.sign(op.delta(inner, 4)))


def alpha_007(df, ctx=None):
    """((RANK(MAX((VWAP-CLOSE),3))+RANK(MIN((VWAP-CLOSE),3)))*RANK(DELTA(VOLUME,3)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    diff = vw - c
    return (op.rank(op.max_(diff, 3)) + op.rank(op.min_(diff, 3))) * op.rank(op.delta(v, 3))


def alpha_008(df, ctx=None):
    """RANK(DELTA(((((HIGH+LOW)/2)*0.2)+(VWAP*0.8)), 4)*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = (h + l) / 2 * 0.2 + vw * 0.8
    return -1 * op.rank(op.delta(inner, 4))


def alpha_009(df, ctx=None):
    """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 7, 2)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = (h + l) / 2 - (op.delay(h, 1) + op.delay(l, 1)) / 2
    b = (h - l) / v
    return op.sma(a * b, 7, 2)


def alpha_010(df, ctx=None):
    """(RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2), 5))"""
    c, o, h, l, v, vw, amt = _cols(df)
    ret = op.retn(c)
    inner = np.where(ret < 0, op.std(ret, 20), c)
    return op.rank(op.max_(pd.Series(inner ** 2, index=c.index), 5))


def alpha_011(df, ctx=None):
    """SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME, 6)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = ((c - l) - (h - c)) / (h - l) * v
    return op.sum_(inner, 6)


def alpha_012(df, ctx=None):
    """(RANK((OPEN-(SUM(VWAP,10)/10))))*(-1*(RANK(ABS((CLOSE-VWAP)))))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.rank(o - op.sum_(vw, 10) / 10) * (-1 * op.rank(abs_(c - vw)))


def alpha_013(df, ctx=None):
    """(((HIGH*LOW)^0.5)-VWAP)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return np.sqrt(h * l) - vw


def alpha_014(df, ctx=None):
    """CLOSE-DELAY(CLOSE,5)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return c - op.delay(c, 5)


def alpha_015(df, ctx=None):
    """OPEN/DELAY(CLOSE,1)-1"""
    c, o, h, l, v, vw, amt = _cols(df)
    return o / op.delay(c, 1) - 1


def alpha_016(df, ctx=None):
    """(-1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.tsmax(op.rank(op.corr(op.rank(v), op.rank(vw), 5)), 5)


def alpha_017(df, ctx=None):
    """RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.rank(vw - op.max_(vw, 15)) ** op.delta(c, 5)


def alpha_018(df, ctx=None):
    """CLOSE/DELAY(CLOSE,5)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return c / op.delay(c, 5)


def alpha_019(df, ctx=None):
    """(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 5)
    branch_eq = np.where(c == prev, 0.0, (c - prev) / c)
    return pd.Series(np.where(c < prev, (c - prev) / prev, branch_eq), index=c.index)


def alpha_020(df, ctx=None):
    """(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 6)
    return (c - prev) / prev * 100


def alpha_021(df, ctx=None):
    """REGBETA(MEAN(CLOSE,6),SEQUENCE(6))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.regbeta(op.mean(c, 6), "sequence", 6)


def alpha_022(df, ctx=None):
    """SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)), 12, 1)
    NOTE: SMEAN is undefined in the source; we treat it as SMA with m=1."""
    c, o, h, l, v, vw, amt = _cols(df)
    m6 = op.mean(c, 6)
    inner = (c - m6) / m6 - op.delay((c - m6) / m6, 3)
    return op.sma(inner, 12, 1)


def alpha_023(df, ctx=None):
    """SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    sd = op.std(c, 20)
    up = np.where(c > prev, sd, 0.0)
    dn = np.where(c <= prev, sd, 0.0)
    up_s = pd.Series(up, index=c.index)
    dn_s = pd.Series(dn, index=c.index)
    a = op.sma(up_s, 20, 1)
    b = op.sma(dn_s, 20, 1)
    return op.safe_divide(a, a + b) * 100


def alpha_024(df, ctx=None):
    """SMA(CLOSE-DELAY(CLOSE,5),5,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.sma(c - op.delay(c, 5), 5, 1)


def alpha_025(df, ctx=None):
    """((-1*RANK((DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9))))))*(1+RANK(SUM(RET,250)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    ret = op.retn(c)
    inner = op.delta(c, 7) * (1 - op.rank(op.decaylinear(v / op.mean(v, 20), 9)))
    return -1 * op.rank(inner) * (1 + op.rank(op.sum_(ret, 250)))


def alpha_026(df, ctx=None):
    """((((SUM(CLOSE,7)/7)-CLOSE))+((CORR(VWAP,DELAY(CLOSE,5),230))))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return (op.sum_(c, 7) / 7 - c) + op.corr(vw, op.delay(c, 5), 230)


def alpha_027(df, ctx=None):
    """WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100, 12)"""
    c, o, h, l, v, vw, amt = _cols(df)
    p3 = op.delay(c, 3); p6 = op.delay(c, 6)
    inner = (c - p3) / p3 * 100 + (c - p6) / p6 * 100
    return op.wma(inner, 12)


def alpha_028(df, ctx=None):
    """3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
       -2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
    NOTE: MAX(HIGH,9) is a doc typo for TSMAX(HIGH,9); TSMAX(LOW,9) is a typo for TSMIN(LOW,9).
    """
    c, o, h, l, v, vw, amt = _cols(df)
    denom = op.tsmax(h, 9) - op.tsmin(l, 9)
    inner = (c - op.tsmin(l, 9)) / denom * 100
    a = op.sma(inner, 3, 1)
    return 3 * a - 2 * op.sma(a, 3, 1)


def alpha_029(df, ctx=None):
    """(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 6)
    return (c - prev) / prev * v


def alpha_030(df, ctx=None):
    """WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML,60))^2,20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    if ctx is None or not all(k in ctx for k in ("mkt", "smb", "hml")):
        raise MissingExternal("Alpha030 requires ctx keys: mkt, smb, hml")
    y = c / op.delay(c, 1) - 1
    resid = op.regresi_multi(y, [ctx["mkt"], ctx["smb"], ctx["hml"]], 60)
    return op.wma(resid ** 2, 20)


def alpha_031(df, ctx=None):
    """(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    m = op.mean(c, 12)
    return (c - m) / m * 100


def alpha_032(df, ctx=None):
    """(-1*SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),3)),3))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.sum_(op.rank(op.corr(op.rank(h), op.rank(v), 3)), 3)


def alpha_033(df, ctx=None):
    """((((-1*TSMIN(LOW,5))+DELAY(TSMIN(LOW,5),5))*RANK(((SUM(RET,240)-SUM(RET,20))/220)))*TSRANK(VOLUME,5))"""
    c, o, h, l, v, vw, amt = _cols(df)
    ret = op.retn(c)
    tmin5 = op.tsmin(l, 5)
    a = -1 * tmin5 + op.delay(tmin5, 5)
    b = op.rank((op.sum_(ret, 240) - op.sum_(ret, 20)) / 220)
    return a * b * op.tsrank(v, 5)


def alpha_034(df, ctx=None):
    """MEAN(CLOSE,12)/CLOSE"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.mean(c, 12) / c


def alpha_035(df, ctx=None):
    """(MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),RANK(DECAYLINEAR(CORR((VOLUME),((OPEN*0.65)+(OPEN*0.35)),17),7)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.delta(o, 1), 15))
    b = op.rank(op.decaylinear(op.corr(v, o * 0.65 + o * 0.35, 17), 7))
    return -1 * op.min_(a, b)


def alpha_036(df, ctx=None):
    """RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP)),6),2)
    Interpretation: SUM over n=2 of (rolling 6-day... wait, CORR needs 3 args).
    Per the standard reference this is RANK(TSRANK(SUM(CORR(RANK(VOLUME),RANK(VWAP),6),2), 2))
    but the original formula reads literally as RANK over the trailing 2-day sum
    of CORR. We follow the literal reading with CORR over 6 days, SUM over 2 days.
    """
    c, o, h, l, v, vw, amt = _cols(df)
    inner = op.sum_(op.corr(op.rank(v), op.rank(vw), 6), 2)
    return op.rank(inner)


def alpha_037(df, ctx=None):
    """(-1*RANK(((SUM(OPEN,5)*SUM(RET,5))-DELAY((SUM(OPEN,5)*SUM(RET,5)),10))))"""
    c, o, h, l, v, vw, amt = _cols(df)
    ret = op.retn(c)
    inner = op.sum_(o, 5) * op.sum_(ret, 5)
    return -1 * op.rank(inner - op.delay(inner, 10))


def alpha_038(df, ctx=None):
    """(((SUM(HIGH,20)/20)<HIGH)?(-1*DELTA(HIGH,2)):0)"""
    c, o, h, l, v, vw, amt = _cols(df)
    cond = op.sum_(h, 20) / 20 < h
    return pd.Series(np.where(cond, -1 * op.delta(h, 2), 0.0), index=c.index)


def alpha_039(df, ctx=None):
    """((RANK(DECAYLINEAR(DELTA((CLOSE),2),8))-RANK(DECAYLINEAR(CORR(((VWAP*0.3)+(OPEN*0.7)),SUM(MEAN(VOLUME,180),37),14),12)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.delta(c, 2), 8))
    b = op.rank(op.decaylinear(op.corr(vw * 0.3 + o * 0.7, op.sum_(op.mean(v, 180), 37), 14), 12))
    return -1 * (a - b)


def alpha_040(df, ctx=None):
    """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    up = np.where(c > prev, v, 0.0)
    dn = np.where(c <= prev, v, 0.0)
    up_s = pd.Series(up, index=c.index)
    dn_s = pd.Series(dn, index=c.index)
    return op.safe_divide(op.sum_(up_s, 26), op.sum_(dn_s, 26)) * 100


def alpha_041(df, ctx=None):
    """(RANK(MAX(DELTA((VWAP),3),5))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.rank(op.max_(op.delta(vw, 3), 5))


def alpha_042(df, ctx=None):
    """((-1*RANK(STD(HIGH,10)))*CORR(HIGH,VOLUME,10))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.rank(op.std(h, 10)) * op.corr(h, v, 10)


def alpha_043(df, ctx=None):
    """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    inner = np.where(c < prev, -v, np.where(c > prev, v, 0.0))
    return op.sum_(pd.Series(inner, index=c.index), 6)


def alpha_044(df, ctx=None):
    """(TSRANK(DECAYLINEAR(CORR(((LOW)),MEAN(VOLUME,10),7),6),4)+TSRANK(DECAYLINEAR(DELTA((VWAP),3),10),15))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.tsrank(op.decaylinear(op.corr(l, op.mean(v, 10), 7), 6), 4)
    b = op.tsrank(op.decaylinear(op.delta(vw, 3), 10), 15)
    return a + b


def alpha_045(df, ctx=None):
    """(RANK(DELTA((((CLOSE*0.6)+(OPEN*0.4))),1))*RANK(CORR(VWAP,MEAN(VOLUME,150),15)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = c * 0.6 + o * 0.4
    return op.rank(op.delta(inner, 1)) * op.rank(op.corr(vw, op.mean(v, 150), 15))


def alpha_046(df, ctx=None):
    """(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return (op.mean(c, 3) + op.mean(c, 6) + op.mean(c, 12) + op.mean(c, 24)) / (4 * c)


def alpha_047(df, ctx=None):
    """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = (op.tsmax(h, 6) - c) / (op.tsmax(h, 6) - op.tsmin(l, 6)) * 100
    return op.sma(inner, 9, 1)


def alpha_048(df, ctx=None):
    """(-1*((RANK(((SIGN((CLOSE-DELAY(CLOSE,1)))+SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2))))+SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3))))))*SUM(VOLUME,5))/SUM(VOLUME,20))"""
    c, o, h, l, v, vw, amt = _cols(df)
    s1 = op.sign(c - op.delay(c, 1))
    s2 = op.sign(op.delay(c, 1) - op.delay(c, 2))
    s3 = op.sign(op.delay(c, 2) - op.delay(c, 3))
    inner = s1 + s2 + s3
    return -1 * (op.rank(inner) * op.sum_(v, 5)) / op.sum_(v, 20)


def alpha_049(df, ctx=None):
    """SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
       / (SUM(up_branch,12)+SUM(down_branch,12))"""
    c, o, h, l, v, vw, amt = _cols(df)
    dh = h - op.delay(h, 1)
    dl = l - op.delay(l, 1)
    cond_up = (h + l) >= (op.delay(h, 1) + op.delay(l, 1))
    cond_dn = (h + l) <= (op.delay(h, 1) + op.delay(l, 1))
    up_branch = np.where(cond_up, 0.0, np.maximum(abs_(dh), abs_(dl)))
    dn_branch = np.where(cond_dn, 0.0, np.maximum(abs_(dh), abs_(dl)))
    up_s = pd.Series(up_branch, index=c.index)
    dn_s = pd.Series(dn_branch, index=c.index)
    su = op.sum_(up_s, 12)
    sd = op.sum_(dn_s, 12)
    return op.safe_divide(su, su + sd)


def alpha_050(df, ctx=None):
    """SUM(down_branch,12)/(SUM(down_branch,12)+SUM(up_branch,12)) - SUM(up_branch,12)/(SUM(up_branch,12)+SUM(down_branch,12))"""
    c, o, h, l, v, vw, amt = _cols(df)
    dh = h - op.delay(h, 1)
    dl = l - op.delay(l, 1)
    cond_up = (h + l) >= (op.delay(h, 1) + op.delay(l, 1))
    cond_dn = (h + l) <= (op.delay(h, 1) + op.delay(l, 1))
    up_branch = np.where(cond_up, 0.0, np.maximum(abs_(dh), abs_(dl)))
    dn_branch = np.where(cond_dn, 0.0, np.maximum(abs_(dh), abs_(dl)))
    up_s = pd.Series(up_branch, index=c.index)
    dn_s = pd.Series(dn_branch, index=c.index)
    su = op.sum_(up_s, 12)
    sd = op.sum_(dn_s, 12)
    return op.safe_divide(sd, sd + su) - op.safe_divide(su, su + sd)


# Re-export common element-wise helpers so we don't need `op.` everywhere
def abs_(s): return op.abs_(s)


# ===========================================================================
# Alpha051 - Alpha100
# ===========================================================================

def alpha_051(df, ctx=None):
    """SUM(down_branch,12)/(SUM(down_branch,12)+SUM(up_branch,12))"""
    c, o, h, l, v, vw, amt = _cols(df)
    dh = h - op.delay(h, 1)
    dl = l - op.delay(l, 1)
    cond_up = (h + l) >= (op.delay(h, 1) + op.delay(l, 1))
    cond_dn = (h + l) <= (op.delay(h, 1) + op.delay(l, 1))
    up_branch = np.where(cond_up, 0.0, np.maximum(abs_(dh), abs_(dl)))
    dn_branch = np.where(cond_dn, 0.0, np.maximum(abs_(dh), abs_(dl)))
    up_s = pd.Series(up_branch, index=c.index)
    dn_s = pd.Series(dn_branch, index=c.index)
    su = op.sum_(up_s, 12)
    sd = op.sum_(dn_s, 12)
    return op.safe_divide(sd, sd + su)


def alpha_052(df, ctx=None):
    """SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    mid = (h + l + c) / 3
    num = op.sum_(np.maximum(0.0, h - op.delay(mid, 1)), 26)
    den = op.sum_(np.maximum(0.0, op.delay(mid, 1) - l), 26)
    return op.safe_divide(num, den) * 100


def alpha_053(df, ctx=None):
    """COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.count(c > op.delay(c, 1), 12) / 12 * 100


def alpha_054(df, ctx=None):
    """(-1*RANK((STD(ABS(CLOSE-OPEN),20)+(CLOSE-OPEN))+CORR(CLOSE,OPEN,10)))
    NOTE: source doc omitted the n-arg of STD; standard reference uses 20."""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = op.std(abs_(c - o), 20) + (c - o) + op.corr(c, o, 10)
    return -1 * op.rank(inner)


def alpha_055(df, ctx=None):
    """Asymmetric DWMA — complex conditional on |HIGH-prev_C|, |LOW-prev_C|, |HIGH-prev_L|.
    Translation follows the standard Alpha191 reference implementation."""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_c = op.delay(c, 1)
    prev_o = op.delay(o, 1)
    prev_l = op.delay(l, 1)
    AHpC = abs_(h - prev_c)
    ALpC = abs_(l - prev_c)
    AHpL = abs_(h - prev_l)
    ApCpO = abs_(prev_c - prev_o)
    HpL = h - prev_l
    CpO_4 = (prev_c - prev_o) / 4

    # Branch1: AHpC > ALpC & AHpC > AHpL
    # branch1 value = AHpC + ALpC/2 + ApCpO/4
    # Branch2: ALpC > AHpL & ALpC > AHpC
    # branch2 value = ALpC + AHpC/2 + ApCpO/4
    # Branch3: else = AHpL + ApCpO/4
    cond1 = (AHpC > ALpC) & (AHpC > AHpL)
    cond2 = (ALpC > AHpL) & (ALpC > AHpC)
    branch1 = AHpC + ALpC / 2 + ApCpO / 4
    branch2 = ALpC + AHpC / 2 + ApCpO / 4
    branch3 = AHpL + ApCpO / 4
    denom = np.where(cond1, branch1, np.where(cond2, branch2, branch3))
    numer = 16 * (c - prev_c + (c - o) / 2 + prev_c - prev_o)
    inner = numer / denom * np.maximum(AHpC, ALpC)
    return op.sum_(pd.Series(inner, index=c.index), 20)


def alpha_056(df, ctx=None):
    """(RANK((OPEN-TSMIN(OPEN,12)))<RANK((RANK(CORR(SUM(((HIGH+LOW)/2),19),SUM(MEAN(VOLUME,40),19),13))^5)))
    Returns 1/0 boolean-as-float."""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(o - op.tsmin(o, 12))
    inner = op.rank(op.corr(op.sum_((h + l) / 2, 19), op.sum_(op.mean(v, 40), 19), 13)) ** 5
    b = op.rank(inner)
    return (a < b).astype(float)


def alpha_057(df, ctx=None):
    """SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = (c - op.tsmin(l, 9)) / (op.tsmax(h, 9) - op.tsmin(l, 9)) * 100
    return op.sma(inner, 3, 1)


def alpha_058(df, ctx=None):
    """COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.count(c > op.delay(c, 1), 20) / 20 * 100


def alpha_059(df, ctx=None):
    """SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_c = op.delay(c, 1)
    cond_up = c > prev_c
    branch = np.where(cond_up, op.min_(l, prev_c), op.max_(h, prev_c))
    inner = np.where(c == prev_c, 0.0, c - branch)
    return op.sum_(pd.Series(inner, index=c.index), 20)


def alpha_060(df, ctx=None):
    """SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = ((c - l) - (h - c)) / (h - l) * v
    return op.sum_(inner, 20)


def alpha_061(df, ctx=None):
    """(MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80),8)),17)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.delta(vw, 1), 12))
    b = op.rank(op.decaylinear(op.rank(op.corr(l, op.mean(v, 80), 8)), 17))
    return -1 * op.max_(a, b)


def alpha_062(df, ctx=None):
    """(-1*CORR(HIGH,RANK(VOLUME),5))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.corr(h, op.rank(v), 5)


def alpha_063(df, ctx=None):
    """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    diff = c - op.delay(c, 1)
    num = op.sma(np.maximum(0.0, diff), 6, 1)
    den = op.sma(abs_(diff), 6, 1)
    return op.safe_divide(num, den) * 100


def alpha_064(df, ctx=None):
    """(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.corr(op.rank(vw), op.rank(v), 4), 4))
    b = op.rank(op.decaylinear(op.max_(op.corr(op.rank(c), op.rank(op.mean(v, 60)), 4), 13), 14))
    return -1 * op.max_(a, b)


def alpha_065(df, ctx=None):
    """MEAN(CLOSE,6)/CLOSE"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.mean(c, 6) / c


def alpha_066(df, ctx=None):
    """(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    m = op.mean(c, 6)
    return (c - m) / m * 100


def alpha_067(df, ctx=None):
    """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    diff = c - op.delay(c, 1)
    num = op.sma(np.maximum(0.0, diff), 24, 1)
    den = op.sma(abs_(diff), 24, 1)
    return op.safe_divide(num, den) * 100


def alpha_068(df, ctx=None):
    """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = (h + l) / 2 - (op.delay(h, 1) + op.delay(l, 1)) / 2
    b = (h - l) / v
    return op.sma(a * b, 15, 2)


def alpha_069(df, ctx=None):
    """(SUM(DTM,20)>SUM(DBM,20)?(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):(SUM(DTM,20)=SUM(DBM,20)?0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
    DTM = IF(OPEN>DELAY(OPEN,1), MAX(HIGH-OPEN, OPEN-DELAY(OPEN,1)), 0)
    DBM = IF(OPEN<DELAY(OPEN,1), MAX(OPEN-LOW, OPEN-DELAY(OPEN,1)), 0)
    """
    c, o, h, l, v, vw, amt = _cols(df)
    prev_o = op.delay(o, 1)
    dtm = np.where(o > prev_o, np.maximum(h - o, o - prev_o), 0.0)
    dbm = np.where(o < prev_o, np.maximum(o - l, o - prev_o), 0.0)
    s_dtm = op.sum_(pd.Series(dtm, index=c.index), 20)
    s_dbm = op.sum_(pd.Series(dbm, index=c.index), 20)
    out = np.where(s_dtm > s_dbm, op.safe_divide(s_dtm - s_dbm, s_dtm),
                   np.where(s_dtm == s_dbm, 0.0, op.safe_divide(s_dtm - s_dbm, s_dbm)))
    return pd.Series(out, index=c.index)


def alpha_070(df, ctx=None):
    """STD(AMOUNT,6)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.std(amt, 6)


def alpha_071(df, ctx=None):
    """(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    m = op.mean(c, 24)
    return (c - m) / m * 100


def alpha_072(df, ctx=None):
    """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = (op.tsmax(h, 6) - c) / (op.tsmax(h, 6) - op.tsmin(l, 6)) * 100
    return op.sma(inner, 15, 1)


def alpha_073(df, ctx=None):
    """((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE),VOLUME,10),16),4),5)-RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.tsrank(op.decaylinear(op.decaylinear(op.corr(c, v, 10), 16), 4), 5)
    b = op.rank(op.decaylinear(op.corr(vw, op.mean(v, 30), 4), 3))
    return -1 * (a - b)


def alpha_074(df, ctx=None):
    """(RANK(CORR(SUM(((LOW*0.35)+(VWAP*0.65)),20),SUM(MEAN(VOLUME,40),20),7))+RANK(CORR(RANK(VWAP),RANK(VOLUME),6)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.corr(op.sum_(l * 0.35 + vw * 0.65, 20), op.sum_(op.mean(v, 40), 20), 7))
    b = op.rank(op.corr(op.rank(vw), op.rank(v), 6))
    return a + b


def alpha_075(df, ctx=None):
    """COUNT(CLOSE>OPEN & BENCH_CLOSE<BENCH_OPEN,50)/COUNT(BENCH_CLOSE<BENCH_OPEN,50)"""
    c, o, h, l, v, vw, amt = _cols(df)
    if ctx is None or "bench_open" not in ctx or "bench_close" not in ctx:
        raise MissingExternal("Alpha075 requires ctx keys: bench_open, bench_close")
    bo, bc = ctx["bench_open"], ctx["bench_close"]
    num = op.count((c > o) & (bc < bo), 50)
    den = op.count(bc < bo, 50)
    return op.safe_divide(num, den)


def alpha_076(df, ctx=None):
    """STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = abs_(c / op.delay(c, 1) - 1) / v
    return op.safe_divide(op.std(inner, 20), op.mean(inner, 20))


def alpha_077(df, ctx=None):
    """MIN(RANK(DECAYLINEAR(((((HIGH+LOW)/2)+HIGH)-(VWAP+HIGH)),20)),RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),3),6)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(((h + l) / 2 + h) - (vw + h), 20))
    b = op.rank(op.decaylinear(op.corr((h + l) / 2, op.mean(v, 40), 3), 6))
    return op.min_(a, b)


def alpha_078(df, ctx=None):
    """((HIGH+LOW+CLOSE)/3 - MA((HIGH+LOW+CLOSE)/3,12)) / (0.015 * MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
    Commodity Channel Index. MA = MEAN."""
    c, o, h, l, v, vw, amt = _cols(df)
    mid = (h + l + c) / 3
    m12 = op.mean(mid, 12)
    denom = 0.015 * op.mean(abs_(c - op.mean(mid, 12)), 12)
    return (mid - m12) / denom


def alpha_079(df, ctx=None):
    """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    diff = c - op.delay(c, 1)
    num = op.sma(np.maximum(0.0, diff), 12, 1)
    den = op.sma(abs_(diff), 12, 1)
    return op.safe_divide(num, den) * 100


def alpha_080(df, ctx=None):
    """(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(v, 5)
    return (v - prev) / prev * 100


def alpha_081(df, ctx=None):
    """SMA(VOLUME,21,2)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.sma(v, 21, 2)


def alpha_082(df, ctx=None):
    """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = (op.tsmax(h, 6) - c) / (op.tsmax(h, 6) - op.tsmin(l, 6)) * 100
    return op.sma(inner, 20, 1)


def alpha_083(df, ctx=None):
    """(-1*RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),5)))  (typo: COVIANCE = COVARIANCE)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.rank(op.covariance(op.rank(h), op.rank(v), 5))


def alpha_084(df, ctx=None):
    """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    inner = np.where(c < prev, -v, np.where(c > prev, v, 0.0))
    return op.sum_(pd.Series(inner, index=c.index), 20)


def alpha_085(df, ctx=None):
    """(TSRANK((VOLUME/MEAN(VOLUME,20)),20)*TSRANK((-1*DELTA(CLOSE,7)),8))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.tsrank(v / op.mean(v, 20), 20)
    b = op.tsrank(-1 * op.delta(c, 7), 8)
    return a * b


def alpha_086(df, ctx=None):
    """Triple-condition on ((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10 - (DELAY(CLOSE,10)-CLOSE)/10)."""
    c, o, h, l, v, vw, amt = _cols(df)
    prev20 = op.delay(c, 20)
    prev10 = op.delay(c, 10)
    x = (prev20 - prev10) / 10 - (prev10 - c) / 10
    out = np.where(x < 0, 1.0,
                   np.where(x > 0.25, -1.0, -1 * (c - op.delay(c, 1))))
    return pd.Series(out, index=c.index)


def alpha_087(df, ctx=None):
    """((RANK(DECAYLINEAR(DELTA(VWAP,4),7))+TSRANK(DECAYLINEAR(((((LOW*0.9)+(LOW*0.1))-VWAP)/(OPEN-((HIGH+LOW)/2))),11),7))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.delta(vw, 4), 7))
    inner = ((l * 0.9 + l * 0.1) - vw) / (o - (h + l) / 2)
    b = op.tsrank(op.decaylinear(inner, 11), 7)
    return -1 * (a + b)


def alpha_088(df, ctx=None):
    """(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 20)
    return (c - prev) / prev * 100


def alpha_089(df, ctx=None):
    """2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.sma(c, 13, 2)
    b = op.sma(c, 27, 2)
    diff = a - b
    return 2 * (diff - op.sma(diff, 10, 2))


def alpha_090(df, ctx=None):
    """(RANK(CORR(RANK(VWAP),RANK(VOLUME),5))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.rank(op.corr(op.rank(vw), op.rank(v), 5))


def alpha_091(df, ctx=None):
    """((RANK((CLOSE-MAX(CLOSE,5)))*RANK(CORR((MEAN(VOLUME,40)),LOW,5)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(c - op.max_(c, 5))
    b = op.rank(op.corr(op.mean(v, 40), l, 5))
    return -1 * a * b


def alpha_092(df, ctx=None):
    """(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.delta(c * 0.35 + vw * 0.65, 2), 3))
    b = op.tsrank(op.decaylinear(abs_(op.corr(op.mean(v, 180), c, 13)), 5), 15)
    return -1 * np.maximum(a, b)


def alpha_093(df, ctx=None):
    """SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_o = op.delay(o, 1)
    inner = np.where(o >= prev_o, 0.0, np.maximum(o - l, o - prev_o))
    return op.sum_(pd.Series(inner, index=c.index), 20)


def alpha_094(df, ctx=None):
    """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    inner = np.where(c < prev, -v, np.where(c > prev, v, 0.0))
    return op.sum_(pd.Series(inner, index=c.index), 30)


def alpha_095(df, ctx=None):
    """STD(AMOUNT,20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.std(amt, 20)


def alpha_096(df, ctx=None):
    """SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = (c - op.tsmin(l, 9)) / (op.tsmax(h, 9) - op.tsmin(l, 9)) * 100
    return op.sma(op.sma(inner, 3, 1), 3, 1)


def alpha_097(df, ctx=None):
    """STD(VOLUME,10)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.std(v, 10)


def alpha_098(df, ctx=None):
    """((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05)||(...==0.05))?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    s100 = op.sum_(c, 100) / 100
    ratio = op.delta(s100, 100) / op.delay(c, 100)
    cond = (ratio < 0.05) | (ratio == 0.05)
    out = np.where(cond, -1 * (c - op.tsmin(c, 100)), -1 * op.delta(c, 3))
    return pd.Series(out, index=c.index)


def alpha_099(df, ctx=None):
    """(-1*RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),5)))  (typo: COVIANCE = COVARIANCE)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.rank(op.covariance(op.rank(c), op.rank(v), 5))


def alpha_100(df, ctx=None):
    """STD(VOLUME,20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.std(v, 20)


# ===========================================================================
# Alpha101 - Alpha150
# ===========================================================================

def alpha_101(df, ctx=None):
    """((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.corr(c, op.sum_(op.mean(v, 30), 37), 15))
    b = op.rank(op.corr(op.rank(h * 0.1 + vw * 0.9), op.rank(v), 11))
    return -1 * (a < b).astype(float)


def alpha_102(df, ctx=None):
    """SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    diff = v - op.delay(v, 1)
    num = op.sma(np.maximum(0.0, diff), 6, 1)
    den = op.sma(abs_(diff), 6, 1)
    return op.safe_divide(num, den) * 100


def alpha_103(df, ctx=None):
    """((20-LOWDAY(LOW,20))/20)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    return (20 - op.lowday(l, 20)) / 20 * 100


def alpha_104(df, ctx=None):
    """(-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * (op.delta(op.corr(h, v, 5), 5) * op.rank(op.std(c, 20)))


def alpha_105(df, ctx=None):
    """(-1*CORR(RANK(OPEN),RANK(VOLUME),10))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.corr(op.rank(o), op.rank(v), 10)


def alpha_106(df, ctx=None):
    """CLOSE-DELAY(CLOSE,20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return c - op.delay(c, 20)


def alpha_107(df, ctx=None):
    """(((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1))))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = -1 * op.rank(o - op.delay(h, 1))
    b = op.rank(o - op.delay(c, 1))
    cc = op.rank(o - op.delay(l, 1))
    return a * b * cc


def alpha_108(df, ctx=None):
    """((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1)
    MIN(HIGH,2) interpreted as TSMIN(HIGH,2)."""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(h - op.tsmin(h, 2))
    b = op.rank(op.corr(vw, op.mean(v, 120), 6))
    return -1 * (a ** b)


def alpha_109(df, ctx=None):
    """SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.sma(h - l, 10, 2)
    return op.safe_divide(a, op.sma(a, 10, 2))


def alpha_110(df, ctx=None):
    """SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    num = op.sum_(np.maximum(0.0, h - prev), 20)
    den = op.sum_(np.maximum(0.0, prev - l), 20)
    return op.safe_divide(num, den) * 100


def alpha_111(df, ctx=None):
    """SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = v * ((c - l) - (h - c)) / (h - l)
    return op.sma(inner, 11, 2) - op.sma(inner, 4, 2)


def alpha_112(df, ctx=None):
    """(SUM(up,12)-SUM(dn,12))/(SUM(up,12)+SUM(dn,12))*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    diff = c - op.delay(c, 1)
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, abs_(diff), 0.0)
    up_s = pd.Series(up, index=c.index)
    dn_s = pd.Series(dn, index=c.index)
    su = op.sum_(up_s, 12)
    sd = op.sum_(dn_s, 12)
    return op.safe_divide(su - sd, su + sd) * 100


def alpha_113(df, ctx=None):
    """(-1*((RANK((SUM(DELAY(CLOSE,5),20)/20))*CORR(CLOSE,VOLUME,2))*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.sum_(op.delay(c, 5), 20) / 20)
    b = op.corr(c, v, 2)
    d = op.rank(op.corr(op.sum_(c, 5), op.sum_(c, 20), 2))
    return -1 * a * b * d


def alpha_114(df, ctx=None):
    """((RANK(DELAY(((HIGH-LOW)/(SUM(CLOSE,5)/5)),2))*RANK(RANK(VOLUME)))/(((HIGH-LOW)/(SUM(CLOSE,5)/5))/(VWAP-CLOSE)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = (h - l) / (op.sum_(c, 5) / 5)
    num = op.rank(op.delay(inner, 2)) * op.rank(op.rank(v))
    den = inner / (vw - c)
    return op.safe_divide(num, den)


def alpha_115(df, ctx=None):
    """(RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.corr(h * 0.9 + c * 0.1, op.mean(v, 30), 10))
    b = op.rank(op.corr(op.tsrank((h + l) / 2, 4), op.tsrank(v, 10), 7))
    return a ** b


def alpha_116(df, ctx=None):
    """REGBETA(CLOSE,SEQUENCE,20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.regbeta(c, "sequence", 20)


def alpha_117(df, ctx=None):
    """((TSRANK(VOLUME,32)*(1-TSRANK(((CLOSE+HIGH)-LOW),16)))*(1-TSRANK(RET,32)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    ret = op.retn(c)
    a = op.tsrank(v, 32)
    b = op.tsrank((c + h) - l, 16)
    cc = op.tsrank(ret, 32)
    return a * (1 - b) * (1 - cc)


def alpha_118(df, ctx=None):
    """SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.safe_divide(op.sum_(h - o, 20), op.sum_(o - l, 20)) * 100


def alpha_119(df, ctx=None):
    """(RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.corr(vw, op.sum_(op.mean(v, 5), 26), 5), 7))
    b = op.rank(op.decaylinear(op.tsrank(op.min_(op.corr(op.rank(o), op.rank(op.mean(v, 15)), 21), 9), 7), 8))
    return a - b


def alpha_120(df, ctx=None):
    """(RANK((VWAP-CLOSE))/RANK((VWAP+CLOSE)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.safe_divide(op.rank(vw - c), op.rank(vw + c))


def alpha_121(df, ctx=None):
    """((RANK((VWAP-MIN(VWAP,12)))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3))*-1)
    MIN(VWAP,12) interpreted as TSMIN(VWAP,12)."""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(vw - op.tsmin(vw, 12))
    b = op.tsrank(op.corr(op.tsrank(vw, 20), op.tsrank(op.mean(v, 60), 2), 18), 3)
    return -1 * (a ** b)


def alpha_122(df, ctx=None):
    """(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(...,1))/DELAY(...,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    s1 = op.sma(np.log(c), 13, 2)
    s2 = op.sma(s1, 13, 2)
    s3 = op.sma(s2, 13, 2)
    return op.safe_divide(s3 - op.delay(s3, 1), op.delay(s3, 1))


def alpha_123(df, ctx=None):
    """((RANK(CORR(SUM(((HIGH+LOW)/2),20),SUM(MEAN(VOLUME,60),20),9))<RANK(CORR(LOW,VOLUME,6)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.corr(op.sum_((h + l) / 2, 20), op.sum_(op.mean(v, 60), 20), 9))
    b = op.rank(op.corr(l, v, 6))
    return -1 * (a < b).astype(float)


def alpha_124(df, ctx=None):
    """(CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.safe_divide(c - vw, op.decaylinear(op.rank(op.tsmax(c, 30)), 2))


def alpha_125(df, ctx=None):
    """(RANK(DECAYLINEAR(CORR((VWAP),MEAN(VOLUME,80),17),20))/RANK(DECAYLINEAR(DELTA(((CLOSE*0.5)+(VWAP*0.5)),3),16)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.corr(vw, op.mean(v, 80), 17), 20))
    b = op.rank(op.decaylinear(op.delta(c * 0.5 + vw * 0.5, 3), 16))
    return op.safe_divide(a, b)


def alpha_126(df, ctx=None):
    """(CLOSE+HIGH+LOW)/3"""
    c, o, h, l, v, vw, amt = _cols(df)
    return (c + h + l) / 3


def alpha_127(df, ctx=None):
    """MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2, 20)^(1/2)
    NOTE: original formula missing the n-arg of MEAN; standard reference uses 20."""
    c, o, h, l, v, vw, amt = _cols(df)
    mx = op.max_(c, 12)
    inner = (100 * (c - mx) / mx) ** 2
    return op.mean(inner, 20) ** 0.5


def alpha_128(df, ctx=None):
    """100 - 100/(1 + SUM(up,14)/SUM(dn,14))   where mid=(H+L+C)/3"""
    c, o, h, l, v, vw, amt = _cols(df)
    mid = (h + l + c) / 3
    prev_mid = op.delay(mid, 1)
    up = np.where(mid > prev_mid, mid * v, 0.0)
    dn = np.where(mid < prev_mid, mid * v, 0.0)
    up_s = pd.Series(up, index=c.index)
    dn_s = pd.Series(dn, index=c.index)
    return 100 - 100 / (1 + op.safe_divide(op.sum_(up_s, 14), op.sum_(dn_s, 14)))


def alpha_129(df, ctx=None):
    """SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)"""
    c, o, h, l, v, vw, amt = _cols(df)
    diff = c - op.delay(c, 1)
    inner = np.where(diff < 0, abs_(diff), 0.0)
    return op.sum_(pd.Series(inner, index=c.index), 12)


def alpha_130(df, ctx=None):
    """(RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),9),10))/RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.corr((h + l) / 2, op.mean(v, 40), 9), 10))
    b = op.rank(op.decaylinear(op.corr(op.rank(vw), op.rank(v), 7), 3))
    return op.safe_divide(a, b)


def alpha_131(df, ctx=None):
    """(RANK(DELTA(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.delta(vw, 1))
    b = op.tsrank(op.corr(c, op.mean(v, 50), 18), 18)
    return a ** b


def alpha_132(df, ctx=None):
    """MEAN(AMOUNT,20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.mean(amt, 20)


def alpha_133(df, ctx=None):
    """((20-HIGHDAY(HIGH,20))/20)*100 - ((20-LOWDAY(LOW,20))/20)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    return (20 - op.highday(h, 20)) / 20 * 100 - (20 - op.lowday(l, 20)) / 20 * 100


def alpha_134(df, ctx=None):
    """(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 12)
    return (c - prev) / prev * v


def alpha_135(df, ctx=None):
    """SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = op.delay(c / op.delay(c, 20), 1)
    return op.sma(inner, 20, 1)


def alpha_136(df, ctx=None):
    """((-1*RANK(DELTA(RET,3)))*CORR(OPEN,VOLUME,10))"""
    c, o, h, l, v, vw, amt = _cols(df)
    ret = op.retn(c)
    return -1 * op.rank(op.delta(ret, 3)) * op.corr(o, v, 10)


def alpha_137(df, ctx=None):
    """SUM((ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)
           * MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))), 20)
    NOTE: source doc truncated; SUM(...,20) wrapper follows standard WorldQuant reference."""
    c, o, h, l, v, vw, amt = _cols(df)
    a = abs_(h - op.delay(l, 1)) + abs_(op.delay(c, 1) - op.delay(o, 1)) / 4
    b = np.maximum(abs_(h - op.delay(c, 1)), abs_(l - op.delay(c, 1)))
    return op.sum_(pd.Series(a * b, index=c.index), 20)


def alpha_138(df, ctx=None):
    """((RANK(DECAYLINEAR(DELTA((((LOW*0.7)+(VWAP*0.3))),3),20))-TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5),19),16),7))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear(op.delta(l * 0.7 + vw * 0.3, 3), 20))
    b = op.tsrank(op.decaylinear(op.tsrank(op.corr(op.tsrank(l, 8), op.tsrank(op.mean(v, 60), 17), 5), 19), 16), 7)
    return -1 * (a - b)


def alpha_139(df, ctx=None):
    """(-1*CORR(OPEN,VOLUME,10))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.corr(o, v, 10)


def alpha_140(df, ctx=None):
    """MIN(RANK(DECAYLINEAR(((RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE))),8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.decaylinear((op.rank(o) + op.rank(l)) - (op.rank(h) + op.rank(c)), 8))
    b = op.tsrank(op.decaylinear(op.corr(op.tsrank(c, 8), op.tsrank(op.mean(v, 60), 20), 8), 7), 3)
    return op.min_(a, b)


def alpha_141(df, ctx=None):
    """(RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * op.rank(op.corr(op.rank(h), op.rank(op.mean(v, 15)), 9))


def alpha_142(df, ctx=None):
    """(((-1*RANK(TSRANK(CLOSE,10)))*RANK(DELTA(DELTA(CLOSE,1),1)))*RANK(TSRANK((VOLUME/MEAN(VOLUME,20)),5)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = -1 * op.rank(op.tsrank(c, 10))
    b = op.rank(op.delta(op.delta(c, 1), 1))
    d = op.rank(op.tsrank(v / op.mean(v, 20), 5))
    return a * b * d


def alpha_143(df, ctx=None):
    """CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    SELF = previous value of this alpha. Initialized to 1.0."""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_c = op.delay(c, 1)
    ret_when_up = (c - prev_c) / prev_c
    cond_up = c > prev_c

    def _fn(g: pd.DataFrame) -> pd.Series:
        up = g["up"].values
        r = g["r"].values
        out = np.empty(len(g))
        last = 1.0
        for i in range(len(g)):
            if np.isnan(up[i]):
                out[i] = np.nan
                continue
            if up[i]:
                last = last * r[i] if not np.isnan(r[i]) else last
            out[i] = last
        return pd.Series(out, index=g.index)

    df_local = pd.DataFrame({"up": cond_up, "r": ret_when_up}, index=c.index)
    return df_local.groupby(level=SYMBOL_LEVEL, group_keys=False).apply(_fn)


def alpha_144(df, ctx=None):
    """SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    inner = abs_(c / prev - 1) / amt
    cond = c < prev
    return op.safe_divide(op.sumif(inner, 20, cond), op.count(cond, 20))


def alpha_145(df, ctx=None):
    """(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    num = op.mean(v, 9) - op.mean(v, 26)
    den = op.mean(v, 12)
    return op.safe_divide(num, den) * 100


def alpha_146(df, ctx=None):
    """MEAN((ret - SMA(ret,61,2)),20) * (ret - SMA(ret,61,2)) / SMA((ret - (ret - SMA(ret,61,2)))^2, 60, 1)
    where ret = (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1). The trailing SMA's m-arg
    is missing in the doc; we use m=1 (the standard Alpha191 reference)."""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    ret = (c - prev) / prev
    sma61 = op.sma(ret, 61, 2)
    diff = ret - sma61
    a = op.mean(diff, 20)
    inner_sq = (ret - diff) ** 2
    c_ = op.sma(inner_sq, 60, 1)
    return op.safe_divide(a * diff, c_)


def alpha_147(df, ctx=None):
    """REGBETA(MEAN(CLOSE,12),SEQUENCE(12))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.regbeta(op.mean(c, 12), "sequence", 12)


def alpha_148(df, ctx=None):
    """((RANK(CORR((OPEN),SUM(MEAN(VOLUME,60),9),6))<RANK((OPEN-TSMIN(OPEN,14))))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.corr(o, op.sum_(op.mean(v, 60), 9), 6))
    b = op.rank(o - op.tsmin(o, 14))
    return -1 * (a < b).astype(float)


def alpha_149(df, ctx=None):
    """REGBETA(FILTER(x, cond), FILTER(y, cond), 252)
    where x = CLOSE/DELAY(CLOSE,1)-1, y = BENCH_CLOSE/DELAY(BENCH_CLOSE,1)-1,
    cond = BENCH_CLOSE < DELAY(BENCH_CLOSE,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    if ctx is None or "bench_close" not in ctx:
        raise MissingExternal("Alpha149 requires ctx key: bench_close")
    bc = ctx["bench_close"]
    bc_prev = op.delay(bc, 1)
    cond = bc < bc_prev
    x = op.filter_(c / op.delay(c, 1) - 1, cond)
    y = op.filter_(bc / bc_prev - 1, cond)
    return op.regbeta(x, y, 252)


def alpha_150(df, ctx=None):
    """(CLOSE+HIGH+LOW)/3*VOLUME"""
    c, o, h, l, v, vw, amt = _cols(df)
    return (c + h + l) / 3 * v


# ===========================================================================
# Alpha151 - Alpha191
# ===========================================================================

def alpha_151(df, ctx=None):
    """SMA(CLOSE-DELAY(CLOSE,20),20,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.sma(c - op.delay(c, 20), 20, 1)


def alpha_152(df, ctx=None):
    """SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    base = op.delay(op.sma(op.delay(c / op.delay(c, 9), 1), 9, 1), 1)
    inner = op.mean(base, 12) - op.mean(base, 26)
    return op.sma(inner, 9, 1)


def alpha_153(df, ctx=None):
    """(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4"""
    c, o, h, l, v, vw, amt = _cols(df)
    return (op.mean(c, 3) + op.mean(c, 6) + op.mean(c, 12) + op.mean(c, 24)) / 4


def alpha_154(df, ctx=None):
    """(((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18)))   -> 1/0 boolean.
    MIN(VWAP,16) interpreted as TSMIN(VWAP,16)."""
    c, o, h, l, v, vw, amt = _cols(df)
    cond = (vw - op.tsmin(vw, 16)) < op.corr(vw, op.mean(v, 180), 18)
    return cond.astype(float)


def alpha_155(df, ctx=None):
    """SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.sma(v, 13, 2)
    b = op.sma(v, 27, 2)
    diff = a - b
    return diff - op.sma(diff, 10, 2)


def alpha_156(df, ctx=None):
    """(MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    base = o * 0.15 + l * 0.85
    inner = op.delta(base, 2) / base * -1
    a = op.rank(op.decaylinear(op.delta(vw, 5), 3))
    b = op.rank(op.decaylinear(inner, 3))
    return -1 * np.maximum(a, b)


def alpha_157(df, ctx=None):
    """(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)+TSRANK(DELAY((-1*RET),6),5))
    CLOSE-1 read literally as series (close - 1)."""
    c, o, h, l, v, vw, amt = _cols(df)
    ret = op.retn(c)
    delta_c1 = op.delta(c - 1, 5)
    inner = -1 * op.rank(delta_c1)
    inner = op.rank(op.rank(inner))
    inner = op.tsmin(inner, 2)
    inner = op.sum_(inner, 1)
    inner = np.log(inner)
    inner = op.rank(op.rank(inner))
    inner = op.prod(inner, 1)
    a = op.tsmin(inner, 5)
    b = op.tsrank(op.delay(-1 * ret, 6), 5)
    return a + b


def alpha_158(df, ctx=None):
    """((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE"""
    c, o, h, l, v, vw, amt = _cols(df)
    s = op.sma(c, 15, 2)
    return ((h - s) - (l - s)) / c


def alpha_159(df, ctx=None):
    """Complex Williams-style accumulation: weighted avg of 3 windows."""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_c = op.delay(c, 1)
    mn = op.min_(l, prev_c)
    mx = op.max_(h, prev_c)
    range_ = mx - mn

    def term(n, k):
        return (c - op.sum_(mn, n)) / op.sum_(range_, n) * k * 24

    numer = term(6, 12) + term(12, 6) + term(24, 6)
    return numer * 100 / (6 * 12 + 6 * 24 + 12 * 24)


def alpha_160(df, ctx=None):
    """SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    inner = np.where(c <= prev, op.std(c, 20), 0.0)
    return op.sma(pd.Series(inner, index=c.index), 20, 1)


def alpha_161(df, ctx=None):
    """MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_c = op.delay(c, 1)
    m1 = np.maximum(h - l, abs_(prev_c - h))
    m2 = np.maximum(m1, abs_(prev_c - l))
    return op.mean(pd.Series(m2, index=c.index), 12)


def alpha_162(df, ctx=None):
    """(RSI - MIN(RSI,12))/(MAX(RSI,12) - MIN(RSI,12))   where RSI = 12-day RSI"""
    c, o, h, l, v, vw, amt = _cols(df)
    diff = c - op.delay(c, 1)
    num = op.sma(np.maximum(0.0, diff), 12, 1)
    den = op.sma(abs_(diff), 12, 1)
    rsi = op.safe_divide(num, den) * 100
    mn = op.tsmin(rsi, 12)
    mx = op.tsmax(rsi, 12)
    return op.safe_divide(rsi - mn, mx - mn)


def alpha_163(df, ctx=None):
    """RANK((-1*RET*MEAN(VOLUME,20)*VWAP*(HIGH-CLOSE)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    ret = op.retn(c)
    return op.rank(-1 * ret * op.mean(v, 20) * vw * (h - c))


def alpha_164(df, ctx=None):
    """SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    diff = c - prev
    safe_inv = np.where(diff > 0, 1.0 / diff.replace(0, np.nan), 1.0)
    inv_s = pd.Series(safe_inv, index=c.index)
    inner = inv_s - op.tsmin(inv_s, 12)
    return op.sma(inner / (h - l) * 100, 13, 2)


def alpha_165(df, ctx=None):
    """MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    SUMAC(x) reads as SUM(x,48); MAX/MIN read as rolling over 48 days."""
    c, o, h, l, v, vw, amt = _cols(df)
    m48 = op.mean(c, 48)
    s48 = op.sum_(c - m48, 48)
    return op.tsmax(s48, 48) - op.safe_divide(op.tsmin(s48, 48), op.std(c, 48))


def alpha_166(df, ctx=None):
    """Skewness-style: -20*(19)^1.5 * SUM((ret-MEAN(ret,20)),20) / (19*18*(SUM(ret^2,20))^1.5)
    where ret = CLOSE/DELAY(CLOSE,1)-1."""
    c, o, h, l, v, vw, amt = _cols(df)
    ret = c / op.delay(c, 1) - 1
    a = op.sum_(ret - op.mean(ret, 20), 20)
    b = op.sum_(ret ** 2, 20)
    return -20 * (19 ** 1.5) * a / (19 * 18 * (b ** 1.5))


def alpha_167(df, ctx=None):
    """SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)"""
    c, o, h, l, v, vw, amt = _cols(df)
    diff = c - op.delay(c, 1)
    inner = np.where(diff > 0, diff, 0.0)
    return op.sum_(pd.Series(inner, index=c.index), 12)


def alpha_168(df, ctx=None):
    """(-1*VOLUME/MEAN(VOLUME,20))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return -1 * v / op.mean(v, 20)


def alpha_169(df, ctx=None):
    """SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    base = op.delay(op.sma(c - op.delay(c, 1), 9, 1), 1)
    inner = op.mean(base, 12) - op.mean(base, 26)
    return op.sma(inner, 10, 1)


def alpha_170(df, ctx=None):
    """((((RANK((1/CLOSE))*VOLUME)/MEAN(VOLUME,20))*((HIGH*RANK((HIGH-CLOSE)))/(SUM(HIGH,5)/5)))-RANK((VWAP-DELAY(VWAP,5))))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(1 / c) * v / op.mean(v, 20)
    b = h * op.rank(h - c) / (op.sum_(h, 5) / 5)
    return a * b - op.rank(vw - op.delay(vw, 5))


def alpha_171(df, ctx=None):
    """((-1*((LOW-CLOSE)*(OPEN^5)))/((CLOSE-HIGH)*(CLOSE^5)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    num = -1 * (l - c) * (o ** 5)
    den = (c - h) * (c ** 5)
    return num / den


def alpha_172(df, ctx=None):
    """DMI-style: MEAN(ABS(LD_pct - HD_pct)/(LD_pct + HD_pct)*100, 6)
    HD = HIGH - DELAY(CLOSE,1), LD = DELAY(CLOSE,1) - LOW, TR = max of (H-L, |HD|, |LD|)."""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_c = op.delay(c, 1)
    hd = h - prev_c
    ld = prev_c - l
    tr = np.maximum.reduce([h - l, np.abs(hd), np.abs(ld)])
    tr = pd.Series(tr, index=c.index).replace(0, np.nan)
    hd_pct = np.where((ld > 0) & (ld > hd), ld, 0.0) * 100 / tr
    ld_pct = np.where((hd > 0) & (hd > ld), hd, 0.0) * 100 / tr
    inner = np.abs(hd_pct - ld_pct) / (hd_pct + ld_pct) * 100
    return op.mean(pd.Series(inner, index=c.index), 6)


def alpha_173(df, ctx=None):
    """3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)"""
    c, o, h, l, v, vw, amt = _cols(df)
    s1 = op.sma(c, 13, 2)
    s2 = op.sma(s1, 13, 2)
    s3 = op.sma(np.log(c), 13, 2)
    s4 = op.sma(s3, 13, 2)
    s5 = op.sma(s4, 13, 2)
    return 3 * s1 - 2 * s2 + s5


def alpha_174(df, ctx=None):
    """SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    inner = np.where(c > prev, op.std(c, 20), 0.0)
    return op.sma(pd.Series(inner, index=c.index), 20, 1)


def alpha_175(df, ctx=None):
    """MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_c = op.delay(c, 1)
    m1 = np.maximum(h - l, abs_(prev_c - h))
    m2 = np.maximum(m1, abs_(prev_c - l))
    return op.mean(pd.Series(m2, index=c.index), 6)


def alpha_176(df, ctx=None):
    """CORR(RANK(((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12)))),RANK(VOLUME),6)"""
    c, o, h, l, v, vw, amt = _cols(df)
    stoch = (c - op.tsmin(l, 12)) / (op.tsmax(h, 12) - op.tsmin(l, 12))
    return op.corr(op.rank(stoch), op.rank(v), 6)


def alpha_177(df, ctx=None):
    """((20-HIGHDAY(HIGH,20))/20)*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    return (20 - op.highday(h, 20)) / 20 * 100


def alpha_178(df, ctx=None):
    """(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev = op.delay(c, 1)
    return (c - prev) / prev * v


def alpha_179(df, ctx=None):
    """(RANK(CORR(VWAP,VOLUME,4))*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    a = op.rank(op.corr(vw, v, 4))
    b = op.rank(op.corr(op.rank(l), op.rank(op.mean(v, 50)), 12))
    return a * b


def alpha_180(df, ctx=None):
    """((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7))):(-1*VOLUME))"""
    c, o, h, l, v, vw, amt = _cols(df)
    d7 = op.delta(c, 7)
    up_branch = -1 * op.tsrank(abs_(d7), 60) * op.sign(d7)
    dn_branch = -1 * v
    cond = op.mean(v, 20) < v
    return pd.Series(np.where(cond, up_branch, dn_branch), index=c.index)


def alpha_181(df, ctx=None):
    """SUM(((ret-MEAN(ret,20))-(bench-MEAN(bench,20))^2),20)/SUM((bench-MEAN(bench,20))^3)
    where ret = CLOSE/DELAY(CLOSE,1)-1."""
    c, o, h, l, v, vw, amt = _cols(df)
    if ctx is None or "bench_close" not in ctx:
        raise MissingExternal("Alpha181 requires ctx key: bench_close")
    bc = ctx["bench_close"]
    ret = c / op.delay(c, 1) - 1
    bm_dev = bc - op.mean(bc, 20)
    num = op.sum_((ret - op.mean(ret, 20)) - bm_dev ** 2, 20)
    den = op.sum_(bm_dev ** 3, 20)
    return op.safe_divide(num, den)


def alpha_182(df, ctx=None):
    """COUNT((CLOSE>OPEN & BC>BO) OR (CLOSE<OPEN & BC<BO),20)/20"""
    c, o, h, l, v, vw, amt = _cols(df)
    if ctx is None or "bench_open" not in ctx or "bench_close" not in ctx:
        raise MissingExternal("Alpha182 requires ctx keys: bench_open, bench_close")
    bo, bc = ctx["bench_open"], ctx["bench_close"]
    cond = ((c > o) & (bc > bo)) | ((c < o) & (bc < bo))
    return op.count(cond, 20) / 20


def alpha_183(df, ctx=None):
    """MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)"""
    c, o, h, l, v, vw, amt = _cols(df)
    m = op.mean(c, 24)
    s = op.sum_(c - m, 24)
    return op.tsmax(s, 24) - op.safe_divide(op.tsmin(s, 24), op.std(c, 24))


def alpha_184(df, ctx=None):
    """RANK(CORR(DELAY((OPEN-CLOSE),1),CLOSE,200))+RANK((OPEN-CLOSE))"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.rank(op.corr(op.delay(o - c, 1), c, 200)) + op.rank(o - c)


def alpha_185(df, ctx=None):
    """RANK((-1*((1-(OPEN/CLOSE))^2)))"""
    c, o, h, l, v, vw, amt = _cols(df)
    inner = -1 * (1 - o / c) ** 2
    return op.rank(inner)


def alpha_186(df, ctx=None):
    """(MEAN(ABS(LD_pct-HD_pct)/(LD_pct+HD_pct)*100,6)+DELAY(MEAN(...,6),6))/2"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_c = op.delay(c, 1)
    hd = h - prev_c
    ld = prev_c - l
    tr = np.maximum.reduce([h - l, np.abs(hd), np.abs(ld)])
    tr = pd.Series(tr, index=c.index).replace(0, np.nan)
    hd_pct = np.where((ld > 0) & (ld > hd), ld, 0.0) * 100 / tr
    ld_pct = np.where((hd > 0) & (hd > ld), hd, 0.0) * 100 / tr
    inner = np.abs(hd_pct - ld_pct) / (hd_pct + ld_pct) * 100
    mean6 = op.mean(pd.Series(inner, index=c.index), 6)
    return (mean6 + op.delay(mean6, 6)) / 2


def alpha_187(df, ctx=None):
    """SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)"""
    c, o, h, l, v, vw, amt = _cols(df)
    prev_o = op.delay(o, 1)
    inner = np.where(o <= prev_o, 0.0, np.maximum(h - o, o - prev_o))
    return op.sum_(pd.Series(inner, index=c.index), 20)


def alpha_188(df, ctx=None):
    """((HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100"""
    c, o, h, l, v, vw, amt = _cols(df)
    s = op.sma(h - l, 11, 2)
    return ((h - l) - s) / s * 100


def alpha_189(df, ctx=None):
    """MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.mean(abs_(c - op.mean(c, 6)), 6)


def alpha_190(df, ctx=None):
    """LOG((COUNT(up_cond,20)-1)*(SUMIF(down_dev^2,20,down_cond))/
          ((COUNT(down_cond,20))*(SUMIF(up_dev^2,20,up_cond))))
    where threshold = (CLOSE/DELAY(CLOSE,19))^(1/20)-1,
          today_ret  = CLOSE/DELAY(CLOSE,1)-1,
          up_cond    = today_ret > threshold, down_cond = today_ret < threshold,
          up_dev     = today_ret - threshold, down_dev = today_ret - threshold."""
    c, o, h, l, v, vw, amt = _cols(df)
    threshold = (c / op.delay(c, 19)) ** (1 / 20) - 1
    today_ret = c / op.delay(c, 1) - 1
    up_cond = today_ret > threshold
    down_cond = today_ret < threshold
    dev = today_ret - threshold
    a = op.count(up_cond, 20) - 1
    b = op.sumif(dev ** 2, 20, down_cond)
    d = op.count(down_cond, 20)
    e = op.sumif(dev ** 2, 20, up_cond)
    inner = op.safe_divide(a * b, d * e)
    return np.log(inner.replace(0, np.nan))


def alpha_191(df, ctx=None):
    """((CORR(MEAN(VOLUME,20),LOW,5)+((HIGH+LOW)/2))-CLOSE)"""
    c, o, h, l, v, vw, amt = _cols(df)
    return op.corr(op.mean(v, 20), l, 5) + (h + l) / 2 - c


# ===========================================================================
# Registry
# ===========================================================================

ALPHA191_REGISTRY: dict[str, callable] = {}
for _name, _obj in list(globals().items()):
    if _name.startswith("alpha_") and callable(_obj):
        _num = _name.split("_", 1)[1]
        ALPHA191_REGISTRY[_num] = _obj
del _name, _obj



