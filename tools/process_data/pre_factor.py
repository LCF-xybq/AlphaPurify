"""
Compute and append technical factors to CSV/Parquet files using TA-Lib.
Supports computing individual factors or all available TA-Lib indicators.

Usage:
    python pre_factor.py --input /path/to/data --factor macd
    python pre_factor.py --input /path/to/data --factor all
    python pre_factor.py --input /path/to/data --factor rsi,macd,atr
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import talib


# ---------------------------------------------------------------------------
# TA-Lib factor registry — grouped by TA-Lib function call
# ---------------------------------------------------------------------------

# Each entry: (factor_name, func, params_dict_or_None)
# func receives (close, high, low, open_arr, volume, params) and returns numpy array
def _sma(c, h, l, o, v, p): return talib.SMA(c, **p)
def _ema(c, h, l, o, v, p): return talib.EMA(c, **p)
def _wma(c, h, l, o, v, p): return talib.WMA(c, **p)
def _dema(c, h, l, o, v, p): return talib.DEMA(c, **p)
def _tema(c, h, l, o, v, p): return talib.TEMA(c, **p)
def _kama(c, h, l, o, v, p): return talib.KAMA(c, **p)
def _mama(c, h, l, o, v, p): return talib.MAMA(c, **p)[0]
def _midprice(c, h, l, o, v, p): return talib.MIDPRICE(h, l, **p)
def _sar(c, h, l, o, v, p): return talib.SAR(h, l, **p)
def _obv(c, h, l, o, v, p): return talib.OBV(c, v, **p)
def _trima(c, h, l, o, v, p): return talib.TRIMA(c, **p)
def _t3(c, h, l, o, v, p): return talib.T3(c, **p)
def _ht_dcperiod(c, h, l, o, v, p): return talib.HT_DCPERIOD(c, **p)
def _ht_dcphase(c, h, l, o, v, p): return talib.HT_DCPHASE(c, **p)
def _ht_trendline(c, h, l, o, v, p): return talib.HT_TRENDLINE(c, **p)
def _ht_trendmode(c, h, l, o, v, p): return talib.HT_TRENDMODE(c, **p)
def _rsi(c, h, l, o, v, p): return talib.RSI(c, **p)
def _cmo(c, h, l, o, v, p): return talib.CMO(c, **p)
def _mom(c, h, l, o, v, p): return talib.MOM(c, **p)
def _trix(c, h, l, o, v, p): return talib.TRIX(c, **p)
def _roc(c, h, l, o, v, p): return talib.ROC(c, **p)
def _rocp(c, h, l, o, v, p): return talib.ROCP(c, **p)
def _rocr(c, h, l, o, v, p): return talib.ROCR(c, **p)
def _rocr100(c, h, l, o, v, p): return talib.ROCR100(c, **p)
def _ppo(c, h, l, o, v, p): return talib.PPO(c, **p)
def _apo(c, h, l, o, v, p): return talib.APO(c, **p)
def _atr(c, h, l, o, v, p): return talib.ATR(h, l, c, **p)
def _natr(c, h, l, o, v, p): return talib.NATR(h, l, c, **p)
def _cci(c, h, l, o, v, p): return talib.CCI(h, l, c, **p)
def _willr(c, h, l, o, v, p): return talib.WILLR(h, l, c, **p)
def _adx(c, h, l, o, v, p): return talib.ADX(h, l, c, **p)
def _adxr(c, h, l, o, v, p): return talib.ADXR(h, l, c, **p)
def _dx(c, h, l, o, v, p): return talib.DX(h, l, c, **p)
def _minus_di(c, h, l, o, v, p): return talib.MINUS_DI(h, l, c, **p)
def _minus_dm(c, h, l, o, v, p): return talib.MINUS_DM(h, l, **p)
def _plus_di(c, h, l, o, v, p): return talib.PLUS_DI(h, l, c, **p)
def _plus_dm(c, h, l, o, v, p): return talib.PLUS_DM(h, l, **p)
def _bop(c, h, l, o, v, p): return talib.BOP(o, h, l, c, **p)
def _ultosc(c, h, l, o, v, p): return talib.ULTOSC(h, l, c, **p)
def _mfi(c, h, l, o, v, p): return talib.MFI(h, l, c, v, **p)
def _ad(c, h, l, o, v, p): return talib.AD(h, l, c, v, **p)
def _adosc(c, h, l, o, v, p): return talib.ADOSC(h, l, c, v, **p)
def _aroon_down(c, h, l, o, v, p): return talib.AROON(h, l, **p)[0]
def _aroon_up(c, h, l, o, v, p): return talib.AROON(h, l, **p)[1]
def _aroonosc(c, h, l, o, v, p): return talib.AROONOSC(h, l, **p)
def _ht_sine(c, h, l, o, v, p): return talib.HT_SINE(c, **p)[0]
def _ht_leadsine(c, h, l, o, v, p): return talib.HT_SINE(c, **p)[1]
def _ht_phasor_inphase(c, h, l, o, v, p): return talib.HT_PHASOR(c, **p)[0]
def _ht_phasor_quadrature(c, h, l, o, v, p): return talib.HT_PHASOR(c, **p)[1]


# Factor definitions: (factor_name, func, default_params)
# All factors call the same underlying TA-Lib function once per factor name.
FACTOR_DEFS = [
    ("sma",              _sma,              {"timeperiod": 20}),
    ("ema",              _ema,              {"timeperiod": 20}),
    ("wma",              _wma,              {"timeperiod": 20}),
    ("dema",             _dema,             {"timeperiod": 20}),
    ("tema",             _tema,             {"timeperiod": 20}),
    ("kama",             _kama,             {"timeperiod": 30}),
    ("mama",             _mama,             {"fastlimit": 0.5, "slowlimit": 0.05}),
    ("midprice",         _midprice,         {"timeperiod": 14}),
    ("sar",              _sar,              {"acceleration": 0.02, "maximum": 0.2}),
    ("obv",              _obv,              {}),
    ("trima",            _trima,            {"timeperiod": 30}),
    ("t3",               _t3,               {"timeperiod": 5, "vfactor": 0.7}),
    ("ht_dcperiod",      _ht_dcperiod,      {}),
    ("ht_dcphase",       _ht_dcphase,       {}),
    ("ht_trendline",     _ht_trendline,     {}),
    ("ht_trendmode",     _ht_trendmode,     {}),
    ("rsi",              _rsi,              {"timeperiod": 14}),
    ("cmo",              _cmo,              {"timeperiod": 14}),
    ("mom",              _mom,              {"timeperiod": 10}),
    ("trix",             _trix,             {"timeperiod": 30}),
    ("roc",              _roc,              {"timeperiod": 10}),
    ("rocp",             _rocp,             {"timeperiod": 10}),
    ("rocr",             _rocr,             {"timeperiod": 10}),
    ("rocr100",          _rocr100,          {"timeperiod": 10}),
    ("ppo",              _ppo,              {}),
    ("apo",              _apo,              {"fastperiod": 12, "slowperiod": 26}),
    ("atr",              _atr,              {"timeperiod": 14}),
    ("natr",             _natr,             {"timeperiod": 14}),
    ("cci",              _cci,              {"timeperiod": 14}),
    ("willr",            _willr,            {"timeperiod": 14}),
    ("adx",              _adx,              {"timeperiod": 14}),
    ("adxr",             _adxr,             {"timeperiod": 14}),
    ("dx",               _dx,               {"timeperiod": 14}),
    ("minus_di",         _minus_di,         {"timeperiod": 14}),
    ("minus_dm",         _minus_dm,         {"timeperiod": 14}),
    ("plus_di",          _plus_di,          {"timeperiod": 14}),
    ("plus_dm",          _plus_dm,          {"timeperiod": 14}),
    ("bop",              _bop,              {}),
    ("ultosc",           _ultosc,           {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28}),
    ("mfi",              _mfi,              {"timeperiod": 14}),
    ("ad",               _ad,               {}),
    ("adosc",            _adosc,            {"fastperiod": 3, "slowperiod": 10}),
    ("aroon_down",       _aroon_down,       {"timeperiod": 14}),
    ("aroon_up",         _aroon_up,         {"timeperiod": 14}),
    ("aroonosc",         _aroonosc,         {"timeperiod": 14}),
    ("ht_sine",          _ht_sine,          {}),
    ("ht_leadsine",      _ht_leadsine,      {}),
    ("ht_phasor_inphase",_ht_phasor_inphase,{}),
    ("ht_phasor_quadrature",_ht_phasor_quadrature,{}),
    # BBANDS — computed once, split into 3 columns
    # MACD  — computed once, split into 3 columns
    # STOCH — computed once, split into 2 columns
]

# Index into ALL_FACTOR_DEFS for multi-output factors to avoid recomputation
# Dynamically computed AFTER ALL_FACTOR_DEFS is built
n_single = len(FACTOR_DEFS)          # 49
n_bb    = 3                          # bb_upper, bb_middle, bb_lower
n_macd  = 3                          # macd, macd_signal, macd_hist
n_stoch = 4                          # stoch_slowk, stoch_slowd, stochf_fastk, stochf_fastd

_IDX_BBANDS = n_single                          # 49
_IDX_MACD   = n_single + n_bb                    # 52
_IDX_STOCH  = n_single + n_bb + n_macd          # 56

ALL_FACTOR_DEFS = FACTOR_DEFS + [
    ("bb_upper",   None, None),   # slot reserved; filled below
    ("bb_middle",  None, None),
    ("bb_lower",   None, None),
    ("macd",       None, None),
    ("macd_signal",None, None),
    ("macd_hist",  None, None),
    ("stoch_slowk",None, None),
    ("stoch_slowd",None, None),
    ("stochf_fastk",None, None),
    ("stochf_fastd",None, None),
]


def compute_talib_factors(df: pd.DataFrame, factor_indices: list[int]) -> pd.DataFrame:
    """
    Compute TA-Lib factors by index into ALL_FACTOR_DEFS.
    Collects all new columns and assigns them once to avoid DataFrame fragmentation.
    """
    n = len(df)
    if n < 5:
        return df

    close  = df["close"].values.astype(float)  if "close"  in df.columns else None
    high   = df["high"].values.astype(float)   if "high"   in df.columns else close
    low    = df["low"].values.astype(float)   if "low"    in df.columns else close
    open_  = df["open"].values.astype(float)  if "open"   in df.columns else close
    volume = df["volume"].values.astype(float) if "volume" in df.columns else None

    if close is None:
        return df

    new_cols = {}   # col_name -> numpy array

    # BBANDS / MACD / STOCH multi-output: track whether already computed
    bb_done   = False
    macd_done = False
    stoch_done = False

    for idx in factor_indices:
        name, func, default_params = ALL_FACTOR_DEFS[idx]
        params = default_params or {}

        if idx == _IDX_BBANDS and not bb_done:
            bb_done = True
            res = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            new_cols["bb_upper"]  = res[0]
            new_cols["bb_middle"] = res[1]
            new_cols["bb_lower"]  = res[2]
            continue
        elif idx == _IDX_MACD and not macd_done:
            macd_done = True
            res = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            new_cols["macd"]        = res[0]
            new_cols["macd_signal"] = res[1]
            new_cols["macd_hist"]   = res[2]
            continue
        elif idx == _IDX_STOCH and not stoch_done:
            stoch_done = True
            res = talib.STOCH(high, low, close,
                               fastk_period=5, slowk_period=3,
                               slowd_period=3, slowk_matype=0, slowd_matype=0)
            new_cols["stoch_slowk"] = res[0]
            new_cols["stoch_slowd"] = res[1]
            res_f = talib.STOCHF(high, low, close,
                                  fastk_period=5, fastd_period=3, fastd_matype=0)
            new_cols["stochf_fastk"] = res_f[0]
            new_cols["stochf_fastd"] = res_f[1]
            continue

        if func is None:
            continue

        val = func(close, high, low, open_, volume, params)

        if isinstance(val, tuple):
            val, col_name = val
        else:
            col_name = name

        new_cols[col_name] = val

    # Assign all new columns in one operation — avoids DataFrame fragmentation
    if new_cols:
        df = df.assign(**new_cols)

    return df


def process_file(filepath: str, factor_indices: list[int], timeout: int = 300) -> dict:
    """Process a single file: read, compute factors, write back."""
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            return {"file": filepath, "status": "skipped", "reason": "unsupported format"}

        if "symbol" not in df.columns or "datetime" not in df.columns:
            return {"file": filepath, "status": "skipped", "reason": "missing symbol/datetime column"}

        df = compute_talib_factors(df, factor_indices)

        if ext == ".csv":
            df.to_csv(filepath, index=False)
        elif ext == ".parquet":
            df.to_parquet(filepath, index=False)

        return {"file": filepath, "status": "success", "rows": len(df), "cols": len(df.columns)}

    except Exception as e:
        return {"file": filepath, "status": "error", "reason": str(e)}


def process_file_alpha(filepath: str, timeout: int = 300) -> dict:
    """Legacy per-file alpha hook (kept for backward compatibility).
    DEPRECATED: most Alpha191 formulas need cross-sectional RANK and cannot be
    computed per-file. Use the panel-mode `--src factors` path instead, which
    merges all files in --input into one panel before computing.
    """
    raise RuntimeError(
        "process_file_alpha is deprecated — alpha191 needs panel mode. "
        "Use the main() --src factors path which calls factors.process_panel_dir."
    )


def main():
    parser = argparse.ArgumentParser(description="Compute TA-Lib factors and append to data files.")
    parser.add_argument("--input", required=True, help="Input directory containing CSV/Parquet files.")
    parser.add_argument(
        "--src",
        default="ta-lib",
        choices=["ta-lib", "factors"],
        help="Source of factor functions: 'ta-lib' (default) or 'factors' (custom).",
    )
    parser.add_argument(
        "--factor",
        required=True,
        help="Factor name(s) to compute. Use 'all' for all TA-Lib factors, or comma-separated list (e.g., 'rsi,macd,atr'). "
             "In --src factors mode, use alpha191 numbers (e.g., '001,002' or 'all').",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per file in seconds.")
    # Alpha191 panel-mode options
    parser.add_argument("--ff3", default=None, help="Optional parquet with MKT/SMB/HML columns for Alpha030/147/149/181/182.")
    parser.add_argument("--benchmark", default=None, help="Optional parquet with open/close columns for benchmark-index alphas.")
    args = parser.parse_args()

    # Collect files
    csv_files    = glob.glob(os.path.join(args.input, "*.csv"))
    parquet_files = glob.glob(os.path.join(args.input, "*.parquet"))
    all_files = csv_files + parquet_files

    if not all_files:
        raise FileNotFoundError(f"No CSV or Parquet files found in {args.input}")

    # === Alpha factors mode (panel: merge all files -> compute -> split back) ===
    if args.src == "factors":
        # Resolve the requested alpha191 numbers
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from factors import process_panel_dir, ALPHA191_REGISTRY

        if args.factor.lower() == "all":
            names = None
        else:
            names = []
            for s in args.factor.split(","):
                s = s.strip()
                if s.lower().startswith("alpha191_"):
                    s = s[len("alpha191_"):]
                names.append(s.zfill(3))
            unknown = [n for n in names if n not in ALPHA191_REGISTRY]
            if unknown:
                raise ValueError(f"Unknown alpha191 ids: {unknown}")

        # Build ctx from optional external data files
        ctx = None
        if args.ff3 or args.benchmark:
            from factors.data import load_external
            # Need a placeholder index; load_external aligns to a MultiIndex reference
            # but here we don't have one yet — pass paths and let process_panel_dir
            # call load_external with the panel's index. For simplicity, we don't
            # support external data via CLI yet; user can pass ctx via Python API.
            print("[factors] NOTE: --ff3 / --benchmark CLI args require panel-mode loading.")
            print("[factors] External-data alphas (030/075/149/181/182) will be skipped.")
            ctx = None

        print(f"Computing Alpha191 factors ({'all 191' if names is None else len(names)} selected)")
        print(f"Found {len(all_files)} files. Panel-mode: merging, computing, splitting back.")
        result = process_panel_dir(args.input, factor_names=names, ctx=ctx)
        print(f"\nDone: {result['success']} succeeded")
        if result["skipped"]:
            print(f"Skipped {len(result['skipped'])} file(s): {result['skipped']}")
        if result["errors"]:
            print(f"Errors: {result['errors']}")
        if result["alpha_skipped"]:
            print(f"All-NaN alpha columns (likely missing external data): {result['alpha_skipped']}")
        return

    # === TA-Lib mode (default) ===
    # Build factor name -> index mapping
    name_to_idx = {name: i for i, (name, _, _) in enumerate(ALL_FACTOR_DEFS)}

    # Special composite factors that share a single TA-Lib call
    COMPOSITE = {
        "bb": ["bb_upper", "bb_middle", "bb_lower"],
        "bollinger": ["bb_upper", "bb_middle", "bb_lower"],
        "stoch": ["stoch_slowk", "stoch_slowd"],
        "stochastic": ["stoch_slowk", "stoch_slowd"],
    }

    # Resolve factor indices
    if args.factor.lower() == "all":
        factor_indices = list(range(len(ALL_FACTOR_DEFS)))
    else:
        factor_indices = []
        for f in args.factor.split(","):
            f = f.strip().lower()
            if f in COMPOSITE:
                for sub in COMPOSITE[f]:
                    if sub in name_to_idx:
                        factor_indices.append(name_to_idx[sub])
            elif f in name_to_idx:
                factor_indices.append(name_to_idx[f])

    if not factor_indices:
        raise ValueError(f"No valid factors found for: {args.factor}")

    print(f"Computing {len(factor_indices)} factor columns")
    print(f"Source: {args.src}")

    # Collect files
    csv_files    = glob.glob(os.path.join(args.input, "*.csv"))
    parquet_files = glob.glob(os.path.join(args.input, "*.parquet"))
    all_files = csv_files + parquet_files

    if not all_files:
        raise FileNotFoundError(f"No CSV or Parquet files found in {args.input}")

    print(f"Found {len(all_files)} files, processing with {args.workers} workers (timeout={args.timeout}s)...")

    success, errors, skipped = 0, 0, 0
    n_files = len(all_files)

    ctx = mp.get_context("spawn")  # safer on Linux, avoids fork issues with C libs
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(process_file, f, factor_indices, args.timeout): f
            for f in all_files
        }
        for future in as_completed(futures, timeout=args.timeout * n_files / args.workers + 60):
            try:
                result = future.result(timeout=args.timeout)
            except TimeoutError:
                result = {"file": futures[future], "status": "error", "reason": "timeout"}
            except Exception as e:
                result = {"file": futures[future], "status": "error", "reason": str(e)}

            if result["status"] == "success":
                success += 1
                print(f"[OK] {result['file']} ({result['rows']} rows, {result['cols']} cols)")
            elif result["status"] == "skipped":
                skipped += 1
                print(f"[SKIP] {result['file']}: {result['reason']}")
            else:
                errors += 1
                print(f"[ERR] {result['file']}: {result['reason']}")

    print(f"\nDone: {success} succeeded, {skipped} skipped, {errors} failed")


if __name__ == "__main__":
    main()
