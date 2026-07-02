"""
Run a full FactorAnalyzer backtest report from a panel dataset.

The primary factor and the
preprocessing methods are read from a YAML combination config (the same file
fetch_data.py consumes):

    factor:      cfg['alpha'][0]      (also accepts cfg['factor'])
    winsorize:   cfg['winsorize'][0]
    standardize: cfg['standardize'][0]

Usage:
    python tools/research/backtest.py --data out.parquet --save ./reports \
        --config data/combination1.yaml

    # Override the primary factor from config
    python tools/research/backtest.py --data out.parquet --save ./reports \
        --config data/combination1.yaml --factor my_alpha
"""

import argparse
import os
from dataclasses import dataclass

import pandas as pd
import yaml

from alphapurify import AlphaPurifier, FactorAnalyzer, PortfolioExposures, PureExposures

# Columns considered metadata, not factors
_META_COLS = {"datetime", "symbol", "open", "high", "low", "close", "volume"}


@dataclass
class BacktestConfig:
    factor_name: str | None
    winsorize: str | None
    standardize: str | None
    exposure_cols: list[str]
    trace_rebalance: str | int | None
    trace_dates: list[str]


def load_config(path: str) -> BacktestConfig:
    """Read combination yaml → BacktestConfig.

    Recognized fields:
      alpha / factor   : primary factor (first entry if list)
      winsorize        : method name (first entry if list)
      standardize      : method name (first entry if list)
      exposure         : list of exposure columns for attribution
                         (PureExposures / PortfolioExposures). Also accepts
                         `exposure_cols` as an alias. Missing/empty → skip
                         attribution. No auto-detection from the panel.
      trace            : dict with two keys —
                           rebalance_period: str | int   (REQUIRED if trace present)
                           dates: list[str] | list[{date: str}]
                         If `trace` is absent, FA.trace() is skipped entirely.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    def _first(seq):
        if not seq:
            return None
        if isinstance(seq, str):
            return seq
        if isinstance(seq, list):
            return seq[0] if seq else None
        for grp in seq.values():
            if grp:
                return grp[0]
        return None

    factor_name = _first(cfg.get("factor") or cfg.get("alpha"))
    winsorize = _first(cfg.get("winsorize"))
    standardize = _first(cfg.get("standardize"))

    exposure_cols = _as_str_list(cfg.get("exposure") or cfg.get("exposure_cols"))

    trace_rebalance: str | int | None = None
    trace_dates: list[str] = []
    raw_trace = cfg.get("trace")
    if raw_trace is not None:
        if not isinstance(raw_trace, dict):
            raise ValueError(
                "`trace` must be a mapping with keys: rebalance_period, dates. "
                f"Got {type(raw_trace).__name__}: {raw_trace!r}"
            )
        if "rebalance_period" not in raw_trace:
            raise ValueError(
                "`trace.rebalance_period` is required when `trace` is present. "
                "Set it to 'W', 'M', 'Q', or an integer number of bars."
            )
        trace_rebalance = raw_trace["rebalance_period"]
        trace_dates = _parse_trace_dates(raw_trace.get("dates"))

    return BacktestConfig(
        factor_name=factor_name,
        winsorize=winsorize,
        standardize=standardize,
        exposure_cols=exposure_cols,
        trace_rebalance=trace_rebalance,
        trace_dates=trace_dates,
    )


def _parse_trace_dates(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and "date" in item:
                out.append(str(item["date"]))
            else:
                raise ValueError(
                    f"trace.dates entries must be str or {{date: str}}; got {item!r}"
                )
        return out
    raise ValueError(f"trace.dates must be a list or str; got {value!r}")


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return []


def load_panel(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError(f"--data must be .parquet or .csv, got '{ext}'")
    return df


def save_fig(fig, path: str):
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.write_html(path)
    print(f"[report] wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate FactorAnalyzer backtest reports from a panel dataset."
    )
    parser.add_argument("--data", required=True, help="Panel data file (.parquet or .csv).")
    parser.add_argument("--save", required=True, help="Output directory for HTML reports.")
    parser.add_argument(
        "--config",
        required=True,
        help="Combination YAML config — supplies factor, winsorize, standardize.",
    )
    parser.add_argument(
        "--factor",
        nargs="*",
        default=[],
        help="Override the primary factor from config (first item used). "
        "Default: empty — read from config's alpha/factor field.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    factor_name, winsorize, standardize = cfg.factor_name, cfg.winsorize, cfg.standardize

    # CLI --factor overrides config's alpha/factor (first entry used)
    if args.factor:
        factor_name = args.factor[0]

    df = load_panel(args.data)
    factor_cols = [c for c in df.columns if c not in _META_COLS]
    if not factor_cols:
        raise ValueError(
            f"No factor columns found (non-meta cols). Got columns: {df.columns.tolist()}"
        )

    # Fall back to first factor column if config didn't specify one
    factor_name = factor_name or factor_cols[0]
    if factor_name not in df.columns:
        raise ValueError(
            f"Config factor '{factor_name}' not in data columns: {df.columns.tolist()}"
        )

    # exposure_cols: must be explicitly declared in config — no auto-detection.
    # Primary factor is always excluded — it cannot be its own exposure.
    exposure_cols = [c for c in cfg.exposure_cols if c != factor_name]
    missing = [c for c in exposure_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"exposure columns {missing} not in data columns: {df.columns.tolist()}"
        )

    print(f"[report] config:         {args.config}")
    print(f"[report] primary factor: {factor_name}")
    print(f"[report] winsorize:      {winsorize or '(skip)'}")
    print(f"[report] standardize:    {standardize or '(skip)'}")
    print(f"[report] exposure cols:  {exposure_cols or '(none)'}")
    print(f"[report] trace rebalance: {cfg.trace_rebalance!r}")
    print(f"[report] trace dates:    {cfg.trace_dates or '(none)'}")

    # ---- preprocess primary factor (README step 1) ----
    pre = AlphaPurifier(
        df, factor_name=factor_name, trade_date_col="datetime", symbol_col="symbol"
    )
    if winsorize:
        pre = pre.winsorize(method=winsorize)
    if standardize:
        pre = pre.standardize(method=standardize)
    proc = pre.to_result()

    os.makedirs(args.save, exist_ok=True)

    # ---- FactorAnalyzer backtest (README step 2) ----
    FA = FactorAnalyzer(
        base_df=proc,
        trade_date_col="datetime",
        symbol_col="symbol",
        price_col="close",
        factor_name=factor_name,
    )
    FA.run()
    save_fig(
        FA.create_long_return_sheet(return_fig=True),
        os.path.join(args.save, "long_return.html"),
    )
    save_fig(
        FA.create_long_short_return_sheet(return_fig=True),
        os.path.join(args.save, "long_short_return.html"),
    )
    save_fig(
        FA.create_short_return_sheet(return_fig=True),
        os.path.join(args.save, "short_return.html"),
    )
    save_fig(
        FA.create_single_fac_ic_sheet(return_fig=True),
        os.path.join(args.save, "single_fac_ic.html"),
    )

    # ---- Attribution (README step 3, needs ≥1 extra factor) ----
    if exposure_cols:
        # PureExposures: cross-sectional regression attribution
        Ex = PureExposures(
            base_df=proc,
            trade_date_col="datetime",
            symbol_col="symbol",
            price_col="close",
            factor_name=factor_name,
            exposure_cols=exposure_cols,
        )
        Ex.run()
        save_fig(
            Ex.plot_pure_exposures(return_fig=True),
            os.path.join(args.save, "pure_exposures.html"),
        )
        save_fig(
            Ex.plot_pure_returns(return_fig=True),
            os.path.join(args.save, "pure_returns.html"),
        )
        save_fig(
            Ex.plot_pure_exposures_and_returns(return_fig=True),
            os.path.join(args.save, "pure_exposures_and_returns.html"),
        )
        save_fig(
            Ex.plot_correlations(return_fig=True),
            os.path.join(args.save, "correlations.html"),
        )

        # PortfolioExposures: quantile-portfolio-level exposure to risk factors
        PE = PortfolioExposures(
            base_df=proc,
            trade_date_col="datetime",
            symbol_col="symbol",
            price_col="close",
            factor_name=factor_name,
            exposure_cols=exposure_cols,
        )
        PE.run()
        save_fig(
            PE.plot_portfolio_exposures(return_fig=True),
            os.path.join(args.save, "portfolio_exposures.html"),
        )
        save_fig(
            PE.plot_portfolio_returns(return_fig=True),
            os.path.join(args.save, "portfolio_returns.html"),
        )
        save_fig(
            PE.plot_portfolio_exposures_and_returns(return_fig=True),
            os.path.join(args.save, "portfolio_exposures_and_returns.html"),
        )
    else:
        print("[report] skipping attribution (need ≥2 factor columns, found 1)")

    # ---- FA.trace snapshots (only when config declares a trace block) ----
    if cfg.trace_rebalance is not None:
        if not cfg.trace_dates:
            print("[report] skipping trace (trace block present but dates is empty)")
        else:
            for date_str in cfg.trace_dates:
                try:
                    fig, _ = FA.trace(
                        rebalance_period=cfg.trace_rebalance,
                        date=date_str,
                        bins=None,
                        position="l",
                        return_fig=True,
                    )
                except ValueError as e:
                    # date not aligned to rebalance calendar — skip but keep going
                    print(f"[report] trace skipped for {date_str}: {e}")
                    continue
                safe_date = str(date_str).replace(":", "-").replace(" ", "_")
                save_fig(fig, os.path.join(args.save, f"trace_{safe_date}.html"))
    else:
        print("[report] skipping trace (no `trace` block in config)")

    print(f"[report] done -> {args.save}")


if __name__ == "__main__":
    main()
