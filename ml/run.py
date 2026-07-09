"""CLI entrypoint for ML multi-factor synthesis via walk-forward LightGBM.

Usage:
    python -m ml.run --panel data/all_factors.parquet \\
        --save-oas results/ml_oos.parquet \\
        --save-report results/ml_report

After running, the OOS parquet can be fed to tools/research/backtest.py:
    python tools/research/backtest.py --data results/ml_oos.parquet \\
        --save results/ml_report/backtest --config data/ml_composite.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow `python ml/run.py` (file invocation) as well as `python -m ml.run`
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.dataset import load_panel                                    # noqa: E402
from ml.walkforward import run_walkforward                          # noqa: E402
from ml.evaluate import evaluate_oos, save_oas_parquet, print_summary  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward LightGBM multi-factor synthesis (OOS)."
    )
    parser.add_argument("--panel", required=True,
                        help="Input all-factors parquet (from fetch_data.py).")
    parser.add_argument("--save-oas", required=True,
                        help="Output OOS parquet: datetime, symbol, close, alpha_composite.")
    parser.add_argument("--save-report", default=None,
                        help="Optional dir to dump month_log.csv + stats.txt.")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward-return horizon in bars (label).")
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--purge-days", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=5)
    args = parser.parse_args()

    print(f"[ml] loading panel: {args.panel}")
    panel = load_panel(args.panel)
    print(f"[ml] panel rows={len(panel)}  symbols={panel['symbol'].nunique()}  "
          f"date range=[{panel['datetime'].min().date()}, "
          f"{panel['datetime'].max().date()}]")

    oos_preds, month_log = run_walkforward(
        panel,
        horizon=args.horizon,
        min_train_months=args.min_train_months,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
    )

    stats = evaluate_oos(oos_preds)
    print_summary(stats, month_log)

    # Save OOS parquet
    save_oas_parquet(oos_preds, args.save_oas)
    print(f"\n[ml] OOS predictions -> {args.save_oas}")

    # Optional report dump
    if args.save_report:
        os.makedirs(args.save_report, exist_ok=True)
        month_log.to_csv(os.path.join(args.save_report, "month_log.csv"), index=False)
        print(f"[ml] month log -> {args.save_report}/month_log.csv")

        # Save stats as text
        if stats:
            with open(os.path.join(args.save_report, "stats.txt"), "w") as f:
                f.write("Walk-forward LightGBM OOS summary\n")
                f.write("=" * 50 + "\n\n")
                for k, v in stats.items():
                    if k == "quantile_means":
                        f.write("quantile_means:\n")
                        for q, val in v.items():
                            f.write(f"  {q}: {val:+.6f}\n")
                    else:
                        f.write(f"{k}: {v}\n")
            print(f"[ml] stats -> {args.save_report}/stats.txt")


if __name__ == "__main__":
    main()
