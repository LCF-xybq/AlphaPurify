#!/usr/bin/env python3
"""
Compare a single column across two CSV files and report the set differences.

Prints the count of values only in csv1, only in csv2, and in both. Optionally
writes the diff rows (with a `source` marker) to a CSV next to csv1.

Usage:
    python tools/process_data/csv_diff.py --csv1 a.csv --csv2 b.csv --column code
    python tools/process_data/csv_diff.py --csv1 a.csv --csv2 b.csv --column code --save
"""
import argparse
from pathlib import Path

import pandas as pd


def diff_column(csv1: Path, csv2: Path, column: str) -> dict:
    """Compare `column` across two CSV files.

    Returns a dict with sets: only_in_1, only_in_2, common, and per-file
    duplicate counts.
    """
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    for path, df in [(csv1, df1), (csv2, df2)]:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in {path}. Available: {list(df.columns)}")

    s1 = df1[column]
    s2 = df2[column]

    set1, set2 = set(s1), set(s2)
    dup1 = s1.duplicated().sum()
    dup2 = s2.duplicated().sum()

    return {
        "df1": df1,
        "df2": df2,
        "only_in_1": set1 - set2,
        "only_in_2": set2 - set1,
        "common": set1 & set2,
        "dup1": int(dup1),
        "dup2": int(dup2),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare a column across two CSV files.")
    parser.add_argument("--csv1", required=True, type=Path, help="First CSV file path")
    parser.add_argument("--csv2", required=True, type=Path, help="Second CSV file path")
    parser.add_argument("--column", required=True, type=str, help="Column name to compare")
    parser.add_argument("--save", action="store_true",
                        help="Write diff rows (only_in_1, only_in_2) to CSV next to csv1")
    parser.add_argument("--show", type=int, default=20,
                        help="Max number of diff values to print per side (default: 20)")
    args = parser.parse_args()

    if not args.csv1.is_file():
        raise FileNotFoundError(f"csv1 not found: {args.csv1}")
    if not args.csv2.is_file():
        raise FileNotFoundError(f"csv2 not found: {args.csv2}")

    result = diff_column(args.csv1, args.csv2, args.column)

    n1 = len(result["df1"])
    n2 = len(result["df2"])
    only1 = result["only_in_1"]
    only2 = result["only_in_2"]
    common = result["common"]

    print("=" * 60)
    print(f"csv1:    {args.csv1}  ({n1} rows)")
    print(f"csv2:    {args.csv2}  ({n2} rows)")
    print(f"column:  {args.column}")
    print("-" * 60)
    print(f"unique in csv1:      {len(only1) + len(common)}")
    print(f"unique in csv2:      {len(only2) + len(common)}")
    print(f"common (intersection): {len(common)}")
    print(f"only in csv1:        {len(only1)}")
    print(f"only in csv2:        {len(only2)}")
    if result["dup1"] or result["dup2"]:
        print(f"[warn] duplicates within csv1: {result['dup1']}, within csv2: {result['dup2']}")
    print("=" * 60)

    def _preview(values, limit):
        if not values:
            return "(none)"
        items = sorted(map(str, values))[:limit]
        suffix = f"  ... (+{len(values) - limit} more)" if len(values) > limit else ""
        return ", ".join(items) + suffix

    print(f"\n[only in csv1] ({len(only1)})")
    print(_preview(only1, args.show))
    print(f"\n[only in csv2] ({len(only2)})")
    print(_preview(only2, args.show))

    if args.save:
        only1_df = result["df1"][result["df1"][args.column].isin(only1)].copy()
        only1_df["__source__"] = "only_in_1"
        only2_df = result["df2"][result["df2"][args.column].isin(only2)].copy()
        only2_df["__source__"] = "only_in_2"

        # Align columns before concat so missing fields become NaN cleanly.
        all_cols = sorted(set(only1_df.columns) | set(only2_df.columns))
        diff_df = pd.concat(
            [only1_df.reindex(columns=all_cols), only2_df.reindex(columns=all_cols)],
            ignore_index=True,
        )

        out_path = args.csv1.with_name(args.csv1.stem + "_diff.csv")
        diff_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(diff_df)} diff rows to: {out_path}")


if __name__ == "__main__":
    main()
