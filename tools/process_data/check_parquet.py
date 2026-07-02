import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Display parquet file contents.")
    parser.add_argument("--input", required=True, help="Path to parquet file.")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows to display.")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    print(f"=== {args.input} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nFirst {args.rows} rows:\n{df.head(args.rows)}")
    print(f"\nLast {args.rows} rows:\n{df.tail(args.rows)}")
    print(f"\nDescriptive Statistics:\n{df.describe()}")


if __name__ == "__main__":
    main()
