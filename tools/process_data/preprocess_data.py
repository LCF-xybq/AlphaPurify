import argparse
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd


def parse_time_to_datetime(time_val):
    s = str(time_val)
    dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
    return dt.strftime("%Y-%m-%d %H:%M")


def process_file(filepath: str) -> dict:
    """Process a single file and return the result."""
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(filepath, encoding="utf-8-sig")
        elif ext == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            return {"file": filepath, "status": "skipped", "reason": "unsupported format"}

        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"])

        if "time" in df.columns:
            df["datetime"] = df["time"].apply(parse_time_to_datetime)
            df = df.drop(columns=["time"])

        if "code" in df.columns:
            df = df.rename(columns={"code": "symbol"})

        if "datetime" in df.columns:
            cols = ["datetime"] + [c for c in df.columns if c != "datetime"]
            df = df[cols]

        out_path = filepath
        if ext == ".csv":
            df.to_csv(out_path, index=False)
        elif ext == ".parquet":
            df.to_parquet(out_path, index=False)

        return {"file": filepath, "status": "success", "rows": len(df)}
    except Exception as e:
        return {"file": filepath, "status": "error", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV/Parquet files")
    parser.add_argument("--input", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    input_dir = args.input
    workers = args.workers

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    all_files = csv_files + parquet_files

    if not all_files:
        raise FileNotFoundError(f"No CSV or Parquet files found in {input_dir}")

    print(f"Found {len(all_files)} files, processing with {workers} workers...")

    success, errors = 0, 0
    batch_size = workers * 4
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch_start in range(0, len(all_files), batch_size):
            batch = all_files[batch_start:batch_start + batch_size]
            futures = {executor.submit(process_file, f): f for f in batch}
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "success":
                    success += 1
                    print(f"[OK] {result['file']} ({result['rows']} rows)")
                elif result["status"] == "error":
                    errors += 1
                    print(f"[ERR] {result['file']}: {result['reason']}")
                else:
                    print(f"[SKIP] {result['file']}: {result['reason']}")

    print(f"\nDone: {success} succeeded, {errors} failed")


if __name__ == "__main__":
    main()
