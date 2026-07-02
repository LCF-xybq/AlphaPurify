#!/usr/bin/env python3
"""
Fetch constituent stocks for a given index and write to results/stocks_from_index_{code}.csv.

All baostock calls go through BaostockClient for unified rate limiting (avoids blacklisting).

Supported indices:
  - sh.000016  SSE 50
  - sh.000300  CSI 300
  - sh.000905  CSI 500
  - all        Union of the three indices above
"""
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.baostock_client import BaostockClient

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Index code -> baostock query method name on BaostockClient
INDEX_METHODS = {
    "sh.000016": "query_sz50_stocks",
    "sh.000300": "query_hs300_stocks",
    "sh.000905": "query_zz500_stocks",
}


def fetch_index_stocks(client: BaostockClient, index_code: str) -> pd.DataFrame:
    if index_code not in INDEX_METHODS:
        supported = ", ".join(INDEX_METHODS.keys())
        raise ValueError(f"Unsupported index: {index_code}. Supported: {supported}")

    method = getattr(client, INDEX_METHODS[index_code])
    rs = method()
    if rs is None:
        return pd.DataFrame()
    data_list = []
    while rs.error_code == "0" and rs.next():
        data_list.append(rs.get_row_data())

    df = pd.DataFrame(data_list, columns=rs.fields)
    df.rename(columns={"code_name": "stock_name"}, inplace=True)
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_index_stocks.py <index_code|all>")
        print("Example: python get_index_stocks.py sh.000300")
        print("        python get_index_stocks.py all")
        sys.exit(1)

    index_code = sys.argv[1]
    client = BaostockClient()

    if index_code == "all":
        dfs = []
        for code in INDEX_METHODS:
            df = fetch_index_stocks(client, code)
            df["source_index"] = code
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        # Deduplicate by stock code, keep first occurrence (priority order)
        df = df.drop_duplicates(subset=["code"], keep="first")
        df = df.sort_values("code").reset_index(drop=True)
    else:
        df = fetch_index_stocks(client, index_code)

    output_path = RESULTS_DIR / f"stocks_from_index_{index_code.replace('.', '_')}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} stocks to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
