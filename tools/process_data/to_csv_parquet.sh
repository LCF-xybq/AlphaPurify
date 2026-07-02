#!/bin/bash


python tools/process_data/to_csv_parquet.py --type index_d --save_type csv

sleep 1

python tools/process_data/to_csv_parquet.py --type index_d --save_type parquet

sleep 1

python tools/process_data/to_csv_parquet.py --type index_w --save_type csv

sleep 1

python tools/process_data/to_csv_parquet.py --type index_w --save_type parquet

sleep 1

python tools/process_data/to_csv_parquet.py --type index_m --save_type csv

sleep 1

python tools/process_data/to_csv_parquet.py --type index_m --save_type parquet

sleep 1

python tools/process_data/to_csv_parquet.py --type stock_d --save_type csv

sleep 1

python tools/process_data/to_csv_parquet.py --type stock_d --save_type parquet

sleep 1

python tools/process_data/to_csv_parquet.py --type stock_w --save_type csv

sleep 1

python tools/process_data/to_csv_parquet.py --type stock_w --save_type parquet

sleep 1

python tools/process_data/to_csv_parquet.py --type stock_m --save_type csv

sleep 1

python tools/process_data/to_csv_parquet.py --type stock_m --save_type parquet
