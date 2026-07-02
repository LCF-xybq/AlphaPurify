#!/usr/bin/env python3
"""
Delete one or more stock codes from the DuckDB database.

Two ways to specify codes:
  --code        One or more stock codes on the command line.
  --lsts <yaml> A YAML file of stock codes (one per line, or a YAML list).

Table selection:
  --tables      Comma-separated list of tables to target.
  Required when only --code is given. When --lsts is given without --tables,
  the script targets every table that contains a `code` column (i.e. delete
  the stocks from all tables). --tables still overrides that default.

By default runs in dry-run mode and only previews the rows that would be
deleted. Pass --yes to commit.

Usage:
    # Preview a single ad-hoc code in one table
    python tools/process_data/delet_stock_from_db.py \
        --code sh.600000 --tables stock_kline_d

    # Delete a bulk list from ALL code-bearing tables
    python tools/process_data/delet_stock_from_db.py \
        --lsts results/delete.yaml --yes

    # Bulk list but restrict tables
    python tools/process_data/delet_stock_from_db.py \
        --lsts results/delete.yaml --tables stock_kline_d,stock_industry --yes

    # List candidate tables that contain a `code` column
    python tools/process_data/delet_stock_from_db.py --list-tables
"""
import argparse
import sys
from pathlib import Path

import duckdb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "quant.db"


def find_code_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return user tables that contain a `code` column."""
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    result = []
    for (table,) in tables:
        cols = [r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()]
        if "code" in cols:
            result.append(table)
    return result


def count_rows(conn: duckdb.DuckDBPyConnection, table: str, codes: list[str]) -> int:
    codes_sql = ", ".join(f"'{c}'" for c in codes)
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE code IN ({codes_sql})").fetchone()[0]


def load_codes_from_file(path: Path) -> list[str]:
    """Read stock codes from a YAML file.

    Accepts:
      - one bare scalar per line (YAML joins multi-line plain scalars with
        spaces, so we split any string result on whitespace)
      - a YAML list (inline or block `- item` style)
      - mixed documents in the same file
    Blank lines and explicit nulls are skipped. Order is preserved, dupes removed.
    """
    text = path.read_text(encoding="utf-8")
    docs = list(yaml.safe_load_all(text))
    codes: list[str] = []
    for doc in docs:
        if doc is None:
            continue
        if isinstance(doc, list):
            items = doc
        elif isinstance(doc, (str, int)):
            items = [doc]
        else:
            raise ValueError(f"Unsupported YAML element in {path}: {doc!r}")
        for item in items:
            if item is None:
                continue
            if not isinstance(item, (str, int)):
                raise ValueError(f"Unsupported list element in {path}: {item!r}")
            # Plain YAML scalars can span multiple lines and get joined with
            # spaces; split on whitespace so "sh.600355 sh.600421" → two codes.
            for token in str(item).split():
                token = token.strip()
                if token:
                    codes.append(token)

    seen = set()
    out = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def main():
    parser = argparse.ArgumentParser(description="Delete stock(s) from the DuckDB database.")
    parser.add_argument("--code", nargs="+",
                        help="One or more stock codes (e.g., sh.600000 sh.600001)")
    parser.add_argument("--lsts", type=Path,
                        help="YAML file of stock codes (one per line or a YAML list). "
                             "When used without --tables, deletes from ALL code-bearing tables.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help=f"DuckDB database path (default: {DEFAULT_DB})")
    parser.add_argument("--tables", type=str,
                        help="Comma-separated list of tables to target. "
                             "Required when only --code is given; optional with --lsts "
                             "(defaults to every code-bearing table).")
    parser.add_argument("--yes", action="store_true",
                        help="Actually commit the deletion. Without this flag the script "
                             "runs in dry-run mode and only previews counts.")
    parser.add_argument("--list-tables", action="store_true",
                        help="Print all tables that contain a `code` column and exit.")
    args = parser.parse_args()

    if not args.db.is_file():
        raise FileNotFoundError(f"Database not found: {args.db}")

    with duckdb.connect(str(args.db), read_only=True) as ro:
        all_tables = find_code_tables(ro)

        if args.list_tables:
            print(f"Tables containing a `code` column in {args.db}:")
            for t in all_tables:
                print(f"  {t}")
            return

    if not args.code and not args.lsts:
        parser.error("Provide --code and/or --lsts (or pass --list-tables).")

    codes: list[str] = []
    if args.code:
        codes.extend(args.code)
    if args.lsts:
        if not args.lsts.is_file():
            raise FileNotFoundError(f"Code list file not found: {args.lsts}")
        loaded = load_codes_from_file(args.lsts)
        print(f"Loaded {len(loaded)} codes from {args.lsts}")
        codes.extend(loaded)

    if not codes:
        raise ValueError("No codes to delete.")

    # Determine target tables:
    #   --tables given       → use it (validated against all_tables)
    #   --lsts given, no --tables → all code-bearing tables
    #   --code only           → --tables required (error here if missing)
    if args.tables:
        target_tables = [t.strip() for t in args.tables.split(",")]
    elif args.lsts:
        target_tables = all_tables
    else:
        parser.error("--tables is required when using --code alone. "
                     f"Available code-bearing tables: {all_tables}")

    unknown = [t for t in target_tables if t not in all_tables]
    if unknown:
        raise ValueError(
            f"Unknown or code-less table(s): {unknown}. "
            f"Available: {all_tables}"
        )

    # Validate codes early — reject anything containing a quote to avoid SQL injection.
    for code in codes:
        if "'" in code or ";" in code or "--" in code:
            raise ValueError(f"Suspicious code string rejected: {code!r}")

    # De-dup while preserving order.
    seen = set()
    codes = [c for c in codes if not (c in seen or seen.add(c))]

    print(f"Database: {args.db}")
    print(f"Codes:    {len(codes)} total")
    if len(codes) <= 10:
        for c in codes:
            print(f"          {c}")
    else:
        for c in codes[:5]:
            print(f"          {c}")
        print(f"          ... (+{len(codes) - 5} more)")
    print(f"Mode:     {'DELETE' if args.yes else 'DRY-RUN (pass --yes to commit)'}")
    print("=" * 60)

    with duckdb.connect(str(args.db), read_only=True) as ro:
        print(f"Target tables ({len(target_tables)}):")
        counts = {}
        for table in target_tables:
            n = count_rows(ro, table, codes)
            counts[table] = n
            marker = "x" if n else " "
            print(f"  [{marker}] {table:<22} {n:>10} rows")
        total = sum(counts.values())
        print("-" * 60)
        print(f"Total rows to delete: {total}")
        print("=" * 60)

    if not args.yes:
        print("\nDry-run complete. No data was modified.")
        print("Re-run with --yes to actually delete these rows.")
        return

    if total == 0:
        print("\nNothing to delete. Exiting.")
        return

    # Real deletion in a single transaction — partial failures roll back.
    codes_sql = ", ".join(f"'{c}'" for c in codes)
    with duckdb.connect(str(args.db)) as conn:
        try:
            conn.execute("BEGIN TRANSACTION")
            for table in target_tables:
                if counts[table] == 0:
                    continue
                conn.execute(f"DELETE FROM {table} WHERE code IN ({codes_sql})")
                print(f"[deleted] {table}: {counts[table]} rows")
            conn.execute("COMMIT")
            print("=" * 60)
            print(f"Done. Deleted {total} rows across {len(target_tables)} tables.")
        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"[error] Deletion rolled back: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
