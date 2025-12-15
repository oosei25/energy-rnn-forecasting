from __future__ import annotations
import sqlite3
import pandas as pd


def list_columns(sqlite_path: str, table: str) -> list[str]:
    with sqlite3.connect(sqlite_path) as con:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [r[1] for r in rows]


def load_opsd_sqlite(
    sqlite_path: str,
    table: str,
    timestamp_col: str,
    columns: list[str],
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> pd.DataFrame:
    # always include timestamp column
    cols = [timestamp_col] + columns
    cols_sql = ", ".join([f'"{c}"' for c in cols])

    where = []
    params: list[str] = []
    if start_utc:
        where.append(f'"{timestamp_col}" >= ?')
        params.append(start_utc)
    if end_utc:
        where.append(f'"{timestamp_col}" < ?')
        params.append(end_utc)

    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    q = f'SELECT {cols_sql} FROM "{table}"{where_sql} ORDER BY "{timestamp_col}" ASC;'

    with sqlite3.connect(sqlite_path) as con:
        df = pd.read_sql_query(q, con, params=params, parse_dates=[timestamp_col])
    return df
