"""Label-free transaction store that the agent's lookup tools query.

The store is a DuckDB file built once from Member 1's split CSVs. It keeps only the
columns an investigator would see. `is_laundering` and the engineered features are
never copied in, so no tool can leak the label, even by accident.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb

SPLITS = ("train", "validation", "test")

# Columns copied from the split CSVs, renamed for the store. The label is deliberately absent.
STORE_SELECT = """
    tx_id,
    CAST("timestamp" AS TIMESTAMP)      AS ts,
    '{split}'                           AS split,
    from_bank,
    to_bank,
    src_account_id                      AS src,
    dst_account_id                      AS dst,
    CAST(amount_paid AS DOUBLE)         AS amount_paid,
    payment_currency,
    CAST(amount_received AS DOUBLE)     AS amount_received,
    receiving_currency,
    payment_format
"""

# Columns the tools may return. `split` stays internal.
PUBLIC_COLUMNS = (
    "tx_id", "ts", "src", "dst", "from_bank", "to_bank", "amount_paid",
    "payment_currency", "amount_received", "receiving_currency", "payment_format",
)
FORBIDDEN_COLUMNS = {"is_laundering", "label", "pattern_id", "typology"}


def find_data_dir(root: Path) -> Path:
    """Where the four split CSVs live: data/processed/ (the data branch contract), else data/."""
    for d in (Path(root) / "data" / "processed", Path(root) / "data"):
        if (d / "train.csv").exists():
            return d
    return Path(root) / "data" / "processed"


def default_csv_paths(data_dir: Path) -> dict[str, Path]:
    return {split: Path(data_dir) / f"{split}.csv" for split in SPLITS}


def build_store(csv_paths: dict[str, Path], db_path: Path, overwrite: bool = False) -> Path:
    """Copy the investigator-visible columns of every split into one DuckDB table `tx`."""
    db_path = Path(db_path)
    if db_path.exists() and not overwrite:
        return db_path

    for path in csv_paths.values():
        if not Path(path).exists():
            raise FileNotFoundError(path)

    # Build into a temp file so an interrupted build never leaves a half-made store.
    tmp_path = db_path.with_name(db_path.name + ".building")
    for p in (tmp_path, Path(str(tmp_path) + ".wal")):
        p.unlink(missing_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    for split, path in csv_paths.items():
        csv = str(Path(path).resolve()).replace("\\", "/").replace("'", "''")
        parts.append(
            f"SELECT {STORE_SELECT.format(split=split)} "
            f"FROM read_csv('{csv}', header = true, all_varchar = true)"
        )

    con = duckdb.connect(str(tmp_path))
    try:
        # Sorting by time lets DuckDB skip whole blocks when a query has a time window.
        con.execute(f"CREATE TABLE tx AS SELECT * FROM ({' UNION ALL '.join(parts)}) ORDER BY ts, tx_id")
        n, n_unique = con.execute("SELECT count(*), count(DISTINCT tx_id) FROM tx").fetchone()
        if n != n_unique:
            raise ValueError(f"tx_id is not unique across splits ({n} rows, {n_unique} ids)")
        leaked = FORBIDDEN_COLUMNS & {c[0] for c in con.execute("DESCRIBE tx").fetchall()}
        if leaked:
            raise ValueError(f"store must not contain {leaked}")
        con.execute("CREATE UNIQUE INDEX tx_id_idx ON tx (tx_id)")
    finally:
        con.close()

    if db_path.exists():
        db_path.unlink()
    tmp_path.rename(db_path)
    return db_path


def _jsonable(value):
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat(sep=" ") if isinstance(value, dt.datetime) else value.isoformat()
    return value


class TransactionStore:
    """Read-only connection to the store. Use `for_alert` to get case-bound tools."""

    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"{db_path} not found. Build it with build_store() first.")
        self.con = duckdb.connect(str(db_path), read_only=True)

    def rows(self, sql: str, params: list | tuple = ()) -> list[dict]:
        # A fresh cursor per query keeps this safe when agent threads run tools in parallel.
        cur = self.con.cursor().execute(sql, list(params))
        names = [d[0] for d in cur.description]
        return [{k: _jsonable(v) for k, v in zip(names, row)} for row in cur.fetchall()]

    def one(self, sql: str, params: list | tuple = ()) -> dict | None:
        found = self.rows(sql, params)
        return found[0] if found else None

    def time_range(self) -> tuple[str, str]:
        r = self.one("SELECT min(ts) AS lo, max(ts) AS hi FROM tx")
        return r["lo"], r["hi"]

    def for_alert(self, tx_id: str, **kwargs):
        from .tools import CaseTools

        return CaseTools(self, tx_id, **kwargs)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
