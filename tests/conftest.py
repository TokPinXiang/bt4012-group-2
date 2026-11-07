"""Shared tiny dataset for the agent tests.

Story: B -> C -> A, then the flagged A -> B closes a cycle. After the alert, B -> D
happens, and the tools must never show it.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.store import TransactionStore, build_store  # noqa: E402

A, B, C, D, E = "1:A", "2:B", "3:C", "4:D", "5:E"
T1, T2, T3 = "HI-Small:00000001", "HI-Small:00000002", "HI-Small:00000003"
FLAG, SAME, FUT = "HI-Small:00000010", "HI-Small:00000011", "HI-Small:00000020"

COLUMNS = ["tx_id", "timestamp", "from_bank", "to_bank", "src_account_id", "dst_account_id",
           "amount_paid", "payment_currency", "amount_received", "receiving_currency",
           "payment_format", "is_laundering", "src_out_count_24h"]

ROWS = {
    "train": [
        (T1, "2022-09-01 09:00:00", A, E, 500, 0),    # A also pays E
        (T2, "2022-09-01 10:00:00", B, C, 1000, 1),   # cycle leg 1
        (T3, "2022-09-01 11:00:00", C, A, 990, 1),    # cycle leg 2
    ],
    "validation": [
        (FLAG, "2022-09-01 12:00:00", A, B, 980, 1),  # closes the cycle
        (SAME, "2022-09-01 12:00:00", C, E, 10, 0),   # same minute: visible
    ],
    "test": [
        (FUT, "2022-09-01 13:00:00", B, D, 970, 1),   # after the alert
    ],
}


def write_split_csvs(directory: Path, rows: dict = ROWS) -> dict[str, Path]:
    paths = {}
    for split, items in rows.items():
        df = pd.DataFrame(
            [(t, ts, s.split(":")[0], d.split(":")[0], s, d, amt, "US Dollar", amt, "US Dollar", "Wire", lab, 0)
             for t, ts, s, d, amt, lab in items],
            columns=COLUMNS,
        )
        paths[split] = directory / f"{split}.csv"
        df.to_csv(paths[split], index=False)
    return paths


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return build_store(write_split_csvs(d), d / "store.duckdb")


@pytest.fixture(scope="session")
def store(db_path):
    with TransactionStore(db_path) as s:
        yield s


@pytest.fixture
def flag_alert():
    return {"tx_id": FLAG, "score": 0.9, "predicted_label": 1, "model_name": "m", "split": "validation", "rank": 1}
