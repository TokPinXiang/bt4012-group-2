"""Lookup tools and evidence building: no labels, no future, correct structure."""

import json

import pandas as pd
import pytest

from conftest import A, B, C, COLUMNS, FLAG, FUT, SAME, T1, T2, T3
from src.agent.alerts import load_alerts
from src.agent.report import assert_no_label_fields, build_alert_report
from src.agent.schemas import AlertReport
from src.agent.store import TransactionStore, build_store
from src.agent.tools import TransactionNotVisible


def test_store_has_no_label(store):
    cols = {r["column_name"] for r in store.rows("DESCRIBE tx")}
    assert "is_laundering" not in cols
    assert store.one("SELECT count(*) AS n FROM tx")["n"] == 6


def test_future_transaction_is_hidden(store):
    case = store.for_alert(FLAG)
    assert case.get_transaction(T3)["tx_id"] == T3
    assert case.get_transaction(SAME)["tx_id"] == SAME
    with pytest.raises(TransactionNotVisible):
        case.get_transaction(FUT)
    with pytest.raises(TransactionNotVisible):
        case.get_transaction("HI-Small:99999999")


def test_history_and_subgraph_stop_at_alert_time(store):
    case = store.for_alert(FLAG)
    hist = case.get_account_history(B)
    assert {t["tx_id"] for t in hist["recent_transactions"]} == {T2, FLAG}
    assert hist["outgoing"]["tx_count"] == 1 and hist["incoming"]["tx_count"] == 1

    sub = case.get_subgraph(hops=2)
    assert {e["tx_id"] for e in sub["edges"]} == {T1, T2, T3, FLAG, SAME}
    assert all(e["ts"] <= case.flagged["ts"] for e in sub["edges"])
    assert case.get_transfers_between(B, "4:D") == []  # B -> D happens only after the alert


def test_cycle_closed_by_flagged_transaction(store):
    cycles = store.for_alert(FLAG).find_cycles()
    assert len(cycles) == 1
    assert cycles[0]["accounts"] == [B, C, A, B]
    assert cycles[0]["tx_ids"] == [T2, T3, FLAG]


def test_cycle_not_visible_before_it_closes(store):
    assert store.for_alert(T3).find_cycles() == []  # the closing leg hasn't happened yet


def test_report_is_label_free_and_matches_contract(store, flag_alert):
    report = build_alert_report(store, flag_alert)
    assert_no_label_fields(report)
    assert "is_laundering" not in json.dumps(report)
    assert FUT not in report["evidence_tx_ids"]
    assert report["structure_signals"]["source_account"]["distinct_recipients"] == 2
    AlertReport.model_validate(report)


def test_missing_alert_becomes_error(store, flag_alert):
    assert "error" in build_alert_report(store, {**flag_alert, "tx_id": "HI-Small:99999999"})


def test_self_transfer_counts_both_ways(tmp_path):
    rows = [("HI-Small:00000001", "2022-09-01 09:00:00", "1", "1", "1:A", "1:A", 100, "US Dollar", 100, "US Dollar", "Wire", 0, 0),
            ("HI-Small:00000002", "2022-09-01 10:00:00", "1", "2", "1:A", "2:B", 50, "US Dollar", 50, "US Dollar", "Wire", 0, 0)]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp_path / "train.csv", index=False)
    with TransactionStore(build_store({"train": tmp_path / "train.csv"}, tmp_path / "s.duckdb")) as s:
        case = s.for_alert("HI-Small:00000002")
        hist = case.get_account_history("1:A")
        sig = case.structure_signals()["source_account"]
        assert hist["outgoing"]["tx_count"] == sig["outgoing_tx"] == 2
        assert hist["incoming"]["tx_count"] == sig["incoming_tx"] == 1
        assert hist["incoming"]["amount_by_currency"] == {"US Dollar": 100.0}
        assert hist["window_tx_count"] == 2 and not hist["recent_transactions_truncated"]


def test_load_alerts_ranks_by_score(tmp_path):
    p = tmp_path / "pred.csv"
    pd.DataFrame({"tx_id": ["a", "b", "c"], "split": ["validation", "validation", "test"],
                  "score": [0.2, 0.9, 0.95], "predicted_label": [0, 1, 1], "model_name": "m"}).to_csv(p, index=False)
    assert [a["tx_id"] for a in load_alerts(p, top_k=2)] == ["c", "b"]
    assert [a["tx_id"] for a in load_alerts(p, top_k=5, split="validation")] == ["b", "a"]
    pd.DataFrame({"tx_id": ["a"], "score": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_alerts(p, top_k=1)
