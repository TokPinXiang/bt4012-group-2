"""The agent reads the team's published files as they are: predictions Parquet + metrics JSON,
and the GNN's alerts_explained.json."""

import json

import pandas as pd
import pytest

from conftest import FLAG, FUT, SAME, T1, T2, T3
from src.agent.alerts import PredictionFormatError, load_alerts, read_predictions
from src.agent.case_view import CaseView, build_case_view
from src.agent.explanations import applies_to, load_explanations
from src.agent.report import build_alert_report, write_alert_reports
from src.agent.schemas import AlertReport
from src.agent.validation import validate_case_reports, validate_predictions


def edge(tx_id):
    return int(tx_id.split(":")[1])


def team_file(tmp_path, threshold=0.5, with_metrics=True):
    """What stage 03/04 publish: edge_id (= source_row), split val/test, score."""
    pred = pd.DataFrame({"edge_id": [edge(FLAG), edge(SAME), edge(FUT)],
                         "split": ["val", "val", "test"], "score": [0.9, 0.2, 0.7]})
    path = tmp_path / "predictions_gnn.parquet"
    pred.to_parquet(path, index=False)
    if with_metrics:
        (tmp_path / "predictions_gnn_metrics.json").write_text(json.dumps({"val": {"threshold": threshold}}))
    return path


def test_team_format_is_converted(tmp_path):
    df, info = read_predictions(team_file(tmp_path, threshold=0.5))
    assert info["format"] == "team" and info["threshold"] == 0.5
    assert df["tx_id"].tolist() == [FLAG, SAME, FUT]                 # 10 -> HI-Small:00000010
    assert df["split"].tolist() == ["validation", "validation", "test"]
    assert df["predicted_label"].tolist() == [1, 0, 1]               # score >= threshold
    assert set(df["model_name"]) == {"gnn"}                          # from the file name


def test_threshold_override_and_missing_metrics(tmp_path):
    path = team_file(tmp_path, with_metrics=False)
    with pytest.raises(PredictionFormatError, match="metrics.json"):
        read_predictions(path)
    df, info = read_predictions(path, threshold=0.8)
    assert info["threshold_source"] == "argument" and df["predicted_label"].tolist() == [1, 0, 0]


def test_load_alerts_accepts_val_or_validation(tmp_path):
    path = team_file(tmp_path)
    assert [a["tx_id"] for a in load_alerts(path, 5, "val")] == [FLAG, SAME]
    assert [a["tx_id"] for a in load_alerts(path, 5, "validation")] == [FLAG, SAME]


def test_validator_accepts_team_format(tmp_path, db_path):
    rep = validate_predictions(team_file(tmp_path), db_path)
    assert rep.ok, rep.render()
    assert rep.stats["format"].startswith("team")


def test_bad_edge_id_is_reported(tmp_path, db_path):
    path = tmp_path / "predictions_gbt.parquet"
    pd.DataFrame({"edge_id": [1.5], "split": ["val"], "score": [0.3]}).to_parquet(path, index=False)
    (tmp_path / "predictions_gbt_metrics.json").write_text(json.dumps({"val": {"threshold": 0.5}}))
    rep = validate_predictions(path, db_path)
    assert not rep.ok and rep.issues[0].check == "format"


# ---------------------------------------------------------------- GNN explanations

def explanation_entry():
    """Shaped like stage 04's alerts_explained.json, for the flagged payment."""
    ctx = lambda tx, hop, share: {"edge_id": edge(tx), "hop": hop, "importance_share": share,
                                  "from_account": "x", "to_account": "y", "payment_format": "Wire",
                                  "time": "2022-09-01 00:00:00"}
    return {"edge_id": edge(FLAG), "score": 0.91, "above_threshold": True,
            "from_account": "1:A", "to_account": "2:B", "payment_format": "Wire", "time": "2022-09-01 12:00:00",
            "share_from_own_transaction": 0.4, "share_from_1hop": 0.45, "share_from_2hop": 0.15,
            "top_own_features": [{"feature": "src_out_count_24h", "value": 2.0, "pushes_towards": "laundering",
                                  "attribution": 0.8},
                                 {"feature": "payment_format = Wire", "value": None, "pushes_towards": "normal",
                                  "attribution": -0.1}],
            "top_context_transactions": [ctx(T3, 1, 0.3), ctx(T1, 1, 0.1), ctx(FUT, 1, 0.2),
                                         {**ctx(T2, 2, 0.05), "edge_id": 99999}]}


def test_explanations_file_is_keyed_by_tx_id(tmp_path):
    path = tmp_path / "alerts_explained.json"
    path.write_text(json.dumps([explanation_entry()]))
    assert set(load_explanations(path)) == {FLAG}
    assert load_explanations(tmp_path / "missing.json") == {}
    assert applies_to({"model_name": "gnn"}) and not applies_to({"model_name": "gbt"})


def test_explanation_becomes_checked_evidence(store, flag_alert, tmp_path):
    report = build_alert_report(store, {**flag_alert, "model_name": "gnn"}, explanation=explanation_entry())
    ex = report["model_explanation"]
    # The future payment and the unknown id are dropped; the rest are fetched from the store.
    assert [t["tx_id"] for t in ex["context_transactions"]] == [T3, T1]
    assert ex["context_not_visible"] == 2
    assert {T3, T1} <= set(report["evidence_tx_ids"]) and FUT not in report["evidence_tx_ids"]
    AlertReport.model_validate(report)

    view = build_case_view(report)
    assert "MODEL EXPLANATION" in view.text and "src_out_count_24h = 2 (towards laundering)" in view.text
    assert "payment_format = Wire (towards normal)" in view.text
    assert view.tx_alias[T3] in view.text                              # context payments are citable
    assert CaseView.restore({**report, "case_aliases": view.aliases()}).aliases() == view.aliases()

    path = write_alert_reports([report], tmp_path / "case_reports_gnn.json", tmp_path / "missing.parquet")
    assert validate_case_reports(path).ok


def test_sample_selects_only_flagged_alerts_with_true_ranks(tmp_path):
    path = tmp_path / "predictions_gnn.parquet"
    edges = [edge(t) for t in (T1, T2, T3, FLAG, SAME)]
    pd.DataFrame({"edge_id": edges, "split": "val", "score": [0.95, 0.9, 0.8, 0.7, 0.1]}).to_parquet(path, index=False)
    (tmp_path / "predictions_gnn_metrics.json").write_text(json.dumps({"val": {"threshold": 0.75}}))
    picked = load_alerts(path, 2, "val", select="sample", seed=1)
    assert len(picked) == 2 and all(a["predicted_label"] == 1 for a in picked)   # never the 0.7 / 0.1 rows
    assert {a["rank"] for a in picked} <= {1, 2, 3}                              # rank in the full queue
    assert picked == load_alerts(path, 2, "val", select="sample", seed=1)        # reproducible
    assert len(load_alerts(path, 10, "val", select="sample")) == 3               # capped at the flagged count


def test_explanation_guidance_in_case_text(store, flag_alert):
    entry = explanation_entry()
    entry["share_from_1hop"], entry["share_from_own_transaction"] = 0.6, 0.3
    entry["top_own_features"].append({"feature": "src_currency_amount_mean", "value": 0.0,
                                      "pushes_towards": "laundering", "attribution": 0.2})
    text = build_case_view(build_alert_report(store, {**flag_alert, "model_name": "gnn"}, explanation=entry)).text
    assert "start with the neighbouring payments" in text
    assert "src_currency_amount_mean = 0 (0 = no earlier history of this kind)" in text
