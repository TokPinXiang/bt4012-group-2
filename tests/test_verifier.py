"""Case view aliases and the verification gate (no LLM needed)."""

import pytest
from pydantic import ValidationError

from conftest import A, B, C, FLAG, FUT, T2, T3
from src.agent.case_view import CaseView, build_case_view
from src.agent.report import build_alert_report
from src.agent.schemas import LLMClaim, LLMNarrative
from src.agent.verifier import verify_claim, verify_narrative


@pytest.fixture
def view(store, flag_alert):
    return build_case_view(build_alert_report(store, flag_alert))


def claim(cited, *checks):
    return LLMClaim.model_validate({"text": "x", "cited_tx": cited, "checks": list(checks)})


def test_case_view_aliases(view):
    assert view.real_tx("T0") == FLAG
    assert view.acct["A1"] == A and view.acct["A2"] == B
    assert FUT not in view.tx_alias and "is_laundering" not in view.text


def test_true_cycle_is_verified(view):
    t = view.tx_alias
    narrative = LLMNarrative.model_validate({
        "typology": "CYCLE", "suspicious": True, "confidence": "high", "summary": "round trip",
        "claims": [{"text": "money went round", "cited_tx": [t[T2], t[T3], "T0"],
                    "checks": [{"kind": "cycle", "accounts": ["A2", view.acct_alias[C], "A1", "A2"]},
                               {"kind": "count", "n": 3}, {"kind": "within_hours", "hours": 2}]}]})
    assert verify_narrative(narrative, view)["status"] == "verified"


@pytest.mark.parametrize("bad", [
    claim(["T0", "T99"], {"kind": "count", "n": 2}),                                         # made-up id
    claim(["T0"], {"kind": "total", "amount": 5000, "currency": "US Dollar"}),                # wrong amount
    claim(["T0"], {"kind": "total", "amount": 980, "currency": "Euro"}),                      # wrong currency
    claim(["T0"], {"kind": "fan_out", "account": "A2", "recipients": 1}),                     # wrong direction
    claim(["T0"], {"kind": "involves", "account": "A99"}),                                    # unknown account
    claim(["T0"], {"kind": "chain", "accounts": ["A2", "A1"]}),                               # wrong route
])
def test_wrong_claims_are_rejected(view, bad):
    assert not verify_claim(bad, view)["passed"]


def test_right_total_passes(view):
    assert verify_claim(claim(["T0"], {"kind": "total", "amount": 980, "currency": "US Dollar"}), view)["passed"]


def test_typology_needs_matching_claim(view):
    narrative = LLMNarrative.model_validate({
        "typology": "CYCLE", "suspicious": True, "confidence": "low", "summary": "s",
        "claims": [{"text": "x", "cited_tx": ["T0"], "checks": [{"kind": "count", "n": 1}]}]})
    result = verify_narrative(narrative, view)
    assert result["status"] == "failed" and not result["typology_supported"]


def test_schema_rejects_malformed_answers():
    with pytest.raises(ValidationError):
        LLMClaim.model_validate({"text": "x", "cited_tx": ["HI-Small:1"], "checks": [{"kind": "count", "n": 1}]})
    with pytest.raises(ValidationError):
        LLMClaim.model_validate({"text": "x", "cited_tx": ["T0"], "checks": [{"kind": "teleport"}]})
    with pytest.raises(ValidationError):
        LLMNarrative.model_validate({"typology": "SMURFING", "suspicious": True, "confidence": "high",
                                     "summary": "s", "claims": []})


def test_restore_rebuilds_same_view(store, flag_alert, view):
    report = {**build_alert_report(store, flag_alert), "case_aliases": view.aliases(), "tool_transactions": []}
    restored = CaseView.restore(report)
    assert restored.acct == view.acct and restored.aliases() == view.aliases()
