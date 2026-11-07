"""The investigator loop end to end with a scripted fake LLM, then the output contract."""

import json

from langchain_core.messages import AIMessage

from conftest import C, FLAG, T2, T3
from src.agent.case_view import build_case_view
from src.agent.investigator import investigate
from src.agent.llm import AgentLLM
from src.agent.report import build_alert_report, write_alert_reports
from src.agent.schemas import AlertReport
from src.agent.specs import load_agent_spec
from src.agent.toolkit import CaseToolkit
from src.agent.trace import Trace, read_traces, render_alert, write_traces
from src.agent.validation import validate_case_reports


class FakeChat:
    """Stands in for ChatOpenAI: returns scripted replies and records what it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.tool_names, self.tool_choices = [], []

    def bind_tools(self, tools, tool_choice=None):
        self.tool_names.append([t.name for t in tools])
        self.tool_choices.append(tool_choice)
        return self

    def invoke(self, messages):
        return self.replies.pop(0)


def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call{i}"}])


def run_agent(store, alert, replies, trace=None):
    agent, tools = load_agent_spec("investigator", toolkit_cls=CaseToolkit)
    fake = FakeChat(replies)
    out = investigate(build_alert_report(store, alert), AgentLLM(fake, "fake"), agent, tools, store, trace=trace)
    return out, fake


def cycle_report(c_alias, tx):
    return {"typology": "CYCLE", "suspicious": True, "confidence": "high",
            "summary": "T0 closes a loop A2 -> " + c_alias + " -> A1 -> A2.",
            "claims": [{"text": "Money went round", "cited_tx": [tx[T2], tx[T3], "T0"],
                        "checks": [{"kind": "cycle", "accounts": ["A2", c_alias, "A1", "A2"]}]}]}


def test_tool_call_then_retry_then_verified(store, flag_alert, tmp_path):
    # Aliases are deterministic: A1=A, A2=B, then accounts in the order the case shows them.
    v = build_case_view(build_alert_report(store, flag_alert))
    c_alias, tx = v.acct_alias[C], v.tx_alias

    wrong = cycle_report(c_alias, tx)
    wrong["claims"].append({"text": "Big total", "cited_tx": ["T0"],
                            "checks": [{"kind": "total", "amount": 1_000_000, "currency": "US Dollar"}]})
    out, fake = run_agent(store, flag_alert, [
        call("get_account_history", {"account": c_alias, "reason": "Check who funds the middle account"}, 1),
        call("submit_case_report", wrong, 2),
        call("submit_case_report", cycle_report(c_alias, tx), 3),
    ])

    assert out["llm"]["tool_calls"][0]["tool"] == "get_account_history"
    assert [a["status"] for a in out["verification"]["attempts"]] == ["partial", "verified"]
    assert out["verification"]["status"] == "verified"
    assert out["triage"]["decision"] == "fast_track"
    assert out["narrative"]["claims"][0]["cited_tx_ids"][-1] == FLAG
    assert "1:A" in out["narrative"]["summary"]  # aliases turned back into real ids
    AlertReport.model_validate(out)

    # The written file passes the full output contract, including re-verification.
    path = write_alert_reports([out], tmp_path / "case_reports_fake.json", tmp_path / "missing.csv")
    rep = validate_case_reports(path)
    assert rep.ok, rep.render()


def test_made_up_ids_end_in_manual_review(store, flag_alert):
    fake_claim = {"typology": "FAN-OUT", "suspicious": True, "confidence": "high", "summary": "s",
                  "claims": [{"text": "A1 paid 3 accounts", "cited_tx": ["T0", "T500", "T501"],
                              "checks": [{"kind": "fan_out", "account": "A1", "recipients": 3}]}]}
    out, _ = run_agent(store, flag_alert, [call("submit_case_report", fake_claim, 1),
                                           call("submit_case_report", fake_claim, 2)])
    assert out["verification"]["status"] == "failed"
    assert out["triage"]["decision"] == "manual_review"
    assert out["narrative"]["claims"] == []
    assert "T500 (not in evidence)" in out["verification"]["dropped_claims"][0]["cited_tx_ids"]


def test_schema_error_is_sent_back(store, flag_alert):
    # A malformed answer first (unknown typology, no claims), then a valid NONE report.
    good = {"typology": "NONE", "suspicious": False, "confidence": "low", "summary": "Routine.",
            "claims": [{"text": "T0 is one payment", "cited_tx": ["T0"], "checks": [{"kind": "count", "n": 1}]}]}
    out, _ = run_agent(store, flag_alert, [call("submit_case_report", {"typology": "SMURFING"}, 1),
                                           call("submit_case_report", good, 2)])
    assert [a["status"] for a in out["verification"]["attempts"]] == ["invalid", "verified"]
    assert out["triage"]["decision"] == "fast_track"


def test_tool_budget_forces_submit(store, flag_alert):
    agent, _ = load_agent_spec("investigator")
    lookups = [call("get_transfers_between", {"from_account": "A1", "to_account": "A2", "reason": "pair history"}, i)
               for i in range(agent.max_tool_calls)]
    good = {"typology": "NONE", "suspicious": False, "confidence": "low", "summary": "Routine.",
            "claims": [{"text": "one payment", "cited_tx": ["T0"], "checks": [{"kind": "count", "n": 1}]}]}
    out, fake = run_agent(store, flag_alert, lookups + [call("submit_case_report", good, 99)])
    assert len(out["llm"]["tool_calls"]) == agent.max_tool_calls
    assert fake.tool_choices[-1] == "submit_case_report"   # forced once the budget ran out
    assert fake.tool_names[-1] == ["submit_case_report"]
    assert json.dumps(out)  # serialisable


def test_trace_records_each_step_and_reason(store, flag_alert, tmp_path):
    v = build_case_view(build_alert_report(store, flag_alert))
    c_alias, tx = v.acct_alias[C], v.tx_alias
    trace = Trace(flag_alert)
    out, _ = run_agent(store, flag_alert, [
        call("get_account_history", {"account": c_alias, "reason": "Check who funds the middle account"}, 1),
        call("submit_case_report", cycle_report(c_alias, tx), 2),
    ], trace=trace)

    kinds = [e["event"] for e in trace.events]
    assert kinds == ["start", "llm_reply", "tool_call", "llm_reply", "submit", "triage"]
    lookup = trace.events[2]
    assert lookup["tool"] == "get_account_history" and lookup["reason"] == "Check who funds the middle account"
    assert lookup["args"] == {"account": c_alias} and lookup["rows"] > 0 and lookup["error"] is None
    assert out["llm"]["tool_calls"][0]["reason"] == "Check who funds the middle account"
    AlertReport.model_validate(out)

    path = write_traces([trace], tmp_path / "trace.jsonl")
    events = read_traces(path)
    assert [e["step"] for e in events] == [1, 2, 3, 4, 5, 6]
    text = render_alert(events)
    assert "because \"Check who funds the middle account\"" in text and "fast_track" in text


def test_lookup_without_reason_is_rejected(store, flag_alert):
    good = {"typology": "NONE", "suspicious": False, "confidence": "low", "summary": "Routine.",
            "claims": [{"text": "one payment", "cited_tx": ["T0"], "checks": [{"kind": "count", "n": 1}]}]}
    out, _ = run_agent(store, flag_alert, [call("get_account_history", {"account": "A1"}, 1),
                                           call("submit_case_report", good, 2)])
    assert "reason" in out["llm"]["tool_calls"][0]["error"]
