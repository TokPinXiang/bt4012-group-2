"""Investigator agent: a LangChain tool-calling loop driven by definitions/agents/investigator.md.

For one alert:
    case view -> LLM investigates (may call lookup tools, within a budget)
              -> LLM calls submit_case_report -> Pydantic schema check -> verifier
              -> if not verified: problems go back as the tool result; LLM may resubmit
              -> keep only claims that pass -> triage
"""

from __future__ import annotations

import re
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from .case_view import CaseView, build_case_view
from .llm import AgentLLM
from .schemas import LLMNarrative
from .specs import AgentSpec, ToolSpec
from .store import TransactionStore
from .toolkit import CaseToolkit
from .trace import Trace
from .verifier import feedback_for_retry, verify_narrative

ALIAS_RE = re.compile(r"\b([AT])(\d+)\b")


def _noop(**_):
    return ""  # tools are executed by the loop below, not by LangChain


def build_langchain_tools(tool_specs: dict[str, ToolSpec]) -> dict[str, StructuredTool]:
    return {name: StructuredTool.from_function(func=_noop, name=name, description=spec.full_description,
                                               args_schema=spec.args_model())
            for name, spec in tool_specs.items()}


def _to_real(text: str, view: CaseView) -> str:
    """Replace aliases in free text with real ids for the final report."""
    def sub(m):
        alias = m.group(0)
        return view.acct.get(alias, alias) if m.group(1) == "A" else (view.real_tx(alias) or alias)
    return ALIAS_RE.sub(sub, text or "")


def _schema_errors(exc: ValidationError, limit: int = 8) -> str:
    return "\n".join(f"- {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:limit])


def triage(alert: dict, status: str) -> dict:
    if not int(alert["predicted_label"]):
        return {"decision": "auto_clear", "reason": "model score below its decision threshold"}
    if status == "verified":
        return {"decision": "fast_track", "reason": "high score and every claim in the narrative was verified"}
    return {"decision": "manual_review",
            "reason": "high score but the AI narrative could not be fully verified; do not rely on it"}


def investigate(report: dict, llm: AgentLLM, agent: AgentSpec, tool_specs: dict[str, ToolSpec],
                store: TransactionStore, window_hours: float = 96, trace: Trace | None = None) -> dict:
    """Fill `narrative`, `verification`, `triage` and `llm` on one alert report (returns a new dict).

    Every step is recorded in `trace`: what the LLM asked for, why, and what came back.
    """
    trace = trace or Trace(report["alert"])
    if report.get("error"):
        decision = {"decision": "manual_review", "reason": report["error"]}
        trace.add("triage", **decision)
        return {**report, "triage": decision}

    view = build_case_view(report)
    toolkit = CaseToolkit(store.for_alert(report["alert"]["tx_id"], window_hours=window_hours), view)
    lc_tools = build_langchain_tools(tool_specs)
    research = [lc_tools[t] for t in agent.tools]
    submit = lc_tools[agent.submit_tool]

    messages = [SystemMessage(agent.prompt), HumanMessage(view.text)]
    usage, attempts = Counter(), []
    tool_calls_used, final, done = 0, None, False
    max_submissions = 1 + agent.max_retries
    trace.add("start", evidence_tx=len(view.tx), case_chars=len(view.text),
              tool_budget=agent.max_tool_calls, max_submissions=max_submissions)

    for _ in range(agent.max_tool_calls + max_submissions + 3):  # hard stop
        out_of_budget = tool_calls_used >= agent.max_tool_calls
        ai = llm.invoke(messages, [submit] if out_of_budget else research + [submit],
                        tool_choice=agent.submit_tool if out_of_budget else None)
        messages.append(ai)
        usage.update({k: v for k, v in (ai.usage_metadata or {}).items() if isinstance(v, int)})
        trace.add("llm_reply", text=str(ai.content or "").strip(), tool_calls=[tc["name"] for tc in ai.tool_calls],
                  forced_submit=out_of_budget, tokens=(ai.usage_metadata or {}).get("total_tokens"))

        if not ai.tool_calls:
            messages.append(HumanMessage(f"Call {agent.submit_tool} with your final report now."))
            continue

        for tc in ai.tool_calls:
            name, args, call_id = tc["name"], tc.get("args") or {}, tc["id"]
            if name == agent.submit_tool:
                try:
                    narrative = LLMNarrative.model_validate(args)
                except ValidationError as exc:
                    attempts.append({"status": "invalid", "detail": _schema_errors(exc, 3)})
                    reply = "Rejected: the report does not match the schema.\n" + _schema_errors(exc) + \
                            f"\nFix it and call {agent.submit_tool} again."
                    trace.add("submit", attempt=len(attempts), status="invalid", feedback=_schema_errors(exc))
                else:
                    check = verify_narrative(narrative, view)
                    attempts.append({"status": check["status"], "claims_passed": check["claims_passed"],
                                     "claims_total": check["claims_total"]})
                    final = (narrative, check)
                    if check["status"] == "verified":
                        reply = "Accepted: every claim was verified."
                    else:
                        reply = ("The checker found problems:\n" + feedback_for_retry(check) +
                                 f"\nFix or remove the failing claims and call {agent.submit_tool} again.")
                    trace.add("submit", attempt=len(attempts), status=check["status"], typology=narrative.typology,
                              claims_passed=check["claims_passed"], claims_total=check["claims_total"],
                              summary=narrative.summary,
                              feedback=None if check["status"] == "verified" else feedback_for_retry(check))
                if (final and final[1]["status"] == "verified") or len(attempts) >= max_submissions:
                    done = True
            elif name in agent.tools and not out_of_budget and tool_calls_used < agent.max_tool_calls:
                tool_calls_used += 1
                reply = toolkit.run(tool_specs[name], args)
                call = toolkit.log[-1]
                trace.add("tool_call", tool=name, args=call["args"], reason=call["reason"], rows=call["rows"],
                          new_transactions=call["new_transactions"], error=call["error"],
                          preview=reply.splitlines()[:4])
            else:
                reply = f"Tool not available now. Call {agent.submit_tool} with your final report."
                trace.add("tool_refused", tool=name,
                          why="lookup budget used up" if name in agent.tools else "not one of this agent's tools")
            messages.append(ToolMessage(reply, tool_call_id=call_id))
        if done:
            break

    llm_run = {"agent": agent.name, "model": llm.model_name, "usage": dict(usage), "tool_calls": toolkit.log}
    base = {**report, "tool_transactions": toolkit.new_transactions, "case_aliases": view.aliases(), "llm": llm_run}

    if final is None:
        verification = {"status": "failed", "typology_supported": False,
                        "typology_reason": "no valid report was submitted", "claims_total": 0, "claims_passed": 0,
                        "dropped_claims": [], "attempts": attempts or [{"status": "invalid", "detail": "no submission"}]}
        decision = triage(report["alert"], "failed")
        trace.add("triage", **decision)
        return {**base, "narrative": None, "verification": verification, "triage": decision}

    narrative, check = final
    claims, dropped = [], []
    for claim, res in zip(narrative.claims, check["claim_results"]):
        real = {"text": _to_real(claim.text, view),
                "cited_tx_ids": [view.real_tx(c) or f"{c} (not in evidence)" for c in claim.cited_tx],
                "checks": [c.model_dump() for c in claim.checks]}
        if res["passed"]:
            claims.append(real)
        else:
            dropped.append({**real, "reasons": res["reasons"]})

    out_narrative = {"typology": narrative.typology, "suspicious": narrative.suspicious,
                     "confidence": narrative.confidence, "summary": _to_real(narrative.summary, view), "claims": claims}
    verification = {"status": check["status"], "typology_supported": check["typology_supported"],
                    "typology_reason": check["typology_reason"], "claims_total": check["claims_total"],
                    "claims_passed": check["claims_passed"], "dropped_claims": dropped, "attempts": attempts}
    decision = triage(report["alert"], check["status"])
    trace.add("triage", **decision)
    return {**base, "narrative": out_narrative, "verification": verification, "triage": decision}
