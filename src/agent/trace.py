"""Step-by-step trace of what the agent did on each alert, and why.

One `Trace` per alert records events in order:
    start         case size, evidence count, tool budget
    llm_reply     what the LLM said (if anything) and which tools it asked for
    tool_call     tool, arguments, the agent's stated reason, what came back
    tool_refused  a call outside the budget or to an unknown tool
    submit        a report submission and what the schema check / verifier said
    triage        the final decision

Events are kept in memory, optionally printed live through `logging`, and written by the
run script to outputs/narratives/trace_<model>.jsonl (one JSON object per line).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger("agent.trace")


def _short(text: str, n: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


class Trace:
    def __init__(self, alert: dict):
        self.tx_id = alert["tx_id"]
        self.rank = alert.get("rank")
        self.events: list[dict] = []
        self._t0 = time.monotonic()

    def add(self, event: str, **data) -> dict:
        entry = {"rank": self.rank, "tx_id": self.tx_id, "step": len(self.events) + 1,
                 "t_ms": round((time.monotonic() - self._t0) * 1000), "event": event, **data}
        self.events.append(entry)
        if log.isEnabledFor(logging.INFO):
            log.info(self._line(entry))
        return entry

    def _line(self, e: dict) -> str:
        head = f"[#{e['rank']} {e['tx_id']}] {e['step']:>2}."
        kind = e["event"]
        if kind == "start":
            return f"{head} start: {e['evidence_tx']} transactions in evidence, budget {e['tool_budget']} lookups"
        if kind == "llm_reply":
            said = f' says "{_short(e["text"], 120)}"' if e.get("text") else ""
            return f"{head} LLM{said} -> {', '.join(e['tool_calls']) or 'no tool call'}"
        if kind == "tool_call":
            outcome = e.get("error") or f"{e['rows']} rows, {e['new_transactions']} new"
            return (f"{head} {e['tool']}({', '.join(f'{k}={v}' for k, v in e['args'].items())}) "
                    f"because \"{_short(e.get('reason'), 120)}\" -> {outcome}")
        if kind == "tool_refused":
            return f"{head} refused {e['tool']}: {e['why']}"
        if kind == "submit":
            if e["status"] == "invalid":
                return f"{head} submit #{e['attempt']}: schema errors -> sent back"
            return (f"{head} submit #{e['attempt']}: {e['typology']}, "
                    f"{e['claims_passed']}/{e['claims_total']} claims pass -> {e['status']}")
        if kind == "triage":
            return f"{head} triage: {e['decision']}"
        return f"{head} {kind}"


def write_traces(traces: list[Trace], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in sorted(traces, key=lambda t: (t.rank or 0, t.tx_id)):
            for e in t.events:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    return path


def read_traces(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_alert(events: list[dict]) -> str:
    """Readable timeline for one alert's events (used by scripts/show_trace.py)."""
    if not events:
        return "no events"
    t = Trace({"tx_id": events[0]["tx_id"], "rank": events[0]["rank"]})
    lines = []
    for e in events:
        lines.append(f"{e['t_ms'] / 1000:6.1f}s  " + t._line(e).split("] ", 1)[1])
        if e["event"] == "tool_call" and e.get("preview"):
            lines += [f"           | {row}" for row in e["preview"]]
        if e["event"] == "submit" and e.get("feedback"):
            lines += [f"           ! {row}" for row in e["feedback"].splitlines()]
        if e["event"] == "submit" and e.get("summary"):
            lines.append(f"           summary: {_short(e['summary'], 300)}")
        if e["event"] == "triage":
            lines.append(f"           {e['reason']}")
    return "\n".join(lines)
