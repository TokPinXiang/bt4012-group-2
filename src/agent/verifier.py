"""Verification gate: plain Python that checks every claim the LLM makes.

The LLM's answer is first parsed into `LLMNarrative` (schemas.py), so malformed claims are
rejected before this runs. Then a claim passes only if
- every cited id is a transaction in the case evidence (so made-up ids fail), and
- every check holds against the cited transactions' real data.

The named typology must also be backed by a passing claim of the matching shape
(e.g. CYCLE needs a passing `cycle` check).
"""

from __future__ import annotations

import datetime as dt

from .case_view import CaseView
from .schemas import (AmountRangeCheck, ChainCheck, CountCheck, CycleCheck, FanInCheck, FanOutCheck, InvolvesCheck,
                      LLMClaim, LLMNarrative, TotalCheck, WithinHoursCheck)

# Which passing check kinds back each typology. Each inner set is one acceptable way.
TYPOLOGY_EVIDENCE = {
    "FAN-OUT": [{"fan_out"}],
    "FAN-IN": [{"fan_in"}],
    "CYCLE": [{"cycle"}],
    "SCATTER-GATHER": [{"fan_out", "fan_in"}],
    "GATHER-SCATTER": [{"fan_in", "fan_out"}],
    "STACK": [{"chain"}, {"fan_out", "fan_in"}],
    "BIPARTITE": [{"fan_out"}, {"fan_in"}],
    "RANDOM": [{"chain"}],
    "NONE": [set()],
}

REL_TOL = 0.01  # amounts may be rounded by up to 1%


def _ts(e: dict) -> dt.datetime:
    return dt.datetime.fromisoformat(e["ts"])


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01, REL_TOL * max(abs(a), abs(b)))


def _follow_route(route: list[str], txs: list[dict]) -> bool:
    """True if txs contain one transfer per leg of the route, in non-decreasing time order."""
    last = None
    for a, b in zip(route, route[1:]):
        legs = sorted((t for t in txs if t["src"] == a and t["dst"] == b and (last is None or _ts(t) >= last)),
                      key=_ts)
        if not legs:
            return False
        last = _ts(legs[0])
    return True


def check_one(check, txs: list[dict], view: CaseView) -> str | None:
    """Return None if the check holds, else a short reason it failed."""
    def real(alias: str) -> str:
        if alias not in view.acct:
            raise LookupError(f"unknown account {alias}")
        return view.acct[alias]

    try:
        if isinstance(check, FanOutCheck):
            a = real(check.account)
            if any(t["src"] != a for t in txs):
                return f"not every cited transaction is sent by {check.account}"
            n = len({t["dst"] for t in txs})
            return None if n == check.recipients else f"cited transactions reach {n} recipients, not {check.recipients}"
        if isinstance(check, FanInCheck):
            a = real(check.account)
            if any(t["dst"] != a for t in txs):
                return f"not every cited transaction is received by {check.account}"
            n = len({t["src"] for t in txs})
            return None if n == check.senders else f"cited transactions come from {n} senders, not {check.senders}"
        if isinstance(check, (CycleCheck, ChainCheck)):
            route = [real(a) for a in check.accounts]
            if isinstance(check, CycleCheck) and route[0] != route[-1]:
                return "a cycle must end at the account it starts from"
            return None if _follow_route(route, txs) else \
                f"cited transactions do not follow {' -> '.join(check.accounts)} in time order"
        if isinstance(check, InvolvesCheck):
            a = real(check.account)
            return None if all(a in (t["src"], t["dst"]) for t in txs) else \
                f"not every cited transaction involves {check.account}"
        if isinstance(check, CountCheck):
            return None if len(txs) == check.n else f"{len(txs)} transactions cited, not {check.n}"
        if isinstance(check, TotalCheck):
            amt_key, cur_key = (("amount_paid", "payment_currency") if check.side == "paid"
                                else ("amount_received", "receiving_currency"))
            if any(t[cur_key] != check.currency for t in txs):
                return f"not every cited transaction is in {check.currency}"
            total = sum(t[amt_key] for t in txs)
            return None if _close(total, check.amount) else \
                f"cited total is {total:,.2f} {check.currency}, not {check.amount:,.2f}"
        if isinstance(check, AmountRangeCheck):
            bad = [t for t in txs if not (check.min * (1 - REL_TOL) <= t["amount_paid"] <= check.max * (1 + REL_TOL))]
            return None if not bad else f"{len(bad)} cited amounts fall outside {check.min:,.2f}-{check.max:,.2f}"
        if isinstance(check, WithinHoursCheck):
            times = [_ts(t) for t in txs]
            span = (max(times) - min(times)).total_seconds() / 3600
            return None if span <= check.hours + 1e-9 else f"cited transactions span {span:.1f}h, more than {check.hours}h"
        return f"unsupported check {type(check).__name__}"
    except LookupError as exc:
        return str(exc)


def verify_claim(claim: LLMClaim, view: CaseView) -> dict:
    reasons = []
    unknown = [c for c in claim.cited_tx if c not in view.tx]
    if unknown:
        reasons.append(f"cites transactions not in the evidence: {unknown}")
    else:
        txs = [view.tx[c] for c in claim.cited_tx]
        reasons += [why for chk in claim.checks if (why := check_one(chk, txs, view))]
    return {"passed": not reasons, "reasons": reasons, "kinds": sorted({c.kind for c in claim.checks})}


def verify_narrative(narrative: LLMNarrative, view: CaseView) -> dict:
    """Check every claim and the typology. Returns per-claim results and an overall status."""
    results = [verify_claim(c, view) for c in narrative.claims]
    passed_kinds = set().union(*[set(r["kinds"]) for r in results if r["passed"]])
    typology_ok = any(need <= passed_kinds for need in TYPOLOGY_EVIDENCE[narrative.typology])
    typology_reason = None if typology_ok else (
        f"{narrative.typology} needs a passing claim with check(s) "
        + " or ".join("+".join(sorted(n)) for n in TYPOLOGY_EVIDENCE[narrative.typology])
    )
    n_pass = sum(r["passed"] for r in results)
    if n_pass == len(results) and typology_ok:
        status = "verified"
    elif n_pass and typology_ok:
        status = "partial"
    else:
        status = "failed"
    return {
        "status": status,
        "typology_supported": typology_ok,
        "typology_reason": typology_reason,
        "claims_total": len(results),
        "claims_passed": n_pass,
        "claim_results": results,
    }


def feedback_for_retry(check: dict) -> str:
    """Plain-language list of problems to send back to the LLM."""
    lines = [f"- claim {i}: " + "; ".join(r["reasons"])
             for i, r in enumerate(check["claim_results"], 1) if not r["passed"]]
    if not check["typology_supported"]:
        lines.append(f"- typology: {check['typology_reason']}")
    return "\n".join(lines)
