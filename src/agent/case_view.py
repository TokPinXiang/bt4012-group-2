"""Turn a JSON alert report into the compact text the LLM reads.

Long ids are swapped for short aliases (accounts A1, A2, ...; transactions T0, T1, ...)
to save tokens and make citations easy to check. T0 is always the flagged transaction,
A1 its sender and A2 its receiver.

The `CaseView` is also the evidence set: the verifier only accepts citations of
transactions registered here. Tools register the extra transactions they fetch, so
those become citable too. `CaseView.restore` rebuilds the exact same view from a saved
case report, which is how the pipeline check re-verifies claims later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _money(amount: float, currency: str) -> str:
    return f"{amount:,.2f} {currency}"


def _amounts(e: dict) -> str:
    paid = _money(e["amount_paid"], e["payment_currency"])
    if e["payment_currency"] == e["receiving_currency"] and abs(e["amount_paid"] - e["amount_received"]) < 0.005:
        return paid
    # Spelled out so the LLM doesn't read a cross-currency payment as a transfer to itself.
    return f"paid {paid}, received {_money(e['amount_received'], e['receiving_currency'])}"


def _totals(side: dict) -> str:
    money = ", ".join(_money(v, k) for k, v in side["amount_by_currency"].items()) or "none"
    return f"{side['tx_count']} tx with {side['counterparties']} accounts ({money})"


@dataclass
class CaseView:
    text: str = ""
    tx: dict[str, dict] = field(default_factory=dict)         # T-alias -> transaction
    tx_alias: dict[str, str] = field(default_factory=dict)    # real tx_id -> T-alias
    acct: dict[str, str] = field(default_factory=dict)        # A-alias -> real account id
    acct_alias: dict[str, str] = field(default_factory=dict)  # real account id -> A-alias

    def account(self, real: str) -> str:
        """Alias for a real account id, creating one if needed."""
        if real not in self.acct_alias:
            alias = f"A{len(self.acct_alias) + 1}"
            self.acct_alias[real], self.acct[alias] = alias, real
        return self.acct_alias[real]

    def register(self, e: dict) -> tuple[str, bool]:
        """Alias for a transaction, adding it to the evidence if new. Returns (alias, was_new)."""
        if e["tx_id"] in self.tx_alias:
            return self.tx_alias[e["tx_id"]], False
        alias = f"T{len(self.tx)}"
        self.tx[alias], self.tx_alias[e["tx_id"]] = e, alias
        self.account(e["src"]), self.account(e["dst"])
        return alias, True

    def real_tx(self, alias: str) -> str | None:
        t = self.tx.get(alias)
        return t["tx_id"] if t else None

    def row(self, alias: str) -> str:
        e = self.tx[alias]
        route = f"{self.account(e['src'])} -> {self.account(e['dst'])}" + (" (self-transfer)" if e["src"] == e["dst"] else "")
        hop = f" | hop {e['hop']}" if "hop" in e else (" | hop 0" if alias == "T0" else "")
        return f"{alias} | {e['ts'][:16]} | {route} | {_amounts(e)} | {e['payment_format']}{hop}"

    def aliases(self) -> dict:
        return {"accounts": dict(self.acct), "transactions": {a: t["tx_id"] for a, t in self.tx.items()}}

    @classmethod
    def restore(cls, report: dict) -> "CaseView":
        """Rebuild the view saved with a case report (same aliases, including tool-fetched evidence)."""
        saved = report.get("case_aliases")
        if not saved:
            return build_case_view(report)
        by_id = {e["tx_id"]: e for e in report["subgraph"]["edges"]}
        by_id.update({e["tx_id"]: e for e in report.get("tool_transactions") or []})
        by_id.setdefault(report["flagged_transaction"]["tx_id"], report["flagged_transaction"])
        v = cls()
        for alias, real in saved["accounts"].items():
            v.acct[alias], v.acct_alias[real] = real, alias
        for alias, tx_id in saved["transactions"].items():
            v.tx[alias], v.tx_alias[tx_id] = by_id[tx_id], alias
        return v


def build_case_view(report: dict) -> CaseView:
    flagged = report["flagged_transaction"]
    edges = sorted(
        (e for e in report["subgraph"]["edges"] if e["tx_id"] != flagged["tx_id"]),
        key=lambda e: (e["ts"], e["tx_id"]),
    )
    v = CaseView()
    v.account(flagged["src"]), v.account(flagged["dst"])
    v.register(flagged)
    for e in edges:
        v.register(e)

    sig = report["structure_signals"]
    src_hist, dst_hist = report["accounts"]["source"], report["accounts"]["destination"]
    sub = report["subgraph"]

    lines = [
        "FLAGGED TRANSACTION",
        v.row("T0"),
        f"Alert time: {report['as_of'][:16]}. Nothing after this is visible. "
        f"Look-back window starts {sub['window']['start'][:16]}.",
        f"Model score: {report['alert']['score']:.3f} ({report['alert']['model_name']}).",
        "",
        "KEY ACCOUNTS (counts are complete for the window, not capped)",
    ]
    for alias, hist in (("A1", src_hist), ("A2", dst_hist)):
        bank = v.acct[alias].split(":")[0]
        lines.append(
            f"{alias} (bank {bank}, first seen {(hist['first_seen'] or '?')[:16]}): "
            f"sent {_totals(hist['outgoing'])}; received {_totals(hist['incoming'])}."
        )
    ph = sig["pair_history"]
    lines += [
        f"Earlier A1 -> A2 transfers: {ph['same_direction_before']}. "
        f"Earlier A2 -> A1 transfers: {ph['reverse_direction_before']}.",
        "",
        "CYCLES CLOSED BY T0 (money that came back round to where it started)",
    ]
    cycles = report.get("cycles_closed_by_flagged") or []
    for c in cycles:
        route = " -> ".join(v.account(a) for a in c["accounts"])
        ids = ", ".join(v.tx_alias[t] for t in c["tx_ids"] if t in v.tx_alias)
        repeat = f" (route seen {c['repeat_count']}x)" if c["repeat_count"] > 1 else ""
        lines.append(f"{route} via {ids}{repeat}")
    if not cycles:
        lines.append("none found")

    lines += [
        "",
        f"TRANSACTIONS IN WINDOW ({len(v.tx)} shown; hop = steps from A1/A2"
        + ("; busy accounts show only their most recent transactions" if sub["truncated"] else "") + ")",
        "id | time | from -> to | amount | format | hop",
    ]
    lines += [v.row(a) for a in list(v.tx) if a != "T0"]
    v.text = "\n".join(lines)
    return v
