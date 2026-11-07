"""Alias-aware tool handlers that the investigator agent can call.

Each handler takes aliases (A7, ...) from the case, runs a CaseTools lookup (so the
no-label and no-future rules still hold), registers any new transactions in the CaseView
so they become citable, and returns compact text for the LLM.

Tool names, descriptions and argument rules live in definitions/tools/*.md. A tool's
`handler` field names the method here.
"""

from __future__ import annotations

from pydantic import ValidationError

from .case_view import CaseView, _totals
from .specs import ToolSpec
from .tools import CaseTools


class ToolInputError(ValueError):
    pass


class CaseToolkit:
    def __init__(self, case: CaseTools, view: CaseView):
        self.case = case
        self.view = view
        self.new_transactions: list[dict] = []
        self.log: list[dict] = []
        self._rows_returned = 0  # rows returned by the current call, for the log

    # ---------------------------------------------------------------- handlers

    def get_account_history(self, account: str, window_hours: float | None = 96, limit: int | None = 25) -> str:
        real = self._account(account)
        h = self.case.get_account_history(real, window_hours=window_hours or 96, limit=limit or 25)
        lines = [
            f"{account} history, {h['window']['start'][:16]} to {h['window']['end'][:16]}: "
            f"first seen {(h['first_seen'] or '?')[:16]}; "
            f"sent {_totals(h['outgoing'])}; received {_totals(h['incoming'])}.",
            f"Most recent {len(h['recent_transactions'])} of {h['window_tx_count']} transactions:",
        ]
        return "\n".join(lines + self._rows(h["recent_transactions"]))

    def get_transfers_between(self, from_account: str, to_account: str, window_hours: float | None = 96) -> str:
        src, dst = self._account(from_account), self._account(to_account)
        txs = self.case.get_transfers_between(src, dst, window_hours=window_hours or 96)
        head = f"{len(txs)} transfers {from_account} -> {to_account} in the last {window_hours or 96:g}h before the alert"
        return "\n".join([head + (":" if txs else ".")] + self._rows(txs))

    # ---------------------------------------------------------------- dispatcher

    def run(self, spec: ToolSpec, args: dict) -> str:
        """Validate args against the tool's spec, run its handler, and log the call."""
        entry = {"tool": spec.name, "args": {k: v for k, v in args.items() if k != "reason"},
                 "reason": args.get("reason"), "rows": 0, "new_transactions": 0, "error": None}
        before = len(self.new_transactions)
        self._rows_returned = 0
        try:
            parsed = spec.args_model().model_validate(args).model_dump()
            parsed.pop("reason", None)  # the agent's stated reason is logged, not passed to the lookup
            result = getattr(self, spec.handler)(**parsed)
        except ValidationError as exc:
            result = entry["error"] = "Invalid arguments: " + "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        except ToolInputError as exc:
            result = entry["error"] = f"Error: {exc}"
        entry["new_transactions"] = len(self.new_transactions) - before
        entry["rows"] = self._rows_returned
        self.log.append(entry)
        return result

    # ---------------------------------------------------------------- helpers

    def _account(self, alias: str) -> str:
        if alias not in self.view.acct:
            raise ToolInputError(f"unknown account {alias}; use an alias that appears in the case")
        return self.view.acct[alias]

    def _rows(self, txs: list[dict]) -> list[str]:
        self._rows_returned += len(txs)
        out = []
        for t in txs:
            alias, new = self.view.register(t)
            if new:
                self.new_transactions.append(t)
            out.append(self.view.row(alias))
        return out
