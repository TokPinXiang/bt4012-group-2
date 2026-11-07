"""Lookup tools the agent uses to investigate one flagged transaction.

Every tool is bound to one alert and enforces two rules itself, so the LLM cannot get
around them:

1. No labels. The store has no `is_laundering` column (see store.py).
2. No future. Only transactions with ts <= the flagged transaction's timestamp are
   visible. Same-minute transactions count as visible, because the data has one-minute
   resolution and they are not later than the alert.

Default windows come from pattern_labels.csv on this dataset: most laundering patterns
span 96 hours or less (gather-scatter can reach about 8 days).
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

from .store import PUBLIC_COLUMNS, TransactionStore

DEFAULT_WINDOW_HOURS = 96
COLS = ", ".join(PUBLIC_COLUMNS)


def _parse_ts(value: str | dt.datetime) -> dt.datetime:
    return value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(value)


class TransactionNotVisible(LookupError):
    """The id does not exist, or it happened after the alert's cutoff."""


class CaseTools:
    def __init__(self, store: TransactionStore, tx_id: str, window_hours: float = DEFAULT_WINDOW_HOURS):
        self.store = store
        flagged = store.one(f"SELECT {COLS} FROM tx WHERE tx_id = ?", [tx_id])
        if flagged is None:
            raise TransactionNotVisible(f"unknown transaction {tx_id}")
        self.flagged = flagged
        self.as_of = _parse_ts(flagged["ts"])
        self.window_hours = window_hours
        self.window_start = self.as_of - dt.timedelta(hours=window_hours)

    # ------------------------------------------------------------------ tools

    def get_transaction(self, tx_id: str) -> dict:
        """One transaction's details, if it exists and is not after the cutoff."""
        row = self.store.one(f"SELECT {COLS} FROM tx WHERE tx_id = ? AND ts <= ?", [tx_id, self.as_of])
        if row is None:
            raise TransactionNotVisible(f"{tx_id} does not exist or is after the alert time")
        return row

    def get_account_history(self, account_id: str, window_hours: float | None = None, limit: int = 25) -> dict:
        """Totals and the most recent transactions of one account, up to the cutoff."""
        start = self._start(window_hours)
        params = [account_id, account_id, start, self.as_of]
        where = "(src = ? OR dst = ?) AND ts >= ? AND ts <= ?"

        # A self-transfer (src = dst, e.g. a currency conversion) counts as both outgoing and
        # incoming, the same way structure_signals counts it.
        totals = self.store.rows(
            """
            SELECT 'outgoing' AS direction, count(*) AS tx_count, count(DISTINCT dst) AS counterparties,
                   min(ts) AS first_ts, max(ts) AS last_ts
            FROM tx WHERE src = $1 AND ts >= $2 AND ts <= $3
            UNION ALL
            SELECT 'incoming', count(*), count(DISTINCT src), min(ts), max(ts)
            FROM tx WHERE dst = $1 AND ts >= $2 AND ts <= $3
            """,
            [account_id, start, self.as_of],
        )
        by_currency = self.store.rows(
            """
            SELECT 'outgoing' AS direction, payment_currency AS currency, round(sum(amount_paid), 2) AS total
            FROM tx WHERE src = $1 AND ts >= $2 AND ts <= $3 GROUP BY 2
            UNION ALL
            SELECT 'incoming', receiving_currency, round(sum(amount_received), 2)
            FROM tx WHERE dst = $1 AND ts >= $2 AND ts <= $3 GROUP BY 2
            ORDER BY 1, 3 DESC
            """,
            [account_id, start, self.as_of],
        )
        lifetime = self.store.one(
            "SELECT min(ts) AS first_seen, count(*) AS tx_count FROM tx WHERE (src = ? OR dst = ?) AND ts <= ?",
            [account_id, account_id, self.as_of],
        )
        in_window = self.store.one(f"SELECT count(*) AS n FROM tx WHERE {where}", params)["n"]
        recent = self.store.rows(
            f"SELECT {COLS} FROM tx WHERE {where} ORDER BY ts DESC, tx_id LIMIT ?", params + [limit]
        )

        summary = {d: {"tx_count": 0, "counterparties": 0, "first_ts": None, "last_ts": None, "amount_by_currency": {}}
                   for d in ("outgoing", "incoming")}
        for t in totals:
            summary[t["direction"]].update({k: t[k] for k in ("tx_count", "counterparties", "first_ts", "last_ts")})
        for c in by_currency:
            summary[c["direction"]]["amount_by_currency"][c["currency"]] = c["total"]

        return {
            "account_id": account_id,
            "window": {"start": self._iso(start), "end": self._iso(self.as_of)},
            "first_seen": lifetime["first_seen"],
            "lifetime_tx_count": lifetime["tx_count"],
            **summary,
            "recent_transactions": recent,
            "window_tx_count": in_window,
            "recent_transactions_truncated": in_window > len(recent),
        }

    def get_subgraph(self, hops: int = 2, window_hours: float | None = None, focus_account_limit: int = 100,
                     per_account_limit: int = 25, max_edges: int = 300) -> dict:
        """Transactions around the flagged one, expanding hop by hop from its two accounts.

        Each account contributes only its most recent transactions, so hub accounts don't
        swamp the graph: up to `focus_account_limit` for the flagged transaction's two
        accounts (so a fan-out's older legs aren't dropped) and `per_account_limit` for the
        rest. The uncapped counts are in `structure_signals`.
        """
        start = self._start(window_hours)
        edges: dict[str, dict] = {self.flagged["tx_id"]: {**self.flagged, "hop": 0}}
        seen_accounts = {self.flagged["src"], self.flagged["dst"]}
        frontier = sorted(seen_accounts)
        truncated = False

        for hop in range(1, hops + 1):
            if not frontier:
                break
            cap = focus_account_limit if hop == 1 else per_account_limit
            found = self.store.rows(
                f"""
                WITH hits AS (
                    SELECT {COLS}, src AS anchor FROM tx
                    WHERE src IN (SELECT unnest($1)) AND ts >= $2 AND ts <= $3
                    UNION ALL
                    SELECT {COLS}, dst AS anchor FROM tx
                    WHERE dst IN (SELECT unnest($1)) AND ts >= $2 AND ts <= $3
                ),
                ranked AS (
                    SELECT *, row_number() OVER (PARTITION BY anchor ORDER BY ts DESC, tx_id) AS rn,
                           count(*) OVER (PARTITION BY anchor) AS anchor_total
                    FROM hits
                )
                SELECT * FROM ranked WHERE rn <= $4 ORDER BY ts DESC, tx_id
                """,
                [frontier, start, self.as_of, cap],
            )
            if any(r["anchor_total"] > cap for r in found):
                truncated = True
            next_frontier = set()
            for r in found:
                if r["tx_id"] in edges:
                    continue
                if len(edges) >= max_edges:
                    truncated = True
                    break
                edges[r["tx_id"]] = {c: r[c] for c in PUBLIC_COLUMNS} | {"hop": hop}
                for acct in (r["src"], r["dst"]):
                    if acct not in seen_accounts:
                        seen_accounts.add(acct)
                        next_frontier.add(acct)
            frontier = sorted(next_frontier)

        edge_list = sorted(edges.values(), key=lambda e: (e["ts"], e["tx_id"]))
        return {
            "hops": hops,
            "window": {"start": self._iso(start), "end": self._iso(self.as_of)},
            "accounts": sorted({e["src"] for e in edge_list} | {e["dst"] for e in edge_list}),
            "edges": edge_list,
            "truncated": truncated,
        }

    def find_cycles(self, subgraph: dict | None = None, max_length: int = 6, max_cycles: int = 10) -> list[dict]:
        """Cycles the flagged transaction closes: earlier money flowing dst -> ... -> src.

        The path's timestamps must not go backwards, so money could really have travelled
        round. Only cycles the flagged transaction *closes* can be seen, because anything
        after it is hidden.
        """
        subgraph = subgraph or self.get_subgraph()
        src, dst, flagged_id = self.flagged["src"], self.flagged["dst"], self.flagged["tx_id"]
        if src == dst:
            return []

        out_edges = defaultdict(list)
        for e in subgraph["edges"]:
            if e["tx_id"] != flagged_id and e["src"] != e["dst"]:
                out_edges[e["src"]].append(e)
        for lst in out_edges.values():
            lst.sort(key=lambda e: e["ts"])

        # Accounts in money order -> the tightest example (latest start) and how many paths repeat it.
        by_route: dict[tuple, dict] = {}
        budget = [5000]  # stop runaway searches around very busy accounts

        def walk(node, last_ts, path, visited):
            if budget[0] <= 0 or len(path) >= max_length - 1:
                return
            for e in out_edges.get(node, []):
                budget[0] -= 1
                if last_ts is not None and e["ts"] < last_ts:
                    continue
                if e["dst"] == src:
                    chain = path + [e]
                    route = tuple([dst] + [x["dst"] for x in chain] + [dst])
                    found = by_route.get(route)
                    if found is None or chain[0]["ts"] > found["start_ts"]:
                        by_route[route] = {
                            "length": len(chain) + 1,
                            "accounts": list(route),
                            "tx_ids": [x["tx_id"] for x in chain] + [flagged_id],
                            "start_ts": chain[0]["ts"],
                            "end_ts": self.flagged["ts"],
                            "repeat_count": found["repeat_count"] if found else 0,
                        }
                    by_route[route]["repeat_count"] += 1
                elif e["dst"] not in visited:
                    walk(e["dst"], e["ts"], path + [e], visited | {e["dst"]})

        walk(dst, None, [], {dst, src})
        # Shortest and most recent first: those are the most suspicious round trips.
        cycles = sorted(by_route.values(), key=lambda c: c["start_ts"], reverse=True)
        cycles.sort(key=lambda c: c["length"])  # stable, so recent-first is kept within each length
        return cycles[:max_cycles]

    def get_transfers_between(self, src: str, dst: str, window_hours: float | None = None, limit: int = 50) -> list[dict]:
        """All transfers from one account to another, up to the cutoff, oldest first."""
        return self.store.rows(
            f"SELECT {COLS} FROM tx WHERE src = ? AND dst = ? AND ts >= ? AND ts <= ? ORDER BY ts, tx_id LIMIT ?",
            [src, dst, self._start(window_hours), self.as_of, limit],
        )

    def structure_signals(self, window_hours: float | None = None) -> dict:
        """Uncapped counts that describe the shape around the flagged transaction."""
        start = self._start(window_hours)
        src, dst = self.flagged["src"], self.flagged["dst"]
        r = self.store.one(
            """
            SELECT
              count(*) FILTER (WHERE src = $1)                       AS src_out_tx,
              count(DISTINCT dst) FILTER (WHERE src = $1)            AS src_out_recipients,
              count(*) FILTER (WHERE dst = $1)                       AS src_in_tx,
              count(DISTINCT src) FILTER (WHERE dst = $1)            AS src_in_senders,
              count(*) FILTER (WHERE dst = $2)                       AS dst_in_tx,
              count(DISTINCT src) FILTER (WHERE dst = $2)            AS dst_in_senders,
              count(*) FILTER (WHERE src = $2)                       AS dst_out_tx,
              count(DISTINCT dst) FILTER (WHERE src = $2)            AS dst_out_recipients
            FROM tx
            WHERE (src IN ($1, $2) OR dst IN ($1, $2)) AND ts >= $3 AND ts <= $4
            """,
            [src, dst, start, self.as_of],
        )
        pair = self.store.one(
            """
            SELECT count(*) FILTER (WHERE src = $1 AND dst = $2 AND tx_id <> $4) AS same_pair_before,
                   count(*) FILTER (WHERE src = $2 AND dst = $1)                  AS reverse_pair_before
            FROM tx WHERE ((src = $1 AND dst = $2) OR (src = $2 AND dst = $1)) AND ts <= $3
            """,
            [src, dst, self.as_of, self.flagged["tx_id"]],
        )
        return {
            "window": {"start": self._iso(start), "end": self._iso(self.as_of)},
            "source_account": {
                "outgoing_tx": r["src_out_tx"], "distinct_recipients": r["src_out_recipients"],
                "incoming_tx": r["src_in_tx"], "distinct_senders": r["src_in_senders"],
            },
            "destination_account": {
                "incoming_tx": r["dst_in_tx"], "distinct_senders": r["dst_in_senders"],
                "outgoing_tx": r["dst_out_tx"], "distinct_recipients": r["dst_out_recipients"],
            },
            "pair_history": {
                "same_direction_before": pair["same_pair_before"],
                "reverse_direction_before": pair["reverse_pair_before"],
            },
        }

    # ---------------------------------------------------------------- helpers

    def _start(self, window_hours: float | None) -> dt.datetime:
        if window_hours is None:
            return self.window_start
        return self.as_of - dt.timedelta(hours=window_hours)

    @staticmethod
    def _iso(t: dt.datetime) -> str:
        return t.isoformat(sep=" ")
