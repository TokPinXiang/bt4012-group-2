"""Score the agent's case reports against pattern_labels.csv (evaluation only).

Metrics (citations = transactions cited by claims that passed verification):
- citation precision: share of cited transactions that are laundering / in the alert's pattern
- citation recall (full): share of the pattern's transactions that were cited
- citation recall (visible): same, but only over pattern transactions at or before the alert
  time, since the agent is not allowed to see later ones
- typology accuracy: named the right scheme, on alerts with a listed pattern
- normal alerts: how often the agent called a normal transaction laundering
- red team: plant a fake id or a wrong amount in each verified claim; share the verifier catches
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

from .case_view import CaseView
from .schemas import LLMClaim
from .verifier import verify_claim


# Scheme names that describe the visible part of each true pattern. With no future data,
# a half-finished scatter-gather can only look like a fan-out (or fan-in, near its end).
COMPATIBLE = {
    "FAN-OUT": {"FAN-OUT"},
    "FAN-IN": {"FAN-IN"},
    "CYCLE": {"CYCLE"},
    "SCATTER-GATHER": {"SCATTER-GATHER", "FAN-OUT", "FAN-IN"},
    "GATHER-SCATTER": {"GATHER-SCATTER", "FAN-IN", "FAN-OUT"},
    "STACK": {"STACK", "BIPARTITE", "FAN-OUT", "FAN-IN"},
    "BIPARTITE": {"BIPARTITE", "FAN-OUT", "FAN-IN", "STACK"},
    "RANDOM": {"RANDOM", "STACK"},
}


def _ts_lookup(db_path: Path, tx_ids: list[str]) -> dict[str, str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("SELECT tx_id, CAST(ts AS VARCHAR) FROM tx WHERE tx_id IN (SELECT unnest(?))",
                           [tx_ids]).fetchall()
    finally:
        con.close()
    return dict(rows)


def evaluate_cases(payload: dict, labels_path: Path, db_path: Path) -> tuple[pd.DataFrame, dict]:
    labels = pd.read_csv(labels_path, dtype=str)
    laundering = set(labels["tx_id"])
    by_tx = labels.set_index("tx_id")
    members = labels.dropna(subset=["pattern_id"]).groupby("pattern_id")["tx_id"].apply(set).to_dict()
    ts = _ts_lookup(db_path, labels["tx_id"].tolist())

    rows = []
    for r in payload["alerts"]:
        tx = r["alert"]["tx_id"]
        narrative = r.get("narrative") or {}
        cited = {t for c in narrative.get("claims", []) for t in c["cited_tx_ids"]}
        row = {
            "tx_id": tx,
            "truth": "NORMAL",
            "predicted_typology": narrative.get("typology"),
            "suspicious": narrative.get("suspicious"),
            "status": (r.get("verification") or {}).get("status", "error"),
            "triage": r["triage"]["decision"],
            "n_cited": len(cited),
            "precision_laundering": len(cited & laundering) / len(cited) if cited else None,
        }
        if tx in by_tx.index:
            info = by_tx.loc[tx]
            row["truth"] = info["typology"]
            pid = info["pattern_id"]
            if isinstance(pid, str):
                pattern = members[pid]
                visible = {t for t in pattern if ts.get(t, "9999") <= r["as_of"]}
                row.update({
                    "pattern_size": len(pattern),
                    "visible_size": len(visible),
                    "precision_pattern": len(cited & pattern) / len(cited) if cited else None,
                    "recall_full": len(cited & pattern) / len(pattern),
                    "recall_visible": len(cited & visible) / len(visible) if visible else None,
                    "typology_correct": narrative.get("typology") == info["typology"],
                    "typology_compatible": narrative.get("typology") in COMPATIBLE.get(info["typology"], set()),
                })
        rows.append(row)

    df = pd.DataFrame(rows)
    named = df[~df["truth"].isin(["NORMAL", "UNKNOWN_UNLISTED"])]
    launder = df[df["truth"] != "NORMAL"]
    normal = df[df["truth"] == "NORMAL"]

    def mean(s):
        s = s.dropna()
        return round(float(s.mean()), 3) if len(s) else None

    summary = {
        "n_alerts": len(df),
        "n_laundering_named_pattern": len(named),
        "n_laundering_unlisted": int((df["truth"] == "UNKNOWN_UNLISTED").sum()),
        "n_normal": len(normal),
        "verification_status": df["status"].value_counts().to_dict(),
        "triage": df["triage"].value_counts().to_dict(),
        "citation_precision_pattern": mean(named.get("precision_pattern", pd.Series(dtype=float))),
        "citation_precision_any_laundering": mean(launder["precision_laundering"]),
        "citation_recall_full": mean(named.get("recall_full", pd.Series(dtype=float))),
        "citation_recall_visible": mean(named.get("recall_visible", pd.Series(dtype=float))),
        "typology_accuracy": mean(named.get("typology_correct", pd.Series(dtype=float)).astype(float)),
        "typology_compatible": mean(named.get("typology_compatible", pd.Series(dtype=float)).astype(float)),
        "laundering_called_suspicious": mean(launder["predicted_typology"].ne("NONE").astype(float)),
        "normal_called_suspicious": mean(normal["predicted_typology"].ne("NONE").astype(float)),
        "per_typology": named.groupby("truth").agg(
            n=("tx_id", "size"),
            typology_correct=("typology_correct", "mean"),
            typology_compatible=("typology_compatible", "mean"),
            precision=("precision_pattern", "mean"),
            recall_visible=("recall_visible", "mean"),
        ).round(2).reset_index().to_dict("records") if len(named) else [],
        "confusion": Counter(zip(df["truth"], df["predicted_typology"].fillna("—"))).most_common(),
    }
    summary["confusion"] = [{"truth": t, "predicted": p, "n": n} for (t, p), n in summary["confusion"]]
    return df, summary


def red_team(payload: dict, seed: int = 4012) -> dict:
    """Corrupt each verified claim two ways and count how often the verifier catches it."""
    rng = random.Random(seed)
    caught, total = Counter(), Counter()
    for r in payload["alerts"]:
        if not r.get("narrative") or "error" in r:
            continue
        view = CaseView.restore(r)

        def check(c: dict) -> bool:
            return verify_claim(LLMClaim.model_validate(c), view)["passed"]

        for claim in r["narrative"]["claims"]:
            alias_claim = {
                "text": claim["text"],
                "cited_tx": [view.tx_alias[t] for t in claim["cited_tx_ids"]],
                "checks": claim["checks"],
            }
            # 1. Fabricated id: swap one citation for an id that was never shown.
            fake = copy.deepcopy(alias_claim)
            fake["cited_tx"][rng.randrange(len(fake["cited_tx"]))] = f"T{len(view.tx) + rng.randint(1, 999)}"
            total["fake_id"] += 1
            caught["fake_id"] += not check(fake)

            # 2. Wrong number: inflate the first numeric fact in the checks by 25%.
            wrong = copy.deepcopy(alias_claim)
            for chk in wrong["checks"]:
                key = next((k for k in ("amount", "recipients", "senders", "n") if k in chk), None)
                if key:
                    chk[key] = (chk[key] + 1) if isinstance(chk[key], int) else round(float(chk[key]) * 1.25, 2)
                    total["wrong_number"] += 1
                    caught["wrong_number"] += not check(wrong)
                    break
    return {k: {"caught": caught[k], "total": total[k], "rate": round(caught[k] / total[k], 3) if total[k] else None}
            for k in total}


# Substring of a verifier reason -> bucket, checked in order.
REASON_BUCKETS = [
    ("not in the evidence", "made-up transaction id"),
    ("cites no transactions", "no citation"),
    ("recipients, not", "wrong count"),
    ("senders, not", "wrong count"),
    ("transactions cited, not", "wrong count"),
    ("cited total is", "wrong amount"),
    ("amounts fall outside", "wrong amount"),
    ("transaction is in ", "wrong currency"),
    ("do not follow", "wrong route or order"),
    ("cycle must", "wrong route or order"),
    ("route", "wrong route or order"),
    ("span", "wrong time span"),
    ("not every cited transaction", "wrong account"),
    ("unknown account", "wrong account"),
]


def failure_reasons(payload: dict) -> dict:
    """Group the reasons claims were dropped, to see what the LLM gets wrong."""
    buckets = Counter()
    for r in payload["alerts"]:
        for d in (r.get("verification") or {}).get("dropped_claims", []):
            for reason in d.get("reasons", []):
                buckets[next((b for key, b in REASON_BUCKETS if key in reason), "malformed or other")] += 1
    return dict(buckets.most_common())
