"""Attach the GNN's own explanation of an alert to its evidence.

Stage 04 (`python scripts/04_train_gnn.py --explain`) writes outputs/agent/alerts_explained.json:
for each of its top alerts, how much of the score came from the payment itself, from payments
touching its two accounts (1 hop) and from payments one step further (2 hops), the payment's
own features that moved the score most, and the 5 neighbouring payments that mattered most.

The file holds no labels or typologies. Its neighbouring payments are still looked up through
CaseTools, so the no-future rule applies to them like any other evidence: one that is unknown
or after the alert time is dropped and counted, never shown.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .alerts import edge_id_to_tx_id
from .tools import CaseTools, TransactionNotVisible

DEFAULT_PATH = Path("outputs") / "agent" / "alerts_explained.json"


def load_explanations(path: Path) -> dict[str, dict]:
    """alerts_explained.json -> {tx_id: raw entry}. Missing file -> {}."""
    path = Path(path)
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not entries:
        return {}
    ids = edge_id_to_tx_id(pd.Series([e["edge_id"] for e in entries]))
    return dict(zip(ids, entries))


def applies_to(alert: dict) -> bool:
    """The file explains the GNN's scores, so it is attached only to GNN alerts."""
    return "gnn" in str(alert.get("model_name", "")).lower()


def build_model_explanation(case: CaseTools, raw: dict) -> dict:
    """Normalise one raw entry; fetch its context payments under the alert's cutoff."""
    context, not_visible = [], 0
    for c in raw.get("top_context_transactions") or []:
        tx_id = edge_id_to_tx_id(pd.Series([c["edge_id"]])).iloc[0]
        try:
            tx = case.get_transaction(tx_id)
        except TransactionNotVisible:
            not_visible += 1
            continue
        context.append({**tx, "hop": int(c["hop"]), "importance_share": float(c["importance_share"])})
    return {
        "source": "gnn",
        "model_score": float(raw["score"]),
        "above_threshold": bool(raw["above_threshold"]),
        "share_own": float(raw["share_from_own_transaction"]),
        "share_1hop": float(raw["share_from_1hop"]),
        "share_2hop": float(raw["share_from_2hop"]),
        "top_features": [
            {"feature": str(f["feature"]),
             "value": None if f.get("value") is None else float(f["value"]),
             "pushes_towards": f["pushes_towards"],
             "attribution": float(f["attribution"])}
            for f in raw.get("top_own_features") or []
        ],
        "context_transactions": context,
        "context_not_visible": not_visible,
    }
