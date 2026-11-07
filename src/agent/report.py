"""Build the JSON alert report: one evidence package per flagged transaction.

Each alert holds everything the LLM may look at, plus empty slots that the next stages
fill in:
    narrative     <- LLM case write-up (claims + cited tx_ids)
    verification  <- verifier result
    triage        <- auto-clear / fast-track / manual review

`evidence_tx_ids` is the full set of transactions shown in the alert. The verifier
will reject any citation outside it.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .store import FORBIDDEN_COLUMNS, TransactionStore
from .tools import DEFAULT_WINDOW_HOURS, TransactionNotVisible

REPORT_VERSION = "alert-report/v2"

DEFAULT_CONFIG = {
    "window_hours": DEFAULT_WINDOW_HOURS,
    "hops": 2,
    "focus_account_limit": 100,
    "per_account_limit": 25,
    "max_edges": 300,
    "history_limit": 25,
    "max_cycle_length": 6,
}


def build_alert_report(store: TransactionStore, alert: dict, config: dict | None = None) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    meta = {
        "rank": int(alert["rank"]),
        "tx_id": alert["tx_id"],
        "score": float(alert["score"]),
        "predicted_label": int(alert["predicted_label"]),
        "model_name": alert["model_name"],
        "split": alert["split"],
    }
    try:
        case = store.for_alert(alert["tx_id"], window_hours=cfg["window_hours"])
    except TransactionNotVisible as exc:
        return {"alert": meta, "error": str(exc)}

    subgraph = case.get_subgraph(
        hops=cfg["hops"], focus_account_limit=cfg["focus_account_limit"],
        per_account_limit=cfg["per_account_limit"], max_edges=cfg["max_edges"],
    )
    src_hist = case.get_account_history(case.flagged["src"], limit=cfg["history_limit"])
    dst_hist = case.get_account_history(case.flagged["dst"], limit=cfg["history_limit"])
    cycles = case.find_cycles(subgraph, max_length=cfg["max_cycle_length"])

    evidence = {e["tx_id"] for e in subgraph["edges"]}
    for hist in (src_hist, dst_hist):
        evidence |= {t["tx_id"] for t in hist["recent_transactions"]}

    return {
        "alert": meta,
        "as_of": case.flagged["ts"],
        "flagged_transaction": case.flagged,
        "accounts": {"source": src_hist, "destination": dst_hist},
        "structure_signals": case.structure_signals(),
        "cycles_closed_by_flagged": cycles,
        "subgraph": subgraph,
        "evidence_tx_ids": sorted(evidence),
        "narrative": None,
        "verification": None,
        "triage": None,
    }


def assert_no_label_fields(obj, path: str = "$") -> None:
    """Fail loudly if any label-like key slipped into the report."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_COLUMNS:
                raise ValueError(f"label field '{k}' found at {path}")
            assert_no_label_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_label_fields(v, f"{path}[{i}]")


# Filled by the LLM stage. These may legitimately name a typology (the LLM's guess).
LLM_FIELDS = ("narrative", "verification", "triage", "llm")


def write_alert_reports(reports: list[dict], out_path: Path, source: Path, config: dict | None = None,
                        extra: dict | None = None) -> Path:
    for r in reports:
        assert_no_label_fields({k: v for k, v in r.items() if k not in LLM_FIELDS})
    payload = {
        "report_version": REPORT_VERSION,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_predictions": str(source),
        "model_names": sorted({r["alert"]["model_name"] for r in reports}),
        "config": {**DEFAULT_CONFIG, **(config or {})},
        "n_alerts": len(reports),
        "n_errors": sum("error" in r for r in reports),
        **(extra or {}),
        "alerts": reports,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
