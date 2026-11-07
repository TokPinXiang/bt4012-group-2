"""Contract checks at the pipeline's boundaries.

validate_predictions   - a predictions_<model>.csv from Member 2 or 3, before the agent uses it
validate_case_reports  - the agent's own output, before anyone relies on it

Both return a CheckReport: a list of errors (must fix) and warnings (look into).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, TypeAdapter, ValidationError

from .alerts import PREDICTION_COLUMNS
from .case_view import CaseView
from .investigator import triage
from .report import LLM_FIELDS, assert_no_label_fields
from .schemas import CaseReportFile, LLMClaim, PredictionRow
from .verifier import verify_claim

MAX_EXAMPLES = 5


class Issue(BaseModel):
    level: Literal["error", "warning"]
    check: str
    message: str
    examples: list[str] = []


class CheckReport(BaseModel):
    target: str
    issues: list[Issue] = []
    stats: dict = {}

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def error(self, check: str, message: str, examples=()):
        self.issues.append(Issue(level="error", check=check, message=message, examples=[str(e) for e in list(examples)[:MAX_EXAMPLES]]))

    def warn(self, check: str, message: str, examples=()):
        self.issues.append(Issue(level="warning", check=check, message=message, examples=[str(e) for e in list(examples)[:MAX_EXAMPLES]]))

    def render(self) -> str:
        lines = [f"{'PASS' if self.ok else 'FAIL'}  {self.target}"]
        for k, v in self.stats.items():
            lines.append(f"   {k}: {v}")
        for i in self.issues:
            lines.append(f"   [{i.level.upper()}] {i.check}: {i.message}")
            lines += [f"       e.g. {e}" for e in i.examples]
        if not self.issues:
            lines.append("   no issues")
        return "\n".join(lines)


# --------------------------------------------------------------------------- predictions

def validate_predictions(path: Path, db_path: Path, labels_path: Path | None = None) -> CheckReport:
    """Check a model's prediction file against the shared contract and the data."""
    rep = CheckReport(target=str(path))
    try:
        df = pd.read_csv(path, dtype={"tx_id": str, "split": str, "model_name": str})
    except Exception as exc:
        rep.error("readable", f"could not read CSV: {exc}")
        return rep

    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        rep.error("columns", f"missing columns {missing}; expected {PREDICTION_COLUMNS}")
        return rep
    extra = [c for c in df.columns if c not in PREDICTION_COLUMNS]
    if extra:
        rep.warn("columns", f"extra columns are ignored: {extra}")
    df = df[PREDICTION_COLUMNS]
    rep.stats["rows"] = len(df)

    # 1. Row schema (types, ranges, id format) via Pydantic.
    try:
        TypeAdapter(list[PredictionRow]).validate_python(df.replace({np.nan: None}).to_dict("records"))
    except ValidationError as exc:
        errs = exc.errors()
        rep.error("row_schema", f"{len(errs)} invalid values",
                  [f"row {e['loc'][0] + 2}, {e['loc'][-1]}: {e['msg']} (got {e.get('input')!r})" for e in errs])
        return rep  # later checks assume valid rows

    # 2. One model per file, no duplicate transactions.
    models = sorted(df["model_name"].unique())
    rep.stats["model_name"] = models
    if len(models) > 1:
        rep.error("one_model", f"file mixes models {models}; use one file per model")
    dup = df[df.duplicated(["tx_id"], keep=False)]["tx_id"].unique()
    if len(dup):
        rep.error("unique_tx", f"{len(dup)} tx_id values appear more than once", dup)

    # 3. Every tx_id exists, in the split it claims to be in.
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.register("pred", df[["tx_id", "split"]])
        unknown = [r[0] for r in con.execute(
            "SELECT p.tx_id FROM pred p LEFT JOIN tx USING (tx_id) WHERE tx.tx_id IS NULL LIMIT 1000").fetchall()]
        wrong_split = con.execute(
            "SELECT p.tx_id, p.split, tx.split FROM pred p JOIN tx USING (tx_id) WHERE p.split <> tx.split LIMIT 1000"
        ).fetchall()
        split_sizes = dict(con.execute("SELECT split, count(*) FROM tx GROUP BY split").fetchall())
    finally:
        con.close()
    if unknown:
        rep.error("tx_exists", f"{len(unknown)}{'+' if len(unknown) == 1000 else ''} tx_id values are not in the data "
                  "(use Min Gu's tx_id unchanged)", unknown)
    if wrong_split:
        rep.error("split_matches", f"{len(wrong_split)} rows have the wrong split",
                  [f"{t}: file says {a}, data says {b}" for t, a, b in wrong_split])

    # 4. Coverage: the agent picks its own top-K, so each scored split should be complete.
    counts = df["split"].value_counts().to_dict()
    rep.stats["rows_by_split"] = counts
    for split, n in counts.items():
        expected = split_sizes.get(split, 0)
        if n < expected:
            rep.warn("coverage", f"{split}: {n:,} of {expected:,} transactions scored ({n / expected:.1%}); "
                     "score every transaction so top-K and metrics are fair")
    if "test" in counts and "validation" not in counts:
        rep.warn("splits", "test is scored but validation is not; tune on validation first")

    # 5. predicted_label must come from one threshold on score.
    for split, g in df.groupby("split"):
        pos, neg = g.loc[g.predicted_label == 1, "score"], g.loc[g.predicted_label == 0, "score"]
        if len(pos) and len(neg) and pos.min() < neg.max():
            overlap = g[((g.predicted_label == 1) & (g.score < neg.max())) | ((g.predicted_label == 0) & (g.score > pos.min()))]
            rep.error("single_threshold", f"{split}: labels are not one threshold on score "
                      f"(lowest flagged {pos.min():.4f} < highest unflagged {neg.max():.4f})",
                      overlap["tx_id"].head(MAX_EXAMPLES))
        rate = len(pos) / len(g)
        rep.stats[f"{split}_flag_rate"] = f"{rate:.3%}" + (f" (threshold ~{pos.min():.4f})" if len(pos) else "")
        if rate == 0:
            rep.warn("flag_rate", f"{split}: nothing is flagged, so every alert will be auto-cleared")
        elif rate > 0.05:
            rep.warn("flag_rate", f"{split}: {rate:.1%} flagged, but only ~0.1% of transactions are laundering")
        if g["score"].nunique() == 1:
            rep.error("constant_score", f"{split}: every score is identical")

    # 6. Optional sanity check against labels: too-good-to-be-true scores usually mean leakage.
    if labels_path and Path(labels_path).exists():
        from sklearn.metrics import average_precision_score, f1_score

        laundering = set(pd.read_csv(labels_path, usecols=["tx_id"], dtype=str)["tx_id"])
        for split, g in df.groupby("split"):
            y = g["tx_id"].isin(laundering).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            ap = average_precision_score(y, g["score"])
            f1 = f1_score(y, g["predicted_label"])
            rep.stats[f"{split}_pr_auc"] = round(float(ap), 4)
            rep.stats[f"{split}_minority_f1"] = round(float(f1), 4)
            if ap > 0.95:
                rep.warn("leakage_smell", f"{split}: PR-AUC {ap:.3f} is suspiciously high for this dataset; "
                         "check that no feature uses future data or the label")
    return rep


# --------------------------------------------------------------------------- agent output

def validate_case_reports(path: Path, predictions_path: Path | None = None) -> CheckReport:
    """Check the agent's output: schema, no labels, no future data, citations, triage, re-verification."""
    import json

    rep = CheckReport(target=str(path))
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        payload = CaseReportFile.model_validate(raw)
    except ValidationError as exc:
        rep.error("schema", f"{len(exc.errors())} schema errors",
                  [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()])
        return rep
    alerts = raw["alerts"]
    rep.stats["alerts"] = len(alerts)

    if payload.n_alerts != len(alerts):
        rep.error("counts", f"n_alerts says {payload.n_alerts} but file has {len(alerts)}")

    future, leaked, outside, triage_bad, reverify_bad, status_bad = [], [], [], [], [], []
    for r in alerts:
        tx = r["alert"]["tx_id"]
        try:
            assert_no_label_fields({k: v for k, v in r.items() if k not in LLM_FIELDS})
        except ValueError as exc:
            leaked.append(f"{tx}: {exc}")
        if r.get("error"):
            continue

        as_of = dt.datetime.fromisoformat(r["as_of"])
        evidence_txs = r["subgraph"]["edges"] + (r.get("tool_transactions") or [])
        future += [f"{tx}: {e['tx_id']} at {e['ts']}" for e in evidence_txs
                   if dt.datetime.fromisoformat(e["ts"]) > as_of]
        future += [f"{tx}: {e['tx_id']} at {e['ts']}" for side in r["accounts"].values()
                   for e in side["recent_transactions"] if dt.datetime.fromisoformat(e["ts"]) > as_of]

        if r.get("triage") and r.get("verification"):
            expected = triage(r["alert"], r["verification"]["status"])["decision"]
            if r["triage"]["decision"] != expected:
                triage_bad.append(f"{tx}: {r['triage']['decision']} but rule gives {expected}")

        narrative = r.get("narrative")
        if not narrative:
            continue
        allowed = {e["tx_id"] for e in evidence_txs}
        for c in narrative["claims"]:
            bad = [t for t in c["cited_tx_ids"] if t not in allowed]
            if bad:
                outside.append(f"{tx}: {bad}")
        v = r["verification"]
        if v["claims_passed"] != len(narrative["claims"]):
            status_bad.append(f"{tx}: claims_passed={v['claims_passed']} but {len(narrative['claims'])} claims kept")

        # Re-run the verifier on every kept claim from the saved data alone.
        view = CaseView.restore(r)
        for c in narrative["claims"]:
            claim = LLMClaim.model_validate({"text": c["text"], "checks": c["checks"],
                                             "cited_tx": [view.tx_alias.get(t, "T999999") for t in c["cited_tx_ids"]]})
            res = verify_claim(claim, view)
            if not res["passed"]:
                reverify_bad.append(f"{tx}: {res['reasons']}")

    for check, items, msg in [
        ("no_labels", leaked, "label-like fields found in evidence"),
        ("no_future", future, "evidence contains transactions after the alert time"),
        ("citations_in_evidence", outside, "kept claims cite transactions outside the evidence"),
        ("triage_rule", triage_bad, "triage decision does not follow the rule"),
        ("claim_counts", status_bad, "verification counts do not match kept claims"),
        ("reverify", reverify_bad, "kept claims fail when re-verified from the saved data"),
    ]:
        if items:
            rep.error(check, f"{len(items)} problems: {msg}", items)

    # Cross-check against the prediction file the run says it used.
    pred_path = Path(predictions_path or payload.source_predictions)
    if pred_path.exists():
        preds = pd.read_csv(pred_path, dtype={"tx_id": str})
        top = preds.sort_values(["score", "tx_id"], ascending=[False, True])
        ranks = {t: i + 1 for i, t in enumerate(top["tx_id"])}
        mismatched = [f"{r['alert']['tx_id']}: rank {r['alert']['rank']} vs {ranks.get(r['alert']['tx_id'])}"
                      for r in alerts if ranks.get(r["alert"]["tx_id"]) is None]
        if mismatched:
            rep.error("from_predictions", f"{len(mismatched)} alerts are not in {pred_path.name}", mismatched)
    else:
        rep.warn("from_predictions", f"{pred_path} not found; skipped the cross-check")

    statuses = pd.Series([(r.get("verification") or {}).get("status", "none") for r in alerts]).value_counts()
    rep.stats["verification"] = statuses.to_dict()
    return rep
