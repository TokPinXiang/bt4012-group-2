"""Shared binary and typology metrics for GBT and GNN predictions.

Choose a threshold on validation only, then pass it explicitly to evaluation.
AUPRC means sklearn average precision, not trapezoidal PR area. Top-k ties are
broken by ascending edge_id, independently of input row ordering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
)

DEFAULT_KS = (500, 1000, 5000)


def validate_scores(y_true, scores):
    y, s = np.asarray(y_true), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError("Labels and scores must be non-empty, equal-length vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1, without missing values")
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("Scores must be finite and within [0, 1]")
    return y.astype(np.int8), s


def choose_threshold(y_val, scores_val):
    """First threshold achieving maximum validation F1, matching both trainers."""
    y, s = validate_scores(y_val, scores_val)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both validation classes")
    p, r, thresholds = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def rank_scores(scores, edge_ids):
    s, ids = np.asarray(scores, dtype=float), np.asarray(edge_ids)
    if ids.ndim != 1 or len(ids) != len(s) or not pd.api.types.is_integer_dtype(ids.dtype):
        raise ValueError("edge_id must be an integer vector aligned to scores")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("edge_id must be unique")
    if not np.isfinite(s).all():
        raise ValueError("Scores must be finite")
    return np.lexsort((ids, -s))


def top_k_metrics(y_true, scores, edge_ids, ks=DEFAULT_KS):
    y, s = validate_scores(y_true, scores)
    order = rank_scores(s, edge_ids)
    positives = int(y.sum())
    rows = []
    for k in ks:
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("Review budgets must be positive integers")
        effective = min(int(k), len(y))
        caught = int(y[order[:effective]].sum())
        cutoff = float(s[order[effective - 1]])
        rows.append({"k": int(k), "reviewed": effective, "caught": caught,
                     "precision": caught / effective,
                     "recall": caught / positives if positives else None,
                     "review_fraction": effective / len(y), "cutoff_score": cutoff,
                     "transactions_tied_at_cutoff": int((s == cutoff).sum())})
    return rows


def evaluate_scores(y_true, scores, threshold, edge_ids, ks=DEFAULT_KS):
    """Evaluate a frozen threshold. Undefined ranking metrics are None, not zero."""
    y, s = validate_scores(y_true, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0, 1]")
    predicted = s >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    positives = tp + fn
    result = {
        "n": len(y), "positives": positives, "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, s)) if positives else None,
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
        "threshold": float(threshold), "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positives if positives else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "alerts": tp + fp, "alert_rate": (tp + fp) / len(y),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    for row in top_k_metrics(y, s, edge_ids, ks):
        for name in ("precision", "recall", "caught"):
            result[f"{name}@{row['k']}"] = row[name]
    return result


def typology_metrics(frame, annotations, threshold, ks=DEFAULT_KS):
    """Return per-typology and annotation-group detection on positive transactions.

    frame is one full model/split, with edge_id, label and score. Global top-k
    membership is calculated BEFORE filtering to laundering transactions.
    Precision by typology is not defined for this binary model. A scheme is
    detected if at least one of its annotated payments in this split is flagged.
    Missing annotations remain MISSING_ANNOTATION, distinct from UNKNOWN_UNLISTED.
    """
    y, s = validate_scores(frame.label, frame.score)
    order = rank_scores(s, frame.edge_id)
    positives = frame.loc[y == 1, ["edge_id"]].copy()
    positives["flagged"] = (s >= threshold)[y == 1]
    for k in ks:
        chosen = np.zeros(len(frame), dtype=bool)
        chosen[order[:k]] = True
        positives[f"caught@{k}"] = chosen[y == 1]
    positives = positives.merge(annotations[["edge_id", "typology", "pattern_id", "annotation_status"]],
                                on="edge_id", how="left", validate="1:1")
    positives["typology"] = positives.typology.fillna("MISSING_ANNOTATION")
    positives["annotation_status"] = positives.annotation_status.fillna("MISSING_ANNOTATION")
    positives["group"] = positives.annotation_status.map({
        "ANNOTATED": "PATTERNED", "UNKNOWN_UNLISTED": "UNPATTERNED",
        "MISSING_ANNOTATION": "MISSING_ANNOTATION"})

    def summarise(group, name, key):
        count, detected = len(group), int(group.flagged.sum())
        annotated = group.loc[group.annotation_status == "ANNOTATED"]
        schemes = annotated.groupby(["typology", "pattern_id"], dropna=False).flagged.max()
        result = {key: name, "positives": count, "detected": detected, "missed": count - detected,
                  "recall": detected / count, "threshold": float(threshold),
                  "schemes": len(schemes) if len(schemes) else None,
                  "schemes_detected": int(schemes.sum()) if len(schemes) else None,
                  "scheme_recall": float(schemes.mean()) if len(schemes) else None}
        for k in ks:
            result[f"caught@{k}"] = int(group[f"caught@{k}"].sum())
            result[f"recall@{k}"] = result[f"caught@{k}"] / count
        return result

    per_type = [summarise(g, name, "typology") for name, g in positives.groupby("typology", sort=True)]
    per_group = [summarise(g, name, "group") for name, g in positives.groupby("group", sort=True)]
    return per_type, per_group
