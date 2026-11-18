"""Validate saved predictions, reproduce trainer metrics, and export comparisons."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .metrics import (DEFAULT_KS, choose_threshold, evaluate_scores, top_k_metrics,
                      typology_metrics, validate_scores)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _ids(series, context):
    if series.isna().any() or not pd.api.types.is_integer_dtype(series.dtype):
        raise ValueError(f"{context}: edge IDs must be nonmissing integers")
    if not series.is_unique:
        raise ValueError(f"{context}: duplicate edge IDs")
    return series.astype("int64")


def _splits(series, allowed, context):
    result = series.replace({"validation": "val"})
    if not result.isin(allowed).all():
        raise ValueError(f"{context}: unexpected/missing split values")
    return result


def load_labels(data_dir):
    frames, paths = [], []
    for split, filename in (("val", "validation.csv"), ("test", "test.csv")):
        path = Path(data_dir) / filename
        d = pd.read_csv(path, usecols=["source_row", "timestamp", "is_laundering"])
        d = d.rename(columns={"source_row": "edge_id", "is_laundering": "label"})
        d["edge_id"] = _ids(d.edge_id, str(path))
        validate_scores(d.label, np.zeros(len(d)))
        if d.label.nunique() != 2:
            raise ValueError(f"{path}: expected both classes for the full evaluation split")
        d["label"] = d.label.astype("int8")
        d["timestamp"] = pd.to_datetime(d.timestamp, errors="raise")
        if d.timestamp.isna().any():
            raise ValueError(f"{path}: missing timestamps")
        d["split"] = split
        frames.append(d)
        paths.append(path)
    if frames[0].timestamp.max() >= frames[1].timestamp.min():
        raise ValueError("Validation and test overlap in time")
    labels = pd.concat(frames, ignore_index=True)
    _ids(labels.edge_id, "validation/test labels")
    return labels.sort_values("edge_id", kind="stable").reset_index(drop=True), paths


def load_predictions(path, labels):
    p = pd.read_parquet(path)
    if not {"edge_id", "split", "score"} <= set(p):
        raise ValueError(f"{path}: need edge_id, split, score")
    p = p[["edge_id", "split", "score"]].copy()
    p["edge_id"] = _ids(p.edge_id, str(path))
    p["split"] = _splits(p.split, ["val", "test"], str(path))
    validate_scores(np.zeros(len(p)), p.score)
    expected, actual = pd.Index(labels.edge_id), pd.Index(p.edge_id)
    missing, extra = expected.difference(actual), actual.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f"{path}: prediction coverage differs from labels: "
                         f"{len(missing)} missing (e.g. {missing[:3].tolist()}), "
                         f"{len(extra)} extra (e.g. {extra[:3].tolist()})")
    aligned = labels.merge(p, on="edge_id", how="left", validate="1:1", suffixes=("", "_prediction"))
    if not aligned.split.eq(aligned.split_prediction).all():
        raise ValueError(f"{path}: prediction split disagrees with label split")
    return aligned.drop(columns="split_prediction")


def load_annotations(path, labels):
    a = pd.read_csv(path, dtype={"pattern_id": "string", "typology": "string", "annotation_status": "string"})
    if "source_row" in a and "edge_id" not in a:
        a = a.rename(columns={"source_row": "edge_id"})
    required = {"edge_id", "pattern_id", "typology", "annotation_status"}
    if not required <= set(a):
        raise ValueError(f"{path}: missing annotation columns {sorted(required - set(a))}")
    a["edge_id"] = _ids(a.edge_id, str(path))
    if "split" in a:
        a["split"] = _splits(a.split, ["train", "val", "test"], str(path))
        advertised_eval = a.loc[a.split.isin(["val", "test"]), "edge_id"]
        if not advertised_eval.isin(labels.edge_id).all():
            raise ValueError("Pattern annotations refer to unknown validation/test transactions")
    a = a.loc[a.edge_id.isin(labels.edge_id)].copy()
    a = a.merge(labels[["edge_id", "label", "split"]], on="edge_id", validate="1:1", suffixes=("_annotation", ""))
    if not a.label.eq(1).all():
        raise ValueError("Pattern annotations include transactions labelled normal")
    if "is_laundering" in a and not a.is_laundering.eq(a.label).all():
        raise ValueError("Annotation laundering labels disagree with evaluation labels")
    if "split_annotation" in a and not a.split_annotation.eq(a.split).all():
        raise ValueError("Annotation split disagrees with evaluation labels")
    if not a.annotation_status.isin(["ANNOTATED", "UNKNOWN_UNLISTED"]).all():
        raise ValueError("Unknown annotation status; do not infer unpatterned laundering from missing data")
    annotated = a.annotation_status.eq("ANNOTATED")
    if a.loc[annotated, ["pattern_id", "typology"]].isna().any().any():
        raise ValueError("Annotated schemes need both pattern_id and typology")
    if a.loc[annotated, "typology"].isin(["UNKNOWN_UNLISTED", "MISSING_ANNOTATION"]).any():
        raise ValueError("Annotated scheme has an invalid typology")
    if not a.loc[~annotated, "typology"].eq("UNKNOWN_UNLISTED").all():
        raise ValueError("Unlisted annotations must explicitly use UNKNOWN_UNLISTED")
    if a.loc[~annotated, "pattern_id"].notna().any():
        raise ValueError("Unlisted transactions must not carry a scheme ID")
    return a[sorted(required)]


def _reference_checks(name, reference, results):
    checks = []
    for split in ("val", "test"):
        for key, expected in reference.get(split, {}).items():
            if key not in results[split] or isinstance(expected, (dict, list, str)):
                continue
            actual = results[split][key]
            equal = actual is expected if actual is None or expected is None else bool(
                np.isclose(actual, expected, rtol=1e-8, atol=1e-10))
            checks.append({"model": name, "split": split, "metric": key,
                           "saved": expected, "recomputed": actual,
                           "absolute_difference": abs(actual - expected) if actual is not None and expected is not None else None,
                           "matches": equal})
    return checks


def fingerprint(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def run_comparison(data_dir, predictions, out_dir, pattern_path=None, references=None,
                   ks=DEFAULT_KS, make_plots=True):
    """Compare named models on exactly the same labelled populations.

    references maps model names to optional trainer metrics JSONs. Saved
    validation thresholds are verified before reuse. Any saved-metric mismatch
    fails evaluation and leaves reference_metric_checks.csv for diagnosis.
    """
    references = references or {}
    ks = tuple(ks)
    if not ks or len(set(ks)) != len(ks) or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must contain distinct positive integer review budgets")
    if len(predictions) < 2:
        raise ValueError("Model comparison requires at least two prediction files")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "status.json", {"state": "running"})
    try:
        return _run(data_dir, predictions, out_dir, pattern_path, references, ks, make_plots)
    except Exception as exc:
        write_json(out_dir / "status.json", {"state": "failed", "error": str(exc)})
        raise


def _run(data_dir, predictions, out_dir, pattern_path, references, ks, make_plots):
    labels, label_paths = load_labels(data_dir)
    annotations = load_annotations(pattern_path, labels) if pattern_path is not None else None
    aligned, results, thresholds, reference_summary = {}, {}, {}, {}
    rows, budget_rows, type_rows, group_rows, checks = [], [], [], [], []
    for name, path in predictions.items():
        print(f"Validating and evaluating {name}: {path}", flush=True)
        frame = load_predictions(path, labels)
        aligned[name] = frame
        val = frame.loc[frame.split == "val"]
        computed = choose_threshold(val.label, val.score)
        threshold = computed
        reference = {}
        if references.get(name) is not None:
            reference = json.loads(Path(references[name]).read_text(encoding="utf-8"))
            if reference.get("summary", {}).get("smoke"):
                raise ValueError(f"{name}: smoke-run results cannot be used for full comparison")
            if "threshold" not in reference.get("val", {}):
                raise ValueError(f"{name}: reference lacks a validation threshold")
            threshold = float(reference["val"]["threshold"])
            if not np.isclose(threshold, computed, rtol=0, atol=1e-12):
                raise ValueError(f"{name}: saved validation threshold {threshold} differs from "
                                 f"recomputed best-F1 threshold {computed}; check paired artifacts")
            reference_summary[name] = reference.get("summary", {})
        thresholds[name] = {"threshold": threshold, "recomputed_validation_threshold": computed,
                            "selection": "first maximum validation F1", "test_threshold_fixed": True,
                            "source": "verified saved validation threshold" if reference else "validation predictions"}
        results[name] = {}
        for split in ("val", "test"):
            d = frame.loc[frame.split == split]
            metrics = evaluate_scores(d.label, d.score, threshold, d.edge_id, ks)
            results[name][split] = metrics
            rows.append({"model": name, "split": split, **metrics})
            budget_rows.extend({"model": name, "split": split, **row}
                               for row in top_k_metrics(d.label, d.score, d.edge_id, ks))
            if annotations is not None:
                types, groups = typology_metrics(d, annotations, threshold, ks)
                type_rows.extend({"model": name, "split": split, **row} for row in types)
                group_rows.extend({"model": name, "split": split, **row} for row in groups)
        checks.extend(_reference_checks(name, reference, results[name]))
    audit = pd.DataFrame(checks, columns=["model", "split", "metric", "saved", "recomputed", "absolute_difference", "matches"])
    audit.to_csv(out_dir / "reference_metric_checks.csv", index=False)
    if len(audit) and not audit.matches.all():
        bad = audit.loc[~audit.matches, ["model", "split", "metric"]].to_dict("records")
        raise ValueError(f"Saved metrics do not match: {bad}. See reference_metric_checks.csv. "
                         "A changed top-k tie policy may affect recall only when the cutoff is tied.")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    comparison.loc[comparison.split == "test"].to_csv(out_dir / "test_comparison.csv", index=False)
    budget = pd.DataFrame(budget_rows)
    budget.to_csv(out_dir / "review_budget.csv", index=False)
    types, groups = pd.DataFrame(type_rows), pd.DataFrame(group_rows)
    if annotations is not None:
        types.to_csv(out_dir / "typology_breakdown.csv", index=False)
        groups.to_csv(out_dir / "typology_groups.csv", index=False)
    population = {}
    for split in ("val", "test"):
        d = labels.loc[labels.split == split]
        missing = int((~d.loc[d.label == 1, "edge_id"].isin(annotations.edge_id)).sum()) if annotations is not None else None
        population[split] = {"rows": len(d), "positives": int(d.label.sum()),
                             "first_timestamp": str(d.timestamp.min()), "last_timestamp": str(d.timestamp.max()),
                             "positive_transactions_missing_annotations": missing}
    protocol = {
        "auprc": "sklearn average_precision_score (not trapezoidal PR area)",
        "thresholds": "model-specific maximum validation F1, fixed for test",
        "top_k": "descending score, then ascending edge_id for ties; k capped at split size",
        "typology": "recall among positive transactions; top-k chosen from full split, not within typology",
        "schemes": "annotated (typology, pattern_id) groups with at least one flagged payment in the evaluated split",
        "missing_annotations": "kept as MISSING_ANNOTATION, never inferred to be unpatterned",
        "scope": "saved individual models; no retraining, ensemble, calibration or significance claim",
        "score_interpretation": "model-specific scores; equal numeric thresholds are not required",
    }
    write_json(out_dir / "thresholds.json", thresholds)
    report = {"protocol": protocol, "population": population, "thresholds": thresholds,
              "models": results, "original_run_summaries": reference_summary,
              "reference_checks": {"checked": len(audit), "all_match": bool(audit.matches.all()) if len(audit) else None},
              "typology_breakdown": type_rows, "typology_groups": group_rows}
    write_json(out_dir / "model_comparison.json", report)
    if make_plots:
        from .plots import save_plots
        save_plots(aligned, comparison, budget, types, out_dir / "figures")
    sources = [*label_paths, *predictions.values(), *[p for p in references.values() if p is not None]]
    if pattern_path is not None:
        sources.append(pattern_path)
    source_code = [Path(__file__), Path(__file__).with_name("metrics.py"), Path(__file__).with_name("plots.py"),
                   Path(__file__).resolve().parents[2] / "scripts" / "05_evaluate.py"]
    write_json(out_dir / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [fingerprint(p) for p in sources],
        "source_code": [fingerprint(p) for p in source_code if p.exists()],
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
                        for p in ("numpy", "pandas", "scikit-learn", "pyarrow", "matplotlib")}},
        "review_budgets": list(ks), "prediction_files": {k: str(Path(v).resolve()) for k, v in predictions.items()},
    })
    notes = """Evaluation outputs

model_comparison.csv / .json: validation and test metrics for each saved model.
test_comparison.csv: report-ready test rows. Scores are fractions, not percentages.
review_budget.csv: precision, recall and laundering caught at each global review budget.
reference_metric_checks.csv: numerical checks against the paired trainer JSON files.
thresholds.json: full-precision, validation-selected thresholds reused on test.
typology_breakdown.csv: positive-class detection by typology (when annotations supplied).
typology_groups.csv: patterned / unpatterned / missing-annotation detection.
figures/: PNG and PDF figures. Confusion matrices contain counts, not percentages.
manifest.json: input hashes, file paths, source hashes and software versions.

No models are fitted. All models must cover every validation/test ID exactly once,
with matching split assignments. Each has its own validation-selected threshold.
AUPRC is average precision. Top-k ties use ascending edge_id, so changing row
order cannot change results. Precision@k uses min(k, split size) as denominator.
Typology recall@k uses the SAME full-split ranked queue as the overall metric.
There is no per-typology precision or multiclass confusion matrix: these models
predict laundering, not typology. Unknown/unlisted and missing annotations are
separate. Scheme recall covers schemes represented in this split, not all time.
These are saved-model comparisons, not multi-seed averages or significance tests.
"""
    (out_dir / "README.txt").write_text(notes, encoding="utf-8")
    write_json(out_dir / "status.json", {"state": "complete"})
    return report
