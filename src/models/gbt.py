"""LightGBM baseline for stage A's CSVs. Selection uses validation, never test.

No dependency on the GNN or shared evaluator. TreeSHAP uses LightGBM's native
pred_contrib API, so a separate shap package is not required.
"""
from __future__ import annotations

import gc
import importlib.metadata
import json
from math import prod
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

CATEGORICAL = ("payment_format", "payment_currency", "receiving_currency")
HISTORY_PREFIXES = ("src_", "dst_", "pair_", "reverse_pair_")
NOT_FEATURES = {
    "source_row", "tx_id", "edge_id", "timestamp", "is_laundering", "label", "split",
    "from_bank", "to_bank", "from_account", "to_account", "src_account_id",
    "dst_account_id", "src_node_id", "dst_node_id", "annotation_status",
}
FILES = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
# Insertion order preserves the original seeded random-search sequence.
SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [50, 100, 200, 500],
    "reg_lambda": [0.0, 1.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
MAX_TRIALS = 3 * (1 + prod(len(values) for values in SEARCH_SPACE.values()))


@dataclass
class TrainConfig:
    features: str = "all"
    seed: int = 0
    trials: int = 30
    max_rounds: int = 1000
    patience: int = 60
    threads: int = 4
    shap_rows: int = 1000
    smoke: bool = False

    def validate(self):
        if self.features not in ("row", "all"):
            raise ValueError("features must be 'row' or 'all'")
        if min(self.trials, self.max_rounds, self.patience, self.threads) < 1:
            raise ValueError("trials, max_rounds, patience and threads must be positive")
        if self.trials > MAX_TRIALS:
            raise ValueError(f"trials exceeds the search-space limit of {MAX_TRIALS}")
        if self.shap_rows < 0 or self.seed < 0:
            raise ValueError("shap_rows and seed must be nonnegative")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def feature_columns(columns, mode="all"):
    """Match the team's row/history split; exclude identifiers and annotations."""
    if mode not in ("row", "all"):
        raise ValueError(mode)
    return [c for c in columns if c.lower() not in NOT_FEATURES
            and not c.lower().endswith("_id")
            and not any(s in c.lower() for s in ("pattern", "typology", "label", "launder"))
            and (mode == "all" or not c.startswith(HISTORY_PREFIXES))]


def load_split(data_dir, split, mode="all", expected_features=None, limit=None):
    """Read only predictors and evaluation keys; float32 limits local RAM usage.

    limit takes first rows for smoke checks only. Pattern labels are never opened.
    """
    path = Path(data_dir) / FILES[split]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"source_row", "timestamp", "is_laundering"}
    if not required <= set(header):
        raise ValueError(f"{path}: missing {sorted(required - set(header))}")
    features = feature_columns(header, mode)
    if not features:
        raise ValueError(f"{path}: no predictors")
    if expected_features is not None and features != list(expected_features):
        raise ValueError(f"{path}: feature schema differs from training")
    dtypes = {c: "string" if c in CATEGORICAL else "float32" for c in features}
    dtypes.update(source_row="int64", is_laundering="float64")
    df = pd.read_csv(path, usecols=["source_row", "timestamp", "is_laundering", *features],
                     dtype=dtypes, nrows=limit)
    if df.empty or df.source_row.isna().any() or not df.source_row.is_unique:
        raise ValueError(f"{path}: empty data or invalid/duplicate source_row")
    if not df.is_laundering.isin([0, 1]).all() or df.is_laundering.nunique() != 2:
        raise ValueError(f"{path}: expected both binary classes (0 and 1)")
    df["is_laundering"] = df.is_laundering.astype("int8")
    df["timestamp"] = pd.to_datetime(df.timestamp, errors="raise")
    if df.timestamp.isna().any():
        raise ValueError(f"{path}: missing timestamps")
    numeric = [c for c in features if c not in CATEGORICAL]
    if np.isinf(df[numeric].to_numpy()).any():
        raise ValueError(f"{path}: infinite feature values; fix upstream")
    return df, features


def check_splits(earlier, later):
    if earlier.timestamp.max() >= later.timestamp.min():
        raise ValueError("Splits overlap in time; use the team's chronological boundaries")
    if np.intersect1d(earlier.source_row, later.source_row).size:
        raise ValueError("Transaction IDs overlap between splits")


def fit_preprocessor(train, features):
    """Learn category levels on training only; unknown levels become missing."""
    categories = {c: sorted(train[c].dropna().astype(str).unique().tolist())
                  for c in features if c in CATEGORICAL}
    return {"features": list(features), "categories": categories,
            "numeric_missing": "native LightGBM missing-value handling",
            "unknown_categories": "mapped to missing; levels fitted on training only"}


def transform(frame, metadata):
    missing = set(metadata["features"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing predictors: {sorted(missing)}")
    x = frame[metadata["features"]].copy()
    for c in x:
        if c in metadata["categories"]:
            x[c] = pd.Categorical(x[c].astype("string"), categories=metadata["categories"][c])
        else:
            x[c] = pd.to_numeric(x[c], errors="raise").astype("float32")
            if np.isinf(x[c].to_numpy()).any():
                raise ValueError(f"Infinite values in {c}")
    return x


def choose_threshold(y, scores):
    """Match the GNN's first maximum-F1 threshold; call on validation only."""
    p, r, thresholds = precision_recall_curve(y, scores)
    f = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(thresholds[int(np.nanargmax(f[:-1]))]) if len(thresholds) else 0.5


def baseline_metrics(y, scores, threshold):
    """Baseline diagnostics; stage 05's shared evaluation is a separate task."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not len(y) or not np.isfinite(scores).all():
        raise ValueError("Invalid labels/scores")
    if not np.isin(y, [0, 1]).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected binary labels and scores in [0, 1]")
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    order = np.argsort(-scores)
    return {
        "n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "alerts": int(predicted.sum()), "alert_rate": float(predicted.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        **{f"recall@{k}": float(y[order[:k]].sum() / max(y.sum(), 1)) for k in (500, 1000, 5000)},
    }


def candidate_parameters(ratio, count, seed):
    """First three isolate weighting; later trials vary capacity and regularisation."""
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Training negative/positive ratio must be finite and positive")
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Trial count must be a positive integer")
    rng = np.random.default_rng(seed)
    weights = list(dict.fromkeys([1.0, float(np.sqrt(ratio)), float(ratio)]))
    # Initial candidates use subsample=1; random candidates use subsample=0.8.
    # The two pools cannot overlap. Balanced data has only one distinct weight.
    maximum = len(weights) * (1 + prod(len(values) for values in SEARCH_SPACE.values()))
    if count > maximum:
        raise ValueError(f"Requested {count} trials, but only {maximum} unique configurations "
                         f"exist for training class ratio {ratio:g}")
    base = dict(learning_rate=0.05, num_leaves=31, max_depth=-1,
                min_child_samples=100, reg_lambda=1.0, reg_alpha=0.0,
                colsample_bytree=1.0, subsample=1.0, subsample_freq=0)
    candidates = [{**base, "scale_pos_weight": w} for w in weights]
    while len(candidates) < count:
        p = {key: rng.choice(values).item() for key, values in SEARCH_SPACE.items()}
        p.update(max_depth=-1, subsample=0.8, subsample_freq=1,
                 scale_pos_weight=float(rng.choice(weights)))
        if p not in candidates:
            candidates.append(p)
    return candidates[:count]


def load_model(run_dir):
    run_dir = Path(run_dir)
    return (lgb.Booster(model_file=str(run_dir / "model.txt")),
            json.loads((run_dir / "preprocessing.json").read_text(encoding="utf-8")))


def predict(model, frame, metadata, threads=4):
    return model.predict(transform(frame, metadata), validate_features=True, num_threads=threads)


def save_interpretation(model, x_val, val_ids, scores, out_dir, sample_size, seed, threads):
    """TreeSHAP contributions sum to raw margin (log-odds), not probability.

    Global importance uses a uniform validation sample. The highest-scored
    validation payment is explained separately, outside global averages.
    """
    out_dir = Path(out_dir)
    importance = pd.DataFrame({"feature": model.feature_name(),
                               "gain": model.feature_importance("gain"),
                               "split_count": model.feature_importance("split")})
    importance.sort_values("gain", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)
    if sample_size == 0:
        return
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(x_val), min(sample_size, len(x_val)), replace=False))
    x = x_val.iloc[indices]
    contributions = model.predict(x, pred_contrib=True, num_threads=threads)
    raw = model.predict(x, raw_score=True, num_threads=threads)
    np.testing.assert_allclose(contributions.sum(axis=1), raw, rtol=1e-5, atol=1e-6)
    saved = pd.DataFrame(contributions[:, :-1], columns=model.feature_name())
    saved.insert(0, "edge_id", np.asarray(val_ids)[indices])
    saved["base_value"] = contributions[:, -1]
    saved.to_parquet(out_dir / "shap_values.parquet", index=False)
    pd.DataFrame({"feature": model.feature_name(),
                  "mean_abs_shap": np.abs(contributions[:, :-1]).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(out_dir / "shap_importance.csv", index=False)
    inputs = x.copy()
    inputs.insert(0, "edge_id", np.asarray(val_ids)[indices])
    inputs.to_parquet(out_dir / "shap_inputs.parquet", index=False)
    top = int(np.argmax(scores))
    local = model.predict(x_val.iloc[[top]], pred_contrib=True, num_threads=threads)[0]
    pd.DataFrame({"feature": model.feature_name(),
                  "value": [None if pd.isna(v) else str(v) for v in x_val.iloc[top]],
                  "shap_log_odds": local[:-1]}).to_csv(out_dir / "shap_top_alert.csv", index=False)
    write_json(out_dir / "shap_metadata.json", {
        "split": "val", "rows": len(indices), "sampling": "uniform without replacement",
        "seed": seed, "method": "LightGBM pred_contrib (TreeSHAP)",
        "units": "raw margin / log-odds; not probability changes or causal effects",
        "top_alert_edge_id": int(np.asarray(val_ids)[top]), "top_alert_score": float(scores[top]),
        "top_alert_base_value": float(local[-1]),
    })


def train_baseline(data_dir, run_dir, config=None):
    """Validation-only search, then test evaluation once for the frozen winner.

    Each run uses a new directory. Smoke runs use CSV prefixes, never report data.
    """
    config = config or TrainConfig()
    config.validate()
    import pyarrow  # fail before expensive fitting if export engine is unavailable

    started = time.perf_counter()
    data_dir, run_dir = Path(data_dir).resolve(), Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {**asdict(config), "data_dir": str(data_dir),
                                        "metric_definition": "sklearn average_precision_score",
                                        "selection": "highest validation AP; first candidate wins ties"})
    versions = {p: importlib.metadata.version(p) for p in
                ("lightgbm", "pandas", "numpy", "scikit-learn", "pyarrow")}
    write_json(run_dir / "environment.json", {"python": platform.python_version(), **versions})
    write_json(run_dir / "status.json", {"state": "running", "smoke": config.smoke})
    train, features = load_split(data_dir, "train", config.features,
                                 limit=100_000 if config.smoke else None)
    val, _ = load_split(data_dir, "val", config.features, features,
                        limit=50_000 if config.smoke else None)
    check_splits(train, val)
    meta = fit_preprocessor(train, features)
    write_json(run_dir / "preprocessing.json", meta)
    x_train, x_val = transform(train, meta), transform(val, meta)
    y_train, y_val = train.is_laundering.to_numpy(), val.is_laundering.to_numpy()
    train_keys = train[["source_row", "timestamp"]].copy()
    val_keys = val[["source_row", "timestamp"]].copy()
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    summaries = {name: _split_summary(frame, data_dir / FILES[name])
                 for name, frame in (("train", train), ("val", val))}
    del train, val
    gc.collect()
    print(f"Data: {len(y_train):,} train, {len(y_val):,} validation; {len(features)} predictors; "
          f"negative/positive ratio={ratio:.2f}", flush=True)
    candidates = candidate_parameters(ratio, config.trials, config.seed)
    write_json(run_dir / "candidates.json", candidates)
    best_ap, best_trial, best_scores = -1.0, None, None
    results = []
    for i, params in enumerate(candidates):
        trial_started = time.perf_counter()
        history = {}
        estimator = lgb.LGBMClassifier(
            objective="binary", metric="average_precision", n_estimators=config.max_rounds,
            random_state=config.seed, n_jobs=config.threads, verbosity=-1,
            deterministic=True, force_col_wise=True, **params)
        estimator.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_names=["val"],
                      categorical_feature=list(meta["categories"]), callbacks=[
                          lgb.early_stopping(config.patience, first_metric_only=True, verbose=False),
                          lgb.record_evaluation(history)])
        scores = estimator.predict_proba(x_val)[:, 1]
        ap = float(average_precision_score(y_val, scores))
        row = {"trial": i, **params, "val_auprc": ap,
               "best_iteration": int(estimator.best_iteration_),
               "seconds": round(time.perf_counter() - trial_started, 3)}
        results.append(row)
        pd.DataFrame(results).to_csv(run_dir / "tuning_results.csv", index=False)
        write_json(run_dir / f"history_trial_{i:02d}.json", history)
        if ap > best_ap:
            best_ap, best_trial, best_scores = ap, row, scores
            estimator.booster_.save_model(str(run_dir / "model.txt"), num_iteration=estimator.best_iteration_)
        print(f"Trial {i + 1}/{len(candidates)}: val AP={ap:.6f}, "
              f"trees={row['best_iteration']}, weight={params['scale_pos_weight']:.2f}", flush=True)
        del estimator
        gc.collect()
    threshold = choose_threshold(y_val, best_scores)
    # Freeze choices before loading test. No train+validation refit, preserving
    # the selected model/threshold pairing and the GNN's validation protocol.
    write_json(run_dir / "selection.json", {"trial": best_trial, "threshold": threshold,
                                           "training_negative_positive_ratio": ratio})
    model, saved_meta = load_model(run_dir)
    restored = model.predict(x_val, validate_features=True, num_threads=config.threads)
    np.testing.assert_allclose(restored, best_scores, rtol=1e-10, atol=1e-12)
    save_interpretation(model, x_val, val_keys.source_row, best_scores, run_dir,
                        config.shap_rows, config.seed, config.threads)
    del x_train, x_val
    gc.collect()
    print("Model and threshold frozen; evaluating test.", flush=True)
    test, _ = load_split(data_dir, "test", config.features, features,
                         limit=50_000 if config.smoke else None)
    check_splits(val_keys, test)
    check_splits(train_keys, test)
    summaries["test"] = _split_summary(test, data_dir / FILES["test"])
    write_json(run_dir / "data_summary.json", summaries)
    test_scores = predict(model, test, saved_meta, config.threads)
    metrics = {"val": baseline_metrics(y_val, best_scores, threshold),
               "test": baseline_metrics(test.is_laundering, test_scores, threshold),
               "summary": {"model": "lightgbm", "features": config.features,
                           "seed": config.seed, "selected_trial": best_trial["trial"],
                           "best_iteration": best_trial["best_iteration"], "smoke": config.smoke,
                           "reload_predictions_verified": True}}
    predictions = pd.concat([
        pd.DataFrame({"edge_id": val_keys.source_row.to_numpy(), "split": "val", "score": best_scores}),
        pd.DataFrame({"edge_id": test.source_row.to_numpy(), "split": "test", "score": test_scores})
    ], ignore_index=True)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    for split, y, scores in (("val", y_val, best_scores), ("test", test.is_laundering, test_scores)):
        p, r, _ = precision_recall_curve(y, scores)
        pd.DataFrame({"precision": p, "recall": r}).to_csv(run_dir / f"pr_curve_{split}.csv", index=False)
    metrics["summary"]["minutes"] = round((time.perf_counter() - started) / 60, 3)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "status.json", {"state": "complete", "smoke": config.smoke})
    return metrics


def _split_summary(frame, path):
    stat = path.stat()
    return {"rows": len(frame), "positives": int(frame.is_laundering.sum()),
            "start": str(frame.timestamp.min()), "end": str(frame.timestamp.max()),
            "missing_by_column": {c: int(n) for c, n in frame.isna().sum().items() if n},
            "source_path": str(path), "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns}
