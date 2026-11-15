"""Train or reload the local LightGBM baseline (owner: Yu Heng).

  python scripts/03_train_gbt.py --smoke
  python scripts/03_train_gbt.py --features all --trials 12
  python scripts/03_train_gbt.py --publish-run outputs/gbt/<completed-run>
  python scripts/03_train_gbt.py --predict-run outputs/gbt/<run> --input payments.csv --output scores.parquet

Paths are resolved against the project root, regardless of working directory.
Cross-model evaluation belongs to stage 05 and is not run here.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def publish(run_dir, overwrite=False):
    """Publish completed full runs only; never replace results silently."""
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete" or status.get("smoke"):
        raise ValueError("Only completed, non-smoke runs can be published")
    out = ROOT / "outputs" / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    pairs = [("predictions.parquet", "predictions_gbt.parquet"),
             ("metrics.json", "predictions_gbt_metrics.json"),
             ("selection.json", "predictions_gbt_selection.json")]
    if not overwrite and any((out / target).exists() for _, target in pairs):
        raise FileExistsError("Published baseline exists. Use --overwrite-published to replace it.")
    for source, target in pairs:
        shutil.copy2(run_dir / source, out / target)
    print(f"Published baseline from {run_dir} to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--publish-run", help="publish an existing completed full run; no fitting")
    mode.add_argument("--predict-run", help="load a saved model and score --input; no fitting")
    ap.add_argument("--input", help="CSV with source_row and the saved model's predictors")
    ap.add_argument("--output", help="new output Parquet for --predict-run")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="outputs/gbt")
    ap.add_argument("--run-name", help="unique output folder name (otherwise timestamped)")
    ap.add_argument("--features", choices=["row", "all"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--max-rounds", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shap-rows", type=int, default=1000, help="uniform validation sample; 0 disables SHAP")
    ap.add_argument("--smoke", action="store_true", help="CSV prefixes, at most 3 trials/30 rounds; not report results")
    ap.add_argument("--publish", action="store_true", help="publish completed full run as predictions_gbt.parquet")
    ap.add_argument("--overwrite-published", action="store_true")
    a = ap.parse_args()
    if a.smoke and a.publish:
        ap.error("Smoke predictions cannot be published")
    if (a.publish_run or a.predict_run) and (a.smoke or a.publish):
        ap.error("Do not combine existing-run modes with --smoke or --publish")
    if a.publish_run:
        publish(project_path(a.publish_run), a.overwrite_published)
        return
    from src.models.gbt import TrainConfig, load_model, predict, train_baseline
    if a.predict_run:
        if not a.input or not a.output:
            ap.error("--predict-run requires --input and --output")
        import pandas as pd
        model, metadata = load_model(project_path(a.predict_run))
        output = project_path(a.output)
        if output.exists():
            raise FileExistsError(output)
        frame = pd.read_csv(project_path(a.input), usecols=["source_row", *metadata["features"]],
                            dtype={"source_row": "int64", **{c: "string" for c in metadata["categories"]}})
        if frame.empty or not frame.source_row.is_unique:
            raise ValueError("Prediction input must contain unique source_row IDs and at least one row")
        scores = predict(model, frame, metadata, a.threads)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"edge_id": frame.source_row, "score": scores}).to_parquet(output, index=False)
        print(f"Wrote {len(scores):,} predictions to {output}")
        return
    if a.input or a.output:
        ap.error("--input and --output are only used with --predict-run")
    cfg = TrainConfig(features=a.features, seed=a.seed, trials=a.trials, max_rounds=a.max_rounds,
                      patience=a.patience, threads=a.threads, shap_rows=a.shap_rows, smoke=a.smoke)
    cfg.validate()
    if a.smoke:
        cfg.trials, cfg.max_rounds = min(cfg.trials, 3), min(cfg.max_rounds, 30)
        cfg.patience, cfg.shap_rows = min(cfg.patience, 5), min(cfg.shap_rows, 100)
    name = a.run_name or f"{'smoke' if a.smoke else 'lgb'}_{a.features}_seed{a.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if name in (".", "..") or "/" in name or "\\" in name or ":" in name:
        ap.error("--run-name must be a folder name, not a path")
    run_dir = project_path(a.out_dir) / name
    print(f"Run: {run_dir}", flush=True)
    result = train_baseline(project_path(a.data_dir), run_dir, cfg)
    print(json.dumps(result, indent=2), flush=True)
    if a.publish:
        publish(run_dir, a.overwrite_published)


if __name__ == "__main__":
    main()
