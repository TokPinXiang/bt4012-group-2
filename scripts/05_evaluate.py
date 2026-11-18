"""Compare the saved LightGBM and GNN scores, without retraining (owner: Yu Heng).

  python scripts/05_evaluate.py
  python scripts/05_evaluate.py --gbt-predictions outputs/predictions/predictions_gbt.parquet --gnn-predictions outputs/predictions_gnn.parquet
  python scripts/05_evaluate.py --no-typology --ks 500 1000 5000

Default discovery checks outputs/predictions/ and outputs/. Explicit paths are
required if both locations contain a file with the same name. Paths are relative
to the project root, independently of the current working directory.
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def project_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def discover(filename):
    matches = [p for p in (ROOT / "outputs" / "predictions" / filename, ROOT / "outputs" / filename) if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} under outputs/predictions or outputs; found {len(matches)}. "
                         "Specify --gbt-predictions / --gnn-predictions explicitly.")
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--gbt-predictions")
    ap.add_argument("--gnn-predictions")
    ap.add_argument("--gbt-metrics", help="optional saved metrics JSON; defaults to the paired file beside predictions")
    ap.add_argument("--gnn-metrics", help="optional saved metrics JSON; defaults to the paired file beside predictions")
    ap.add_argument("--patterns", help="default: pattern_labels.csv in --data-dir")
    ap.add_argument("--no-typology", action="store_true")
    ap.add_argument("--ks", nargs="+", type=int, default=[500, 1000, 5000])
    ap.add_argument("--out-dir", default="outputs/evaluation")
    ap.add_argument("--run-name", help="new folder name; defaults to a Singapore-time timestamp")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    if a.no_typology and a.patterns:
        ap.error("Use either --patterns or --no-typology")
    if any(k < 1 for k in a.ks) or len(set(a.ks)) != len(a.ks):
        ap.error("--ks must contain distinct positive integers")
    name = a.run_name or "comparison_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    if name in (".", "..") or any(c in name for c in ("/", "\\", ":")):
        ap.error("--run-name must be a folder name, not a path")
    predictions, references = {}, {}
    for model, prediction_arg, metric_arg, filename in (
        ("lightgbm", a.gbt_predictions, a.gbt_metrics, "predictions_gbt.parquet"),
        ("gnn", a.gnn_predictions, a.gnn_metrics, "predictions_gnn.parquet"),
    ):
        p = project_path(prediction_arg) if prediction_arg else discover(filename)
        if not p.is_file():
            raise FileNotFoundError(p)
        paired = project_path(metric_arg) if metric_arg else p.with_name(p.stem + "_metrics.json")
        if metric_arg and not paired.is_file():
            raise FileNotFoundError(paired)
        predictions[model] = p
        references[model] = paired if paired.is_file() else None
        if references[model] is None:
            print(f"{model}: no paired metrics JSON; threshold will be selected from validation scores.")
    pattern_path = None if a.no_typology else (project_path(a.patterns) if a.patterns else project_path(a.data_dir) / "pattern_labels.csv")
    if pattern_path is not None and not pattern_path.is_file():
        raise FileNotFoundError(f"{pattern_path}; supply --patterns or explicitly use --no-typology")
    from src.eval.compare import run_comparison
    out_dir = project_path(a.out_dir) / name
    report = run_comparison(project_path(a.data_dir), predictions, out_dir,
                            pattern_path=pattern_path, references=references,
                            ks=tuple(sorted(a.ks)), make_plots=not a.no_plots)
    import pandas as pd
    table = pd.DataFrame({model: result["test"] for model, result in report["models"].items()}).T
    print("\nTest results (each model uses its own validation-selected threshold):")
    print(table[["auprc", "roc_auc", "precision", "recall", "f1", "alerts"]].round(6).to_string())
    print(f"\nReference checks: {report['reference_checks']}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
