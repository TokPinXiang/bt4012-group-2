"""Stage 04: train the GNN edge classifier, write predictions_gnn.parquet and the agent's explanations.

  python scripts/04_train_gnn.py                          # every run in configs/config.yaml (finished runs skipped)
  python scripts/04_train_gnn.py --runs gin_plain multigin
  python scripts/04_train_gnn.py --runs pna_final --seeds 1 2
  python scripts/04_train_gnn.py --publish                # predictions file from the final run's best-val seed
  python scripts/04_train_gnn.py --explain [--pattern_labels data/raw/pattern_labels.csv]
  python scripts/04_train_gnn.py --watchlist --pattern_labels data/raw/pattern_labels.csv   # watchlist evaluation
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.gnn.data import build_graph, read_table
from src.models.gnn.explain import explain
from src.models.gnn.train import DEFAULTS, train_run
from src.models.gnn.watchlist import evaluate as watchlist_eval


def p(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def best_final(cfg):
    res = pd.read_csv(p(os.path.join(cfg["out_dir"], "results.csv")))
    fin = res[res["name"].str.startswith(cfg["final"] + "_seed")]
    if fin.empty:
        raise SystemExit(f"no finished {cfg['final']} run yet")
    return fin.sort_values("val_auprc").iloc[-1]["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--runs", nargs="*", help="run names from the config (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, help="override the seeds of the selected runs")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--pattern_labels", default=None, help="evaluation only: adds the group summary of explanations")
    ap.add_argument("--watchlist", action="store_true", help="evaluate the watchlist follow-up (needs --pattern_labels)")
    ap.add_argument("--max_eval", type=int, default=0, help="quick checks only: cap val/test edges scored")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["gnn"]
    data_dir, out_dir, cache = p(cfg["data_dir"]), p(cfg["out_dir"]), p(cfg["cache_dir"])
    if not (a.publish or a.explain or a.watchlist):
        todo = [r for r in cfg["runs"] if not a.runs or r["name"] in a.runs]
        if a.runs and len(todo) != len(a.runs):
            raise SystemExit(f"unknown run in {a.runs}")
        g, g_key = None, None
        for r in todo:
            c = {**DEFAULTS, **cfg.get("defaults", {}), **{k: v for k, v in r.items() if k not in ("name", "seeds")}}
            for seed in (a.seeds or r.get("seeds", [0])):
                name = f"{r['name']}_seed{seed}"
                if os.path.exists(os.path.join(out_dir, "runs", name, "metrics.json")):
                    print(f"{name}: already finished, skipped")
                    continue
                key = (c["features"], c["ports"])
                if key != g_key:  # one graph per feature set, shared by the runs that use it
                    g, g_key = None, None
                    g, g_key = build_graph(data_dir, c["features"], c["ports"], cache), key
                train_run(name, c, data_dir, out_dir, seed=seed, cache_dir=cache, max_eval=a.max_eval, g=dict(g))
    if a.publish:
        name = best_final(cfg)
        pred = pd.read_parquet(os.path.join(out_dir, "runs", name, "predictions.parquet"))
        os.makedirs(os.path.dirname(p(cfg["predictions"])), exist_ok=True)
        pred[["edge_id", "split", "score"]].to_parquet(p(cfg["predictions"]), index=False)
        shutil.copy(os.path.join(out_dir, "runs", name, "metrics.json"),
                    os.path.join(os.path.dirname(p(cfg["predictions"])), "predictions_gnn_metrics.json"))
        print(f"wrote {cfg['predictions']} from {name} ({len(pred)} rows)")
    if a.explain:
        name = best_final(cfg)
        e = cfg.get("explain", {})
        explain(os.path.join(out_dir, "runs", name), p(cfg["explain_out"]), split=e.get("split", "test"),
                n_alerts=e.get("n_alerts", 100), pattern_labels=a.pattern_labels, cache_dir=cache, data_dir=data_dir)
    if a.watchlist:  # evaluation only: labels stand in for the agent's verdicts
        if not a.pattern_labels:
            raise SystemExit("--watchlist needs --pattern_labels (to split patterned / unpatterned recall)")
        name = best_final(cfg)
        run = os.path.join(out_dir, "runs", name)
        thr = json.load(open(os.path.join(run, "metrics.json")))["val"]["threshold"]
        pred = pd.read_parquet(os.path.join(run, "predictions.parquet"))
        pl = pd.read_csv(a.pattern_labels)
        key = "source_row" if "source_row" in pl.columns else "edge_id"
        patterned = pl.loc[pl["annotation_status"].eq("ANNOTATED"), key].astype("int64")
        table, _, _ = read_table(data_dir, cache)
        w = cfg.get("watchlist", {})
        res = watchlist_eval(table, pred, thr, patterned, split=w.get("split", "test"),
                             delay_hours=w.get("delay_hours", 24), caps=(w.get("max_prior_payments", 20), None))
        res.to_csv(os.path.join(out_dir, "watchlist_eval.csv"), index=False)
        print(f"watchlist follow-up on {name}, threshold {thr:.3f} (from validation):")
        print(res[["key", "confirmations", "max_prior_payments", "extra_alerts", "extra_alerts_pct_of_payments", "extra_alert_precision",
                   "unpatterned_recall_gnn", "unpatterned_recall_gnn+watchlist",
                   "all_recall_gnn+watchlist"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
