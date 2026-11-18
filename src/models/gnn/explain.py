"""Explains the GNN's top test alerts for the investigation agent (stage 06).

For each alert:
1. Replay the scoring pass exactly (same minute batches, same sampling seed), so the alert is explained on the same
   neighbourhood that produced its score in the predictions file.
2. Measure how much each feature of each payment in that neighbourhood pushed the score up or down
   (gradient x input; features are standardised, so the comparison point is an average payment).
3. Summarise: the share of the decision that came from the payment itself, from payments touching its two accounts
   (1 hop) and from payments one step further (2 hops); the payment's own features that mattered most; and the
   neighbouring payments that mattered most.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from .data import SPLITS, build_graph
from .sampler import EdgeSampler
from .train import DEV, batches, make_model, pna_degree_hist


def load_run(run_dir, cache_dir=None, data_dir=None):
    ck = torch.load(os.path.join(run_dir, "model.pt"), weights_only=False, map_location="cpu")
    c = ck["config"]
    g = build_graph(data_dir or c["data_dir"], c["features"], c["ports"], cache_dir)
    if g["feat_names"] != c["feat_names"]:
        raise ValueError("the data's features differ from the ones this model was trained on")
    split = g["split"].numpy()
    model = make_model(c, c["edge_dim"], pna_degree_hist(g, np.where(split == 0)[0]) if c["model"] == "pna" else None)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return g, model, c


def replay(model, g, c, split, wanted, seed=12345):
    """Re-run the scoring pass for one split exactly as train.score did (same minute batches, same sampling seed) and
    explain the wanted target rows. Returns {row: (sampled edge ids, attributions, hop of each edge, probability)};
    hop 0 = the payment itself, 1 = touches its accounts, 2 = one step further."""
    s = SPLITS[split]
    split_arr, ts = g["split"].numpy(), g["ts"].numpy()
    smp = EdgeSampler(g["edge_index"], g["n_nodes"], split_arr <= s, ts=ts if c["causal"] else None)
    smp.rng = np.random.default_rng(seed)
    eids = np.where(split_arr == s)[0]
    out = {}
    for tgt in batches(eids, ts, c["batch"] * 2, c["causal"]):
        mt = ts[tgt].max() if c["causal"] else None
        sub_eids, nid, sub_ei, n_t = smp.sample(tgt, c["fanout"], max_time=mt)
        pos = [j for j, e in enumerate(tgt) if int(e) in wanted]
        if not pos:
            continue
        x = g["edge_attr"][sub_eids].to(DEV).clone().requires_grad_(True)
        ei_dev = sub_ei.to(DEV)
        logits = model(ei_dev, x, len(nid), target_nodes=ei_dev[:, :n_t].reshape(-1), n_out=n_t)
        prob = torch.softmax(logits.detach(), -1)[:, 1].cpu().numpy()
        ei = sub_ei.numpy()
        for j in pos:
            (grad,) = torch.autograd.grad(logits[j, 1] - logits[j, 0], x, retain_graph=True)
            ends = [ei[0, j], ei[1, j]]
            h1 = np.isin(ei[0], ends) | np.isin(ei[1], ends)
            near = np.unique(ei[:, h1])
            h2 = ~h1 & (np.isin(ei[0], near) | np.isin(ei[1], near))
            keep = np.flatnonzero(h1 | h2)
            keep = np.concatenate([[j], keep[keep != j]])
            hop = np.where(h1[keep], 1, 2)
            hop[0] = 0
            attr = (grad * x).detach().cpu().numpy()[keep]
            out[int(tgt[j])] = (sub_eids.numpy()[keep], attr, hop, float(prob[j]))
    return out


def shares(attr, hop):
    imp = np.abs(attr).sum(1)
    tot = imp.sum() + 1e-12
    return imp, [float(imp[hop == h].sum() / tot) for h in (0, 1, 2)]


def explain(run_dir, out, split="test", n_alerts=100, pattern_labels=None, n_group=200, seed=0, cache_dir=None,
            data_dir=None):
    """Writes alerts_explained.json to `out` and the label files to <run_dir>/explain_eval/."""
    eval_out = os.path.join(run_dir, "explain_eval")
    os.makedirs(out, exist_ok=True)
    os.makedirs(eval_out, exist_ok=True)
    g, model, c = load_run(run_dir, cache_dir, data_dir)
    df = g["table"]
    thr = json.load(open(os.path.join(run_dir, "metrics.json")))["val"]["threshold"]
    pred = pd.read_parquet(os.path.join(run_dir, "predictions.parquet"))
    pred = pred[pred["split"] == split]
    row_of = pd.Series(np.arange(len(df)), index=df["edge_id"].values)
    names = g["feat_names"]
    show = {k[5:]: k for k in df.columns if k.startswith("show_")}
    fmt = "payment_format" if "payment_format" in df.columns else None

    def describe(r):
        d = {"from_account": str(r["from_account"]), "to_account": str(r["to_account"])}
        d.update({k: (round(float(r[v]), 2) if k.startswith("amount") else str(r[v])) for k, v in show.items()})
        d["payment_format"] = str(r[fmt]) if fmt else None
        d["time"] = str(r["timestamp"])
        return d

    top_alerts = pred.nlargest(n_alerts, "score")
    groups = {}
    if pattern_labels:  # evaluation only: outcome groups for the summary of where decisions come from
        pl = pd.read_csv(pattern_labels)
        key = "source_row" if "source_row" in pl.columns else "edge_id"
        patterned = set(pl.loc[pl["annotation_status"].eq("ANNOTATED"), key].astype(np.int64))
        y = df["label"].values[row_of[pred["edge_id"].values].values]
        lab = pred.assign(y=y, hit=pred["score"] >= thr, pattern=pred["edge_id"].isin(patterned))
        rng = np.random.default_rng(seed)
        pick = lambda d: d.iloc[rng.permutation(len(d))[:n_group]]
        groups = {"caught_pattern": pick(lab[(lab.y == 1) & lab.hit & lab.pattern]),
                  "missed_pattern": pick(lab[(lab.y == 1) & ~lab.hit & lab.pattern]),
                  "caught_nopattern": pick(lab[(lab.y == 1) & lab.hit & ~lab.pattern]),
                  "missed_nopattern": pick(lab[(lab.y == 1) & ~lab.hit & ~lab.pattern]),
                  "false_alarm_top": lab[lab.y == 0].nlargest(n_group, "score"),
                  "normal_random": pick(lab[lab.y == 0])}
    wanted = {int(row_of[int(i)]) for d in [top_alerts, *groups.values()] for i in d["edge_id"].values}
    res = replay(model, g, c, split, wanted)
    gap = max(abs(res[int(row_of[int(i)])][3] - s) for i, s in zip(top_alerts["edge_id"], top_alerts["score"]))
    print(f"largest gap between recomputed and reported score: {gap:.2e}")

    alerts, evals = [], []
    for _, p in top_alerts.iterrows():
        e = int(row_of[int(p.edge_id)])
        eids, attr, hop, _ = res[e]
        imp, (s0, s1, s2) = shares(attr, hop)
        r = df.iloc[e]
        ctx = []
        for k in np.argsort(-imp[1:])[:5] + 1:
            ctx.append({"edge_id": int(df["edge_id"].iat[eids[k]]), "hop": int(hop[k]),
                        "importance_share": round(float(imp[k] / imp.sum()), 4), **describe(df.iloc[eids[k]])})
        top = sorted(zip(names, attr[0]), key=lambda z: -abs(z[1]))[:5]
        alerts.append({"edge_id": int(p.edge_id), "score": round(float(p.score), 4),
                       "above_threshold": bool(p.score >= thr), **describe(r),
                       "share_from_own_transaction": round(s0, 4), "share_from_1hop": round(s1, 4),
                       "share_from_2hop": round(s2, 4),
                       "top_own_features": [{"feature": n, "value": float(r[n]) if n in g["raw_numeric"] else None,
                                             "pushes_towards": "laundering" if v > 0 else "normal",
                                             "attribution": round(float(v), 4)} for n, v in top],
                       "top_context_transactions": ctx})
        evals.append({"edge_id": int(p.edge_id), "label": int(df["label"].iat[e])})
    json.dump(alerts, open(os.path.join(out, "alerts_explained.json"), "w"), indent=2)
    ev = pd.DataFrame(evals)

    if pattern_labels:
        ev["in_pattern"] = ev["edge_id"].isin(patterned)
        rows = []
        for gname, d in groups.items():
            if not len(d):
                continue
            sh, nctx = [], []
            for t in d["edge_id"].values:
                _, attr, hop, _ = res[int(row_of[int(t)])]
                sh.append(shares(attr, hop)[1])
                nctx.append(len(hop) - 1)
            sh = np.array(sh)
            rows.append({"group": gname, "n": len(d), "mean_score": float(d["score"].mean()),
                         "share_own_tx": sh[:, 0].mean(), "share_1hop": sh[:, 1].mean(),
                         "share_2hop": sh[:, 2].mean(), "median_context_edges": float(np.median(nctx))})
        pd.DataFrame(rows).to_csv(os.path.join(eval_out, "explain_summary.csv"), index=False)
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    ev.to_csv(os.path.join(eval_out, "alerts_eval.csv"), index=False)
    print(f"wrote {len(alerts)} alerts to {out}/alerts_explained.json")
    return alerts
