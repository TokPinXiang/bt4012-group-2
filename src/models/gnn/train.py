"""Trains one GNN run, scores validation and test, and saves the results.

- What each stage sees: training sees only training payments; validation scoring also sees training payments as
  background; test scoring sees all payments. Background payments give context, but their labels are never fed to
  the model (as in Altman et al. 2023 and Egressy et al. 2024).
- What it saves: each run gets its own folder (settings, training history, model, scores, metrics) and one row in
  results.csv. A finished run is skipped; an interrupted run restarts from its last completed epoch.
- No look-ahead (time-respecting runs): in training, payments are grouped in time order and each group only sees
  context up to its last payment. The validation check used for early stopping works the same way.
- Reported scores (time-respecting runs): made minute by minute, like a bank checking at the end of each minute,
  using only payments up to that minute.
- Metrics printed here are for early stopping and logging only; the team comparison uses the saved scores file.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from .data import build_graph
from .model import EdgeGNN
from .sampler import EdgeSampler

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULTS = dict(model="gin", features="all", ports=True, rmp=True, ego=True, eu=True, causal=False, pair_agg=False,
                hidden=64, layers=2, fanout=[100, 100], batch=4096, epochs=15, patience=4, lr=1e-3, dropout=0.1,
                pos_weight=6.0, neg_frac=0.2)
KS = (500, 1000, 5000)


def metrics(y, s, threshold=None):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
    out = {"auprc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))}
    if threshold is None:  # best-F1 threshold (chosen on validation, then reused on test)
        p, r, t = precision_recall_curve(y, s)
        f = 2 * p * r / np.clip(p + r, 1e-12, None)
        threshold = float(t[int(np.nanargmax(f[:-1]))]) if len(t) else 0.5
    out["threshold"] = float(threshold)
    out["f1"] = float(f1_score(y, s >= threshold, zero_division=0))
    for k in KS:
        out[f"recall@{k}"] = float(y[np.argsort(-s)[:k]].sum() / max(y.sum(), 1))
    return out


def pna_degree_hist(g, train_eids):
    """In-degree histogram for PNA's scalers, from the training graph only."""
    d = torch.bincount(g["edge_index"][1, torch.from_numpy(train_eids)], minlength=g["n_nodes"])
    return torch.bincount(d.clamp(max=500))


def make_model(c, edge_dim, deg=None):
    return EdgeGNN(edge_dim, c["model"], c["hidden"], c["layers"], c["eu"], c["rmp"], c["ego"], c["dropout"],
                   deg=deg, pair_agg=c["pair_agg"]).to(DEV)


def forward_batch(model, g, sampler, tgt, fanout, causal, x=None):
    """Sample the neighbourhood of target edges `tgt` and return their logits (and the sampled edge ids)."""
    mt = g["ts"].numpy()[tgt].max() if causal else None  # time-respecting: no context later than the batch's last payment
    sub_eids, nid, sub_ei, n_t = sampler.sample(tgt, fanout, max_time=mt)
    sub_ei = sub_ei.to(DEV)
    attr = g["edge_attr"][sub_eids].to(DEV) if x is None else x
    return model(sub_ei, attr, len(nid), target_nodes=sub_ei[:, :n_t].reshape(-1), n_out=n_t), sub_eids


def batches(eids, ts, size, by_minute):
    """Consecutive chunks of time-sorted target edges; by_minute: never mix two minutes in one chunk."""
    if not by_minute:
        return [eids[i:i + size] for i in range(0, len(eids), size)]
    cuts = np.flatnonzero(np.diff(ts[eids])) + 1
    return [m[i:i + size] for m in np.split(eids, cuts) for i in range(0, len(m), size)]


@torch.no_grad()
def score(model, g, sampler, eids, c, strict=False, seed=12345):
    """P(laundering) for target edges (time-sorted), with a fixed sampling seed so scores are repeatable.
    strict (causal runs): one minute per batch, so no payment sees context from a later minute."""
    model.eval()
    sampler.rng = np.random.default_rng(seed)
    out = []
    for tgt in batches(eids, g["ts"].numpy(), c["batch"] * 2, strict and c["causal"]):
        logits, _ = forward_batch(model, g, sampler, tgt, c["fanout"], c["causal"])
        out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def val_halves(y, s, ts):
    """Validation AUPRC on its earlier and later half (by time): a change counts only if it wins on both."""
    cut = np.median(ts)
    return [float(average_precision_score(y[m], s[m])) for m in (ts <= cut, ts > cut)]


def train_run(name, cfg, data_dir, out_dir, seed=0, cache_dir=None, max_eval=0, g=None, _stop_after=None):
    """_stop_after: tests only, simulates a disconnect after that epoch's checkpoint."""
    c = {**DEFAULTS, **cfg}
    run_dir = os.path.join(out_dir, "runs", name)
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        print(f"{name}: already finished, skipped")
        return json.load(open(os.path.join(run_dir, "metrics.json")))["summary"]
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(seed)
    t0 = time.time()
    if g is None:
        g = build_graph(data_dir, c["features"], c["ports"], cache_dir)
    g.pop("table", None)
    split, y, ts = g["split"].numpy(), g["y"].numpy(), g["ts"].numpy()
    views = {0: split == 0, 1: split <= 1, 2: np.ones_like(split, bool)}
    samplers = {k: EdgeSampler(g["edge_index"], g["n_nodes"], v, seed=seed + k, ts=ts if c["causal"] else None)
                for k, v in views.items()}
    tr_e, va_e, te_e = (np.where(split == s)[0] for s in range(3))
    if max_eval:  # quick checks only
        va_e, te_e = va_e[:max_eval], te_e[:max_eval]
    deg = pna_degree_hist(g, tr_e) if c["model"] == "pna" else None
    model = make_model(c, g["edge_attr"].shape[1], deg)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    conf = {**c, "name": name, "seed": seed, "data_dir": data_dir, "edge_dim": int(g["edge_attr"].shape[1]),
            "feat_names": g["feat_names"], "n_params": sum(p.numel() for p in model.parameters()),
            "n_nodes": g["n_nodes"], "n_edges": int(len(y)), "device": str(DEV)}
    json.dump(conf, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    rng = np.random.default_rng(seed)
    hist, best, best_state, bad, start = [], -1.0, None, 0, 0
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    if os.path.exists(ckpt):  # resume an interrupted run
        k = torch.load(ckpt, weights_only=False, map_location="cpu")
        model.load_state_dict(k["model"]), opt.load_state_dict(k["opt"])
        hist, best, best_state, bad, start = k["hist"], k["best"], k["best_state"], k["bad"], k["epoch"] + 1
        rng.bit_generator.state = k["rng"]
        samplers[0].rng.bit_generator.state = k["sampler_rng"]
        torch.set_rng_state(k["torch_rng"])
        print(f"{name}: resuming after epoch {k['epoch']}")
    print(f"{name}: {conf['n_params']} params, {conf['edge_dim']} edge features, {len(tr_e)} train edges", flush=True)
    w = torch.tensor([1.0, c["pos_weight"]], device=DEV)
    pos, neg = tr_e[y[tr_e] == 1], tr_e[y[tr_e] == 0]
    for ep in range(start, c["epochs"]):
        if bad >= c["patience"]:
            break
        model.train()
        te0 = time.time()
        ep_e = np.concatenate([pos, rng.choice(neg, int(len(neg) * c["neg_frac"]), replace=False)])
        if c["causal"]:  # time-contiguous batches (edges are time-sorted), batches in random order
            ep_e = np.sort(ep_e)
            starts = rng.permutation(np.arange(0, len(ep_e), c["batch"]))
        else:
            rng.shuffle(ep_e)
            starts = np.arange(0, len(ep_e), c["batch"])
        tot = 0.0
        for i in starts:
            tgt = ep_e[i:i + c["batch"]]
            logits, _ = forward_batch(model, g, samplers[0], tgt, c["fanout"], c["causal"])
            loss = F.cross_entropy(logits, torch.from_numpy(y[tgt]).to(DEV), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        s_va = score(model, g, samplers[1], va_e, c)
        m = metrics(y[va_e], s_va)
        hist.append({"epoch": ep, "loss": tot / max(len(starts), 1), "val_auprc": m["auprc"], "val_f1": m["f1"],
                     "secs": round(time.time() - te0)})
        print(json.dumps(hist[-1]), flush=True)
        if m["auprc"] > best:
            best, bad = m["auprc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "best": best,
                    "best_state": best_state, "bad": bad, "epoch": ep, "rng": rng.bit_generator.state,
                    "sampler_rng": samplers[0].rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, ckpt)
        if _stop_after == ep:
            raise KeyboardInterrupt("simulated disconnect")
    pd.DataFrame(hist).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": conf}, os.path.join(run_dir, "model.pt"))
    t1 = time.time()
    s_va = score(model, g, samplers[1], va_e, c, strict=True)
    s_te = score(model, g, samplers[2], te_e, c, strict=True)
    m_va = metrics(y[va_e], s_va)
    m_te = metrics(y[te_e], s_te, threshold=m_va["threshold"])  # threshold fixed on validation
    eid = g["edge_id"].numpy()
    pd.concat([pd.DataFrame({"edge_id": eid[va_e], "split": "val", "score": s_va}),
               pd.DataFrame({"edge_id": eid[te_e], "split": "test", "score": s_te})]
              ).to_parquet(os.path.join(run_dir, "predictions.parquet"), index=False)
    h1, h2 = val_halves(y[va_e], s_va, ts[va_e])
    summary = {"name": name, "seed": seed, "model": c["model"], "features": c["features"], "ports": c["ports"],
               "rmp": c["rmp"], "ego": c["ego"], "eu": c["eu"], "causal": c["causal"], "pair_agg": c["pair_agg"],
               "epochs_run": len(hist), "minutes": round((sum(h["secs"] for h in hist) + time.time() - t1) / 60, 1),
               "scoring": "minute by minute" if c["causal"] else "whole graph (not time-respecting)",
               "val_auprc": m_va["auprc"], "val_auprc_half1": h1, "val_auprc_half2": h2, "val_f1": m_va["f1"],
               **{f"test_{k}": m_te[k] for k in ("auprc", "roc_auc", "f1", *[f"recall@{k}" for k in KS])}}
    res = os.path.join(out_dir, "results.csv")  # results row first: metrics.json marks the run as finished
    old = pd.read_csv(res) if os.path.exists(res) else pd.DataFrame()
    pd.concat([old[old["name"] != name] if len(old) else old, pd.DataFrame([summary])],
              ignore_index=True).to_csv(res, index=False)
    json.dump({"val": m_va, "test": m_te, "summary": summary}, open(os.path.join(run_dir, "metrics.json"), "w"),
              indent=2)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(json.dumps(summary, indent=2))
    return summary
