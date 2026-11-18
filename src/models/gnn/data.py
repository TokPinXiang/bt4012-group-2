"""Load the team's processed data and turn it into graph tensors for the GNN.

Input (stage A output, read-only):
  - the shared files data/processed/{transactions,split,features}.parquet (README contract), or
  - per-split train.csv / validation.csv / test.csv (all columns in one table per split).
Graph: accounts are nodes, transactions are directed edges (repeated pairs stay separate edges).
Edge features: the stage-A feature columns, plus two port numbers. Scaling is fitted on the train split only.

Feature sets (for the ablation):
  row = columns that describe the payment itself (amounts, currencies, format, time, cross-bank flags)
  all = row + the account-history columns (names starting src_, dst_, pair_, reverse_pair_)
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

SPLITS = {"train": 0, "val": 1, "validation": 1, "test": 2}
CSV_FILES = ("train.csv", "validation.csv", "test.csv")
CSV_CATEGORICAL = ("payment_format", "payment_currency", "receiving_currency")
HISTORY_PREFIXES = ("src_", "dst_", "pair_", "reverse_pair_")
NOT_FEATURES = {"source_row", "tx_id", "edge_id", "timestamp", "is_laundering", "label", "split",
                "from_bank", "to_bank", "from_account", "to_account", "src_account_id", "dst_account_id",
                "src_node_id", "dst_node_id"}


def is_history(name):
    return name.startswith(HISTORY_PREFIXES)


def _not_feature(c):
    c = c.lower()
    return c in NOT_FEATURES or c.endswith("_id") or "pattern" in c or "typology" in c


def _from_csv(d):
    """Per-split CSVs from stage A. edge_id = source_row (1-based row in HI-Small_Trans.csv)."""
    ids = {c: str for c in ("tx_id", "from_bank", "from_account", "to_bank", "to_account",
                            "src_account_id", "dst_account_id")}
    parts = []
    for s, f in enumerate(CSV_FILES):
        p = pd.read_csv(os.path.join(d, f), dtype=ids)
        p["split"] = s
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({"edge_id": df["source_row"].astype(np.int64), "timestamp": pd.to_datetime(df["timestamp"]),
                        "from_account": df["src_account_id"], "to_account": df["dst_account_id"],
                        "split": df["split"].astype(np.int8), "label": df["is_laundering"].astype(np.int8)})
    cats = [c for c in CSV_CATEGORICAL if c in df.columns]
    num = [c for c in df.columns if c not in cats and c != "split" and not _not_feature(c)
           and pd.api.types.is_numeric_dtype(df[c])]
    feats = df[num + cats].copy()
    show = {"amount": df["amount_paid"].values, "currency": df["payment_currency"].values} \
        if {"amount_paid", "payment_currency"} <= set(df.columns) else {}
    return out, feats, num, cats, show


def _from_parquet(d):
    """README contract: transactions.parquet + split.parquet + features.parquet, joined on edge_id."""
    t = pd.read_parquet(os.path.join(d, "transactions.parquet"))
    s = pd.read_parquet(os.path.join(d, "split.parquet"))
    f = pd.read_parquet(os.path.join(d, "features.parquet"))
    t = t.merge(s[["edge_id", "split"]], on="edge_id", how="inner", validate="1:1")
    t = t.merge(f, on="edge_id", how="inner", validate="1:1", suffixes=("", "_feat"))
    if len(t) != len(s):
        raise ValueError(f"{len(s) - len(t)} edges in split.parquet are missing from the other files")
    out = pd.DataFrame({"edge_id": t["edge_id"].astype(np.int64), "timestamp": pd.to_datetime(t["timestamp"]),
                        "from_account": t["from_account"].astype(str), "to_account": t["to_account"].astype(str),
                        "split": t["split"].map(lambda x: SPLITS[x] if isinstance(x, str) else x).astype(np.int8),
                        "label": t["label"].astype(np.int8)})
    fcols = [c for c in f.columns if c != "edge_id" and not _not_feature(c)]
    num = [c for c in fcols if pd.api.types.is_numeric_dtype(t[c]) and not pd.api.types.is_bool_dtype(t[c])]
    num += [c for c in fcols if pd.api.types.is_bool_dtype(t[c])]
    cats = [c for c in fcols if c not in num]
    if "payment_format" in t.columns and "payment_format" not in cats:
        cats.append("payment_format")
    feats = t[num + cats].copy()
    show = {"amount_usd": t["amount_usd"].values} if "amount_usd" in t.columns else {}
    if "currency" in t.columns:
        show["currency"] = t["currency"].values
    return out, feats, num, cats, show


def read_table(data_dir, cache_dir=None):
    """One row per transaction, sorted by (timestamp, edge_id). Cached as parquet (keyed on the input files)."""
    if all(os.path.exists(os.path.join(data_dir, f)) for f in ("transactions.parquet", "split.parquet",
                                                               "features.parquet")):
        files, reader = ("transactions.parquet", "split.parquet", "features.parquet"), _from_parquet
    elif all(os.path.exists(os.path.join(data_dir, f)) for f in CSV_FILES):
        files, reader = CSV_FILES, _from_csv
    else:
        raise FileNotFoundError(f"{data_dir}: need transactions/split/features.parquet or {', '.join(CSV_FILES)}")
    h = hashlib.md5()
    for f in files:  # key = size + first and last MB of each input file (file times change when Colab copies them)
        path = os.path.join(data_dir, f)
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            h.update(f"{f}:{size}".encode() + fh.read(2**20))
            fh.seek(max(size - 2**20, 0))
            h.update(fh.read())
    cache = os.path.join(cache_dir, f"table_{h.hexdigest()[:12]}.parquet") if cache_dir else None
    if cache and os.path.exists(cache) and os.path.exists(cache.replace(".parquet", ".json")):
        df = pd.read_parquet(cache)
        meta = json.load(open(cache.replace(".parquet", ".json")))
        return df, meta["numeric"], meta["categorical"]
    out, feats, num, cats, show = reader(data_dir)
    if not out["edge_id"].is_unique:
        raise ValueError("edge_id is not unique")
    df = pd.concat([out, feats.reset_index(drop=True)], axis=1)
    for k, v in show.items():
        df["show_" + k] = v
    for c in cats:
        df[c] = df[c].astype(str)
    df = df.sort_values(["timestamp", "edge_id"], kind="stable").reset_index(drop=True)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(cache + ".tmp", index=False)
        os.replace(cache + ".tmp", cache)  # a half-written cache is never picked up
        json.dump({"numeric": num, "categorical": cats}, open(cache.replace(".parquet", ".json"), "w"))
    return df, num, cats


def add_ports(src, dst, ts):
    d = pd.DataFrame({"src": src, "dst": dst, "ts": ts})
    out = {}
    for node, nbr, name in (("src", "dst", "out_port"), ("dst", "src", "in_port")):
        first = d.groupby([node, nbr], sort=False)["ts"].min().reset_index()
        first[name] = first.groupby(node)["ts"].rank(method="min") - 1
        out[name] = d[[node, nbr]].merge(first, on=[node, nbr], how="left")[name].to_numpy(np.float32)
    return out["out_port"], out["in_port"]


def build_graph(data_dir, features="all", ports=True, cache_dir=None):
    """Graph tensors for training. Edges are in time order; edge k of every tensor is row k of `table`."""
    df, num, cats = read_table(data_dir, cache_dir)
    if features not in ("row", "all"):
        raise ValueError(features)
    num = [c for c in num if features == "all" or not is_history(c)]
    accs = pd.Index(pd.unique(np.concatenate([df["from_account"].values, df["to_account"].values])))
    src, dst = accs.get_indexer(df["from_account"]), accs.get_indexer(df["to_account"])
    ts = (df["timestamp"].values.astype("datetime64[s]").astype(np.int64))
    tr = df["split"].values == 0
    X = np.nan_to_num(df[num].to_numpy(np.float32, na_value=0.0).copy())
    names = list(num)
    if ports:
        op, ip = add_ports(src, dst, ts)
        X = np.concatenate([X, np.log1p(np.stack([op, ip], 1))], 1)
        names += ["out_port", "in_port"]
    heavy = np.abs(X[tr]).max(0) > 100  # heavy-tailed counts, amounts, durations: signed log (decided on train only)
    X[:, heavy] = np.sign(X[:, heavy]) * np.log1p(np.abs(X[:, heavy]))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6  # scaling fitted on train only
    X = np.clip((X - mu) / sd, -10, 10)  # hub accounts give huge z-scores even after the log
    blocks = [X]
    for c in cats:
        levels = sorted(df[c].unique())
        codes = pd.Categorical(df[c], categories=levels).codes
        blocks.append(np.eye(len(levels), dtype=np.float32)[codes])
        names += [f"{c} = {v}" for v in levels]
    X = np.concatenate(blocks, 1).astype(np.float32)
    return {
        "edge_index": torch.from_numpy(np.stack([src, dst]).astype(np.int64)),
        "edge_attr": torch.from_numpy(X),
        "y": torch.from_numpy(df["label"].values.astype(np.int64)),
        "split": torch.from_numpy(df["split"].values.astype(np.int64)),
        "edge_id": torch.from_numpy(df["edge_id"].values.astype(np.int64)),
        "ts": torch.from_numpy(ts),
        "n_nodes": len(accs), "feat_names": names, "raw_numeric": num, "table": df,
    }
