"""Watchlist follow-up for the investigation agent (stage 06), applied after the GNN.

Once a laundering case is confirmed, its sender (keys="sender") or both accounts (keys="either") are watched. Later
payments of a watched account are raised for review from `delay_hours` after the confirmation, regardless of the GNN
score. Very busy accounts can be excluded with `max_prior_payments`. Recommended: keys="sender",
max_prior_payments=20.

Deployment uses `watchlist_from_confirmed` and `watchlist_alerts` with analyst or agent verdicts and reads no labels.
`evaluate` is for reporting only: labels stand in for the agent's verdicts.
"""
import numpy as np
import pandas as pd

KEYS = ("sender", "either")


def _accounts(table, edge_ids, keys):
    t = table.loc[table["edge_id"].isin(edge_ids), ["edge_id", "timestamp", "from_account", "to_account"]]
    parts = [t.rename(columns={"from_account": "account"})[["edge_id", "timestamp", "account"]]]
    if keys == "either":
        parts.append(t.rename(columns={"to_account": "account"})[["edge_id", "timestamp", "account"]])
    return pd.concat(parts, ignore_index=True)


def watchlist_from_confirmed(table, confirmed, keys="sender", delay_hours=24, source="agent"):
    """confirmed: DataFrame with edge_id and optionally confirmed_at (default: the payment's own time).
    Returns one row per watched account: account, active_from (earliest confirmation + delay), source_edge_id, source.
    keys="sender": only the confirmed payment's sender is watched; "either": sender and receiver."""
    if keys not in KEYS:
        raise ValueError(keys)
    c = confirmed.drop_duplicates("edge_id")
    acc = _accounts(table, c["edge_id"], keys)
    if "confirmed_at" in c.columns:
        acc = acc.merge(c[["edge_id", "confirmed_at"]], on="edge_id", how="left")
        acc["confirmed_at"] = pd.to_datetime(acc["confirmed_at"]).fillna(acc["timestamp"])
    else:
        acc["confirmed_at"] = acc["timestamp"]
    acc["active_from"] = acc["confirmed_at"] + pd.Timedelta(hours=delay_hours)
    acc = acc.sort_values(["active_from", "edge_id"]).drop_duplicates("account")
    w = pd.DataFrame({"account": acc["account"].values, "confirmed_at": acc["confirmed_at"].values,
                      "active_from": acc["active_from"].values, "source_edge_id": acc["edge_id"].values,
                      "source": source})
    return add_prior_payments(table, w)


def add_prior_payments(table, watchlist):
    """Add prior_payments: payments the account sent or received strictly before its confirmation."""
    w = watchlist.drop(columns=["prior_payments"], errors="ignore")
    t = table[["timestamp", "from_account", "to_account"]]
    side = pd.concat([t[["from_account", "timestamp"]].rename(columns={"from_account": "account"}),
                      t[["to_account", "timestamp"]].rename(columns={"to_account": "account"})])
    side = side[side["account"].isin(set(w["account"]))].merge(w[["account", "confirmed_at"]], on="account")
    cnt = side[side["timestamp"] < side["confirmed_at"]].groupby("account").size()
    w["prior_payments"] = cnt.reindex(w["account"]).fillna(0).astype(np.int64).values
    return w


def merge_watchlists(*wls):
    """Union of watchlists; an account keeps its earliest active_from."""
    w = pd.concat(wls, ignore_index=True).sort_values(["active_from", "source_edge_id"])
    return w.drop_duplicates("account").reset_index(drop=True)


def watchlist_alerts(table, watchlist, keys="sender", edge_ids=None, max_prior_payments=None):
    """Payments whose sender (or, with keys="either", receiver) is watched and that happen at or after active_from.
    max_prior_payments: skip watched accounts that had more payments than this before their confirmation.
    Returns edge_id, timestamp, from_account, to_account, watched_account, active_from, source_edge_id, reason."""
    if keys not in KEYS:
        raise ValueError(keys)
    t = table[["edge_id", "timestamp", "from_account", "to_account"]]
    if edge_ids is not None:
        t = t[t["edge_id"].isin(edge_ids)]
    w = watchlist
    if max_prior_payments is not None:
        w = w[w["prior_payments"] <= max_prior_payments]
    w = w[["account", "active_from", "source_edge_id"]]
    hits = []
    for col, role in (("from_account", "sender"), ("to_account", "receiver")):
        if role == "receiver" and keys == "sender":
            continue
        h = t.merge(w, left_on=col, right_on="account", how="inner")
        h = h[h["timestamp"] >= h["active_from"]]
        hits.append(h.assign(watched_account=h[col], role=role))
    if not hits or not sum(len(h) for h in hits):
        return pd.DataFrame(columns=["edge_id", "timestamp", "from_account", "to_account", "watched_account",
                                     "active_from", "source_edge_id", "reason"])
    h = pd.concat(hits).sort_values(["edge_id", "active_from"]).drop_duplicates("edge_id")
    h["reason"] = ("payment " + h["role"] + " " + h["watched_account"].astype(str)
                   + " is on the watchlist since " + h["active_from"].dt.strftime("%Y-%m-%d %H:%M")
                   + " (confirmed case, payment " + h["source_edge_id"].astype(str) + ")")
    cols = ["edge_id", "timestamp", "from_account", "to_account", "watched_account", "active_from",
            "source_edge_id", "reason"]
    return h[cols].sort_values(["timestamp", "edge_id"]).reset_index(drop=True)


def evaluate(table, pred, threshold, patterned_ids, split="test", keys=KEYS, delay_hours=24, caps=(20, None)):
    """Evaluation table; labels stand in for the agent's verdicts.
    pred: edge_id, split, score for val and test. patterned_ids: edge_ids in a listed typology.
    Confirmation scenarios:
      gnn_confirmed   true GNN alerts (score >= threshold) in val and test
      +past_cases     plus laundering from the training period (recommended)
    caps: max_prior_payments values to compare (None = no cap).
    Recall counts payments flagged by the GNN or the watchlist; a confirmation itself is not counted as caught."""
    df = table[["edge_id", "timestamp", "from_account", "to_account", "split", "label"]]
    p = pred[["edge_id", "score"]].merge(df, on="edge_id", how="inner")
    split_code = {"val": 1, "test": 2}[split]
    ev = p[p["split"] == split_code].copy()
    ev["patterned"] = ev["edge_id"].isin(set(patterned_ids))
    ev["gnn"] = ev["score"] >= threshold
    lab = ev["label"] == 1
    groups = {"all": lab, "patterned": lab & ev["patterned"], "unpatterned": lab & ~ev["patterned"]}

    gnn_true = p.loc[(p["score"] >= threshold) & (p["label"] == 1), ["edge_id"]]
    past = df.loc[(df["split"] == 0) & (df["label"] == 1), ["edge_id"]]
    scen = [("gnn_confirmed", [gnn_true]), ("gnn_confirmed+past_cases", [gnn_true, past])]

    rows = []
    n = len(ev)
    for k in keys:
        for name, seeds in scen:
            wl = merge_watchlists(*[watchlist_from_confirmed(df, s, k, delay_hours) for s in seeds])
            for cap in caps:
                al = watchlist_alerts(df, wl, k, edge_ids=ev["edge_id"], max_prior_payments=cap)
                wflag = ev["edge_id"].isin(set(al["edge_id"]))
                extra = wflag & ~ev["gnn"]
                row = {"key": k, "confirmations": name, "max_prior_payments": "none" if cap is None else cap,
                       "watched_accounts": int(len(wl) if cap is None else (wl["prior_payments"] <= cap).sum()),
                       "extra_alerts": int(extra.sum()), "extra_alerts_pct_of_payments": 100 * extra.sum() / n,
                       "extra_alert_precision": float(ev.loc[extra, "label"].mean()) if extra.any() else np.nan}
                for gname, m in groups.items():
                    tot = max(int(m.sum()), 1)
                    row[f"{gname}_recall_gnn"] = float((m & ev["gnn"]).sum() / tot)
                    row[f"{gname}_recall_gnn+watchlist"] = float((m & (ev["gnn"] | wflag)).sum() / tot)
                    row[f"{gname}_extra_caught"] = int((m & extra).sum())
                rows.append(row)
    return pd.DataFrame(rows)
