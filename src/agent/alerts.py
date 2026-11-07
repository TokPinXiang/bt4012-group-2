"""Read model prediction files, and make a stand-in one until Members 2 and 3 deliver.

Shared prediction format (one row per scored transaction):
    tx_id, split, score, predicted_label, model_name
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PREDICTION_COLUMNS = ["tx_id", "split", "score", "predicted_label", "model_name"]
TYPOLOGIES = ["FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER", "GATHER-SCATTER", "STACK", "BIPARTITE", "RANDOM"]


def load_alerts(path: Path, top_k: int, split: str | None = None) -> list[dict]:
    """Top-K rows by score from a prediction file, highest first, with a 1-based rank."""
    df = pd.read_csv(path, dtype={"tx_id": str, "split": str, "model_name": str})
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; expected {PREDICTION_COLUMNS}")
    if split:
        df = df[df["split"] == split]
    df = df.sort_values(["score", "tx_id"], ascending=[False, True]).head(top_k).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df[PREDICTION_COLUMNS + ["rank"]].to_dict("records")


def make_standin_alerts(
    data_dir: Path,
    out_path: Path,
    split: str = "validation",
    per_typology: int = 3,
    n_unlisted: int = 6,
    n_normal_hard: int = 10,
    n_normal_random: int = 10,
    seed: int = 4012,
) -> pd.DataFrame:
    """A fake prediction file with a known mix of laundering and normal transactions.

    - Laundering: `per_typology` patterns per named typology (one random transaction each),
      plus `n_unlisted` laundering transactions with no listed pattern.
    - Hard normals: legitimate transactions from busy accounts (many sends or receives in
      the past 24h). These look like what a real model would flag by mistake.
    - Random normals: ordinary legitimate transactions.

    Scores are made up. They exist only so the file has the same shape as a model's.
    """
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)

    labels = pd.read_csv(data_dir / "pattern_labels.csv", dtype=str)
    labels = labels[labels["split"] == split]

    picked = []
    for typ in TYPOLOGIES:
        pats = labels.loc[labels["typology"] == typ, "pattern_id"].dropna().unique()
        for pid in rng.choice(pats, size=min(per_typology, len(pats)), replace=False):
            members = labels.loc[labels["pattern_id"] == pid, "tx_id"].to_numpy()
            picked.append(("laundering", rng.choice(members)))
    unlisted = labels.loc[labels["annotation_status"] == "UNKNOWN_UNLISTED", "tx_id"].to_numpy()
    picked += [("laundering", t) for t in rng.choice(unlisted, size=min(n_unlisted, len(unlisted)), replace=False)]

    csv = str((data_dir / f"{split}.csv").resolve()).replace("\\", "/").replace("'", "''")
    con = duckdb.connect()
    # A seeded hash of tx_id gives the same "random" order on every run. DuckDB's random()
    # is not reproducible here because it reads the CSV in parallel threads.
    shuffle = f"ORDER BY hash(tx_id || '{int(seed)}'), tx_id"
    busy = "(CAST(src_out_count_24h AS DOUBLE) >= 10 OR CAST(dst_in_count_24h AS DOUBLE) >= 10)"
    hard = con.execute(
        f"""
        SELECT tx_id FROM read_csv('{csv}', all_varchar = true)
        WHERE is_laundering = '0' AND {busy}
        {shuffle} LIMIT {int(n_normal_hard)}
        """
    ).fetchall()
    # Ordinary normals exclude busy accounts, so the two groups never overlap.
    easy = con.execute(
        f"""
        SELECT tx_id FROM read_csv('{csv}', all_varchar = true)
        WHERE is_laundering = '0' AND NOT {busy}
        {shuffle} LIMIT {int(n_normal_random)}
        """
    ).fetchall()
    con.close()
    picked += [("normal_hard", t) for (t,) in hard] + [("normal_random", t) for (t,) in easy]

    score_range = {"laundering": (0.55, 0.99), "normal_hard": (0.50, 0.95), "normal_random": (0.30, 0.80)}
    rows = [
        {"tx_id": tx, "split": split, "score": round(float(rng.uniform(*score_range[kind])), 4)}
        for kind, tx in picked
    ]
    df = pd.DataFrame(rows).drop_duplicates("tx_id")
    df["predicted_label"] = (df["score"] >= 0.5).astype(int)
    df["model_name"] = "standin"
    df = df.sort_values("score", ascending=False)[PREDICTION_COLUMNS]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
