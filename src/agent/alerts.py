"""Read model prediction files, and make a stand-in one for testing without the models.

Two input formats are accepted; both come out as the agent's format
    tx_id, split, score, predicted_label, model_name

1. The team's published format (Members 2 and 3), read as it is:
       predictions_<model>.parquet          edge_id (= source_row), split (val / test), score
       predictions_<model>_metrics.json     ["val"]["threshold"] = the alert threshold
   converted as: tx_id = "HI-Small:" + edge_id padded to 8 digits; val -> validation;
   predicted_label = score >= threshold; model_name from the file name.
2. The agent's own format, as a CSV or Parquet with the five columns above.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PREDICTION_COLUMNS = ["tx_id", "split", "score", "predicted_label", "model_name"]
TEAM_COLUMNS = ["edge_id", "split", "score"]
SPLIT_NAMES = {"val": "validation", "validation": "validation", "test": "test", "train": "train"}
TYPOLOGIES = ["FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER", "GATHER-SCATTER", "STACK", "BIPARTITE", "RANDOM"]


class PredictionFormatError(ValueError):
    """The file is in neither accepted format, or its threshold cannot be found."""


def edge_id_to_tx_id(edge_id: pd.Series) -> pd.Series:
    """source_row / edge_id 4310127 -> 'HI-Small:04310127' (the data's own tx_id rule)."""
    as_num = pd.to_numeric(edge_id, errors="coerce")
    bad = as_num.isna() | (as_num < 0) | (as_num % 1 != 0)
    if bad.any():
        raise PredictionFormatError(f"{int(bad.sum())} edge_id values are not whole non-negative numbers, "
                                    f"e.g. {edge_id[bad].head(3).tolist()}")
    return "HI-Small:" + as_num.astype("int64").astype(str).str.zfill(8)


def metrics_path_for(path: Path) -> Path:
    """predictions_gnn.parquet -> predictions_gnn_metrics.json (the name stage 03/04 publish)."""
    path = Path(path)
    return path.with_name(f"{path.stem}_metrics.json")


def model_name_from_path(path: Path) -> str:
    stem = Path(path).stem
    return stem[len("predictions_"):] if stem.startswith("predictions_") else stem


def read_threshold(metrics_path: Path) -> float:
    try:
        return float(json.loads(Path(metrics_path).read_text(encoding="utf-8"))["val"]["threshold"])
    except FileNotFoundError:
        raise PredictionFormatError(f"no threshold: {metrics_path.name} not found next to the predictions "
                                    "(copy it from the model owner, or pass --threshold)") from None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PredictionFormatError(f"{metrics_path.name} has no usable ['val']['threshold']: {exc}") from None


def read_predictions(path: Path, threshold: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Read either accepted format and return (agent-format DataFrame, info about the conversion)."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"tx_id": str, "split": str, "model_name": str})

    if set(PREDICTION_COLUMNS) <= set(df.columns):
        extra = [c for c in df.columns if c not in PREDICTION_COLUMNS]
        out = df[PREDICTION_COLUMNS].copy()
        out["split"] = out["split"].map(lambda s: SPLIT_NAMES.get(s, s))
        return out, {"format": "agent", "extra_columns": extra}

    if set(TEAM_COLUMNS) <= set(df.columns):
        source = "argument"
        if threshold is None:
            threshold, source = read_threshold(metrics_path_for(path)), metrics_path_for(path).name
        out = pd.DataFrame({
            "tx_id": edge_id_to_tx_id(df["edge_id"]),
            "split": df["split"].astype(str).map(lambda s: SPLIT_NAMES.get(s, s)),
            "score": pd.to_numeric(df["score"], errors="coerce"),
        })
        out["predicted_label"] = (out["score"] >= threshold).astype(int)
        out["model_name"] = model_name_from_path(path)
        extra = [c for c in df.columns if c not in TEAM_COLUMNS]
        return out, {"format": "team", "threshold": float(threshold), "threshold_source": source,
                     "extra_columns": extra}

    raise PredictionFormatError(
        f"{path.name} has columns {list(df.columns)}; expected the team format {TEAM_COLUMNS} "
        f"(+ {metrics_path_for(path).name}) or the agent format {PREDICTION_COLUMNS}")


def load_alerts(path: Path, top_k: int, split: str | None = None, threshold: float | None = None,
                select: str = "top", seed: int = 4012) -> list[dict]:
    """Choose the alerts to investigate, each with its 1-based rank by score within the split.

    select="top"     the top_k highest scores (what an analyst would see first)
    select="sample"  a seeded random sample of top_k from every payment above the model's
                     threshold. The very top alerts are almost all laundering, so a sample
                     across the whole alert queue also tests how the agent handles false alarms.
    Ties are broken by tx_id, which follows edge_id order, matching the team evaluator.
    """
    df, _ = read_predictions(path, threshold)
    if split:
        df = df[df["split"] == SPLIT_NAMES.get(split, split)]
    df = df.sort_values(["score", "tx_id"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = df.index + 1
    if select == "top":
        df = df.head(top_k)
    elif select == "sample":
        flagged = df[df["predicted_label"] == 1]
        pick = np.random.default_rng(seed).choice(len(flagged), size=min(top_k, len(flagged)), replace=False)
        df = flagged.iloc[np.sort(pick)]
    else:
        raise ValueError(f"select must be 'top' or 'sample', not {select!r}")
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
