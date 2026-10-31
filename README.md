# BT4012 Group 2: AML Detection with GNN + Verified LLM Agent

Detect money laundering in the IBM AML HI-Small dataset with a GNN edge classifier,
then explain the top alerts with an LLM agent whose evidence is checked against the graph.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Put your Kaggle token at `~/.kaggle/kaggle.json`, then copy `.env.example` to `.env` and add your LLM API key.

## Run the pipeline

```bash
python scripts/00_download_data.py
python scripts/01_preprocess.py
python scripts/02_build_features.py
python scripts/03_train_gbt.py
python scripts/04_train_gnn.py
python scripts/05_evaluate.py
python scripts/06_run_agent.py
```

## Layout

| Folder | What goes there | Owner |
|---|---|---|
| `src/data/`, `src/features/` | Cleaning, split, features | A |
| `src/models/gbt.py`, `src/eval/` | Baseline and shared metrics | B |
| `src/models/gnn/` | GNN edge classifier | C |
| `src/agent/` | Agent tools, LLM loop, verification gate | D |
| `scripts/` | Thin entry points, one per stage | |
| `notebooks/` | Exploration only, one notebook per person | |
| `configs/config.yaml` | Paths, split dates, k values | |

## Shared file formats

| File | Columns |
|---|---|
| `data/processed/transactions.parquet` | edge_id, timestamp, from_account, to_account, amount_usd, currency, payment_format, label |
| `data/processed/split.parquet` | edge_id, split (train / val / test) |
| `data/processed/features.parquet` | edge_id, feature columns |
| `outputs/predictions/predictions_<model>.parquet` | edge_id, score |

## Rules

- Stages talk only through the files above, not through each other's code.
- Graph and temporal features use past transactions only (no leakage).
- `Patterns.txt` and labels are for evaluation only; the agent's tools must not read them.
- Never commit data, outputs, `.env` or `kaggle.json`.
