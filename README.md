# BT4012 Group 2: AML Detection with GNN + Verified LLM Agent

Detect money laundering in the IBM AML HI-Small dataset with a GNN edge classifier,
then explain the top alerts with an LLM agent whose evidence is checked against the graph.

## Setup

Run commands from the project root (the folder containing `scripts/`). Python 3.10
was used for the reported baseline experiment.

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The baseline and saved-prediction evaluation need no Kaggle or LLM credentials.
Configure credentials separately when downloading raw data or running the agent.

## Prepare the data

Obtain the team's processed files and place them in `data/processed/`:

- `train.csv`, `validation.csv`, `test.csv`: the fixed chronological splits, with
  `source_row`, `timestamp`, `is_laundering` and transaction/history features.
- `pattern_labels.csv`: annotations for typology evaluation only.

The baseline and shared evaluator use these CSVs. The GNN additionally supports
the three-Parquet input contract listed below. In this checkout, stages 00-02
and 06 are placeholders; do not rely on them to generate data or agent outputs.

## Train and publish predictions

```powershell
python scripts/03_train_gbt.py --trials 30 --publish
python scripts/04_train_gnn.py
python scripts/04_train_gnn.py --publish
```

The explicit publish steps put both models' validation/test scores in
`outputs/predictions/`, alongside their metrics JSONs. They must finish before
stage 05. GNN training may be run separately in Colab; see
[the GNN instructions](src/models/gnn/README.md).

The baseline defaults to all transaction/history features, 30 trials, seed 0,
at most 1,000 boosting rounds and early-stopping patience 60. The script, Python
configuration and local notebook use the same trial budget. Selection uses
validation average precision; the maximum-F1 validation threshold is then fixed
for test. The selected model is not refitted on training plus validation.
Each run saves its model, preprocessing metadata, configuration, trials, metrics,
predictions and explanations in a new `outputs/gbt/` folder. The current search
allows at most 2,595 distinct candidates for imbalanced data (865 for balanced
classes); larger requests are rejected.

For a quick setup check, use `python scripts/03_train_gbt.py --smoke`.
Smoke results are not report results and cannot be published. The local
[baseline notebook](notebooks/baseline_lightgbm.ipynb) starts in smoke mode;
set `MODE = 'full'` to train on the full splits. To inspect a completed run,
set `MODE = 'view'` and `VIEW_RUN` to its folder. The notebook creates plots.

To publish an existing baseline run without retraining:

```powershell
python scripts/03_train_gbt.py --publish-run "outputs/gbt/YOUR_RUN_FOLDER"
```

If replacing an already published baseline, add `--overwrite-published` to the
training or publishing command. Original timestamped runs are preserved.

## Evaluate saved predictions

If the team has already trained the models, copy `predictions_gbt.parquet`,
`predictions_gnn.parquet` and their matching `predictions_*_metrics.json` files
into `outputs/predictions/`, then run:

```powershell
python scripts/05_evaluate.py
```

This stage does not retrain either model. It also discovers predictions directly
under `outputs/`; if both locations contain the same filename, specify
`--gbt-predictions` and `--gnn-predictions` explicitly. It checks exact transaction
coverage and split alignment, verifies each saved validation threshold and
recomputes metrics against the supplied labels. Saved metrics must match within
numerical tolerance. Ranking ties use ascending `edge_id` for reproducibility.

Results are saved to a new `outputs/evaluation/comparison_<timestamp>/` folder:
comparison CSV/JSON, review-budget metrics, typology and scheme detection tables,
reference checks, input hashes and PNG/PDF figures. Typology recall at k uses the
full transaction ranking, not a separate queue per typology. If annotations are
unavailable, explicitly use `--no-typology`. Never tune against test results.

## Checks

```powershell
python -m unittest discover -s tests -v
```

## Layout

| Folder | What goes there | Owner |
|---|---|---|
| `src/data/`, `src/features/` | Cleaning, split, features | A |
| `src/models/gbt.py`, `src/eval/` | Baseline and shared metrics | B |
| `src/models/gnn/` | GNN edge classifier | C |
| `src/agent/` | Agent tools, LLM loop, verification gate | D |
| `scripts/` | Thin entry points, one per stage | |
| `notebooks/` | Experiment runners, plots and report tables | |
| `configs/config.yaml` | GNN paths and experiment settings | C |

## Shared file formats

| File | Columns |
|---|---|
| `data/processed/{train,validation,test}.csv` | source_row, timestamp, is_laundering, feature columns and account metadata |
| `data/processed/pattern_labels.csv` | source_row, split, is_laundering, pattern_id, typology, annotation_status and descriptive fields |
| `data/processed/transactions.parquet` | edge_id, timestamp, from_account, to_account, amount_usd, currency, payment_format, label |
| `data/processed/split.parquet` | edge_id, split (train / val / test) |
| `data/processed/features.parquet` | edge_id, feature columns |
| `outputs/predictions/predictions_<model>.parquet` | edge_id (= source_row), split (val / test), score |

## Rules

- Stages talk only through the files above, not through each other's code.
- Graph and temporal features use past transactions only (no leakage).
- Ground-truth labels support supervised training and evaluation. Pattern
  annotations are evaluation-only. The agent's tools must read neither labels
  nor pattern annotations.
- Never commit data, outputs, `.env` or `kaggle.json`.
