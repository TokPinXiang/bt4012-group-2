# BT4012 Group 2: AML Detection with GNN + Verified LLM Agent

Detect money laundering in the IBM AML HI-Small dataset with a tabular baseline and a GNN
edge classifier, then explain the top alerts with an LLM agent whose evidence is checked
against the transaction data.

## Setup

Run commands from the project root (the folder containing `scripts/`). Python 3.10
was used for the reported baseline experiment and the agent.

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The baseline and saved-prediction evaluation need no Kaggle or LLM credentials.
Configure credentials separately when downloading raw data or running the agent.

## Prepare the data

Most of the team only needs the processed files: obtain them and place them in
`data/processed/`:

- `train.csv`, `validation.csv`, `test.csv`: the fixed chronological splits, with
  `source_row`, `timestamp`, `is_laundering` and transaction/history features.
- `pattern_labels.csv`: annotations for typology evaluation only.

The baseline and shared evaluator use these CSVs. The GNN additionally supports
the three-Parquet input contract listed below. Stage 00 is a placeholder; use the
manual download below.

### Regenerating the processed files

The data notebook was validated on Python 3.12 with its own pinned environment:

```bash
python -m pip install -r requirements-data.txt
```

Download these two files from [IBM AML on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
and place them, unchanged, in `data/raw/`:

- `HI-Small_Trans.csv`
- `HI-Small_Patterns.txt`

Original row order, textual bank IDs and decimal strings matter for stable IDs and
exact pattern matching. The notebook records source SHA256 hashes. An LLM API key
or Kaggle token is **not required to process files already downloaded**.

```bash
python scripts/01_preprocess.py --check-inputs
python scripts/01_preprocess.py
```

The script executes [the self-contained submission notebook](notebooks/01_prepare_final_data.ipynb),
including cleaning, feature engineering, pattern-label mapping, splitting and validation.
The notebook can also be opened directly from the repository root or `notebooks/`
and run top to bottom. A full run needs several GB of memory and disk space.

`scripts/02_build_features.py` is an alias for the same complete workflow; run **one**
of the two entry points. No separate feature-engineering implementation is maintained.
The CLI saves its executed notebook under ignored `outputs/data-preparation/`.

### Shared data contract

| Output in `data/processed/` | Purpose |
|---|---|
| `train.csv` | 3,554,955 chronological training transactions; 2,856 positives |
| `validation.csv` | 761,746 tuning transactions; 760 positives |
| `test.csv` | 761,635 final evaluation transactions; 1,561 positives |
| `pattern_labels.csv` | Evaluation-only annotations for all 5,177 positive transactions |

The three model files each contain **62 columns = 50 predictors + `is_laundering`
target + 11 metadata columns**. Section 9 of the notebook provides the explicit
`FEATURE_COLUMNS` allowlist, metadata exclusions and loading code. Select the same
50 names in the same order for every split. Three predictors are categorical:
`payment_currency`, `receiving_currency`, `payment_format`. Fit their handling on
training only, or use the chosen model's categorical support.

`tx_id` and `source_row` both identify a transaction: `tx_id` is `HI-Small:` followed by
`source_row` padded to 8 digits (`source_row` 4310127 = `HI-Small:04310127`), and
`source_row` preserves the original one-based CSV data row. Keep bank/account columns
as text, including leading zeros. `src_account_id` and `dst_account_id` identify
bank-account pairs; `src_node_id` and `dst_node_id` are graph indices, not scalar
predictors. The timestamp is metadata for ordering and causal filtering.

Amounts retain their native currencies in `amount_paid`/`payment_currency` and
`amount_received`/`receiving_currency`. No FX source was supplied, so **no `amount_usd`
is fabricated**.

### Time and label rules

- The split is approximately 70/15/15 by chronological row count, keeping whole timestamp
  batches together. Train ends `2022-09-07 14:55`; validation ends `2022-09-09 03:16`.
- A historical feature at time `t` uses only observations with timestamp **strictly less
  than `t`**. Same-minute transactions cannot see one another. Earlier unlabeled
  validation/test transactions can update history, matching streaming detection.
- GNN message-passing context must follow the same earlier-edge restriction. Do not
  use a complete validation/test graph to classify earlier transactions.
- The agent may use the model CSVs only after excluding `is_laundering`, and must never
  read `pattern_labels.csv`. Evidence must obey the alert time or a documented investigation cutoff.
- Patterns are evaluation labels, not predictors. There are 3,209 annotated positives
  and 1,968 `UNKNOWN_UNLISTED` positives. Unknown typology does not mean legitimate.
- The sparse September 11+ tail contains 655 positives among 1,108 rows: 42% of test
  positives. Keep the common test set and include period-specific diagnostic results.

The notebook checks every exported row/field, source integrity, IDs, chronological
separation, annotation coverage, and 3,200 independently recomputed historical values.
It also tests same-time exclusion and future invariance. Nine exact duplicate excess
rows are removed under a documented keep-first assumption; all 5,177 positive labels
are retained. Benford features are exploratory, not proof of laundering.

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

## Investigation agent

For each top-K alert, the agent gathers evidence, writes a case with cited transactions,
has every claim checked by code, and triages the alert:

1. **Evidence**: a label-free DuckDB copy of the transactions (`is_laundering` is never
   copied in) feeds lookup tools that only return transactions at or before the alert
   minute. Same-minute transactions count as visible because the data has one-minute
   resolution; this is the agent's documented investigation cutoff.
2. **Case**: an LLM agent (LangChain + NUS SoCLaaS, `qwen3.6:35b`) reads a compact case,
   may call lookup tools within a budget, and submits a report via a tool whose arguments
   are a Pydantic schema. Each claim cites transactions and carries machine-checkable facts.
3. **Verification**: plain Python checks every cited id and fact against the data. Failures
   go back to the agent once; claims that still fail are dropped.
4. **Triage**: high score + verified → `fast_track`; high score + not verified →
   `manual_review`; below threshold → `auto_clear`.

The agent and its tools are defined in Markdown files with YAML frontmatter
(`src/agent/definitions/agents/*.md`, `src/agent/definitions/tools/*.md`): the frontmatter
holds settings (model, tool list, budgets, argument rules) and the body holds the prompt.
They are validated with Pydantic when loaded.

### Model prediction files

The agent reads the published model files as they are:
`outputs/predictions/predictions_<model>.parquet` (`edge_id`, `split` = val / test, `score`)
plus the matching `predictions_<model>_metrics.json`. It converts them on load:

| Published | Agent |
|---|---|
| `edge_id` (= `source_row`) | `tx_id` = `HI-Small:` + `edge_id` padded to 8 digits |
| `split` `val` / `test` | `validation` / `test` |
| `["val"]["threshold"]` in the metrics JSON | `predicted_label` = 1 if `score` ≥ threshold, else 0 |
| file name `predictions_gnn` | `model_name` = `gnn` |

A CSV already in the agent's own columns (`tx_id, split, score, predicted_label,
model_name`) is also accepted. Check a file before running the agent:

```bash
python scripts/validate_predictions.py outputs/predictions/predictions_gnn.parquet
```

If the GNN's explanations file (`outputs/agent/alerts_explained.json`, from
`python scripts/04_train_gnn.py --explain`) exists, GNN alerts it covers also get a
"model explanation" section: how much of the score came from the payment itself and
its 1- and 2-hop neighbourhood, the top features, and the neighbouring payments that
mattered most. Those payments are looked up in the agent's store under the same
no-future rule and become citable evidence. The explanation is a lead, not proof.

### Run

SoCLaaS settings are read from `~/.config/soclaas/soclaas.env` (`SOCLAAS_BASE_URL`,
`SOCLAAS_API_KEY`; see `.env.example`). Never commit the key.

```bash
python scripts/make_standin_alerts.py      # fake alerts until real model files exist
python scripts/06_run_agent.py --predictions outputs/predictions/predictions_gnn.parquet --split validation
python scripts/07_evaluate_agent.py --cases outputs/narratives/case_reports_gnn.json
python scripts/show_trace.py --rank 1      # what the agent did on alert #1: each tool, why, what came back
python scripts/check_pipeline.py           # re-check every input and output contract
```

`06_run_agent.py` validates the prediction file first and the case reports last
(schema, no labels, no future data, citations inside the evidence, triage rule, and
re-verification of every kept claim). JSON Schemas for these contracts are in
`docs/contracts/` (regenerate with `python -m src.agent.schemas`).

`pattern_labels.csv` is read only by `07_evaluate_agent.py`, the prediction validator's
optional PR-AUC sanity check, and the stand-in alert generator (which simulates a model).
The agent itself never reads it.

## Checks

```powershell
python -m unittest discover -s tests -v    # baseline and evaluation tests
python -m pytest tests                     # all tests, including the agent's (offline, fake LLM)
```

## Layout

| Folder | What goes there | Owner |
|---|---|---|
| `notebooks/01_prepare_final_data.ipynb`, `scripts/01_preprocess.py`, `scripts/02_build_features.py` | Cleaning, split, features | A |
| `src/models/gbt.py`, `src/eval/` | Baseline and shared metrics | B |
| `src/models/gnn/` | GNN edge classifier | C |
| `src/agent/` | Agent store, tools, LangChain loop, verifier, Pydantic contracts | D |
| `src/agent/definitions/` | Agent and tool definitions (Markdown + YAML frontmatter) | D |
| `scripts/` | Thin entry points, one per stage; agent checks in `validate_predictions.py`, `check_pipeline.py`, `show_trace.py` | |
| `notebooks/` | Experiment runners, plots and report tables | |
| `configs/config.yaml` | GNN paths and experiment settings | C |
| `tests/` | Baseline, evaluation and agent tests | |
| `docs/contracts/` | JSON Schemas for the agent's file formats | D |
| `data/raw/`, `data/processed/`, `outputs/` | Local data and run outputs, ignored by Git | |

## Shared file formats

| File | Columns |
|---|---|
| `data/processed/{train,validation,test}.csv` | source_row, tx_id, timestamp, is_laundering, feature columns and account metadata |
| `data/processed/pattern_labels.csv` | source_row, tx_id, split, is_laundering, pattern_id, typology, annotation_status and descriptive fields |
| `data/processed/transactions.parquet` | edge_id, timestamp, from_account, to_account, amount_usd, currency, payment_format, label |
| `data/processed/split.parquet` | edge_id, split (train / val / test) |
| `data/processed/features.parquet` | edge_id, feature columns |
| `outputs/predictions/predictions_<model>.parquet` | edge_id (= source_row), split (val / test), score |
| `outputs/predictions/predictions_<model>_metrics.json` | per-split metrics; `["val"]["threshold"]` is the alert threshold |
| `outputs/agent/alerts_explained.json` | GNN explanations of its top alerts, for the agent |

The three Parquet data files are the GNN's alternative input; the four CSVs are the
team's delivered data.

## Rules

- Stages talk only through the files above, not through each other's code.
- Graph and temporal features use past transactions only (no leakage).
- Ground-truth labels support supervised training and evaluation. Pattern
  annotations are evaluation-only. The agent's tools must read neither labels
  nor pattern annotations.
- Never commit data, outputs, `.env`, API keys or `kaggle.json`. Share the final
  CSV bundle through the team's agreed file-sharing location, or regenerate it from
  the two source files.

Data source and license: [IBM AML-Data](https://github.com/IBM/AML-Data) describes the
synthetic dataset and identifies the data license as CDLA-Sharing-1.0. Notebook code
development and review were assisted by OpenAI Codex; acknowledge this assistance in
the final submission and be able to explain the implementation.
