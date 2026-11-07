# BT4012 Group 2: AML Detection with GNN and a Verified LLM Agent

Detect laundering transactions in IBM AML HI-Small with a tabular baseline and a GNN,
then investigate alerts using an LLM agent backed by transaction evidence.

The `data` branch implements **data preparation and validation**. The `agent` branch
builds on it with the **investigation agent** (see [Investigation agent](#investigation-agent)).
Model training and shared model evaluation remain assigned to their owners; those
scripts are still placeholders.

## Prepare the data

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows alternative: .venv\Scripts\activate
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
`scripts/00_download_data.py` remains a placeholder; use the manual download above.
The CLI saves its executed notebook under ignored `outputs/data-preparation/`.

`requirements.txt` includes the data dependencies plus the existing team's model
and agent dependencies. The data-only installation avoids installing those unrelated
packages. The model/agent environments have not been validated by this data change.

## Shared data contract

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

Use `tx_id` as the common transaction key for predictions, alerts and annotation joins.
`source_row` preserves the original one-based CSV data row. Keep bank/account columns
as text, including leading zeros. `src_account_id` and `dst_account_id` identify
bank-account pairs; `src_node_id` and `dst_node_id` are graph indices, not scalar
predictors. The timestamp is metadata for ordering and causal filtering.

Amounts retain their native currencies in `amount_paid`/`payment_currency` and
`amount_received`/`receiving_currency`. No FX source was supplied, so **no `amount_usd`
is fabricated**. These four CSVs replace the initial placeholder contract for separate
`transactions.parquet`, `split.parquet` and `features.parquet`; split membership is
given by each model filename. Downstream prediction files follow
[Model prediction files](#model-prediction-files) below (proposed by the agent owner;
to be confirmed with the model owners).

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

`outputs/predictions/predictions_<model>.csv`, one row per transaction, every row of
validation (and test, once, at the end):

| Column | Meaning |
|---|---|
| `tx_id` | `tx_id` from the data CSVs, unchanged, e.g. `HI-Small:03970858` |
| `split` | `validation` or `test` |
| `score` | probability of laundering in [0, 1]; higher = more suspicious |
| `predicted_label` | 1 if `score` ≥ the threshold tuned on validation, else 0 |
| `model_name` | short name, e.g. `xgb`, `gnn` |

GNN predictions must be per transaction (edge), not per account. Check a file before
handing it over:

```bash
python scripts/validate_predictions.py outputs/predictions/predictions_xgb.csv
```

### Run

SoCLaaS settings are read from `~/.config/soclaas/soclaas.env` (`SOCLAAS_BASE_URL`,
`SOCLAAS_API_KEY`; see `.env.example`). Never commit the key.

```bash
python scripts/make_standin_alerts.py      # fake alerts until real model files exist
python scripts/06_run_agent.py --predictions outputs/predictions/predictions_xgb.csv --split validation
python scripts/07_evaluate_agent.py --cases outputs/narratives/case_reports_xgb.json
python scripts/show_trace.py --rank 1      # what the agent did on alert #1: each tool, why, what came back
python scripts/check_pipeline.py           # re-check every input and output contract
python -m pytest tests                     # offline tests with a scripted fake LLM
```

`06_run_agent.py` validates the prediction file first and the case reports last
(schema, no labels, no future data, citations inside the evidence, triage rule, and
re-verification of every kept claim). JSON Schemas for these contracts are in
`docs/contracts/` (regenerate with `python -m src.agent.schemas`).

`pattern_labels.csv` is read only by `07_evaluate_agent.py`, the prediction validator's
optional PR-AUC sanity check, and the stand-in alert generator (which simulates a model).
The agent itself never reads it.

## Repository layout

| Location | Purpose |
|---|---|
| `notebooks/01_prepare_final_data.ipynb` | Canonical data implementation and submission notebook; Kang Min Gu |
| `scripts/01_preprocess.py` | Thin CLI entry point to that notebook |
| `scripts/02_build_features.py` | Compatibility alias; no duplicated feature logic |
| `data/raw/` | Local source files, ignored by Git |
| `data/processed/` | Four regenerated CSVs, ignored by Git |
| `scripts/03_train_gbt.py`–`05_evaluate.py`, `src/models/`, `src/eval/` | Existing placeholders for the remaining team work |
| `src/agent/` | Investigation agent: store, tools, LangChain loop, verifier, Pydantic contracts |
| `src/agent/definitions/` | Agent and tool definitions (Markdown + YAML frontmatter) |
| `scripts/06_run_agent.py`, `07_evaluate_agent.py` | Run the agent; score it against `pattern_labels.csv` |
| `scripts/validate_predictions.py`, `check_pipeline.py` | Contract checks for model files and agent outputs |
| `tests/` | Agent tests (offline, fake LLM) |
| `docs/contracts/` | JSON Schemas for the shared file formats |
| `outputs/` | Local execution artifacts and future model/agent outputs, ignored by Git |

Do not commit raw/generated data, `.env`, API keys or `kaggle.json`. Share the final
CSV bundle through the team's agreed file-sharing location, or regenerate it from
the two source files. No team download link has been configured in this branch.

Data source and license: [IBM AML-Data](https://github.com/IBM/AML-Data) describes the
synthetic dataset and identifies the data license as CDLA-Sharing-1.0. Notebook code
development and review were assisted by OpenAI Codex; acknowledge this assistance in
the final submission and be able to explain the implementation.
