# BT4012 Group 2: AML Detection with GNN and a Verified LLM Agent

Detect laundering transactions in IBM AML HI-Small with a tabular baseline and a GNN,
then investigate alerts using an LLM agent backed by transaction evidence.

The `data` branch implements **data preparation and validation**. Model training,
shared model evaluation and the investigation agent remain assigned to their owners.
The original model/agent scripts are placeholders, not working implementations yet.

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
given by each model filename. Downstream prediction files should contain `tx_id, score`
(Parquet or CSV as agreed by the model owners).

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

## Repository layout

| Location | Purpose |
|---|---|
| `notebooks/01_prepare_final_data.ipynb` | Canonical data implementation and submission notebook; Kang Min Gu |
| `scripts/01_preprocess.py` | Thin CLI entry point to that notebook |
| `scripts/02_build_features.py` | Compatibility alias; no duplicated feature logic |
| `data/raw/` | Local source files, ignored by Git |
| `data/processed/` | Four regenerated CSVs, ignored by Git |
| `scripts/03_train_gbt.py`–`06_run_agent.py`, `src/` | Existing placeholders for the remaining team work |
| `outputs/` | Local execution artifacts and future model/agent outputs, ignored by Git |

Do not commit raw/generated data, `.env`, API keys or `kaggle.json`. Share the final
CSV bundle through the team's agreed file-sharing location, or regenerate it from
the two source files. No team download link has been configured in this branch.

Data source and license: [IBM AML-Data](https://github.com/IBM/AML-Data) describes the
synthetic dataset and identifies the data license as CDLA-Sharing-1.0. Notebook code
development and review were assisted by OpenAI Codex; acknowledge this assistance in
the final submission and be able to explain the implementation.
