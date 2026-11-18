# GNN edge classifier (stage 04, owner C)

Accounts are nodes and payments are directed edges. Each payment is classified from its own features and a sampled
two-hop neighbourhood of payments made up to its minute. Backbone: PNA with the Multi-GNN adaptations (reverse message passing,
port numbers, ego IDs, edge updates; Egressy et al., AAAI 2024) and MEGA-style pooling of parallel payments.

## Run
```
python scripts/04_train_gnn.py                 # every run in configs/config.yaml (gnn section); finished runs are skipped
python scripts/04_train_gnn.py --publish       # outputs/predictions/predictions_gnn.parquet
python scripts/04_train_gnn.py --explain       # outputs/agent/alerts_explained.json
python scripts/04_train_gnn.py --watchlist --pattern_labels <pattern_labels.csv>   # outputs/gnn/watchlist_eval.csv
```
On Colab, `notebooks/gnn_edge_classifier.ipynb` runs all of the above and builds the report tables.

## Input and output
- Reads stage A's output from `gnn.data_dir`: `transactions/split/features.parquet` (README contract) or
  `train.csv`, `validation.csv`, `test.csv`. The files are never changed. The model rescales its own copy in
  memory: large counts and amounts are log-transformed, every column is standardised with the training set's mean and
  spread, and extreme values are capped at ±10.
- Labels are used for training only. Pattern labels are never read during training.
- `predictions_gnn.parquet`: `edge_id` (= `source_row`), `split` (val / test), `score` (P(laundering)).
- Run folders and `results.csv` in `outputs/gnn/` (not committed).

## Files
| File | What it does |
|---|---|
| `data.py` | loads the data, builds graph tensors and port numbers, scaling fitted on train |
| `sampler.py` | k-hop neighbourhood sampler; time-respecting mode never uses later payments as context |
| `model.py` | GIN / PNA edge classifier |
| `train.py` | training loop, early stopping on validation AUPRC, logging, resume after a disconnect |
| `explain.py` | explains the top alerts for the agent (gradient x input), on the same neighbourhood that produced each score |
| `watchlist.py` | follow-up after confirmed cases: watch the sender, raise its later payments (post-processing for the agent) |

## Explanations file for the agent (`outputs/agent/alerts_explained.json`)
No labels, typologies or pattern information. Labels for scoring the agent are kept in the run folder
(`explain_eval/alerts_eval.csv`).

| field | meaning |
|---|---|
| `edge_id` | payment id (`source_row`) |
| `score`, `above_threshold` | GNN laundering probability; above the alert threshold chosen on validation (best F1) |
| `from_account`, `to_account` | `bank:account` |
| `amount`, `currency` | amount paid and payment currency (display only); `amount_usd` instead if stage A provides it |
| `payment_format`, `time` | as in the data |
| `share_from_own_transaction` / `_1hop` / `_2hop` | share of the decision from the payment itself, from payments touching its two accounts, and from payments one step further |
| `top_own_features` | 5 own features that moved the score most: `feature` (column, or `column = value` for categories, or `out_port` / `in_port`), `value` (raw value, null for categories and ports), `pushes_towards` (laundering / normal), `attribution` (size; compare within an alert only) |
| `top_context_transactions` | 5 neighbouring payments that moved the score most: `hop`, `importance_share`, and the same display fields |

A history feature with value 0 usually means "no earlier history" (e.g. `src_currency_amount_mean = 0`: the sender
has not paid in this currency before). Only payments at or before the alert's time are used as context.

## Watchlist follow-up (`watchlist.py`)
After a case is confirmed, `watchlist_from_confirmed` adds the sender to a watchlist and `watchlist_alerts` raises
that account's later payments from 24 h after the confirmation, excluding accounts with more than 20 earlier payments.
This targets laundering outside the listed typologies, which recurs at the same sender. No labels are read;
`evaluate` simulates confirmations with labels for reporting. Related work: taint analysis (Moser et al. 2014) and
client risk profiling (Jensen and Iosifidis 2023).

The model design re-implements IBM Multi-GNN (Apache-2.0, https://github.com/IBM/Multi-GNN).
