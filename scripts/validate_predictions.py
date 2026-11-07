"""Check a prediction file before handing it to the agent. Members 2 and 3: run this on yours.

    python scripts/validate_predictions.py outputs/predictions/predictions_xgb.csv
    python scripts/validate_predictions.py outputs/predictions/*.csv

Checks: columns and types, tx_id format and existence, correct split, no duplicates, one
model per file, labels follow one threshold, coverage, flag rate, and (using
pattern_labels.csv) PR-AUC / F1 with a warning if they look too good to be true.
Exit code 1 if any file has errors.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.store import build_store, default_csv_paths, find_data_dir  # noqa: E402
from src.agent.validation import validate_predictions  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--db", type=Path, default=ROOT / "data/processed/agent_store.duckdb")
    ap.add_argument("--labels", type=Path, default=find_data_dir(ROOT) / "pattern_labels.csv")
    ap.add_argument("--no-labels", action="store_true", help="skip the PR-AUC / leakage sanity check")
    a = ap.parse_args()

    if not a.db.exists():
        print(f"Building the transaction store at {a.db} (one-off) ...")
        build_store(default_csv_paths(find_data_dir(ROOT)), a.db)

    ok = True
    for f in a.files:
        rep = validate_predictions(f, a.db, None if a.no_labels else a.labels)
        print(rep.render(), end="\n\n")
        ok &= rep.ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
