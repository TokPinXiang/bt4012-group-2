"""Score the agent's case reports against pattern_labels.csv.

    python scripts/07_evaluate_agent.py
    python scripts/07_evaluate_agent.py --cases outputs/narratives/case_reports_xgb.json

Writes outputs/narratives/agent_eval_<name>.json and a per-alert CSV next to it.
Labels are read here only, after the agent has finished.

Owner: Member 4 (agent)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.evaluate import evaluate_cases, failure_reasons, red_team  # noqa: E402
from src.agent.store import find_data_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=ROOT / "outputs/narratives/case_reports_standin.json")
    ap.add_argument("--labels", type=Path, default=find_data_dir(ROOT) / "pattern_labels.csv")
    ap.add_argument("--db", type=Path, default=ROOT / "data/processed/agent_store.duckdb")
    a = ap.parse_args()

    payload = json.loads(a.cases.read_text(encoding="utf-8"))
    per_alert, summary = evaluate_cases(payload, a.labels, a.db)
    summary["red_team"] = red_team(payload)
    summary["dropped_claim_reasons"] = failure_reasons(payload)
    summary["llm_model"] = payload.get("llm_model")
    summary["source"] = str(a.cases)

    name = a.cases.stem.replace("case_reports_", "")
    out = a.cases.with_name(f"agent_eval_{name}.json")
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    per_alert.to_csv(out.with_suffix(".csv"), index=False)

    skip = {"per_typology", "confusion"}
    for k, v in summary.items():
        if k not in skip:
            print(f"{k:36s} {v}")
    print("\nper typology:")
    for row in summary["per_typology"]:
        print("  ", row)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
