"""Check every agent-pipeline artefact that exists right now.

    python scripts/check_pipeline.py

1. Definitions: agent and tool .md files parse and refer to things that exist.
2. Inputs:      every outputs/predictions/predictions_*.csv passes the prediction contract.
3. Outputs:     every outputs/narratives/case_reports_*.json passes the case-report contract
                (schema, no labels, no future data, citations inside evidence, triage rule,
                and every kept claim re-verified from the saved data).
Exit code 1 if anything fails.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.specs import load_agent_spec  # noqa: E402
from src.agent.store import find_data_dir  # noqa: E402
from src.agent.toolkit import CaseToolkit  # noqa: E402
from src.agent.validation import validate_case_reports, validate_predictions  # noqa: E402


def main():
    ok = True
    db = ROOT / "data/processed/agent_store.duckdb"

    print("1. Agent and tool definitions")
    try:
        agent, tools = load_agent_spec("investigator", toolkit_cls=CaseToolkit)
        print(f"PASS  {agent.name} ({agent.model}) with tools {list(tools)}\n")
    except Exception as exc:
        print(f"FAIL  {exc}\n")
        ok = False

    print("2. Prediction files")
    folder = ROOT / "outputs/predictions"
    preds = sorted([*folder.glob("predictions_*.csv"), *folder.glob("predictions_*.parquet")])
    for p in preds:
        rep = validate_predictions(p, db, find_data_dir(ROOT) / "pattern_labels.csv")
        print(rep.render(), end="\n\n")
        ok &= rep.ok
    if not preds:
        print("   none found\n")

    print("3. Case reports")
    cases = sorted((ROOT / "outputs/narratives").glob("case_reports_*.json"))
    for c in cases:
        rep = validate_case_reports(c)
        print(rep.render(), end="\n\n")
        ok &= rep.ok
    if not cases:
        print("   none found\n")

    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
