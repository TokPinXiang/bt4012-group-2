"""Show what the agent did on an alert, step by step: each tool it called, why, and what came back.

    python scripts/show_trace.py                       # summary of every alert in the trace
    python scripts/show_trace.py --rank 3              # full timeline for alert #3
    python scripts/show_trace.py --tx HI-Small:02180323
    python scripts/show_trace.py --trace outputs/narratives/trace_xgb.jsonl --rank 1

The trace is written by scripts/06_run_agent.py next to the case reports.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.trace import read_traces, render_alert  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, default=ROOT / "outputs/narratives/trace_standin.jsonl")
    ap.add_argument("--rank", type=int, help="alert rank to show in full")
    ap.add_argument("--tx", help="alert tx_id to show in full")
    a = ap.parse_args()

    by_alert = defaultdict(list)
    for e in read_traces(a.trace):
        by_alert[(e["rank"], e["tx_id"])].append(e)

    if a.rank is not None or a.tx:
        hits = [k for k in by_alert if k[0] == a.rank or k[1] == a.tx]
        if not hits:
            sys.exit("No such alert in the trace.")
        for rank, tx in hits:
            print(f"Alert #{rank}  {tx}\n")
            print(render_alert(by_alert[(rank, tx)]))
        return

    # Overview: one line per alert, then which tools were used and why.
    tools = Counter()
    print(f"{'rank':>4}  {'tx_id':<18} {'lookups':>7} {'submits':>7}  {'final status':<13} triage")
    for (rank, tx), events in sorted(by_alert.items()):
        calls = [e for e in events if e["event"] == "tool_call"]
        submits = [e for e in events if e["event"] == "submit"]
        tools.update(e["tool"] for e in calls)
        final = submits[-1]["status"] if submits else "none"
        decision = next((e["decision"] for e in events if e["event"] == "triage"), "?")
        print(f"{rank:>4}  {tx:<18} {len(calls):>7} {len(submits):>7}  {final:<13} {decision}")
    print(f"\nLookups by tool: {dict(tools)}")
    print("Show one alert in full with --rank N")


if __name__ == "__main__":
    main()
