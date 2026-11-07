"""Top-K alerts -> verified case reports in outputs/narratives/.

    python scripts/06_run_agent.py                                   # stand-in alerts
    python scripts/06_run_agent.py --predictions outputs/predictions/predictions_xgb.csv --split validation
    python scripts/06_run_agent.py --skip-llm                        # evidence only, no LLM calls

Steps:
  0. validate the prediction file against the shared contract (stops on errors)
  1. gather label-free, no-future evidence for each alert (JSON alert report)
  2. investigator agent (definitions/agents/investigator.md, LangChain + SoCLaaS) investigates,
     may call lookup tools, and submits a case report
  3. Pydantic schema check + verifier on every claim; problems go back to the agent once
  4. triage: fast_track / manual_review / auto_clear
  5. validate the output file (schema, no labels, no future, citations, re-verification)

LLM replies are cached in outputs/narratives/llm_cache/, so re-runs are free.

Owner: Member 4 (agent)
"""

import argparse
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.alerts import load_alerts  # noqa: E402
from src.agent.investigator import investigate  # noqa: E402
from src.agent.llm import AgentLLM, make_chat_model  # noqa: E402
from src.agent.report import DEFAULT_CONFIG, build_alert_report, write_alert_reports  # noqa: E402
from src.agent.specs import load_agent_spec  # noqa: E402
from src.agent.store import TransactionStore, build_store, default_csv_paths, find_data_dir  # noqa: E402
from src.agent.toolkit import CaseToolkit  # noqa: E402
from src.agent.trace import Trace, write_traces  # noqa: E402
from src.agent.validation import validate_case_reports, validate_predictions  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=ROOT / "outputs/predictions/predictions_standin.csv")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--split", default=None, help="only alerts from this split (e.g. validation)")
    ap.add_argument("--data-dir", type=Path, default=find_data_dir(ROOT))
    ap.add_argument("--db", type=Path, default=ROOT / "data/processed/agent_store.duckdb")
    ap.add_argument("--rebuild-store", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="default: outputs/narratives/case_reports_<model>.json")
    ap.add_argument("--skip-llm", action="store_true", help="only build the evidence (step 1)")
    ap.add_argument("--agent", default="investigator", help="definition in src/agent/definitions/agents/")
    ap.add_argument("--llm-model", default=None, help="override the model named in the agent definition")
    ap.add_argument("--workers", type=int, default=4, help="alerts investigated in parallel")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--trace", action="store_true",
                    help="print each agent step live (tool, reason, result); always saved to trace_<model>.jsonl")
    for key, val in DEFAULT_CONFIG.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    a = ap.parse_args()
    if a.trace:
        logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
        logging.getLogger("agent.trace").setLevel(logging.INFO)

    if not a.db.exists() or a.rebuild_store:
        print(f"Building label-free store at {a.db} ...")
        build_store(default_csv_paths(a.data_dir), a.db, overwrite=True)

    check = validate_predictions(a.predictions, a.db)
    print("0. " + check.render().replace("\n", "\n   "))
    if not check.ok:
        sys.exit("Prediction file failed validation; fix it before running the agent.")

    alerts = load_alerts(a.predictions, a.top_k, a.split)
    if not alerts:
        sys.exit(f"No alerts found in {a.predictions}")
    config = {k: getattr(a, k) for k in DEFAULT_CONFIG}
    models = "_".join(sorted({al["model_name"] for al in alerts}))

    t0 = time.time()
    with TransactionStore(a.db) as store:
        reports = [build_alert_report(store, alert, config) for alert in alerts]
        print(f"1. Evidence for {len(reports)} alerts ({time.time() - t0:.1f}s)")

        extra = {}
        if a.skip_llm:
            out = a.out or ROOT / f"outputs/narratives/alert_reports_{models}.json"
        else:
            agent, tool_specs = load_agent_spec(a.agent, toolkit_cls=CaseToolkit)
            model = a.llm_model or agent.model
            llm = AgentLLM(make_chat_model(model, agent.temperature, agent.max_tokens), model,
                           cache_dir=None if a.no_cache else ROOT / "outputs/narratives/llm_cache")
            print(f"2-4. Agent '{agent.name}' on {model}, tools {agent.tools}")
            t0, done = time.time(), [0]
            traces = [Trace(r["alert"]) for r in reports]

            def run(pair):
                report, trace = pair
                result = investigate(report, llm, agent, tool_specs, store, window_hours=a.window_hours, trace=trace)
                done[0] += 1
                if not a.trace:  # with --trace the step lines already show progress
                    print(f"\r     {done[0]}/{len(reports)} alerts", end="", flush=True)
                return result

            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                reports = list(pool.map(run, zip(reports, traces)))
            print(f"  ({time.time() - t0:.0f}s)")
            trace_path = write_traces(traces, (a.out or ROOT / f"outputs/narratives/case_reports_{models}.json")
                                      .with_name(f"trace_{models}.jsonl"))
            print(f"     trace:        {trace_path}  (view: python scripts/show_trace.py --rank 1)")

            status = Counter((r.get("verification") or {}).get("status", "error") for r in reports)
            decisions = Counter(r["triage"]["decision"] for r in reports)
            tokens, tool_calls = Counter(), Counter()
            for r in reports:
                tokens.update((r.get("llm") or {}).get("usage") or {})
                tool_calls.update(c["tool"] for c in (r.get("llm") or {}).get("tool_calls", []))
            extra = {"agent": agent.name, "llm_model": model, "verification_counts": dict(status),
                     "triage_counts": dict(decisions), "llm_tokens": dict(tokens)}
            print(f"     verification: {dict(status)}")
            print(f"     triage:       {dict(decisions)}")
            print(f"     tool calls:   {dict(tool_calls)}")
            print(f"     tokens:       {dict(tokens)}")
            out = a.out or ROOT / f"outputs/narratives/case_reports_{models}.json"

    write_alert_reports(reports, out, a.predictions, config, extra)
    print(f"Wrote {out}")

    if not a.skip_llm:
        result = validate_case_reports(out, a.predictions)
        print("5. " + result.render().replace("\n", "\n   "))
        if not result.ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
