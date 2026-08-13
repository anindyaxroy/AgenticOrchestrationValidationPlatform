#!/usr/bin/env python3
"""
run_pipeline_cli.py — Test the full pipeline on any XES/CSV WITHOUT Docker.

Usage:
    cd backend
    pip install -r requirements.txt
    python run_pipeline_cli.py /path/to/pdc_2016_1.xes

With live agent reasoning:
    ANTHROPIC_API_KEY=sk-ant-... python run_pipeline_cli.py /path/to/pdc_2016_1.xes \
        --ask "Where is the bottleneck and did the agent learn?"
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.knowledge_store import KnowledgeStore
from app.pipeline.reasoning_agent import make_graph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--ask", default=None)
    ap.add_argument("--chroma", default="/tmp/bpmn_chroma")
    ap.add_argument("--runs",   default="/tmp/bpmn_runs")
    ap.add_argument("--json",   action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")

    store = KnowledgeStore(persist_dir=args.chroma)
    dataset_id = "cli-" + os.path.basename(args.file)
    out = run_pipeline(args.file, dataset_id=dataset_id,
                       episodes=args.episodes, store=store, log_dir=args.runs)

    W = 70
    print("=" * W)
    print(f"FILE:         {out['log_summary']['source_filename']}")
    print(f"RUN ID:       {out['run_id']}")
    print(f"CONTENT HASH: {out['content_hash']}  ← lineage anchor")
    print("=" * W)
    s = out["log_summary"]
    print(f"cases={s['n_cases']}  events={s['n_events']}  "
          f"activities={s['n_activities']}  timestamps={s['has_timestamps']}")
    print(f"trace attributes: {s['trace_attr_keys']}")

    print("\nPROCESS MINING")
    print(f"  variants: {len(out['mining']['variants'])}")
    b0 = out["mining"]["bottlenecks"][0]
    print(f"  top bottleneck: {b0['activity']} (score {b0['score']})")
    print(f"  conformance: {out['mining']['conformance']}")
    print("  findings (embedded for the agent):")
    for f in out["mining"]["findings_text"]:
        print(f"    ✦ {f}")

    print("\nFEATURES (derived — not hardcoded)")
    print(f"  vector: {out['features']['vector']}")
    print(f"  time-based available: {out['features']['time_based_available']}")
    print("  provenance:")
    for k, v in out["features"]["provenance"].items():
        print(f"    {k}: {v[:80]}")

    print("\nRL LEARNING PROOF (held-out seeds agent never trained on)")
    print(f"  learned mean return: {out['rl']['learned_eval']}")
    for name, c in out["rl"]["proof"]["comparisons"].items():
        verdict = "✓ BEATS" if c["learned_better"] else "✗ does NOT beat"
        print(f"  vs {name:<12}: {verdict}  "
              f"advantage={c['mean_advantage']}  win_rate={c['win_rate']}  d={c['cohens_d']}")

    print("\nAGENT REASONING (LangGraph — 5 nodes)")
    print("  nodes:", [a["node"] for a in out["agent"]["node_audit"]])
    print("  summary:")
    print("   ", out["agent"]["summary"][:600].replace("\n", "\n    "))

    print("\nAUDIT LOG (append-only — every line carries content_hash)")
    for e in out["audit_log"]:
        print(f"  #{e['seq']} [{e['content_hash']}] {e['event']}")

    if args.ask:
        print(f"\n{'=' * W}")
        print(f"AGENT QUESTION: {args.ask}")
        proof = out["rl"]["proof"]
        graph = make_graph(store, lambda _d: proof)
        res = graph.invoke({"dataset_id": dataset_id, "question": args.ask})
        print(res.get("summary", ""))

    if args.json:
        with open("pipeline_result.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("\nFull result → pipeline_result.json")

if __name__ == "__main__":
    main()
