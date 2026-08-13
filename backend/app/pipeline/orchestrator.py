"""
orchestrator.py — End-to-end pipeline under ONE traceable run_id.
Passes run_id to rl_agent so episode logs are named {run_id}_episodes.jsonl.
"""
from __future__ import annotations
import os
from typing import Any, Optional
from .log_loader         import load_event_log
from .feature_extraction import extract_features
from .process_mining     import run_mining
from .knowledge_store    import KnowledgeStore
from .rl_agent           import train_and_prove_per_trace
from .reasoning_agent    import make_graph
from .audit              import RunContext


def run_pipeline(file_path: str, dataset_id: str,
                 question: str = (
                     "Characterise this process: main bottleneck, "
                     "conformance profile, and did the RL agent learn?"
                 ),
                 episodes: int = 300,
                 store: Optional[KnowledgeStore] = None,
                 log_dir: str = "/app/data/runs") -> dict[str, Any]:

    # Stage 0: load + lineage
    log = load_event_log(file_path, dataset_id=dataset_id)
    ctx = RunContext.create(dataset_id=dataset_id,
                            source_filename=log.source_filename,
                            content_hash=log.content_hash,
                            log_dir=log_dir)
    ctx.log("log_loaded", log.summary())

    # Stage 1: process mining
    mr = run_mining(log)
    assert ctx.lineage_check(mr.content_hash), "mining hash mismatch"
    ctx.log("mining_done", {
        "n_variants":       len(mr.variants),
        "top_bottleneck":   mr.bottlenecks[0]["activity"] if mr.bottlenecks else None,
        "conformance_rate": mr.conformance.get("conformance_rate"),
        "n_findings":       len(mr.findings_text()),
    })

    # Stage 2: feature extraction
    fv = extract_features(log)
    assert ctx.lineage_check(fv.content_hash), "features hash mismatch"
    ctx.log("features_extracted", {
        "vector":               fv.as_array(),
        "time_based_available": fv.time_based_available,
        "provenance":           fv.provenance,
    })

    # Stage 3: embed findings
    store = store or KnowledgeStore()
    n_emb = store.add_findings(mr.dataset_id, mr.content_hash,
                               mr.findings_text(), kind="mining")
    ctx.log("findings_embedded", {
        "n_embedded":  n_emb,
        "store_count": store.count(dataset_id),
        "findings":    mr.findings_text(),
    })

    # Stage 4: RL training + proof, on REAL per-trace episodes (see trace_env.py
    # and rl_agent.train_and_prove_per_trace for why this replaced the old
    # single-synthetic-point environment). Episode log named after run_id.
    rl = train_and_prove_per_trace(log, fv, episodes=episodes,
                                   log_dir=log_dir, run_id=ctx.run_id)
    assert ctx.lineage_check(rl.content_hash), "rl hash mismatch"
    ctx.log("rl_trained", {
        "episodes":          rl.episodes,
        "learned_mean":      rl.learned_eval["mean"],
        "beats_random":      rl.proof["comparisons"]["random"]["learned_better"],
        "beats_greedy":      rl.proof["comparisons"]["greedy"]["learned_better"],
        "q_states":          rl.q_table_size,
        "convergence_ep":    rl.metrics["training"]["convergence_episode"],
        "policy_entropy":    rl.metrics["training"]["policy_entropy"],
        "episode_log_path":  rl.episode_log_path,
    })

    # Stage 5: agent reasoning
    def rl_provider(_did): return rl.proof
    graph     = make_graph(store, rl_provider)
    raw       = graph.invoke({"data": {"dataset_id": dataset_id, "question": question}})
    agent_data = raw.get("data", {})
    agent_out  = {
        "reasoning":  agent_data.get("reasoning", ""),
        "summary":    agent_data.get("summary", ""),
        "node_audit": agent_data.get("audit", []),
    }
    ctx.log("agent_reasoned", {
        "nodes_run":     [a["node"] for a in agent_out["node_audit"]],
        "summary_chars": len(agent_out["summary"]),
    })
    ctx.log("run_completed", {"status": "ok"})

    return {
        "run_id":            ctx.run_id,
        "content_hash":      log.content_hash[:16],
        "log_summary":       log.summary(),
        "mining":            mr.to_dict(),
        "features":          fv.to_dict(),
        "rl":                rl.to_dict(),
        "agent":             agent_out,
        "audit_log":         ctx.read_log(),
        "episode_log_path":  rl.episode_log_path,
    }
