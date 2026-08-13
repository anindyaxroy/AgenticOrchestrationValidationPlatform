"""
pipeline.py — The real pipeline API. Every result is computed from the uploaded file.
Supervisor points: lineage (run_id+content_hash), real mining, derived features,
embedded findings, RL learning proof, LangGraph agent reasoning, audit log.

Supports both single-dataset runs (legacy, still used directly by some UI flows)
and multi-dataset BATCH runs, so uploading several files no longer means
processing them one at a time through the UI.
"""
import os, uuid, asyncio, traceback
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Body, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional

from app.core.database import get_db, AsyncSessionLocal
from app.core.config import settings
from app.models.orm_models import Dataset, PipelineRun
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.knowledge_store import KnowledgeStore
from app.pipeline.reasoning_agent import make_graph
from app.pipeline.report_generator import build_run_report_pdf, build_run_report_html, build_comparison_report_pdf

router = APIRouter()

# Process-wide shared store (Chroma persists to /app/data/chroma)
_store = KnowledgeStore(persist_dir=settings.CHROMA_DIR)
# In-memory run cache (survives single-worker session; backed by DB for durability)
_CACHE: dict[str, dict] = {}
_RL_PROOF: dict[str, dict] = {}
# In-memory batch tracker: batch_id -> {status, episodes, items: {dataset_id: {...}}, created_at}
_BATCHES: dict[str, dict] = {}


async def _get_file(dataset_id: str, db: AsyncSession) -> tuple[str, str]:
    res = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    ds = res.scalar_one_or_none()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    path = ds.file_path
    if not path or not os.path.exists(path):
        raise HTTPException(400, f"File not on disk: {path}")
    return path, ds.id


def _summarise(out: dict) -> dict:
    """Same trimmed shape returned by the single-run endpoint, reused for batch items."""
    return {
        "run_id": out["run_id"],
        "content_hash": out["content_hash"],
        "log_summary": out["log_summary"],
        "mining": {
            "n_variants": len(out["mining"]["variants"]),
            "bottlenecks": out["mining"]["bottlenecks"][:5],
            "bottleneck_basis": out["mining"].get("bottleneck_basis", ""),
            "conformance": out["mining"]["conformance"],
            "directly_follows": out["mining"]["directly_follows"][:5],
            "findings_text": out["mining"]["findings_text"],
        },
        "features": out["features"],
        "rl": {
            "episodes": out["rl"]["episodes"],
            "learned_eval": out["rl"]["learned_eval"],
            "baseline_eval": out["rl"]["baseline_eval"],
            "proof": out["rl"]["proof"],
            "reward_curve": out["rl"]["reward_curve"],
            "reward_curve_smoothed": out["rl"].get("reward_curve_smoothed", []),
            "metrics": out["rl"].get("metrics", {}),
        },
        "agent": out["agent"],
    }


def _slim(out: dict) -> dict:
    """Compact version persisted in the DB (no full reward curve / audit log).

    Deliberately carries enough of mining + RL to drive the Compare page and
    PDF export even after the in-memory _CACHE has been evicted (worker
    restart, container recycle) — see api/pipeline.py compare_runs /
    _row_from_db, and report_generator.build_comparison_report_pdf.
    """
    mining = out["mining"]
    rl = out["rl"]
    return {
        "content_hash": out["content_hash"],
        "log_summary": out["log_summary"],
        "features": {
            "vector": out["features"]["vector"],
            "values": out["features"].get("values", {}),
        },
        "mining_summary": {
            "n_variants": len(mining["variants"]),
            "top_bottleneck": mining["bottlenecks"][0] if mining["bottlenecks"] else None,
            "conformance": mining["conformance"],
        },
        "rl_summary": {
            "episodes": rl["episodes"],
            "learned_eval": rl["learned_eval"],
            "proof": rl["proof"],
            "training": rl.get("metrics", {}).get("training", {}),
        },
    }


async def _run_one(dataset_id: str, episodes: int, db: AsyncSession) -> dict:
    """Run the pipeline for a single dataset, off the event loop thread, and persist it."""
    path, did = await _get_file(dataset_id, db)
    out = await asyncio.to_thread(
        run_pipeline, path, dataset_id=did, episodes=episodes,
        store=_store, log_dir=settings.RUN_LOG_DIR,
    )
    _CACHE[out["run_id"]] = out
    _RL_PROOF[did] = out["rl"]["proof"]
    pr = PipelineRun(
        id=out["run_id"], dataset_id=did,
        content_hash=out["content_hash"], status="complete",
        result=_slim(out), completed_at=datetime.utcnow(),
    )
    db.add(pr)
    await db.commit()
    return _summarise(out)


@router.post("/run/{dataset_id}")
async def run(dataset_id: str, episodes: int = 300, db: AsyncSession = Depends(get_db)):
    return await _run_one(dataset_id, episodes, db)


class BatchRunRequest(BaseModel):
    dataset_ids: List[str]
    episodes: int = 300


async def _execute_batch(batch_id: str, dataset_ids: List[str], episodes: int):
    """Background worker: processes datasets SEQUENTIALLY.

    Sequential, not concurrent, is a deliberate choice: RL training + a live
    Claude API call per file are each meaningfully CPU/IO-heavy, and this
    platform runs as a single backend worker with no task queue (see
    docker-compose.yml — Redis is present but no Celery worker is deployed).
    Running files sequentially avoids resource contention and rate-limit
    collisions that concurrent execution would risk, at the cost of wall-clock
    time. Each file's own progress is still independently visible via polling.
    """
    batch = _BATCHES[batch_id]
    for ds_id in dataset_ids:
        batch["items"][ds_id]["status"] = "running"
        batch["items"][ds_id]["started_at"] = datetime.utcnow().isoformat()
        try:
            async with AsyncSessionLocal() as db:
                result = await _run_one(ds_id, episodes, db)
            batch["items"][ds_id]["status"] = "complete"
            batch["items"][ds_id]["result"] = result
        except Exception as e:
            batch["items"][ds_id]["status"] = "error"
            batch["items"][ds_id]["error"] = str(e)
            batch["items"][ds_id]["traceback"] = traceback.format_exc()
        batch["items"][ds_id]["finished_at"] = datetime.utcnow().isoformat()
    batch["status"] = "complete"
    batch["finished_at"] = datetime.utcnow().isoformat()


@router.post("/run-batch")
async def run_batch(req: BatchRunRequest):
    if not req.dataset_ids:
        raise HTTPException(400, "dataset_ids must be a non-empty list")
    batch_id = str(uuid.uuid4())
    _BATCHES[batch_id] = {
        "batch_id": batch_id,
        "status": "running",
        "episodes": req.episodes,
        "created_at": datetime.utcnow().isoformat(),
        "items": {
            ds_id: {"dataset_id": ds_id, "status": "queued"}
            for ds_id in req.dataset_ids
        },
    }
    asyncio.create_task(_execute_batch(batch_id, req.dataset_ids, req.episodes))
    return _BATCHES[batch_id]


@router.get("/batch/{batch_id}")
async def batch_status(batch_id: str):
    batch = _BATCHES.get(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found — may be on another worker or server restarted")
    return batch


@router.get("/runs/{dataset_id}")
async def list_runs(dataset_id: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(
        select(PipelineRun)
        .where(PipelineRun.dataset_id == dataset_id)
        .order_by(PipelineRun.created_at.desc())
    )
    return [{"run_id": r.id, "content_hash": r.content_hash,
             "status": r.status, "created_at": r.created_at.isoformat()}
            for r in res.scalars().all()]


@router.get("/runs")
async def list_all_runs(db: AsyncSession = Depends(get_db)):
    """Every completed run across every dataset, newest first — feeds the
    Compare Datasets picker (and works after a page refresh, since it's
    DB-backed rather than depending on the in-memory _CACHE)."""
    res = await db.execute(
        select(PipelineRun, Dataset.name)
        .join(Dataset, Dataset.id == PipelineRun.dataset_id)
        .order_by(PipelineRun.created_at.desc())
    )
    out = []
    for pr, dataset_name in res.all():
        slim = pr.result or {}
        out.append({
            "run_id": pr.id,
            "dataset_id": pr.dataset_id,
            "dataset_name": dataset_name,
            "content_hash": pr.content_hash,
            "status": pr.status,
            "created_at": pr.created_at.isoformat(),
            "completed_at": pr.completed_at.isoformat() if pr.completed_at else None,
            "source_filename": (slim.get("log_summary") or {}).get("source_filename"),
            "cached": pr.id in _CACHE,
        })
    return out


class CompareRequest(BaseModel):
    run_ids: List[str]


def _row_from_cache(out: dict, dataset_name: str) -> dict:
    mining, rl, feats = out["mining"], out["rl"], out["features"]
    cmp = (rl.get("proof") or {}).get("comparisons", {})
    return {
        "run_id": out["run_id"], "dataset_name": dataset_name,
        "source_filename": out["log_summary"].get("source_filename"),
        "content_hash": out["content_hash"],
        "n_cases": out["log_summary"].get("n_cases"),
        "n_events": out["log_summary"].get("n_events"),
        "has_timestamps": out["log_summary"].get("has_timestamps"),
        "n_variants": len(mining.get("variants", [])),
        "top_bottleneck": mining["bottlenecks"][0]["activity"] if mining.get("bottlenecks") else None,
        "bottleneck_score": (feats.get("values") or {}).get("bottleneck_score"),
        "rework_probability": (feats.get("values") or {}).get("rework_probability"),
        "conformance_rate": (mining.get("conformance") or {}).get("conformance_rate"),
        "episodes": rl.get("episodes"),
        "learned_mean": (rl.get("learned_eval") or {}).get("mean"),
        "learned_std": (rl.get("learned_eval") or {}).get("std"),
        "comparisons": cmp,
        "convergence_episode": (rl.get("metrics", {}).get("training", {}) or {}).get("convergence_episode"),
        "data_completeness": "cache_full",
    }


def _row_from_db(pr: PipelineRun, dataset_name: str) -> dict:
    slim = pr.result or {}
    ls  = slim.get("log_summary", {}) or {}
    ms  = slim.get("mining_summary", {}) or {}
    rs  = slim.get("rl_summary", {}) or {}
    ft  = slim.get("features", {}) or {}
    complete = "db_summary" if (ms or rs) else "db_legacy"
    return {
        "run_id": pr.id, "dataset_name": dataset_name,
        "source_filename": ls.get("source_filename"),
        "content_hash": pr.content_hash,
        "n_cases": ls.get("n_cases"), "n_events": ls.get("n_events"),
        "has_timestamps": ls.get("has_timestamps"),
        "n_variants": ms.get("n_variants"),
        "top_bottleneck": (ms.get("top_bottleneck") or {}).get("activity"),
        "bottleneck_score": (ft.get("values") or {}).get("bottleneck_score"),
        "rework_probability": (ft.get("values") or {}).get("rework_probability"),
        "conformance_rate": (ms.get("conformance") or {}).get("conformance_rate"),
        "episodes": rs.get("episodes"),
        "learned_mean": (rs.get("learned_eval") or {}).get("mean"),
        "learned_std": (rs.get("learned_eval") or {}).get("std"),
        "comparisons": (rs.get("proof") or {}).get("comparisons", {}),
        "convergence_episode": (rs.get("training") or {}).get("convergence_episode"),
        "data_completeness": complete,
    }


async def _compare_rows(run_ids: List[str], db: AsyncSession) -> dict:
    if not run_ids:
        raise HTTPException(400, "run_ids must be a non-empty list")
    res = await db.execute(
        select(PipelineRun, Dataset.name)
        .where(PipelineRun.id.in_(run_ids))
        .join(Dataset, Dataset.id == PipelineRun.dataset_id)
    )
    by_id = {pr.id: (pr, name) for pr, name in res.all()}
    rows, missing = [], []
    for rid in run_ids:
        if rid in _CACHE:
            _, name = by_id.get(rid, (None, None))
            out = _CACHE[rid]
            rows.append(_row_from_cache(out, name or out["log_summary"].get("dataset_id", "")))
        elif rid in by_id:
            pr, name = by_id[rid]
            rows.append(_row_from_db(pr, name))
        else:
            missing.append(rid)
    return {"rows": rows, "missing": missing}


@router.post("/compare")
async def compare_runs(req: CompareRequest, db: AsyncSession = Depends(get_db)):
    return await _compare_rows(req.run_ids, db)

@router.get("/result/{run_id}")
async def result(run_id: str):
    if run_id not in _CACHE:
        raise HTTPException(404, "Run not in cache — may be on another worker")
    return _CACHE[run_id]

@router.get("/audit/{run_id}")
async def audit(run_id: str):
    import json
    path = os.path.join(settings.RUN_LOG_DIR, f"{run_id}.jsonl")
    if not os.path.exists(path):
        raise HTTPException(404, "Audit log not found")
    with open(path) as f:
        return {"run_id": run_id, "events": [json.loads(l) for l in f if l.strip()]}


@router.get("/report/{run_id}")
async def run_report(run_id: str, fmt: str = "pdf", db: AsyncSession = Depends(get_db)):
    """Full report for one run — every stage the pipeline executed, in
    order: dataset characterisation, mining findings, MDP feature vector,
    RL proof, agent reasoning, and the audit trail. `fmt` is 'pdf' (default)
    or 'html'. Needs the run in _CACHE (same requirement as /result) since
    that's the only place the full mining/audit detail lives."""
    if fmt not in ("pdf", "html"):
        raise HTTPException(400, "fmt must be 'pdf' or 'html'")
    if run_id not in _CACHE:
        raise HTTPException(
            404,
            "This run's full detail isn't in the server's live cache (e.g. after a restart). "
            "Open its results in the app once — or re-run the pipeline — to regenerate it."
        )
    out = _CACHE[run_id]
    res = await db.execute(
        select(Dataset.name).join(PipelineRun, PipelineRun.dataset_id == Dataset.id)
        .where(PipelineRun.id == run_id)
    )
    dataset_name = res.scalar_one_or_none() or out["log_summary"].get("dataset_id", "")

    if fmt == "html":
        html = build_run_report_html(out, dataset_name, out.get("audit_log"))
        filename = f"bpmn_report_{out['content_hash']}.html"
        return Response(content=html, media_type="text/html",
                         headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    pdf_bytes = build_run_report_pdf(out, dataset_name, out.get("audit_log"))
    filename = f"bpmn_report_{out['content_hash']}.pdf"
    return Response(content=pdf_bytes, media_type="application/pdf",
                     headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/report/compare")
async def compare_report(req: CompareRequest, db: AsyncSession = Depends(get_db)):
    """PDF comparing several runs (possibly across different datasets) side
    by side. Runs missing from both cache and DB are listed as skipped in
    the report rather than failing the whole export."""
    if len(req.run_ids) < 2:
        raise HTTPException(400, "Select at least 2 runs to compare")
    result = await _compare_rows(req.run_ids, db)
    if len(result["rows"]) < 2:
        raise HTTPException(400, "Fewer than 2 of the selected runs have data available to compare")
    pdf_bytes = build_comparison_report_pdf(result["rows"], result["missing"])
    return Response(content=pdf_bytes, media_type="application/pdf",
                     headers={"Content-Disposition": 'attachment; filename="bpmn_comparison_report.pdf"'})

@router.post("/ask/{dataset_id}")
async def ask(dataset_id: str, question: str = Body(..., embed=True),
              db: AsyncSession = Depends(get_db)):
    if _store.count(dataset_id) == 0:
        raise HTTPException(400, "No findings embedded yet — run the pipeline first")
    proof = _RL_PROOF.get(dataset_id, {})
    graph = make_graph(_store, lambda _d: proof)
    raw = await asyncio.to_thread(graph.invoke, {"data": {"dataset_id": dataset_id, "question": question}})
    d = raw.get("data", {})
    out = {"reasoning": d.get("reasoning",""), "summary": d.get("summary",""), "node_audit": d.get("audit",[])}
    return {
        "dataset_id": dataset_id, "question": question,
        "reasoning": out.get("reasoning", ""),
        "summary": out.get("summary", ""),
        "node_audit": out.get("audit", []),
    }


@router.get("/episodes/{run_id}")
async def episode_log(run_id: str, mode: str = "all",
                      limit: int = 50, offset: int = 0):
    """
    Return per-episode trace log for a run.
    mode: 'all' | 'train' | 'eval_learned' | 'eval_random' | 'eval_greedy' | 'eval_do_nothing'
    Each entry includes: episode, mode, seed, total_reward, epsilon, q_table_size,
    initial_state, final_state, steps (with action, reward_delta, state_before/after, q_values).
    """
    import json
    path = os.path.join(settings.RUN_LOG_DIR, f"{run_id}_episodes.jsonl")
    if not os.path.exists(path):
        raise HTTPException(404, "Episode log not found — run the pipeline first")
    with open(path) as f:
        all_entries = [json.loads(l) for l in f if l.strip()]
    if mode != "all":
        all_entries = [e for e in all_entries if e.get("mode") == mode]
    total = len(all_entries)
    return {
        "run_id":  run_id,
        "mode":    mode,
        "total":   total,
        "offset":  offset,
        "limit":   limit,
        "entries": all_entries[offset: offset + limit],
    }
