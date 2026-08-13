"""
datasets.py — Multi-file upload with real data preview.

Supports: .xes, .csv, .tsv, .gz (gzipped XES)
Multiple files can be uploaded in one request.
Each uploaded file gets a preview computed immediately so the
Data Ingestion page can show what is actually in the file.
"""
import os, uuid, json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from app.core.config   import settings
from app.models.orm_models import Dataset

# Import the real parser so we can compute a preview on upload
from app.pipeline.log_loader import load_event_log, _detect_format

router = APIRouter()

ALLOWED_EXT = {".xes", ".csv", ".tsv", ".gz", ".xes.gz"}


def _ext(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".xes.gz"):
        return ".xes.gz"
    return os.path.splitext(name)[1]


def _safe_name(filename: str) -> str:
    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base)
    if base.lower().endswith(".xes.gz"):
        stem = base[:-7]; ext = ".xes.gz"
    return f"{stem}_{uuid.uuid4().hex[:8]}{ext}"


def _compute_preview(path: str, dataset_id: str) -> dict:
    """Parse the real file and return a rich preview dict."""
    try:
        log = load_event_log(path, dataset_id=dataset_id)
        
        # Activity frequency
        from collections import Counter
        act_counts = Counter()
        for t in log.traces:
            act_counts.update(t.activities)
        top_acts = [{"activity": a, "count": c}
                    for a, c in act_counts.most_common(10)]
        
        # Trace length stats
        lengths = [t.length for t in log.traces] if log.traces else [0]
        mean_len = round(sum(lengths) / len(lengths), 1)
        
        # Sample traces (first 3)
        samples = []
        for t in log.traces[:3]:
            samples.append({
                "case_id":    t.case_id,
                "length":     t.length,
                "sequence":   t.activities[:12],
                "attributes": {k: str(v) for k, v in t.attributes.items()},
            })
        
        # Label balance
        label_key = next((k for k in log.trace_attr_keys
                          if "ispos" in k.lower() or "label" in k.lower()), None)
        label_balance = None
        if label_key:
            pos = sum(1 for t in log.traces if t.attributes.get(label_key) is True)
            label_balance = {"key": label_key, "positive": pos,
                             "negative": log.n_cases - pos}
        
        # Cost attribute
        cost_key = next((k for k in log.trace_attr_keys
                         if "cost" in k.lower()), None)
        cost_stats = None
        if cost_key:
            vals = [float(t.attributes.get(cost_key, 0)) for t in log.traces]
            cost_stats = {
                "key":  cost_key,
                "min":  round(min(vals), 2),
                "max":  round(max(vals), 2),
                "mean": round(sum(vals) / len(vals), 2),
            }
        
        return {
            "status":          "ok",
            "format":          log.fmt,
            "n_cases":         log.n_cases,
            "n_events":        log.n_events,
            "n_activities":    len(log.activity_alphabet),
            "has_timestamps":  log.has_timestamps,
            "trace_attr_keys": log.trace_attr_keys,
            "activity_alphabet": log.activity_alphabet[:30],
            "top_activities":  top_acts,
            "mean_trace_length": mean_len,
            "min_trace_length":  min(lengths),
            "max_trace_length":  max(lengths),
            "sample_traces":   samples,
            "label_balance":   label_balance,
            "cost_stats":      cost_stats,
            "content_hash":    log.content_hash[:16],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _ds_dict(ds: Dataset) -> dict:
    return {
        "id":         ds.id,
        "name":       ds.name,
        "source":     ds.source,
        "file_path":  ds.file_path,
        "file_size":  ds.file_size,
        "status":     ds.status,
        "metadata":   ds.metadata_ or {},
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
    }


async def _get_or_404(dataset_id: str, db: AsyncSession) -> Dataset:
    res = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    ds = res.scalar_one_or_none()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return ds


# ── Routes ────────────────────────────────────────────────────

@router.get("/")
async def list_datasets(db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    return [_ds_dict(d) for d in res.scalars().all()]


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    ds = await _get_or_404(dataset_id, db)
    return _ds_dict(ds)


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """Return a rich data preview for the ingestion page."""
    ds = await _get_or_404(dataset_id, db)
    if not ds.file_path or not os.path.exists(ds.file_path):
        raise HTTPException(400, "File not on disk")
    # Return cached preview if available
    meta = ds.metadata_ or {}
    if meta.get("preview"):
        return meta["preview"]
    # Compute on demand
    preview = _compute_preview(ds.file_path, ds.id)
    return preview


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a dataset record and its uploaded file. Referenced by the frontend
    delete button, which previously had no matching route."""
    ds = await _get_or_404(dataset_id, db)
    if ds.file_path and os.path.exists(ds.file_path):
        try:
            os.remove(ds.file_path)
        except OSError:
            pass
    await db.delete(ds)
    await db.commit()
    return {"deleted": dataset_id}


@router.post("/upload")
async def upload_single(file: UploadFile = File(...),
                        db: AsyncSession = Depends(get_db)):
    """Upload a single file (backward compatible)."""
    results = await _ingest_files([file], db)
    return results[0]


@router.post("/upload-many")
async def upload_many(files: List[UploadFile] = File(...),
                      db: AsyncSession = Depends(get_db)):
    """Upload multiple files at once."""
    return await _ingest_files(files, db)


async def _ingest_files(files: List[UploadFile],
                        db: AsyncSession) -> list:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    results = []
    for file in files:
        fname = file.filename or "upload"
        ext = _ext(fname)
        if ext not in ALLOWED_EXT:
            results.append({
                "name":  fname,
                "error": f"Unsupported type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}"
            })
            continue
        safe = _safe_name(fname)
        dest = os.path.join(settings.UPLOAD_DIR, safe)
        raw  = await file.read()
        with open(dest, "wb") as f:
            f.write(raw)
        # Compute preview immediately
        ds_id_tmp = str(uuid.uuid4())
        preview = _compute_preview(dest, ds_id_tmp)
        ds = Dataset(
            name      = fname,
            source    = "upload",
            file_path = dest,
            file_size = len(raw),
            status    = "ready" if preview["status"] == "ok" else "error",
            metadata_ = {
                "original_name": fname,
                "ext":           ext,
                "preview":       preview,
            },
        )
        db.add(ds)
        await db.flush()
        await db.refresh(ds)
        d = _ds_dict(ds)
        d["preview"] = preview
        results.append(d)
    return results
