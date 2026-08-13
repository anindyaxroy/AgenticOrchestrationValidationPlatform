"""
BPMN Agentic Orchestration Validation Platform
Anindya Roy — UvA MBA AI Thesis 2026
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.core.database import create_tables
from app.api import datasets, pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    # ensure data dirs exist at startup
    for d in ["/app/data/uploads", "/app/data/runs", "/app/data/chroma"]:
        os.makedirs(d, exist_ok=True)
    yield

app = FastAPI(
    title="BPMN Agentic Orchestration Platform",
    description="Thesis validation platform — Anindya Roy, UvA MBA AI 2026",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "bpmn-agentic-platform", "version": "2.0.0"}

@app.get("/api/info")
async def info():
    return {
        "title": "BPMN Agentic Orchestration Validation Platform",
        "version": "2.0.0",
        "researcher": "Anindya Roy",
        "institution": "University of Amsterdam — MBA in AI, Data & Analytics",
        "thesis": "Autonomous AI Agents for Optimization of Real-Time BPMN Workflow Orchestration",
        "endpoints": {
            "upload": "POST /api/datasets/upload",
            "run_pipeline": "POST /api/pipeline/run/{dataset_id}",
            "audit": "GET /api/pipeline/audit/{run_id}",
            "ask_agent": "POST /api/pipeline/ask/{dataset_id}",
            "docs": "/docs",
        }
    }
