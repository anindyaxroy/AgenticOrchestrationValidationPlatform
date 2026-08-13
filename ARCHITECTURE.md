# Architecture

This document describes how the BPMN Agentic Orchestration Validation Platform is put together: the service topology, the six-stage processing pipeline, and how data flows from an uploaded event log to a generated recommendation.

## Service topology

```
                        ┌─────────────────────────────────────────┐
                        │              nginx (:8080)               │
                        │  single external entry point              │
                        └───────────────┬───────────────────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
        location /              location /api/           location /docs
        location /ws/           →  backend:8000          location /openapi.json
                │                       │                       │
                ▼                       ▼                       ▼
        ┌───────────────┐      ┌─────────────────┐     FastAPI auto-docs
        │   frontend      │      │    backend       │     (Swagger UI)
        │  React / Vite   │      │  FastAPI + Uvicorn│
        │  served via     │      └────────┬──────────┘
        │  its own nginx  │               │
        └───────────────┘      ┌──────────┼──────────┐
                                │          │          │
                                ▼          ▼          ▼
                        ┌───────────┐ ┌────────┐ ┌──────────┐
                        │ Postgres 16│ │ Redis 7 │ │  Chroma   │
                        │  (state)   │ │ (cache) │ │(knowledge │
                        │            │ │         │ │  store)   │
                        └───────────┘ └────────┘ └──────────┘
```

Five containers, defined in `docker-compose.yml`:

| Service | Image / build | Port (host) | Role |
|---|---|---|---|
| `nginx` | `nginx:alpine` | `8080` | Single external entry point. Routes `/` to the frontend, `/api/`, `/docs`, `/openapi.json`, `/health` to the backend, `/ws/` for WebSocket updates. |
| `frontend` | built from `frontend/Dockerfile` | internal only | React/Vite single-page app, served behind its own nginx layer inside the container. Not directly exposed — reached only through the top-level nginx. |
| `backend` | built from `backend/Dockerfile` | `8000` (also exposed directly for local dev/debugging) | FastAPI application. Owns the entire six-stage pipeline described below. |
| `db` | `postgres:16-alpine` | internal only | Dataset, run, and audit-log metadata. Initialised from `scripts/init_db.sql`. |
| `redis` | `redis:7-alpine` | internal only | Caching / job-state layer. |

All five share a single Docker network (`bpmn_net`), so only `nginx`'s `8080` (and `backend`'s `8000`, exposed for convenience during development) need to be reachable from outside the Docker host at all — everything else talks to everything else over the internal network only.

## Data and file layout

Everything persistent lives under `./data`, bind-mounted into the backend container at `/app/data`:

```
data/
  uploads/     raw uploaded event logs (.xes, .xes.gz, .csv, .tsv)
  runs/        per-run output: episode logs (JSONL), reports, audit trail
  chroma/      the embedded knowledge-store vector index
```

This directory is intentionally **not** committed to git (see `.gitignore`) — it's runtime state and user data, not source.

## The six-stage pipeline

Every dataset, regardless of format or timestamp availability, passes through the identical six stages. This is the architectural principle the whole platform is organised around: the pipeline doesn't branch by dataset type, only individual feature computations do (Stage 3).

```
 1. Log loading        2. Process mining      3. Feature extraction
    + lineage hashing  ────────────────────►  ────────────────────►
    (log_loader.py)        (process_mining.py)    (feature_extraction.py)
        │                       │                       │
        ▼                       ▼                       ▼
   EventLog +              MiningResult           6-dim FeatureVector
   SHA-256 hash          (variants, DFG,          (real-timestamp branch
                          bottlenecks,              or sequence-position
                          conformance)               proxy branch)
                                                        │
        ┌───────────────────────────────────────────────┘
        ▼
 4. Embedding           5. RL training          6. Agent reasoning
    ────────────────►   ────────────────────►   ────────────────────►
    (knowledge_store.py)    (rl_agent.py +          (reasoning_agent.py,
                              trace_env.py)           LangGraph + Claude)
        │                       │                       │
        ▼                       ▼                       ▼
   Findings embedded      Trained Q-table +       Grounded narrative:
   into Chroma            baseline comparison      executive summary,
   (retrievable text)     (random/do-nothing/       bottleneck analysis,
                            greedy)                  recommendations
```

### Stage 1 — Log loading and lineage hashing (`log_loader.py`)

Detects format (`.xes`, `.xes.gz`, `.csv`, `.tsv`) by extension or content-sniffing, normalises into a single internal `EventLog`/`Trace` representation, and computes a SHA-256 hash of the raw file bytes before any parsing occurs. That hash is attached to every downstream artefact so any reported number traces back to the exact input file.

### Stage 2 — Process mining (`process_mining.py`)

Deterministic, code-computed exploratory analysis: variant discovery, activity frequency, the directly-follows graph (and self-loop detection within it), bottleneck ranking, and — where a conformance label is present — a conformance summary. Produces auto-generated natural-language findings text, which is what Stage 4 embeds (not raw statistics).

### Stage 3 — Feature extraction (`feature_extraction.py`)

Computes the six-dimensional MDP state vector (bottleneck score, SLA risk index, cost variance, dominant activity share, rework probability, resource utilisation). Three of the six branch between a real-timestamp formula and a sequence-position proxy, gated by a single per-file coverage check (`≥80%` usable timestamp coverage). Every feature carries an explicit provenance string recording which formula and data source produced it.

### Stage 4 — Embedding (`knowledge_store.py`)

Pushes the Stage 2 findings text into a Chroma vector store, tagged with dataset ID and content hash, so Stage 6 can retrieve exactly the relevant findings for a given run.

### Stage 5 — RL training (`rl_agent.py`, `trace_env.py`)

Trains a tabular Q-learning agent per-trace against the six-feature state space (discretised into 5 bins/dimension), evaluates it on a held-out trace split (zero case-ID overlap with training, verified programmatically) against three baselines — random, do-nothing, and one-step greedy — and reports mean advantage, win rate, relative improvement, MAE, and Cohen's d for each comparison.

### Stage 6 — Agent reasoning (`reasoning_agent.py`)

A LangGraph pipeline: `retrieve_node` pulls relevant findings from the Stage 4 knowledge store, `rl_node` pulls the Stage 5 comparison table, `reason_node` feeds both to Claude as grounding context. The model narrates and interprets pre-computed evidence — it does not compute any statistic itself. An offline fallback path (`_grounded_markdown`) produces the same structure without a live model call, for environments without API access.

## Why this matters for reviewing the platform

Because every stage's inputs and outputs are disclosed (the hash in Stage 1, the provenance strings in Stage 3, the retrievable findings text in Stage 4, the comparison table in Stage 5), any number appearing in a generated report can be traced back through exactly one deterministic path to the original uploaded file. Nothing in the pipeline output is invented at the reasoning stage — see `SYSTEM_DESIGN.md` for the reasoning behind that constraint.
