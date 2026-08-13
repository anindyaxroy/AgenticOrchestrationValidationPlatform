# BPMN Agentic Orchestration Validation Platform

**Anindya Roy — University of Amsterdam, MBA in AI, Data & Analytics — Thesis, 2026**

## What this is

A validation platform demonstrating that an autonomous, reinforcement-learning-based agent can learn from process-mining findings on real BPMN event logs and generate defensible process-optimisation recommendations under human-in-the-loop oversight.

Every number displayed by the platform is computed from the uploaded file through a disclosed, six-stage pipeline. Nothing is hardcoded, and every reasoning-stage recommendation is grounded in pre-computed, retrievable evidence rather than generated freely by a language model.

## Documentation

| Document | Covers |
|---|---|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Service topology, the six-stage pipeline, data flow |
| [`SYSTEM_DESIGN.md`](./SYSTEM_DESIGN.md) | Design rationale — state representation, RL formulation, disclosed corrections, open questions |
| [`USAGE.md`](./USAGE.md) | Setup, running a pipeline, API reference, troubleshooting |

## Quick start — local (no Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline_cli.py /path/to/your_file.xes

# With live agent reasoning:
export ANTHROPIC_API_KEY=sk-ant-...
python run_pipeline_cli.py /path/to/your_file.xes --ask "Where is the bottleneck?"
```

## Quick start — Docker (full stack)

```bash
# 1. Copy .env.example to .env and fill in real values (never commit .env — see below)
cp .env.example .env

# 2. Build and start everything
docker compose up -d --build

# 3. Open the platform
open http://localhost:8080

# 4. Check logs
docker compose logs -f backend

# 5. Tear down completely (removes all state)
docker compose down -v
```

See [`USAGE.md`](./USAGE.md) for the full workflow, API reference, and troubleshooting (large-file uploads, Windows PowerShell issues, etc.).

## Secrets

`docker-compose.yml` reads `SECRET_KEY`, the database password, and `ANTHROPIC_API_KEY` from environment variables — none of these are hardcoded in a committed file. Copy `.env.example` to `.env` and fill in real values; `.env` is gitignored and must never be committed. If you're sharing access to a running instance with someone outside this repo (e.g., a reviewer), see the access-control notes in `USAGE.md` — that access layer's credentials follow the same rule: generated locally, never committed.

## License / attribution

This is a thesis research artefact. See the thesis document itself for full methodology, citations, and academic context.
