-- Fresh schema for BPMN Agentic Platform (clean build)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS datasets (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    name        VARCHAR(255) NOT NULL,
    source      VARCHAR(50)  NOT NULL DEFAULT 'upload',
    file_path   TEXT,
    file_size   BIGINT,
    status      VARCHAR(20)  NOT NULL DEFAULT 'ready',
    metadata    JSONB        DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id           TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    dataset_id   TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    content_hash TEXT,
    status       VARCHAR(20) NOT NULL DEFAULT 'running',
    result       JSONB       DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON pipeline_runs(dataset_id);
