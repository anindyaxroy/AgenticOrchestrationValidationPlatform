"""
audit.py — Lineage spine + append-only audit log.

Your requirement: "our application also produces log and traceability so we can
audit the steps." And supervisor point 9: one dataset must be the traceable
context across the whole run.

Design:
  - A RunContext is created once per pipeline execution. It holds run_id,
    dataset_id, source_filename, and content_hash (the lineage anchor).
  - Every stage logs a structured event via the same context. Each event records
    the run_id + content_hash, so any output can be traced to (a) the run that
    produced it and (b) the exact bytes of the source file.
  - The log is append-only JSONL: tamper-evident-ish, replayable, and trivial to
    show an examiner ("here is exactly what the system did, step by step").
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunContext:
    run_id: str
    dataset_id: str
    source_filename: str
    content_hash: str
    started_at: str
    log_path: str
    _seq: int = 0

    @classmethod
    def create(cls, dataset_id: str, source_filename: str, content_hash: str,
               log_dir: str = "/app/data/runs") -> "RunContext":
        os.makedirs(log_dir, exist_ok=True)
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        log_path = os.path.join(log_dir, f"{run_id}.jsonl")
        ctx = cls(run_id=run_id, dataset_id=dataset_id,
                  source_filename=source_filename, content_hash=content_hash,
                  started_at=_now_iso(), log_path=log_path)
        ctx.log("run_started", {
            "dataset_id": dataset_id,
            "source_filename": source_filename,
            "content_hash": content_hash,
        })
        return ctx

    def log(self, event: str, detail: dict[str, Any], status: str = "ok") -> dict:
        self._seq += 1
        entry = {
            "seq": self._seq,
            "ts": _now_iso(),
            "run_id": self.run_id,
            "content_hash": self.content_hash[:16],   # lineage on EVERY line
            "event": event,
            "status": status,
            "detail": detail,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def read_log(self) -> list[dict]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def lineage_check(self, claimed_hash: str) -> bool:
        """Verify a downstream artifact belongs to THIS run's source file."""
        return claimed_hash[:16] == self.content_hash[:16]
