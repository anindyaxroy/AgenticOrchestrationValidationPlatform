"""
log_loader.py — Format-aware event-log loader with lineage.

Supervisor points addressed:
  (4)(9) traceability: every loaded log carries a dataset_id + content hash, so
         every downstream number can be traced back to THIS exact file.
  (6)    real data: we parse the actual file, we do not transcribe constants.

Design: different incoming file types are detected and parsed differently
(XES vs CSV today; extensible). The output is a single normalized EventLog
object so every downstream stage sees the same shape regardless of source format.
"""
from __future__ import annotations

import hashlib
import os
import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Trace:
    case_id: str
    activities: list[str]                      # ordered activity sequence
    timestamps: list[Optional[datetime]]       # parallel to activities; None if absent
    attributes: dict[str, Any] = field(default_factory=dict)   # trace-level (e.g. pdc:isPos, pdc:costs)

    @property
    def length(self) -> int:
        return len(self.activities)


@dataclass
class EventLog:
    dataset_id: str
    source_filename: str
    content_hash: str            # sha256 of raw bytes — the lineage anchor
    fmt: str                     # 'xes' | 'csv'
    traces: list[Trace]
    has_timestamps: bool
    trace_attr_keys: list[str]   # which trace-level attributes exist (e.g. pdc:isPos)
    activity_alphabet: list[str] # sorted unique activities

    # ---- convenience accessors used by downstream stages ----
    @property
    def n_cases(self) -> int:
        return len(self.traces)

    @property
    def n_events(self) -> int:
        return sum(t.length for t in self.traces)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_filename": self.source_filename,
            "content_hash": self.content_hash[:16],   # short form for display
            "format": self.fmt,
            "n_cases": self.n_cases,
            "n_events": self.n_events,
            "n_activities": len(self.activity_alphabet),
            "has_timestamps": self.has_timestamps,
            "trace_attr_keys": self.trace_attr_keys,
        }


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _detect_format(filename: str, raw: bytes) -> str:
    low = filename.lower()
    if low.endswith(".gz"):
        low = low[:-3]
    if low.endswith(".xes"):
        return "xes"
    if low.endswith((".csv", ".tsv")):
        return "csv"
    # sniff: XES is XML with <log>
    head = raw[:512].lstrip()
    if head.startswith(b"<?xml") or head.startswith(b"<log"):
        return "xes"
    return "csv"


def _coerce_ts(val: str) -> Optional[datetime]:
    if not val:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(val.split("+")[0] if "+" in val and "%z" not in fmt else val, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _parse_xes(raw: bytes) -> tuple[list[Trace], bool, set]:
    content = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    root = ET.fromstring(content.decode("utf-8", errors="replace"))

    # strip namespace if present
    def tag(e):
        return e.tag.split("}")[-1]

    traces: list[Trace] = []
    has_ts = False
    attr_keys: set = set()

    for tr in [c for c in root if tag(c) == "trace"]:
        case_id = ""
        tattrs: dict[str, Any] = {}
        acts: list[str] = []
        tss: list[Optional[datetime]] = []

        for child in tr:
            t = tag(child)
            key = child.get("key", "")
            if t in ("string", "float", "int", "boolean", "date") and key:
                # trace-level attribute
                val = child.get("value", "")
                if key == "concept:name":
                    case_id = val
                else:
                    attr_keys.add(key)
                    if t == "float":
                        tattrs[key] = float(val) if val else 0.0
                    elif t == "int":
                        tattrs[key] = int(val) if val else 0
                    elif t == "boolean":
                        tattrs[key] = (val.lower() == "true")
                    else:
                        tattrs[key] = val
            elif t == "event":
                act = None
                ts = None
                for ec in child:
                    ek = ec.get("key", "")
                    if ek == "concept:name":
                        act = ec.get("value", "")
                    elif ek == "time:timestamp":
                        ts = _coerce_ts(ec.get("value", ""))
                        has_ts = has_ts or ts is not None
                if act is not None:
                    acts.append(act)
                    tss.append(ts)

        traces.append(Trace(case_id=case_id or str(len(traces) + 1),
                            activities=acts, timestamps=tss, attributes=tattrs))
    return traces, has_ts, attr_keys


def _parse_csv(raw: bytes) -> tuple[list[Trace], bool, set]:
    import csv as _csv
    import io
    text = raw.decode("utf-8", errors="replace")
    sniff = _csv.Sniffer()
    try:
        dialect = sniff.sniff(text[:2048])
    except _csv.Error:
        dialect = _csv.excel
    reader = _csv.DictReader(io.StringIO(text), dialect=dialect)
    headers = reader.fieldnames or []

    def pick(cands):
        hl = {h.lower().replace(" ", "_").replace(":", "_"): h for h in headers}
        for c in cands:
            k = c.lower().replace(":", "_")
            if k in hl:
                return hl[k]
        return None

    c_case = pick(["case:concept:name", "case_id", "caseid", "case", "trace_id"])
    c_act = pick(["concept:name", "activity", "task", "event", "name"])
    c_ts = pick(["time:timestamp", "timestamp", "start_time", "end_time", "time"])

    grouped: dict[str, list[tuple]] = {}
    has_ts = False
    for row in reader:
        cid = str(row.get(c_case, "")) if c_case else "1"
        act = str(row.get(c_act, "")) if c_act else ""
        ts = _coerce_ts(str(row.get(c_ts, ""))) if c_ts else None
        has_ts = has_ts or ts is not None
        grouped.setdefault(cid, []).append((act, ts))

    traces = []
    for cid, evs in grouped.items():
        traces.append(Trace(case_id=cid,
                            activities=[a for a, _ in evs],
                            timestamps=[t for _, t in evs]))
    return traces, has_ts, set()


def load_event_log(path: str, dataset_id: str) -> EventLog:
    """Load a file into a normalized EventLog with a lineage hash."""
    with open(path, "rb") as f:
        raw = f.read()
    filename = os.path.basename(path)
    fmt = _detect_format(filename, raw)
    chash = _sha256(raw)

    if fmt == "xes":
        traces, has_ts, attr_keys = _parse_xes(raw)
    else:
        traces, has_ts, attr_keys = _parse_csv(raw)

    alphabet = sorted({a for t in traces for a in t.activities})
    return EventLog(
        dataset_id=dataset_id,
        source_filename=filename,
        content_hash=chash,
        fmt=fmt,
        traces=traces,
        has_timestamps=has_ts,
        trace_attr_keys=sorted(attr_keys),
        activity_alphabet=alphabet,
    )
