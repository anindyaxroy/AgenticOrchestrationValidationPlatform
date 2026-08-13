"""
feature_extraction.py — Derive the MDP state vector from a real EventLog.

Supervisor points addressed:
  (3) "show how features are extracted / defined" — every feature has an
      explicit, documented formula computed FROM the log. No hardcoded vectors.
  (6) features are derived from the actual process data, with provenance.

Honesty about time-dependent features: when a log has NO timestamps (PDC
2016), bottleneck_score / sla_risk_index / resource_utilisation are computed
on a SEQUENCE-POSITION basis instead of real duration, and this is disclosed
via provenance strings and time_based_available.

When a log DOES carry timestamps (e.g. BPI Challenge 2012/2017), this module
now actually uses them -- computing real per-activity waiting time and real
per-case cycle time -- rather than silently falling back to the sequence
proxy regardless of what data is available. That fallback-regardless-of-data
gap existed in earlier versions of this module and meant uploading a
timestamped file never actually unlocked real duration-based features; this
version fixes that. If timestamp COVERAGE is too incomplete to trust (fewer
than MIN_TIMESTAMP_COVERAGE of traces have a usable first+last timestamp),
it still falls back to the proxy, and says so explicitly in provenance and in
the new `time_based_used` field -- which is deliberately kept separate from
`time_based_available`, because a log can HAVE timestamps without there being
ENOUGH of them to trust for this file specifically.

The six state features (matching the thesis MDP design):
  0 bottleneck_score          — concentration of the most position-critical activity
  1 sla_risk_index            — proxy for at-risk cases (here: share of non-conforming/negative or long traces)
  2 cost_variance_norm        — normalized spread of case cost (or trace-length spread if no cost attr)
  3 dominant_activity_share   — frequency share of the single most common activity
  4 rework_probability        — share of cases where an activity repeats within the case
  5 resource_utilisation      — proxy for load (here: mean trace length normalized by max)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .log_loader import EventLog, Trace


FEATURE_NAMES = [
    "bottleneck_score",
    "sla_risk_index",
    "cost_variance_norm",
    "dominant_activity_share",
    "rework_probability",
    "resource_utilisation",
]

# Which features need timestamps to be "true" operational measures.
# When the log has no (or insufficient) timestamps we fall back to
# sequence-based proxies and flag it via provenance + time_based_used.
TIME_DEPENDENT = {"bottleneck_score", "sla_risk_index", "resource_utilisation"}

# Fraction of traces that need a usable first+last real timestamp before we
# trust real-duration features for this file. Below this, a file that
# technically "has timestamps" still isn't reliable enough to compute real
# cycle times from, so we fall back to the sequence proxy and say so.
MIN_TIMESTAMP_COVERAGE = 0.8


def _trace_cycle_time(t: Trace) -> Optional[float]:
    """Real elapsed seconds from this trace's first to last timestamped
    event, or None if fewer than two timestamps are present."""
    ts = [x for x in t.timestamps if x is not None]
    if len(ts) < 2:
        return None
    return (max(ts) - min(ts)).total_seconds()


def _real_time_context(log: EventLog):
    """Computes real per-case cycle times and real per-activity mean waiting
    time (time since the previous event in the SAME case), using only actual
    timestamp deltas. Returns (usable, cycle_times, activity_wait, coverage)
    so callers can decide whether coverage is good enough to trust."""
    cycle_times: dict[str, float] = {}
    for t in log.traces:
        ct = _trace_cycle_time(t)
        if ct is not None:
            cycle_times[t.case_id] = ct
    coverage = len(cycle_times) / max(len(log.traces), 1)

    wait_sum: dict[str, float] = {}
    wait_n: dict[str, int] = {}
    for t in log.traces:
        prev_ts = None
        for act, ts in zip(t.activities, t.timestamps):
            if prev_ts is not None and ts is not None:
                wait = (ts - prev_ts).total_seconds()
                if wait >= 0:
                    wait_sum[act] = wait_sum.get(act, 0.0) + wait
                    wait_n[act] = wait_n.get(act, 0) + 1
            if ts is not None:
                prev_ts = ts
    activity_wait = {a: wait_sum[a] / wait_n[a] for a in wait_sum if wait_n.get(a)}

    usable = bool(log.has_timestamps) and coverage >= MIN_TIMESTAMP_COVERAGE and len(cycle_times) >= 2
    return usable, cycle_times, activity_wait, coverage


@dataclass
class FeatureVector:
    dataset_id: str
    content_hash: str            # lineage: ties this vector to the exact source file
    values: dict[str, float]
    provenance: dict[str, str]   # feature_name -> human-readable formula actually used
    time_based_available: bool     # log.has_timestamps -- timestamps exist SOMEWHERE in the file
    time_based_used: bool = False  # whether real durations were actually usable for THIS file
    timestamp_coverage: float = 0.0  # fraction of traces with a usable first+last timestamp

    def as_array(self) -> list[float]:
        return [round(self.values[n], 4) for n in FEATURE_NAMES]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "content_hash": self.content_hash[:16],
            "values": {k: round(v, 4) for k, v in self.values.items()},
            "vector": self.as_array(),
            "provenance": self.provenance,
            "time_based_available": self.time_based_available,
            "time_based_used": self.time_based_used,
            "timestamp_coverage": round(self.timestamp_coverage, 3),
        }


def _normalize(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def extract_features(log: EventLog) -> FeatureVector:
    traces = log.traces
    n = max(len(traces), 1)
    prov: dict[str, str] = {}

    # ---- activity frequency table (used by several features) ----
    act_counts: Counter = Counter()
    for t in traces:
        act_counts.update(t.activities)
    total_events = max(sum(act_counts.values()), 1)
    most_common_act, most_common_n = (act_counts.most_common(1)[0]
                                       if act_counts else ("", 0))

    # 0. dominant_activity_share — freq share of single most common activity
    dominant_share = most_common_n / total_events
    prov["dominant_activity_share"] = (
        f"count('{most_common_act}')={most_common_n} / total_events={total_events}"
    )

    # ---- real-time context: only computed once, used by 3 features below ----
    rt_usable, cycle_times, activity_wait, rt_coverage = _real_time_context(log)

    # 3. bottleneck_score — position-criticality of the most "blocking" activity.
    if rt_usable and activity_wait:
        # REAL: activity with the highest mean actual waiting time before it,
        # weighted by how often it occurs (Little's Law intuition: high WIP
        # at a stage shows up here as high real wait time before that stage).
        weighted_wait = {a: activity_wait[a] * (act_counts.get(a, 0) / total_events)
                         for a in activity_wait}
        max_possible = max(activity_wait.values()) or 1.0
        btl_act = max(weighted_wait, key=weighted_wait.get)
        bottleneck_score = min(1.0, weighted_wait[btl_act] / max_possible)
        prov["bottleneck_score"] = (
            f"[REAL] mean_real_wait_time*freq_share, normalized by max observed wait time; "
            f"top activity='{btl_act}' (timestamp coverage {rt_coverage:.0%})"
        )
    else:
        # PROXY: no usable timestamps -- fall back to sequence position.
        # NOTE: previously this divided the top score by itself (top / max(top,...))
        # which is always exactly 1.0 regardless of the data -- a real bug, now
        # fixed by normalizing against the formula's own theoretical bound instead
        # of self-referentially against the observed max.
        pos_weight: dict[str, float] = {}
        for t in traces:
            L = max(t.length - 1, 1)
            for idx, a in enumerate(t.activities):
                pos_weight[a] = pos_weight.get(a, 0.0) + (idx / L)
        btl_scores = {}
        for a, c in act_counts.items():
            mean_pos = pos_weight.get(a, 0.0) / c
            btl_scores[a] = mean_pos * (c / total_events)
        if btl_scores:
            btl_act = max(btl_scores, key=btl_scores.get)
            bottleneck_score = min(1.0, btl_scores[btl_act])  # already in [0,1]: mean_pos<=1, freq_share<=1
        else:
            bottleneck_score, btl_act = 0.0, ""
        reason = "no timestamps" if not log.has_timestamps else f"timestamp coverage only {rt_coverage:.0%} (below {MIN_TIMESTAMP_COVERAGE:.0%} threshold)"
        prov["bottleneck_score"] = (
            f"[PROXY] max over activities of mean_relative_position*freq_share; "
            f"top activity='{btl_act}' (sequence-based -- {reason})"
        )

    # 4. rework_probability — share of cases where ANY activity repeats in-case
    rework_cases = sum(1 for t in traces if len(set(t.activities)) < len(t.activities))
    rework_probability = rework_cases / n
    prov["rework_probability"] = f"cases_with_repeated_activity={rework_cases} / n_cases={n}"

    # 2. cost_variance_norm — spread of per-case cost if a cost attr exists,
    #    else spread of trace length (a sequence proxy).
    cost_key = next((k for k in log.trace_attr_keys if "cost" in k.lower()), None)
    if cost_key:
        vals = [float(t.attributes.get(cost_key, 0.0)) for t in traces]
        basis = f"trace attribute '{cost_key}'"
    else:
        vals = [float(t.length) for t in traces]
        basis = "trace length (no cost attribute present)"
    if vals:
        mean_v = sum(vals) / len(vals)
        var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        # coefficient of variation, squashed to [0,1]
        cv = (std / mean_v) if mean_v else 0.0
        cost_variance_norm = min(1.0, cv)
    else:
        cost_variance_norm = 0.0
    prov["cost_variance_norm"] = f"coefficient_of_variation of {basis}, clipped to [0,1]"

    # 1. sla_risk_index — TIME-DEPENDENT.
    label_key = next((k for k in log.trace_attr_keys
                      if k.lower() in ("pdc:ispos", "ispos", "label")), None)
    if rt_usable:
        # REAL: share of cases whose REAL cycle time exceeds the file's own
        # 75th-percentile real cycle time.
        ct_vals = sorted(cycle_times.values())
        p75_ct = ct_vals[int(0.75 * (len(ct_vals) - 1))] if ct_vals else 0.0
        risky = sum(1 for v in cycle_times.values() if v > p75_ct)
        # cases with no usable cycle time can't be judged -- excluded from
        # both numerator and denominator rather than silently assumed safe.
        sla_risk_index = risky / max(len(cycle_times), 1)
        sla_basis = f"share of cases with real cycle_time > file's 75th-percentile real cycle_time (timestamp coverage {rt_coverage:.0%})"
        prov["sla_risk_index"] = f"[REAL] {sla_basis}"
    elif label_key:
        risky = sum(1 for t in traces if t.attributes.get(label_key) is False)
        sla_risk_index = risky / n
        sla_basis = f"share of cases with {label_key}=False (conformance label proxy)"
        prov["sla_risk_index"] = f"[PROXY] {sla_basis}"
    else:
        lengths = sorted(t.length for t in traces)
        p75 = lengths[int(0.75 * (len(lengths) - 1))] if lengths else 0
        risky = sum(1 for t in traces if t.length > p75)
        sla_risk_index = risky / n
        reason = "no timestamps" if not log.has_timestamps else f"timestamp coverage only {rt_coverage:.0%}"
        sla_basis = f"share of cases longer than 75th-percentile trace length ({reason})"
        prov["sla_risk_index"] = f"[PROXY] {sla_basis}"

    # 5. resource_utilisation — TIME-DEPENDENT.
    if rt_usable:
        # REAL: mean real cycle time / max real cycle time -- actual relative
        # case duration load, not just event count.
        ct_vals = list(cycle_times.values())
        mean_ct = sum(ct_vals) / len(ct_vals)
        resource_utilisation = mean_ct / (max(ct_vals) or 1)
        prov["resource_utilisation"] = (
            f"[REAL] mean_real_cycle_time / max_real_cycle_time (timestamp coverage {rt_coverage:.0%})"
        )
    else:
        lengths = [t.length for t in traces] or [0]
        mean_len = sum(lengths) / len(lengths)
        resource_utilisation = mean_len / (max(lengths) or 1)
        reason = "no timestamps" if not log.has_timestamps else f"timestamp coverage only {rt_coverage:.0%}"
        prov["resource_utilisation"] = (
            f"[PROXY] mean_trace_length / max_trace_length ({reason} -> structural load proxy)"
        )

    values = {
        "bottleneck_score": float(bottleneck_score),
        "sla_risk_index": float(sla_risk_index),
        "cost_variance_norm": float(cost_variance_norm),
        "dominant_activity_share": float(dominant_share),
        "rework_probability": float(rework_probability),
        "resource_utilisation": float(resource_utilisation),
    }

    return FeatureVector(
        dataset_id=log.dataset_id,
        content_hash=log.content_hash,
        values=values,
        provenance=prov,
        time_based_available=log.has_timestamps,
        time_based_used=rt_usable,
        timestamp_coverage=rt_coverage,
    )
