"""
process_mining.py — Real mining computed FROM the EventLog (no constants).

Supervisor points addressed:
  (1) the outcome of process mining is an actual function of the data here.
  (4) numbers trace to the loaded file via content_hash.
  (6) computed on the real traces.

Outputs feed two consumers:
  - the knowledge store (embedded findings the agent retrieves)
  - the RL environment (the state vector is the mining-derived FeatureVector)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .log_loader import EventLog
from .feature_extraction import _real_time_context


@dataclass
class MiningResult:
    dataset_id: str
    content_hash: str
    n_cases: int
    n_events: int
    n_activities: int
    has_timestamps: bool
    variants: list[dict[str, Any]]        # distinct activity sequences + frequency
    activity_frequency: list[dict[str, Any]]
    bottlenecks: list[dict[str, Any]]     # criticality ranking (real wait-time based if usable, else sequence-based)
    conformance: dict[str, Any]           # label-based conformance summary (PDC)
    directly_follows: list[dict[str, Any]]  # top DF edges (the process "shape")
    bottleneck_basis: str = ""            # discloses whether bottlenecks used REAL wait time or a sequence PROXY

    def findings_text(self) -> list[str]:
        """Natural-language findings — these are what get embedded for the agent."""
        out = []
        top_v = self.variants[0] if self.variants else None
        if top_v:
            out.append(
                f"The most frequent process variant covers {top_v['cases']} of "
                f"{self.n_cases} cases ({top_v['frequency_pct']}%), following the path "
                f"{top_v['path']}."
            )
        if self.bottlenecks:
            b = self.bottlenecks[0]
            basis_note = "using real wall-clock waiting time" if self.bottleneck_basis.startswith("[REAL]") \
                        else "using a sequence-position proxy, since real timestamps were not usable for this criticality measure"
            out.append(
                f"The most position-critical activity is '{b['activity']}' "
                f"(criticality score {b['score']}, {basis_note})."
            )
        if self.conformance.get("has_labels"):
            out.append(
                f"Conformance: {self.conformance['positive']} positive (conforming) and "
                f"{self.conformance['negative']} negative (non-conforming) cases. "
                f"Negative cases carry mean cost {self.conformance['neg_mean_cost']} versus "
                f"{self.conformance['pos_mean_cost']} for positive cases."
            )
        if self.directly_follows:
            edges = ", ".join(f"{e['from']}->{e['to']}({e['count']})"
                              for e in self.directly_follows[:3])
            out.append(f"Dominant directly-follows transitions: {edges}.")
        out.append(
            f"Process variability is {'high' if len(self.variants) >= self.n_cases else 'moderate'}: "
            f"{len(self.variants)} distinct variants across {self.n_cases} cases."
        )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "content_hash": self.content_hash[:16],
            "n_cases": self.n_cases,
            "n_events": self.n_events,
            "n_activities": self.n_activities,
            "has_timestamps": self.has_timestamps,
            "variants": self.variants,
            "activity_frequency": self.activity_frequency,
            "bottlenecks": self.bottlenecks,
            "bottleneck_basis": self.bottleneck_basis,
            "conformance": self.conformance,
            "directly_follows": self.directly_follows,
            "findings_text": self.findings_text(),
        }


def run_mining(log: EventLog) -> MiningResult:
    traces = log.traces
    n = max(len(traces), 1)

    # ---- variants ----
    variant_counter: Counter = Counter()
    for t in traces:
        variant_counter[tuple(t.activities)] += 1
    variants = []
    for i, (seq, cnt) in enumerate(variant_counter.most_common()):
        variants.append({
            "id": i + 1,
            "path": " -> ".join(seq) if len(seq) <= 12 else " -> ".join(seq[:12]) + " ...",
            "length": len(seq),
            "cases": cnt,
            "frequency_pct": round(100 * cnt / n, 1),
        })

    # ---- activity frequency ----
    act_counts: Counter = Counter()
    for t in traces:
        act_counts.update(t.activities)
    total_ev = max(sum(act_counts.values()), 1)
    activity_frequency = [
        {"activity": a, "count": c, "share_pct": round(100 * c / total_ev, 1)}
        for a, c in act_counts.most_common()
    ]

    # ---- directly-follows graph ----
    df_counter: Counter = Counter()
    for t in traces:
        for a, b in zip(t.activities, t.activities[1:]):
            df_counter[(a, b)] += 1
    directly_follows = [
        {"from": a, "to": b, "count": c}
        for (a, b), c in df_counter.most_common(10)
    ]

    # ---- bottlenecks ----
    # Previously this ALWAYS used sequence position (late + frequent) and
    # normalized the top score by dividing it by itself (score/max_score),
    # which is mathematically always exactly 1.0 regardless of the data --
    # the same bug independently present in feature_extraction.py, now fixed
    # there and here. This also now uses REAL per-activity wait time when the
    # file's timestamps are usable (same _real_time_context helper
    # feature_extraction.py uses, rather than a third duplicate
    # implementation), instead of ignoring timestamps unconditionally.
    rt_usable, cycle_times, activity_wait, rt_coverage = _real_time_context(log)

    if rt_usable and activity_wait:
        weighted = {a: activity_wait[a] * (act_counts.get(a, 0) / total_ev) for a in activity_wait}
        max_wait = max(activity_wait.values()) or 1.0
        btl = [(a, min(1.0, w / max_wait), None, act_counts.get(a, 0)) for a, w in weighted.items()]
        bottleneck_basis = f"[REAL] mean real wait time before activity * freq share (timestamp coverage {rt_coverage:.0%})"
    else:
        pos_accum: dict[str, float] = defaultdict(float)
        for t in traces:
            L = max(t.length - 1, 1)
            for idx, a in enumerate(t.activities):
                pos_accum[a] += idx / L
        btl = []
        for a, c in act_counts.items():
            mean_pos = pos_accum[a] / c
            score = mean_pos * (c / total_ev)  # already in [0,1]: mean_pos<=1, freq_share<=1 -- no self-referential division needed
            btl.append((a, min(1.0, score), mean_pos, c))
        reason = "no timestamps" if not log.has_timestamps else f"timestamp coverage only {rt_coverage:.0%}"
        bottleneck_basis = f"[PROXY] mean_relative_position * freq_share, sequence-based ({reason})"

    btl.sort(key=lambda x: x[1], reverse=True)
    bottlenecks = [
        {"activity": a, "score": round(s, 3),
         "mean_relative_position": (round(mp, 3) if mp is not None else None), "occurrences": c}
        for a, s, mp, c in btl[:8]
    ]

    # ---- conformance (label-based; PDC has explicit labels) ----
    label_key = next((k for k in log.trace_attr_keys
                      if k.lower() in ("pdc:ispos", "ispos", "label")), None)
    cost_key = next((k for k in log.trace_attr_keys if "cost" in k.lower()), None)
    if label_key:
        pos = [t for t in traces if t.attributes.get(label_key) is True]
        neg = [t for t in traces if t.attributes.get(label_key) is False]
        def mean_cost(group):
            if not cost_key or not group:
                return None
            return round(sum(float(t.attributes.get(cost_key, 0.0)) for t in group) / len(group), 2)
        conformance = {
            "has_labels": True,
            "positive": len(pos),
            "negative": len(neg),
            "conformance_rate": round(len(pos) / n, 3),
            "pos_mean_cost": mean_cost(pos),
            "neg_mean_cost": mean_cost(neg),
        }
    else:
        conformance = {"has_labels": False}

    return MiningResult(
        dataset_id=log.dataset_id,
        content_hash=log.content_hash,
        n_cases=log.n_cases,
        n_events=log.n_events,
        n_activities=len(log.activity_alphabet),
        has_timestamps=log.has_timestamps,
        variants=variants,
        activity_frequency=activity_frequency,
        bottlenecks=bottlenecks,
        bottleneck_basis=bottleneck_basis,
        conformance=conformance,
        directly_follows=directly_follows,
    )
