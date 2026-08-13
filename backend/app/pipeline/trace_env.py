"""
trace_env.py — Per-trace RL environment: walks through REAL trace event
sequences instead of perturbing one synthetic aggregate point.

Why this file exists (read this before touching ACTION effects below):

The original environment (see BPMNEnv in rl_agent.py) trained and evaluated
by taking ONE aggregate 6-dim feature vector per uploaded file and applying
Gaussian noise to it -- there was no per-trace state-action data anywhere,
and "held-out evaluation" meant "a different random seed perturbing the same
single point," not a different real case the agent never saw. That made two
things in the thesis false as originally written: (1) hypotheses claiming to
mine "n≈200 traces" had no such data to mine against, and (2) the
"rework_probability requires lookahead" explanation for why greedy competes
with RL had no mechanism to hang on, because every original action effect was
a static one-step delta with no delayed consequence.

This module fixes both: one episode = one real trace, walked through its own
real activity sequence. Three of the seven actions (no_op, skip_optional,
reorder_activities) apply directly to the REAL remaining activities in that
trace -- a genuine, trace-specific counterfactual, not a universal constant.
The other four (parallelise, reallocate_resource, escalate_to_human,
insert_compliance_agent) still use a documented, fixed-effect assumption
layered on top of the real state, because their true causal effect cannot be
observed from a historical log without a randomized trial -- this is the same
limitation any offline heuristic workflow-redesign evaluation has (Reijers,
2003), and it is disclosed here rather than hidden.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from .log_loader import Trace, EventLog
from .feature_extraction import FEATURE_NAMES, _real_time_context, MIN_TIMESTAMP_COVERAGE

ACTIONS = {
    0: "no_op",
    1: "parallelise",
    2: "skip_optional",
    3: "reallocate_resource",
    4: "escalate_to_human",
    5: "insert_compliance_agent",
    6: "reorder_activities",
}

# Which actions apply to the REAL remaining sequence of this specific trace,
# vs. which ones layer a documented fixed-effect assumption on top of the
# real state because the true effect isn't observable from a single
# historical log (see module docstring).
GROUNDED_ACTIONS = {0, 2, 6}
ASSUMPTION_ACTIONS = {1, 3, 4, 5}

ASSUMPTION_EFFECTS = {
    1: {"resource_utilisation": -0.05},                            # parallelise: assumed concurrency saves load
    3: {"resource_utilisation": -0.10},                            # reallocate_resource: no per-event resource field exists in this log schema (Trace has no per-event resource attribute) -> assumption
    4: {"sla_risk_index": -0.12, "cost_variance_norm": +0.05},     # escalate_to_human: assumed hand-off overhead
    5: {"rework_probability": -0.10, "cost_variance_norm": -0.03}, # insert_compliance_agent: hypothetical step, not present in the data
}

REWARD_WEIGHTS = {
    "bottleneck_score":        0.25,
    "sla_risk_index":          0.25,
    "rework_probability":      0.25,
    "resource_utilisation":    0.15,
    "cost_variance_norm":      0.10,
}

# Activities at or below this frequency percentile (file-wide) count as
# "optional" -- skip_optional only has a real effect on these.
OPTIONAL_FREQ_PERCENTILE = 0.25


class FileContext:
    """Precomputed, file-level reference values that every trace's per-step
    state is scored against. Computed ONCE per file from real data, mirroring
    feature_extraction.py's own formulas so the per-trace features stay
    consistent with the file-level ones shown elsewhere in the UI."""

    def __init__(self, log: EventLog):
        act_counts: Counter = Counter()
        for t in log.traces:
            act_counts.update(t.activities)
        self.activity_counts = act_counts
        total_events = max(sum(act_counts.values()), 1)

        pos_weight_sum: dict[str, float] = {}
        for t in log.traces:
            L = max(t.length - 1, 1)
            for idx, a in enumerate(t.activities):
                pos_weight_sum[a] = pos_weight_sum.get(a, 0.0) + (idx / L)
        self.pos_weight = {
            a: (pos_weight_sum[a] / act_counts[a]) * (act_counts[a] / total_events)
            for a in act_counts
        }
        self.pos_weight_max = max(self.pos_weight.values()) if self.pos_weight else 1.0

        lengths = sorted(t.length for t in log.traces) or [0]
        self.mean_length = sum(lengths) / len(lengths)
        self.p75_length = lengths[int(0.75 * (len(lengths) - 1))] if lengths else 1

        freqs = sorted(act_counts.values())
        cut_idx = int(OPTIONAL_FREQ_PERCENTILE * (len(freqs) - 1)) if freqs else 0
        optional_threshold = freqs[cut_idx] if freqs else 0
        self.optional_activities = {a for a, c in act_counts.items() if c <= optional_threshold}

        cost_key = next((k for k in log.trace_attr_keys if "cost" in k.lower()), None)
        self.cost_key = cost_key
        if cost_key:
            vals = [float(t.attributes.get(cost_key, 0.0)) for t in log.traces]
        else:
            vals = [float(t.length) for t in log.traces]
        n = len(vals) or 1
        self.cost_mean = sum(vals) / n
        var = sum((v - self.cost_mean) ** 2 for v in vals) / n
        self.cost_std = var ** 0.5 or 1.0

        # Real-timestamp context -- SAME logic and threshold as
        # feature_extraction.py, so the per-trace RL state and the file-level
        # dataset characterisation never disagree about whether timestamps
        # were actually usable for this file.
        self.rt_usable, self.cycle_times, self.activity_wait, self.rt_coverage = _real_time_context(log)
        if self.rt_usable and self.cycle_times:
            ct_vals = sorted(self.cycle_times.values())
            self.p75_cycle_time = ct_vals[int(0.75 * (len(ct_vals) - 1))] if ct_vals else 1.0
            self.max_cycle_time = max(ct_vals) or 1.0
            self.max_activity_wait = max(self.activity_wait.values()) if self.activity_wait else 1.0
        else:
            self.p75_cycle_time = 1.0
            self.max_cycle_time = 1.0
            self.max_activity_wait = 1.0

    def is_optional(self, activity: str) -> bool:
        return activity in self.optional_activities

    def trace_cost_anomaly(self, t: Trace) -> float:
        """How anomalous THIS trace's own cost (or length, if no cost
        attribute) is relative to the file. Trace-invariant by construction
        (cost is only known at case completion), same design choice
        feature_extraction.py already makes for time-dependent proxies."""
        v = float(t.attributes.get(self.cost_key, 0.0)) if self.cost_key else float(t.length)
        return max(0.0, min(1.0, abs(v - self.cost_mean) / self.cost_std))


def _state_from_prefix(prefix: list[tuple[str, object]], full_length: int,
                       ctx: FileContext, own_cycle_time: Optional[float]) -> list[float]:
    """Compute the 6-dim state vector from a REAL (possibly counterfactually
    modified) trace prefix. `prefix` is a list of (activity, timestamp)
    pairs -- timestamps travel WITH their activity through skip/reorder so
    real elapsed time stays computable even after a counterfactual edit.
    Index 2 (cost_variance_norm) is filled in by the caller since it's
    trace-invariant, not prefix-dependent."""
    names = [a for a, _ in prefix]
    ts_vals = sorted(ts for _, ts in prefix if ts is not None)
    n = max(len(names), 1)
    counts = Counter(names)
    current = names[-1] if names else ""

    dominant_activity_share = counts.most_common(1)[0][1] / n if counts else 0.0
    rework_probability = 1.0 if any(c > 1 for c in counts.values()) else 0.0

    # bottleneck_score: real per-activity mean wait time when this file's
    # timestamps are usable AND this specific activity has observed wait-time
    # data; otherwise the sequence-position fallback (some activities, e.g.
    # always-first ones, never have a "previous event" to measure a wait
    # against even in an otherwise-usable file).
    if ctx.rt_usable and current in ctx.activity_wait:
        bottleneck_score = min(1.0, ctx.activity_wait[current] / ctx.max_activity_wait) if ctx.max_activity_wait else 0.0
    else:
        bottleneck_score = (ctx.pos_weight.get(current, 0.0) / ctx.pos_weight_max) if ctx.pos_weight_max else 0.0

    # sla_risk_index / resource_utilisation: real elapsed time (max-min over
    # the prefix's own timestamps, robust to reordering) when usable for
    # THIS trace specifically; else the length-based proxy.
    if ctx.rt_usable and len(ts_vals) >= 2:
        elapsed = (ts_vals[-1] - ts_vals[0]).total_seconds()
        sla_risk_index = min(1.0, max(0.0, elapsed / ctx.p75_cycle_time)) if ctx.p75_cycle_time else 0.0
        if own_cycle_time:
            resource_utilisation = min(1.0, max(0.0, elapsed / own_cycle_time))
        else:
            resource_utilisation = min(1.0, len(names) / max(full_length, 1))
    else:
        sla_risk_index = min(1.0, len(names) / ctx.p75_length) if ctx.p75_length else 0.0
        resource_utilisation = min(1.0, len(names) / max(full_length, 1))

    return [bottleneck_score, sla_risk_index, 0.0, dominant_activity_share,
            rework_probability, resource_utilisation]


class TraceEnv:
    """One episode = one real trace. State is recomputed from the real (or
    counterfactually modified) prefix at every step -- not from a synthetic
    perturbation of a single aggregate point. Prefix/remaining carry
    (activity, timestamp) pairs so real elapsed time survives skip/reorder."""

    def __init__(self, ctx: FileContext, max_steps: int = 12):
        self.ctx = ctx
        self.max_steps = max_steps
        self.trace: Optional[Trace] = None
        self.remaining: list[tuple[str, object]] = []
        self.prefix: list[tuple[str, object]] = []
        self.cost_anomaly = 0.0
        self.own_cycle_time: Optional[float] = None
        self.state: list[float] = []
        self.t = 0

    def reset_on_trace(self, trace: Trace) -> tuple[int, ...]:
        self.trace = trace
        pairs = list(zip(trace.activities, trace.timestamps))
        self.prefix = pairs[:1] if pairs else []
        self.remaining = pairs[1:]
        self.cost_anomaly = self.ctx.trace_cost_anomaly(trace)
        self.own_cycle_time = self.ctx.cycle_times.get(trace.case_id)
        self.t = 0
        self._recompute_state()
        return self._disc(self.state)

    def _recompute_state(self):
        s = _state_from_prefix(self.prefix, self.trace.length, self.ctx, self.own_cycle_time)
        s[2] = self.cost_anomaly
        self.state = [max(0.0, min(1.0, v)) for v in s]

    def _health(self, s: list[float]) -> float:
        d = dict(zip(FEATURE_NAMES, s))
        return -sum(REWARD_WEIGHTS[k] * d[k] for k in REWARD_WEIGHTS)

    def step(self, action: int):
        prev = self._health(self.state)
        state_before = list(self.state)

        if action in GROUNDED_ACTIONS:
            self._apply_grounded(action)
        else:
            self._advance_one_real_step()
            self._recompute_state()
            d = dict(zip(FEATURE_NAMES, self.state))
            for feat, delta in ASSUMPTION_EFFECTS.get(action, {}).items():
                d[feat] = max(0.0, min(1.0, d[feat] + delta))
            self.state = [d[k] for k in FEATURE_NAMES]

        reward = self._health(self.state) - prev
        self.t += 1
        done = self.t >= self.max_steps or not self.remaining
        return self._disc(self.state), reward, done, state_before

    def _advance_one_real_step(self):
        if self.remaining:
            self.prefix = self.prefix + [self.remaining[0]]
            self.remaining = self.remaining[1:]

    def _apply_grounded(self, action: int):
        if action == 0:
            self._advance_one_real_step()
        elif action == 2:
            # Only a real intervention if the NEXT real event is classified
            # optional; otherwise honestly falls back to no_op rather than
            # pretending every activity is always skippable.
            if self.remaining and self.ctx.is_optional(self.remaining[0][0]):
                self.remaining = self.remaining[1:]
                self._advance_one_real_step()
            else:
                self._advance_one_real_step()
        elif action == 6:
            if len(self.remaining) >= 2:
                self.remaining = [self.remaining[1], self.remaining[0]] + self.remaining[2:]
            self._advance_one_real_step()
        self._recompute_state()

    @staticmethod
    def _disc(s: list[float], bins: int = 5) -> tuple[int, ...]:
        return tuple(min(bins - 1, int(v * bins)) for v in s)
