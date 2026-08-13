"""
rl_agent.py — Q-learning agent with per-episode traceability.

Every episode records:
  - episode number, seed, total reward
  - action chosen at each step
  - feature state BEFORE and AFTER each action
  - Q-values for all actions at each step
  - reward delta at each step

This episode log is written to /app/data/runs/{run_id}_episodes.jsonl
so it can be audited step-by-step.
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .feature_extraction import FeatureVector, FEATURE_NAMES
from .log_loader import EventLog
from .trace_env import TraceEnv, FileContext, GROUNDED_ACTIONS, ASSUMPTION_ACTIONS


# ── Environment ──────────────────────────────────────────────────────────────
ACTIONS = {
    0: "no_op",
    1: "parallelise",
    2: "skip_optional",
    3: "reallocate_resource",
    4: "escalate_to_human",
    5: "insert_compliance_agent",
    6: "reorder_activities",
}

ACTION_EFFECTS = {
    0: {},
    1: {"bottleneck_score": -0.15, "resource_utilisation": -0.05},
    2: {"bottleneck_score": -0.10, "sla_risk_index": +0.06},
    3: {"resource_utilisation": -0.18, "bottleneck_score": -0.08},
    4: {"sla_risk_index": -0.12, "resource_utilisation": +0.06, "cost_variance_norm": +0.05},
    5: {"rework_probability": -0.22, "sla_risk_index": +0.05, "cost_variance_norm": -0.08},
    6: {"bottleneck_score": -0.09, "rework_probability": -0.12,
        "resource_utilisation": -0.06, "dominant_activity_share": -0.05},
}

REWARD_WEIGHTS = {
    "bottleneck_score":        0.25,
    "sla_risk_index":          0.25,
    "rework_probability":      0.25,
    "resource_utilisation":    0.15,
    "cost_variance_norm":      0.10,
}


class BPMNEnv:
    def __init__(self, base_state: list[float], max_steps: int = 12, seed: int = 0):
        self.base      = list(base_state)
        self.max_steps = max_steps
        self.rng       = random.Random(seed)
        self.state: list[float] = []
        self.t = 0

    def reset(self, perturb: float = 0.05) -> tuple[int, ...]:
        self.state = [min(1.0, max(0.0, v + self.rng.gauss(0, perturb)))
                      for v in self.base]
        self.t = 0
        return self._disc(self.state)

    def _health(self, s: list[float]) -> float:
        d = dict(zip(FEATURE_NAMES, s))
        return -sum(REWARD_WEIGHTS[k] * d[k] for k in REWARD_WEIGHTS)

    def step(self, action: int):
        prev  = self._health(self.state)
        state_before = list(self.state)
        d     = dict(zip(FEATURE_NAMES, self.state))
        for feat, delta in ACTION_EFFECTS[action].items():
            d[feat] = min(1.0, max(0.0, d[feat] + delta))
        self.state = [d[k] for k in FEATURE_NAMES]
        reward = self._health(self.state) - prev
        self.t += 1
        return self._disc(self.state), reward, self.t >= self.max_steps, state_before

    @staticmethod
    def _disc(s: list[float], bins: int = 5) -> tuple[int, ...]:
        return tuple(min(bins - 1, int(v * bins)) for v in s)


# ── Q-learning agent ─────────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, n_actions: int, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995, seed=0):
        self.q: dict[tuple, list[float]] = {}
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps = eps_start
        self.eps_end, self.eps_decay = eps_end, eps_decay
        self.rng = random.Random(seed)

    def _row(self, s):
        if s not in self.q:
            self.q[s] = [0.0] * self.n_actions
        return self.q[s]

    def act(self, s, greedy=False) -> int:
        row = self._row(s)
        if not greedy and self.rng.random() < self.eps:
            return self.rng.randrange(self.n_actions)
        m = max(row)
        return self.rng.choice([i for i, v in enumerate(row) if v == m])

    def q_values(self, s) -> dict[str, float]:
        """Return Q-values for all actions at state s."""
        row = self._row(s)
        return {ACTIONS[i]: round(v, 4) for i, v in enumerate(row)}

    def update(self, s, a, r, s2, done):
        row = self._row(s)
        nxt = 0.0 if done else max(self._row(s2))
        row[a] += self.alpha * (r + self.gamma * nxt - row[a])

    def decay(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def policy_entropy(self) -> float:
        if not self.q:
            return 0.0
        entropies = []
        for row in self.q.values():
            m = max(row)
            if all(v == m for v in row):
                entropies.append(0.0)
            else:
                s  = sum(math.exp(v) for v in row)
                ps = [math.exp(v) / s for v in row]
                entropies.append(-sum(p * math.log(p + 1e-10) for p in ps))
        return round(sum(entropies) / len(entropies), 4)


# ── Baselines ─────────────────────────────────────────────────────────────────
def baseline_policy(name: str) -> Callable:
    if name == "random":
        rng = random.Random(123)
        return lambda s, env: rng.randrange(len(ACTIONS))
    if name == "do_nothing":
        return lambda s, env: 0
    if name == "greedy":
        def _greedy(s, env):
            best_a, best_r = 0, -1e9
            for a in range(len(ACTIONS)):
                d = dict(zip(FEATURE_NAMES, env.state))
                prev = env._health(env.state)
                for feat, delta in ACTION_EFFECTS[a].items():
                    d[feat] = min(1.0, max(0.0, d[feat] + delta))
                r = env._health([d[k] for k in FEATURE_NAMES]) - prev
                if r > best_r:
                    best_r, best_a = r, a
            return best_a
        return _greedy
    raise ValueError(name)


# ── Statistics helpers ─────────────────────────────────────────────────────────
def _stats(xs: list[float]) -> dict:
    n   = len(xs)
    m   = sum(xs) / n
    v   = sum((x - m) ** 2 for x in xs) / n
    sd  = v ** 0.5
    med = sorted(xs)[n // 2]
    return {"mean": round(m, 4), "std": round(sd, 4),
            "median": round(med, 4), "min": round(min(xs), 4),
            "max": round(max(xs), 4), "n": n}

def _mae(a: list[float], b: list[float]) -> float:
    return round(sum(abs(x - y) for x, y in zip(a, b)) / len(a), 4)

def _cohens_d(a: list[float], b: list[float]) -> float | None:
    """Returns None when pooled variance is ~0 (degenerate/constant returns),
    rather than silently reporting an effect size of exactly 0.0. A pooled-
    variance-zero case is undefined, not "no effect" — conflating the two
    previously made a 100% win rate render as d=0.0000 in the report."""
    ma = sum(a) / len(a); mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / len(a)
    vb = sum((x - mb) ** 2 for x in b) / len(b)
    ps = ((va + vb) / 2) ** 0.5
    if ps <= 1e-9:
        return None
    return round((ma - mb) / ps, 3)

def _convergence_episode(curve: list[float], pct: float = 0.90) -> int:
    """Episode where the curve first gets within (1-pct) of its final value,
    in absolute terms. The previous version used `final * (2 - pct)` as the
    threshold, which for a NEGATIVE final value (true here, since the health
    function is always <= 0) produces a threshold MORE negative than final
    itself -- meaning almost every episode trivially satisfied it, and
    convergence_episode was reporting ~1 regardless of actual training
    dynamics. Using an absolute tolerance around final fixes this for
    negative, positive, or zero reward curves alike."""
    if not curve: return -1
    final = curve[-1]
    tol = abs(final) * (1 - pct)
    for i, v in enumerate(curve):
        if abs(v - final) <= tol:
            return i + 1
    return len(curve)


# ── Episode log entry ─────────────────────────────────────────────────────────
def _run_episode_traced(policy, env: BPMNEnv, seed: int,
                        agent: Optional[QLearningAgent] = None,
                        ep_num: int = 0, mode: str = "train") -> tuple[float, dict]:
    """Run one episode and return (total_reward, episode_trace)."""
    env.rng = random.Random(seed)
    s = env.reset()
    total, done = 0.0, False
    steps = []
    initial_state = list(env.state)

    while not done:
        qvals = agent.q_values(s) if agent else {}
        a = policy(s, env)
        s2, r, done, state_before = env.step(a)
        total += r
        steps.append({
            "step":          env.t,
            "action":        ACTIONS[a],
            "action_id":     a,
            "reward_delta":  round(r, 4),
            "state_before":  {k: round(v, 4) for k, v in zip(FEATURE_NAMES, state_before)},
            "state_after":   {k: round(v, 4) for k, v in zip(FEATURE_NAMES, env.state)},
            "q_values":      qvals,
        })
        s = s2

    return total, {
        "episode":        ep_num,
        "mode":           mode,
        "seed":           seed,
        "total_reward":   round(total, 4),
        "n_steps":        env.t,
        "initial_state":  {k: round(v, 4) for k, v in zip(FEATURE_NAMES, initial_state)},
        "final_state":    {k: round(v, 4) for k, v in zip(FEATURE_NAMES, env.state)},
        "steps":          steps,
    }


def _run_episode(policy, env: BPMNEnv, seed: int) -> float:
    """Fast evaluation without tracing."""
    env.rng = random.Random(seed)
    s = env.reset()
    total, done = 0.0, False
    while not done:
        a = policy(s, env)
        s2, r, done, _ = env.step(a)
        total += r
        s = s2
    return total


# ── TrainingResult ─────────────────────────────────────────────────────────────
@dataclass
class TrainingResult:
    dataset_id:    str
    content_hash:  str
    episodes:      int
    reward_curve:  list[float]
    learned_eval:  dict
    baseline_eval: dict
    proof:         dict
    q_table_size:  int
    metrics:       dict
    episode_log_path: Optional[str]   # path to JSONL file with per-episode traces
    reward_curve_smoothed: list[float] = field(default_factory=list)  # centered moving average, for readable plots

    def to_dict(self):
        return {
            "dataset_id":        self.dataset_id,
            "content_hash":      self.content_hash[:16],
            "episodes":          self.episodes,
            "reward_curve":      [round(x, 4) for x in self.reward_curve],
            "reward_curve_smoothed": [round(x, 4) for x in self.reward_curve_smoothed],
            "learned_eval":      self.learned_eval,
            "baseline_eval":     self.baseline_eval,
            "proof":             self.proof,
            "q_table_size":      self.q_table_size,
            "metrics":           self.metrics,
            "episode_log_path":  self.episode_log_path,
        }


def _moving_average(curve: list[float], window: int) -> list[float]:
    """Centered moving average, for a readable training-progress plot.
    Per-trace episodes are inherently noisier than the old single-point
    environment's episodes (each one is a DIFFERENT real case with its own
    difficulty, not small noise around one point), so the raw curve alone is
    hard to read as a learning trend. This does NOT replace the raw curve --
    both are kept, since the raw curve is the actual truth and the smoothed
    one is a readability aid, not a substitute statistic."""
    if not curve:
        return []
    window = max(1, min(window, len(curve)))
    half = window // 2
    out = []
    for i in range(len(curve)):
        lo = max(0, i - half)
        hi = min(len(curve), i + half + 1)
        out.append(sum(curve[lo:hi]) / (hi - lo))
    return out


# ── Main training function ─────────────────────────────────────────────────────
def train_and_prove(fv: FeatureVector, episodes: int = 400,
                    train_seed: int = 0, n_eval: int = 100,
                    log_dir: str = "/app/data/runs",
                    run_id: str = "") -> TrainingResult:

    base_state = fv.as_array()
    env   = BPMNEnv(base_state, seed=train_seed)
    agent = QLearningAgent(n_actions=len(ACTIONS), seed=train_seed)

    # ── Episode log file (append-only JSONL) ──
    os.makedirs(log_dir, exist_ok=True)
    ep_log_path = os.path.join(log_dir, f"{run_id or fv.dataset_id}_episodes.jsonl") if log_dir else None

    def _write_ep(entry: dict):
        if ep_log_path:
            with open(ep_log_path, "a") as f:
                f.write(json.dumps({
                    **entry,
                    "dataset_id":   fv.dataset_id,
                    "content_hash": fv.content_hash[:16],
                    "epsilon":      round(agent.eps, 4),
                    "q_table_size": len(agent.q),
                }) + "\n")

    # ── TRAINING LOOP with per-episode logging ──
    reward_curve = []
    # Log first 10 episodes fully (step-by-step), then every 10th, then last 5
    trace_episodes = set(
        list(range(min(10, episodes))) +
        list(range(0, episodes, max(1, episodes // 20))) +
        list(range(max(0, episodes - 5), episodes))
    )

    for ep in range(episodes):
        seed = 10_000 + ep
        env.rng = random.Random(seed)

        if ep in trace_episodes:
            # Full step-by-step trace
            pol = lambda s, e: agent.act(s)
            total, trace = _run_episode_traced(pol, env, seed, agent, ep, "train")
            trace["epsilon"]      = round(agent.eps, 4)
            trace["q_table_size"] = len(agent.q)
            _write_ep(trace)
            # Re-run for Q updates (traced run already consumed env state)
            env.rng = random.Random(seed)
            s = env.reset()
            done = False
            while not done:
                a = agent.act(s)
                s2, r, done, _ = env.step(a)
                agent.update(s, a, r, s2, done)
                s = s2
        else:
            # Fast loop
            env.rng = random.Random(seed)
            s = env.reset()
            done, total = False, 0.0
            while not done:
                a = agent.act(s)
                s2, r, done, _ = env.step(a)
                agent.update(s, a, r, s2, done)
                s = s2; total += r
            _write_ep({
                "episode": ep, "mode": "train_compact",
                "seed": seed, "total_reward": round(total, 4),
                "epsilon": round(agent.eps, 4), "q_table_size": len(agent.q),
            })

        agent.decay()
        reward_curve.append(total)

    # ── HELD-OUT EVALUATION (full traces for first 5 eval episodes) ──
    eval_seeds    = [900_000 + i for i in range(n_eval)]
    learned_rets  = []
    baseline_rets = {b: [] for b in ("random", "do_nothing", "greedy")}

    for i, sd in enumerate(eval_seeds):
        ev = BPMNEnv(base_state, seed=train_seed)
        if i < 5:
            total, trace = _run_episode_traced(
                lambda s, e: agent.act(s, greedy=True), ev, sd, agent, i, "eval_learned"
            )
            _write_ep(trace)
        else:
            total = _run_episode(lambda s, e: agent.act(s, greedy=True), ev, sd)
        learned_rets.append(total)

    for bname in baseline_rets:
        pol = baseline_policy(bname)
        for i, sd in enumerate(eval_seeds):
            ev = BPMNEnv(base_state, seed=train_seed)
            if i < 3:
                total, trace = _run_episode_traced(pol, ev, sd, None, i, f"eval_{bname}")
                _write_ep(trace)
            else:
                total = _run_episode(pol, ev, sd)
            baseline_rets[bname].append(total)

    # ── Stats ──
    learned_stats  = _stats(learned_rets)
    baseline_stats = {b: _stats(v) for b, v in baseline_rets.items()}

    comparisons = {}
    for bname, brets in baseline_rets.items():
        deltas   = [l - b for l, b in zip(learned_rets, brets)]
        dm       = sum(deltas) / len(deltas)
        win_rate = sum(1 for d in deltas if d > 0) / len(deltas)
        rel_impr = (dm / abs(baseline_stats[bname]["mean"]) * 100
                    if baseline_stats[bname]["mean"] != 0 else 0.0)
        comparisons[bname] = {
            "mean_advantage":            round(dm, 4),
            "win_rate":                  round(win_rate, 3),
            "cohens_d":                  _cohens_d(learned_rets, brets),
            "relative_improvement_pct":  round(rel_impr, 1),
            "mae_vs_learned":            _mae(brets, learned_rets),
            "learned_better":            dm > 0,
        }

    # Action distribution
    action_counts: Counter = Counter()
    for sd in eval_seeds[:20]:
        ev = BPMNEnv(base_state, seed=train_seed)
        ev.rng = random.Random(sd)
        s = ev.reset()
        done = False
        while not done:
            a = agent.act(s, greedy=True)
            action_counts[ACTIONS[a]] += 1
            s, _, done, _ = ev.step(a)
    tot = sum(action_counts.values()) or 1
    action_dist = {k: round(v / tot, 3) for k, v in action_counts.most_common()}

    n_half = len(reward_curve) // 2
    fhm    = sum(reward_curve[:n_half]) / n_half if n_half else 0
    shm    = sum(reward_curve[n_half:]) / (len(reward_curve) - n_half) if n_half else 0

    metrics = {
        "training": {
            "episodes":                  episodes,
            "convergence_episode":       _convergence_episode(reward_curve),
            "first_half_mean_return":    round(fhm, 4),
            "second_half_mean_return":   round(shm, 4),
            "learning_improvement_pct":  round((shm - fhm) / abs(fhm) * 100 if fhm else 0, 1),
            "final_epsilon":             round(agent.eps, 4),
            "policy_entropy":            agent.policy_entropy(),
            "q_table_size":              len(agent.q),
            "q_table_coverage_pct":      round(len(agent.q) / (5 ** len(FEATURE_NAMES)) * 100, 4),
        },
        "held_out":  {"learned": learned_stats, "baselines": baseline_stats},
        "proof":     {"learned_mean": learned_stats["mean"],
                      "learned_std":  learned_stats["std"],
                      "comparisons":  comparisons},
        "action_distribution": action_dist,
        "episode_log_path": ep_log_path,
        "metric_definitions": {
            "mean_return":                "Average cumulative reward per episode on held-out seeds.",
            "std_return":                 "Standard deviation of returns — lower = more consistent policy.",
            "cohens_d":                   "Effect size (|d|>0.8 = large). Scale-free comparison.",
            "win_rate":                   "Fraction of held-out episodes where learned > baseline.",
            "mae_vs_learned":             "Mean Absolute Error of baseline vs learned, per episode.",
            "relative_improvement_pct":   "% improvement of learned mean return over baseline.",
            "policy_entropy":             "Action distribution entropy. Near 0 = deterministic policy.",
            "convergence_episode":        "Episode where reward first hit 90% of final value.",
            "q_table_coverage_pct":       "% of 5^6 theoretical state space actually visited.",
        }
    }

    return TrainingResult(
        dataset_id       = fv.dataset_id,
        content_hash     = fv.content_hash,
        episodes         = episodes,
        reward_curve     = reward_curve,
        learned_eval     = {"mean": learned_stats["mean"], "std": learned_stats["std"]},
        baseline_eval    = {b: {"mean": s["mean"], "std": s["std"]}
                            for b, s in baseline_stats.items()},
        proof            = {"learned_mean": learned_stats["mean"],
                            "learned_std":  learned_stats["std"],
                            "comparisons":  comparisons},
        q_table_size     = len(agent.q),
        metrics          = metrics,
        episode_log_path = ep_log_path,
    )


# ── Per-trace training (replaces the synthetic single-point environment) ──────
#
# The functions above (BPMNEnv, train_and_prove) train and evaluate against ONE
# synthetic point perturbed with Gaussian noise -- there is no per-trace
# state-action data in that path, and "held-out" there means "a different
# random seed on the same point," not a different real case. They are kept
# here, unused by the orchestrator, only so the difference is auditable.
#
# train_and_prove_per_trace below is what the orchestrator actually calls now.
# One episode = one real trace (see trace_env.TraceEnv). Training and
# evaluation use a genuine trace-level split, so "held-out" means real cases
# the agent never trained on.

def _run_trace_episode_traced(policy, env: TraceEnv, trace, agent: Optional[QLearningAgent],
                              ep_num: int, mode: str) -> tuple[float, dict]:
    s = env.reset_on_trace(trace)
    total, done = 0.0, False
    steps = []
    initial_state = list(env.state)
    while not done:
        qvals = agent.q_values(s) if agent else {}
        a = policy(s, env)
        s2, r, done, state_before = env.step(a)
        total += r
        steps.append({
            "step":          env.t,
            "action":        ACTIONS[a],
            "action_id":     a,
            "action_kind":   "grounded" if a in GROUNDED_ACTIONS else "assumption",
            "reward_delta":  round(r, 4),
            "state_before":  {k: round(v, 4) for k, v in zip(FEATURE_NAMES, state_before)},
            "state_after":   {k: round(v, 4) for k, v in zip(FEATURE_NAMES, env.state)},
            "q_values":      qvals,
        })
        s = s2
    return total, {
        "episode":        ep_num,
        "mode":            mode,
        "case_id":         trace.case_id,
        "trace_length":    trace.length,
        "total_reward":    round(total, 4),
        "n_steps":         env.t,
        "initial_state":   {k: round(v, 4) for k, v in zip(FEATURE_NAMES, initial_state)},
        "final_state":     {k: round(v, 4) for k, v in zip(FEATURE_NAMES, env.state)},
        "steps":           steps,
    }


def _run_trace_episode(policy, env: TraceEnv, trace) -> float:
    s = env.reset_on_trace(trace)
    total, done = 0.0, False
    while not done:
        a = policy(s, env)
        s2, r, done, _ = env.step(a)
        total += r
        s = s2
    return total


def _trace_greedy_policy(s, env: TraceEnv) -> int:
    """One-step lookahead using the environment's OWN health function --
    same idea as the original baseline_policy('greedy'), but now evaluated
    against real per-trace state, including the two grounded actions whose
    availability depends on this specific trace's remaining sequence (e.g.
    skip_optional only helps if the next real event is actually optional)."""
    best_a, best_r = 0, -1e9
    for a in range(len(ACTIONS)):
        # Cheaply probe each action on a throwaway copy of the trace pointer
        probe = TraceEnv(env.ctx, max_steps=env.max_steps)
        probe.trace, probe.prefix, probe.remaining = env.trace, list(env.prefix), list(env.remaining)
        probe.cost_anomaly, probe.t = env.cost_anomaly, env.t
        probe.state = list(env.state)
        prev = probe._health(probe.state)
        _, r, _, _ = probe.step(a)
        if r > best_r:
            best_r, best_a = r, a
    return best_a


def _split_traces(traces: list, eval_frac: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = list(traces)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n <= 1:
        return shuffled, shuffled  # degenerate case: not enough traces to hold anything out
    n_eval = max(1, round(n * eval_frac))
    n_eval = min(n_eval, n - 1)  # always leave at least 1 training trace
    return shuffled[n_eval:], shuffled[:n_eval]


def train_and_prove_per_trace(log: EventLog, fv: FeatureVector, episodes: int = 300,
                              train_seed: int = 0, eval_frac: float = 0.2,
                              log_dir: str = "/app/data/runs", run_id: str = "",
                              max_steps: int = 12) -> "TrainingResult":
    """Train and evaluate a Q-learning agent on REAL per-trace episodes from
    this file, with a genuine trace-level train/eval split. Returns the same
    TrainingResult shape as train_and_prove so orchestrator.py and the API
    layer don't need to change what they consume."""

    ctx = FileContext(log)
    train_traces, eval_traces = _split_traces(log.traces, eval_frac, train_seed)

    agent = QLearningAgent(n_actions=len(ACTIONS), seed=train_seed)
    env = TraceEnv(ctx, max_steps=max_steps)

    os.makedirs(log_dir, exist_ok=True)
    ep_log_path = os.path.join(log_dir, f"{run_id or fv.dataset_id}_episodes.jsonl") if log_dir else None

    def _write_ep(entry: dict):
        if ep_log_path:
            with open(ep_log_path, "a") as f:
                f.write(json.dumps({
                    **entry,
                    "dataset_id":   fv.dataset_id,
                    "content_hash": fv.content_hash[:16],
                    "epsilon":      round(agent.eps, 4),
                    "q_table_size": len(agent.q),
                }) + "\n")

    # ── Training: cycle through TRAIN traces only, shuffled each pass ──
    reward_curve = []
    rng = random.Random(train_seed)
    trace_cycle = list(train_traces)
    trace_log_budget = set(
        list(range(min(10, episodes))) +
        list(range(0, episodes, max(1, episodes // 20))) +
        list(range(max(0, episodes - 5), episodes))
    )

    for ep in range(episodes):
        if ep % len(trace_cycle) == 0:
            rng.shuffle(trace_cycle)
        trace = trace_cycle[ep % len(trace_cycle)]

        if ep in trace_log_budget:
            pol = lambda s, e: agent.act(s)
            total, trace_rec = _run_trace_episode_traced(pol, env, trace, agent, ep, "train")
            trace_rec["epsilon"] = round(agent.eps, 4)
            trace_rec["q_table_size"] = len(agent.q)
            _write_ep(trace_rec)
            # Re-run same trace for the actual Q-updates (traced run above didn't update the table)
            s = env.reset_on_trace(trace)
            done = False
            while not done:
                a = agent.act(s)
                s2, r, done, _ = env.step(a)
                agent.update(s, a, r, s2, done)
                s = s2
        else:
            s = env.reset_on_trace(trace)
            done, total = False, 0.0
            while not done:
                a = agent.act(s)
                s2, r, done, _ = env.step(a)
                agent.update(s, a, r, s2, done)
                s = s2; total += r
            _write_ep({
                "episode": ep, "mode": "train_compact", "case_id": trace.case_id,
                "total_reward": round(total, 4),
            })

        agent.decay()
        reward_curve.append(total)

    # ── Held-out evaluation: REAL traces the agent never trained on ──
    learned_rets = []
    baseline_rets = {b: [] for b in ("random", "do_nothing", "greedy")}
    rand_rng = random.Random(999)

    for i, trace in enumerate(eval_traces):
        if i < 5:
            total, trace_rec = _run_trace_episode_traced(
                lambda s, e: agent.act(s, greedy=True), env, trace, agent, i, "eval_learned"
            )
            _write_ep(trace_rec)
        else:
            total = _run_trace_episode(lambda s, e: agent.act(s, greedy=True), env, trace)
        learned_rets.append(total)

    for bname in baseline_rets:
        if bname == "random":
            pol = lambda s, e: rand_rng.randrange(len(ACTIONS))
        elif bname == "do_nothing":
            pol = lambda s, e: 0
        else:
            pol = _trace_greedy_policy
        for i, trace in enumerate(eval_traces):
            if i < 3:
                total, trace_rec = _run_trace_episode_traced(pol, env, trace, None, i, f"eval_{bname}")
                _write_ep(trace_rec)
            else:
                total = _run_trace_episode(pol, env, trace)
            baseline_rets[bname].append(total)

    # ── Stats (n = number of held-out traces -- disclosed, not padded) ──
    learned_stats  = _stats(learned_rets)
    baseline_stats = {b: _stats(v) for b, v in baseline_rets.items()}

    comparisons = {}
    for bname, brets in baseline_rets.items():
        deltas   = [l - b for l, b in zip(learned_rets, brets)]
        dm       = sum(deltas) / len(deltas)
        win_rate = sum(1 for d in deltas if d > 0) / len(deltas)
        rel_impr = (dm / abs(baseline_stats[bname]["mean"]) * 100
                    if baseline_stats[bname]["mean"] != 0 else 0.0)
        comparisons[bname] = {
            "mean_advantage":            round(dm, 4),
            "win_rate":                  round(win_rate, 3),
            "cohens_d":                  _cohens_d(learned_rets, brets),
            "relative_improvement_pct":  round(rel_impr, 1),
            "mae_vs_learned":            _mae(brets, learned_rets),
            "learned_better":            dm > 0,
        }

    action_counts: Counter = Counter()
    for trace in eval_traces:
        s = env.reset_on_trace(trace)
        done = False
        while not done:
            a = agent.act(s, greedy=True)
            action_counts[ACTIONS[a]] += 1
            s, _, done, _ = env.step(a)
    tot = sum(action_counts.values()) or 1
    action_dist = {k: round(v / tot, 3) for k, v in action_counts.most_common()}

    n_half = len(reward_curve) // 2
    fhm = sum(reward_curve[:n_half]) / n_half if n_half else 0
    shm = sum(reward_curve[n_half:]) / (len(reward_curve) - n_half) if n_half else 0

    metrics = {
        "training": {
            "episodes":                 episodes,
            "n_train_traces":           len(train_traces),
            "n_eval_traces":            len(eval_traces),
            "train_case_ids":           [t.case_id for t in train_traces],
            "eval_case_ids":            [t.case_id for t in eval_traces],
            "convergence_episode":      _convergence_episode(reward_curve),
            "first_half_mean_return":   round(fhm, 4),
            "second_half_mean_return":  round(shm, 4),
            "learning_improvement_pct": round((shm - fhm) / abs(fhm) * 100 if fhm else 0, 1),
            "final_epsilon":            round(agent.eps, 4),
            "policy_entropy":           agent.policy_entropy(),
            "q_table_size":             len(agent.q),
            "q_table_coverage_pct":     round(len(agent.q) / (5 ** len(FEATURE_NAMES)) * 100, 4),
        },
        "held_out": {"learned": learned_stats, "baselines": baseline_stats},
        "proof": {"learned_mean": learned_stats["mean"], "learned_std": learned_stats["std"],
                  "comparisons": comparisons},
        "action_distribution": action_dist,
        "episode_log_path": ep_log_path,
        "environment_note": (
            "Episodes are REAL traces from this file (see trace_env.py). Actions "
            f"{sorted(GROUNDED_ACTIONS)} apply to this trace's own real remaining "
            f"activity sequence; actions {sorted(ASSUMPTION_ACTIONS)} use a "
            "documented fixed-effect assumption layered on the real state because "
            "their true effect can't be observed from a historical log alone."
        ),
        "metric_definitions": {
            "mean_return":              "Average cumulative reward per HELD-OUT real trace (never trained on).",
            "std_return":               "Standard deviation of returns across held-out traces.",
            "cohens_d":                 "Effect size (|d|>0.8 = large). Scale-free comparison.",
            "win_rate":                 "Fraction of held-out traces where learned > baseline.",
            "mae_vs_learned":           "Mean Absolute Error of baseline vs learned, per held-out trace.",
            "relative_improvement_pct": "% improvement of learned mean return over baseline.",
            "policy_entropy":           "Action distribution entropy. Near 0 = deterministic policy.",
            "convergence_episode":      "Episode where reward first hit 90% of final value.",
            "n_eval_traces":            "Real, disclosed held-out sample size -- small for a single file; "
                                        "aggregate across a batch run for a larger n.",
        }
    }

    smoothing_window = max(5, episodes // 30)
    reward_curve_smoothed = _moving_average(reward_curve, smoothing_window)
    metrics["training"]["smoothing_window"] = smoothing_window

    return TrainingResult(
        dataset_id       = fv.dataset_id,
        content_hash     = fv.content_hash,
        episodes         = episodes,
        reward_curve     = reward_curve,
        reward_curve_smoothed = reward_curve_smoothed,
        learned_eval     = {"mean": learned_stats["mean"], "std": learned_stats["std"]},
        baseline_eval    = {b: {"mean": s["mean"], "std": s["std"]}
                            for b, s in baseline_stats.items()},
        proof            = {"learned_mean": learned_stats["mean"], "learned_std": learned_stats["std"],
                            "comparisons": comparisons},
        q_table_size     = len(agent.q),
        metrics          = metrics,
        episode_log_path = ep_log_path,
    )
