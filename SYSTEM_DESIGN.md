# System Design

This document explains *why* the platform is built the way it is: the design decisions behind the state representation, the reinforcement-learning formulation, and the evaluation protocol. For *what* each component does mechanically, see `ARCHITECTURE.md`. For exact formulas and academic citations, see the thesis's Methodology chapter.

## Design principle: one pipeline, not two

An early version of this project's evaluation design ran non-timestamped and timestamped datasets through structurally separate paths. That design was retracted. The platform now applies one six-stage pipeline to every dataset; the only thing that changes between a timestamped log (BPI Challenge 2012/2017/2019) and a non-timestamped one (PDC 2016) is which of two disclosed formula branches computes three of the six state features. This matters architecturally because it means a bug fix, a metric change, or a new baseline applies identically everywhere — there is no separate code path to keep in sync.

## State representation

Process state at each decision step is a six-dimensional vector, each dimension grounded in a specific theoretical construct rather than chosen empirically:

| Feature | Grounding |
|---|---|
| Bottleneck score | Little's Law — work-in-progress accumulation as a capacity-mismatch signal |
| SLA risk index | Deadline-driven prioritisation (a standard workflow-redesign heuristic) |
| Rework probability | Process deviation / exception signal from process-mining theory |
| Cost variance, dominant activity share, resource utilisation | Descriptive covariates, not independently hypothesis-bearing |

**This is theory-driven feature engineering, not data-driven feature selection.** No filter, wrapper, or embedded feature-selection method was run to choose these six. If you're evaluating this platform, that's a deliberate scope boundary, not an oversight — it's stated explicitly rather than implied.

Three of the six features require duration data and therefore branch:

- **Real-timestamp branch**: used when a file has ≥80% usable timestamp coverage. Computes genuine wait times, cycle times, and elapsed durations.
- **Sequence-position proxy branch**: used otherwise. Substitutes relative position within the trace for actual elapsed time.

This branch is decided once per file and applied consistently across all three time-dependent features — a file can't end up using real data for one and a proxy for another for the same underlying reason.

## Action space: grounded vs. assumption-based

Seven candidate actions, split into two categories that differ in how their effect is computed:

- **Grounded** (`no_op`, `skip_optional`, `reorder_activities`) — applied directly to the trace's own real remaining event sequence. Their effect is a genuine, trace-specific counterfactual.
- **Assumption-based** (`parallelise`, `reallocate_resource`, `escalate_to_human`, `insert_compliance_agent`) — apply a fixed, documented delta on top of the naturally-advanced real state, because their true causal effect cannot be estimated from a historical log without a randomised controlled trial.

This split is deliberate and disclosed per-action, not left implicit in an undifferentiated action set. It bounds what can honestly be claimed about a "recommendation": statistical validity of the policy comparison (Section below) is one claim; causal validity of an individual assumption-based action's effect is a different, unestablished claim. The two should never be conflated when interpreting platform output.

## Reinforcement learning: what and why

**Algorithm**: tabular Q-learning (Watkins & Dayan, 1992) — not deep RL. The state space is low-dimensional (6 features × 5 discretised bins = at most 15,625 states), so a neural function approximator would be solving a problem this state space doesn't have. The Q-table is a sparse dictionary; only visited states get an entry.

**Reward**: dense, per-step, derived from the state itself via a fixed linear "health" function — a weighted sum of the "badness" features (bottleneck, SLA risk, rework weighted highest; resource utilisation and cost variance weighted lower). Reward is the change in health from one step to the next. This is not an externally supplied ground-truth reward; it's fully determined by the same six features that define state.

**Why a greedy baseline, specifically**: the platform evaluates the trained policy against three baselines — random, do-nothing, and a one-step greedy policy with perfect knowledge of the reward function. Greedy is the theoretically interesting comparator: it performs no discretisation and no value propagation, recomputing the true best immediate action fresh at every step. If the trained agent can't beat it, that's informative about the transition dynamics of the specific dataset (whether the environment rewards lookahead at all), not simply a training shortfall.

**Held-out evaluation**: traces are split at the case level (not the event level), 80/20, with zero case-ID overlap between train and held-out pools verified programmatically on every run. The agent's Q-table is only ever updated from the training pool.

## Evaluation metrics — and a disclosed caveat

Per baseline comparison: mean advantage, win rate, relative improvement, MAE, and Cohen's d. All are paired at the trace level (same held-out trace evaluated under both policies), except Cohen's d, which uses the **independent-samples** formula — pooling the standard deviation of the two raw return arrays rather than the standard deviation of their paired differences. This is a generally conservative choice relative to the textbook paired estimator (d_z), and it's disclosed here rather than silently assumed correct. If you're extending this platform's statistical layer, implementing the paired d_z as an alternative/cross-check is a natural next contribution.

## Disclosed implementation corrections

Three implementation errors were found during this project's own validation work and corrected — listed here because they're the kind of thing a code reviewer should be able to find without re-deriving the whole pipeline:

1. **Bottleneck-score normalisation** (sequence-position proxy branch) — an earlier version divided the top-ranked score by itself, which is mathematically always exactly 1.0 regardless of data. Present independently in two code modules computing the same formula. Fixed by removing the redundant self-referential step (both constituent terms are already bounded in [0,1]).
2. **Convergence-episode metric** — used a threshold of `final_reward × 1.1`. Because this environment's reward is constructed to be consistently negative, that threshold was *more negative* than the final value, so the metric trivially reported episode 1 regardless of actual training progress. Fixed with an absolute tolerance band, symmetric for either sign of reward curve.
3. **Cohen's d on zero-variance held-out samples** — silently returned `0.0000` when the pooled standard deviation of two compared return arrays collapsed to zero (e.g., a baseline producing an identical return on every held-out trace). A returned `0.0` is indistinguishable from a genuine null effect in a results table, which it isn't — a zero-variance denominator makes the effect size undefined, not zero. Fixed to return an explicit undefined marker instead.

If you're picking up this codebase, it's worth running a quick regression check (re-run a previously validated dataset) after any change to `rl_agent.py`'s statistics functions, precisely because this class of bug — correct on well-conditioned data, silently wrong on degenerate input — has already bitten this project three times.

## Known open questions (not yet resolved in code)

- Whether the RL-vs-greedy performance gap correlates with training-pool size, rework probability, or some combination — evidence to date is mixed across validated datasets, and the platform doesn't yet have a matched-pool re-run mode built in.
- A full seven-action distribution tabulation on held-out traces, to check whether the action space's current incentive structure (favouring assumption-based actions, which have a guaranteed-direction effect) limits the interpretability of certain agent-behaviour hypotheses.

These aren't bugs — they're the actual open research questions this platform was built to investigate, and they should stay visible to anyone extending the code rather than getting buried in a thesis PDF.
