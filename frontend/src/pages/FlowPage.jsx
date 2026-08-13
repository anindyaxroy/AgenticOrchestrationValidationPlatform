import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { pipelineApi } from '../services/api'
import { useStore } from '../services/store'
import RunSwitcher from '../components/RunSwitcher'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
         BarChart, Bar, Cell, ReferenceLine } from 'recharts'

// ── Colour palette ────────────────────────────────────────────────────────────
const C = {
  teal:   '#085041', blue:   '#185FA5', purple: '#534AB7',
  amber:  '#B45309', violet: '#7C3AED', slate:  '#0369A1',
  red:    '#D85A30', gray:   '#6B7280',
}

const FEAT_COLORS = [C.teal, C.blue, C.purple, C.amber, C.violet, C.slate]
const FEAT_NAMES  = [
  'bottleneck_score','sla_risk_index','cost_variance_norm',
  'dominant_activity_share','rework_probability','resource_utilisation',
]
const FEAT_SHORT  = ['Bottleneck','SLA Risk','Cost Var','Dom.Act','Rework','Resource']

// ── Pipeline stages definition ────────────────────────────────────────────────
const STAGES = [
  {
    id: 'upload', label: 'Upload', icon: 'ti-upload', color: C.slate,
    module: 'log_loader.py',
    what: 'Browser POSTs the file to POST /api/datasets/upload. FastAPI writes it to /app/data/uploads/{name}_{uuid}.xes/csv.',
    output: 'dataset_id + file_path stored in Postgres datasets table.',
    code: `# datasets.py
raw = await file.read()
with open(dest, "wb") as f:
    f.write(raw)
preview = _compute_preview(dest, ds_id)   # parses immediately
ds = Dataset(name=fname, file_path=dest, ...)
db.add(ds)`,
  },
  {
    id: 'load', label: 'Load + Lineage', icon: 'ti-file-import', color: C.teal,
    module: 'log_loader.py',
    what: 'Opens the file, detects format (XES vs CSV vs GZ), parses it into a normalised EventLog. Computes SHA-256 of raw bytes — this becomes the content_hash that appears on every audit line.',
    output: 'EventLog(traces, activities, timestamps, trace_attr_keys, content_hash)',
    code: `# log_loader.py
raw  = open(path, "rb").read()
chash = hashlib.sha256(raw).hexdigest()   # lineage anchor
fmt  = _detect_format(filename, raw)       # 'xes' | 'csv'
if fmt == "xes":
    traces, has_ts, attr_keys = _parse_xes(raw)
else:
    traces, has_ts, attr_keys = _parse_csv(raw)
return EventLog(..., content_hash=chash)`,
  },
  {
    id: 'mine', label: 'Process Mining', icon: 'ti-chart-dots', color: C.blue,
    module: 'process_mining.py',
    what: 'Computes real process statistics from the parsed traces. No hardcoded values — every number comes from the actual log.',
    output: 'Variants, directly-follows graph, sequence bottlenecks, conformance, findings text[]',
    code: `# process_mining.py
variant_counter = Counter(tuple(t.activities) for t in traces)
# sequence bottleneck: late + frequent
for t in traces:
    for idx, a in enumerate(t.activities):
        pos_accum[a] += idx / max(t.length-1, 1)
# conformance via labels (pdc:isPos)
pos = [t for t in traces if t.attrs.get("pdc:isPos") is True]
# natural-language findings → will be embedded for the agent
findings = mr.findings_text()`,
  },
  {
    id: 'features', label: 'Feature Extraction', icon: 'ti-vector', color: C.purple,
    module: 'feature_extraction.py',
    what: 'Derives the 6-dimensional MDP state vector. Each feature has a documented formula. Time-dependent features use proxies when no timestamps exist.',
    output: 'FeatureVector(values, provenance, time_based_available)',
    code: `# feature_extraction.py  — one example feature
# dominant_activity_share
most_common_n = act_counts.most_common(1)[0][1]
dominant_share = most_common_n / total_events

# rework_probability
rework = sum(1 for t in traces
             if len(set(t.activities)) < len(t.activities))
rework_probability = rework / n_cases

# provenance string stored alongside value:
prov["rework_probability"] = (
  f"cases_with_repeated_activity={rework} / n_cases={n_cases}"
)`,
  },
  {
    id: 'embed', label: 'Embed Findings', icon: 'ti-database', color: C.amber,
    module: 'knowledge_store.py',
    what: 'Natural-language mining findings are vector-embedded into Chroma. The agent retrieves them by semantic similarity at reasoning time. Each embedding is tagged with dataset_id + content_hash.',
    output: 'N findings stored in Chroma collection pm_findings',
    code: `# knowledge_store.py
def add_findings(self, dataset_id, content_hash, findings, kind):
    ids, metas = [], []
    for i, f in enumerate(findings):
        fid = sha256(f"{content_hash}:{kind}:{i}:{f}").hex()[:24]
        metas.append({"dataset_id": dataset_id,
                       "content_hash": content_hash[:16]})
    self.col.upsert(ids=ids, documents=findings, metadatas=metas)

# later, the agent queries:
hits = store.query("What is the bottleneck?",
                    dataset_id=dataset_id, k=5)`,
  },
  {
    id: 'rl', label: 'RL Training + Proof', icon: 'ti-brain', color: C.violet,
    module: 'rl_agent.py',
    what: 'Trains a tabular Q-learning agent. State = discretised FeatureVector. Actions = 7 process interventions. Reward = health improvement (lower bottleneck/SLA/rework). Evaluates on 100 HELD-OUT seeds the agent never trained on. Compares against 3 baselines.',
    output: 'Q-table, reward_curve, metrics (MAE, Cohen\'s d, win_rate, convergence), episode JSONL log',
    code: `# rl_agent.py — Bellman update
def update(self, s, a, r, s2, done):
    row = self._row(s)
    nxt = 0.0 if done else max(self._row(s2))
    row[a] += alpha * (r + gamma * nxt - row[a])
    #         └─ learning rate × TD error ─┘

# evaluation on HELD-OUT seeds (disjoint from training)
eval_seeds = [900_000 + i for i in range(100)]
learned_returns = [run_episode(agent_greedy, ev, sd)
                    for sd in eval_seeds]

# Cohen's d effect size
cohens_d = (mean_learned - mean_baseline) / pooled_std

# Episode trace written to JSONL:
_write_ep({"episode": ep, "action": ACTIONS[a],
            "state_before": {...}, "state_after": {...},
            "q_values": agent.q_values(s), "reward": r})`,
  },
  {
    id: 'agent', label: 'Agent Reasoning', icon: 'ti-robot', color: C.red,
    module: 'reasoning_agent.py + LangGraph',
    what: '5-node LangGraph state machine. init_node plans → retrieve_node queries Chroma for relevant findings → rl_node loads the learning proof → reason_node calls Claude Sonnet with grounded evidence → summarise_node produces business summary. Output is structured Markdown.',
    output: 'Markdown reasoning, executive summary, node audit trail',
    code: `# reasoning_agent.py
g = StateGraph(AgentState)  # single 'data' key — no LangGraph clash
g.add_node("init_node",      init_node)
g.add_node("retrieve_node",  retrieve_node)   # Chroma query
g.add_node("rl_node",        rl_node)          # loads proof
g.add_node("reason_node",    reason_node)      # Claude Sonnet
g.add_node("summarise_node", summarise_node)   # exec summary
g.set_entry_point("init_node")
# ...edges...
result = graph.invoke({"data": {"question": q,
                                 "dataset_id": dataset_id}})`,
  },
  {
    id: 'audit', label: 'Audit Log', icon: 'ti-shield-check', color: C.teal,
    module: 'audit.py + orchestrator.py',
    what: 'Every stage appends a JSONL event to /app/data/runs/{run_id}.jsonl. Each line carries the content_hash, so any output can be traced to the exact source file bytes. The episode log is a separate {run_id}_episodes.jsonl.',
    output: 'Append-only JSONL audit trail + per-episode trace log',
    code: `# audit.py
def log(self, event: str, detail: dict) -> dict:
    self._seq += 1
    entry = {
        "seq":          self._seq,
        "ts":           _now_iso(),
        "run_id":       self.run_id,
        "content_hash": self.content_hash[:16],  # lineage on EVERY line
        "event":        event,
        "detail":       detail,
    }
    with open(self.log_path, "a") as f:
        f.write(json.dumps(entry) + "\\n")`,
  },
]

// ── Episode trace viewer ──────────────────────────────────────────────────────
function EpisodeViewer({ runId }) {
  const [episodes, setEpisodes]   = useState([])
  const [loading,  setLoading]    = useState(false)
  const [mode,     setMode]       = useState('train')
  const [selected, setSelected]   = useState(null)
  const [error,    setError]      = useState(null)

  const load = async () => {
    if (!runId) return
    setLoading(true); setError(null)
    try {
      const r = await pipelineApi.episodes(runId, mode, 30)
      setEpisodes(r.entries || [])
      setSelected(r.entries?.[0] || null)
    } catch(e) {
      setError(e.response?.data?.detail || e.message)
    } finally { setLoading(false) }
  }

  useEffect(() => { if (runId) load() }, [runId, mode])

  const MODES = [
    {id:'train',       label:'Training'},
    {id:'train_compact', label:'Training (compact)'},
    {id:'eval_learned',label:'Eval: Learned'},
    {id:'eval_random', label:'Eval: Random'},
    {id:'eval_greedy', label:'Eval: Greedy'},
  ]

  if (!runId) return (
    <div className="text-xs text-gray-400 text-center py-8">Run the pipeline to view episode traces.</div>
  )

  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        {MODES.map(m => (
          <button key={m.id} onClick={() => setMode(m.id)}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors
              ${mode===m.id ? 'border-teal-400 bg-teal-50 text-teal-700' : 'border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
            {m.label}
          </button>
        ))}
      </div>

      {error && <div className="text-xs text-red-500 bg-red-50 rounded-lg p-2">{error}</div>}
      {loading && <div className="text-xs text-gray-400 text-center py-4"><i className="ti ti-loader animate-spin mr-1"/>Loading episodes…</div>}

      {!loading && episodes.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Episode list */}
          <div className="space-y-1 max-h-96 overflow-y-auto">
            {episodes.map((ep, i) => (
              <div key={i} onClick={() => setSelected(ep)}
                className={`p-2.5 rounded-lg border cursor-pointer transition-all text-xs
                  ${selected===ep ? 'border-teal-400 bg-teal-50' : 'border-gray-100 hover:border-gray-200'}`}>
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-gray-700">Ep {ep.episode}</span>
                  <span className={`font-mono font-bold ${ep.total_reward > 0 ? 'text-teal-600' : 'text-red-500'}`}>
                    {ep.total_reward > 0 ? '+' : ''}{ep.total_reward}
                  </span>
                </div>
                <div className="text-gray-400 mt-0.5">
                  {ep.mode} · ε={ep.epsilon ?? '—'} · Q={ep.q_table_size ?? '—'}
                </div>
              </div>
            ))}
          </div>

          {/* Episode detail */}
          <div className="md:col-span-2">
            {selected ? (
              <div className="space-y-3">
                {/* Summary */}
                <div className="grid grid-cols-3 gap-2">
                  {[
                    {label:'Total Reward', val: selected.total_reward, color: selected.total_reward > 0 ? C.teal : C.red},
                    {label:'Steps',        val: selected.n_steps || selected.steps?.length || '—'},
                    {label:'Epsilon',      val: selected.epsilon ?? '—'},
                  ].map(({label,val,color}) => (
                    <div key={label} className="bg-gray-50 rounded-xl p-2.5 text-center">
                      <div className="text-xs text-gray-400">{label}</div>
                      <div className="font-mono font-bold text-sm" style={{color: color || '#333'}}>{val}</div>
                    </div>
                  ))}
                </div>

                {/* Initial vs final state */}
                {selected.initial_state && (
                  <div>
                    <div className="text-xs font-semibold text-gray-600 mb-2">State evolution (initial → final)</div>
                    <div className="space-y-1">
                      {FEAT_NAMES.map((k, i) => {
                        const ini = selected.initial_state?.[k] ?? 0
                        const fin = selected.final_state?.[k] ?? 0
                        const delta = fin - ini
                        return (
                          <div key={k} className="flex items-center gap-2 text-xs">
                            <div className="w-24 text-gray-500 truncate">{FEAT_SHORT[i]}</div>
                            <div className="flex-1 relative h-4 bg-gray-100 rounded-full overflow-hidden">
                              <div className="absolute h-full rounded-full opacity-30"
                                style={{width:`${ini*100}%`, background: FEAT_COLORS[i]}}/>
                              <div className="absolute h-full rounded-full"
                                style={{width:`${fin*100}%`, background: FEAT_COLORS[i], opacity: 0.7}}/>
                            </div>
                            <div className="font-mono w-12 text-right">{fin.toFixed(3)}</div>
                            <div className={`font-mono w-14 text-right ${delta < 0 ? 'text-teal-600' : delta > 0 ? 'text-red-500' : 'text-gray-400'}`}>
                              {delta > 0 ? '+' : ''}{delta.toFixed(3)}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Step-by-step trace */}
                {selected.steps?.length > 0 && (
                  <div>
                    <div className="text-xs font-semibold text-gray-600 mb-2">Step-by-step action trace</div>
                    <div className="space-y-1.5 max-h-72 overflow-y-auto">
                      {selected.steps.map((step, si) => (
                        <div key={si} className="bg-gray-50 rounded-lg p-2 text-xs border border-gray-100">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-mono text-gray-400 w-6">t={step.step}</span>
                            <span className="font-mono font-bold px-2 py-0.5 rounded"
                              style={{background: C.teal+'20', color: C.teal}}>
                              {step.action}
                            </span>
                            <span className={`font-mono ml-auto font-bold ${step.reward_delta > 0 ? 'text-teal-600' : step.reward_delta < 0 ? 'text-red-500' : 'text-gray-400'}`}>
                              {step.reward_delta > 0 ? '+' : ''}{step.reward_delta}
                            </span>
                          </div>
                          {/* Q-values for this step */}
                          {step.q_values && Object.keys(step.q_values).length > 0 && (
                            <div className="flex gap-1 flex-wrap mt-1">
                              {Object.entries(step.q_values)
                                .sort((a,b) => b[1]-a[1])
                                .slice(0,4)
                                .map(([act,qv]) => (
                                  <span key={act} className={`font-mono text-xs px-1.5 py-0.5 rounded
                                    ${act === step.action ? 'bg-teal-100 text-teal-700 font-bold' : 'bg-gray-100 text-gray-500'}`}>
                                    {act.replace('_',' ')}={qv}
                                  </span>
                                ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-xs text-gray-400 text-center py-8">Select an episode to inspect</div>
            )}
          </div>
        </div>
      )}
      {!loading && episodes.length === 0 && !error && (
        <div className="text-xs text-gray-400 text-center py-6">No {mode} episodes in this log.</div>
      )}
    </div>
  )
}

// ── Main FlowPage ─────────────────────────────────────────────────────────────
export default function FlowPage() {
  const { currentRun } = useStore()
  const navigate       = useNavigate()
  const [expanded, setExpanded] = useState(null)
  const [tab,      setTab]      = useState('flow')  // 'flow' | 'episodes'

  return (
    <div className="max-w-5xl space-y-5">
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-medium text-gray-900">Pipeline Architecture & Episode Traces</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            How every file byte becomes a trained agent decision — and how to audit each step.
          </p>
        </div>
        <RunSwitcher/>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-gray-200">
        {[
          {id:'flow',     label:'Architecture & Code',   icon:'ti-sitemap'},
          {id:'episodes', label:'Episode Trace Log',     icon:'ti-timeline'},
        ].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium border-b-2 transition-colors -mb-px
              ${tab===t.id ? 'border-teal-600 text-teal-700' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>
            <i className={`ti ${t.icon}`}/>{t.label}
          </button>
        ))}
      </div>

      {/* ── Architecture tab ── */}
      {tab === 'flow' && (
        <div className="space-y-3">
          <div className="text-xs text-gray-500 bg-blue-50 border border-blue-200 rounded-lg p-3">
            <i className="ti ti-info-circle mr-1 text-blue-500"/>
            Click any stage to expand the code that runs it. Every stage writes to the audit log
            with the same <span className="font-mono">content_hash</span> — tracing all outputs
            back to the uploaded file.
          </div>

          {STAGES.map((stage, idx) => {
            const isOpen   = expanded === stage.id
            const isLast   = idx === STAGES.length - 1
            return (
              <div key={stage.id}>
                {/* Stage card */}
                <div className={`rounded-xl border-2 transition-all cursor-pointer
                  ${isOpen ? 'shadow-lg' : 'hover:shadow-md'}`}
                  style={{borderColor: isOpen ? stage.color : '#e5e7eb'}}
                  onClick={() => setExpanded(isOpen ? null : stage.id)}>

                  <div className="flex items-center gap-4 p-4">
                    {/* Step number + icon */}
                    <div className="flex flex-col items-center flex-shrink-0">
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{background: stage.color}}>
                        <i className={`ti ${stage.icon} text-white text-lg`}/>
                      </div>
                      <div className="font-mono text-xs text-gray-300 mt-1">0{idx+1}</div>
                    </div>

                    {/* Stage info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-bold text-gray-900">{stage.label}</span>
                        <code className="font-mono text-xs px-2 py-0.5 rounded-full"
                          style={{background: stage.color+'18', color: stage.color}}>
                          {stage.module}
                        </code>
                      </div>
                      <p className="text-xs text-gray-500 leading-relaxed">{stage.what}</p>
                    </div>

                    {/* Output badge */}
                    <div className="text-right flex-shrink-0 hidden md:block max-w-48">
                      <div className="text-xs text-gray-400 mb-1">outputs</div>
                      <div className="text-xs text-gray-600 leading-snug">{stage.output}</div>
                    </div>

                    <i className={`ti ${isOpen?'ti-chevron-up':'ti-chevron-down'} text-gray-300 flex-shrink-0`}/>
                  </div>

                  {/* Expanded code section */}
                  {isOpen && (
                    <div className="border-t-2 p-4 space-y-3" style={{borderColor: stage.color+'40'}}>
                      <div>
                        <div className="text-xs font-semibold text-gray-600 mb-1">What runs:</div>
                        <div className="text-xs text-gray-600 leading-relaxed">{stage.what}</div>
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-gray-600 mb-1.5">Output:</div>
                        <div className="text-xs font-mono bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-gray-600">
                          {stage.output}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-gray-600 mb-1.5">Core code:</div>
                        <pre className="bg-gray-900 text-green-300 rounded-xl p-4 text-xs font-mono overflow-x-auto leading-relaxed whitespace-pre-wrap">
                          {stage.code.trim()}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>

                {/* Arrow connector */}
                {!isLast && (
                  <div className="flex justify-center my-1">
                    <div className="flex flex-col items-center">
                      <div className="w-0.5 h-4 bg-gray-200"/>
                      <i className="ti ti-arrow-down text-gray-300 text-xs"/>
                      <div className="text-xs text-gray-300 font-mono">audit_log.append(event + content_hash)</div>
                    </div>
                  </div>
                )}
              </div>
            )
          })}

          {/* Lineage explanation */}
          <div className="card bg-teal-50 border-teal-200 mt-4">
            <div className="section-label">How Lineage Threads Through the Entire Pipeline</div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs text-gray-600">
              {[
                {icon:'ti-file-import', label:'1. Upload', text:'SHA-256 of file bytes → content_hash'},
                {icon:'ti-link',        label:'2. Every stage', text:'Asserts input content_hash matches run_id\'s hash before executing'},
                {icon:'ti-writing',     label:'3. Audit log', text:'Every event writes content_hash. Swap the file → different hash → different chain'},
                {icon:'ti-search',      label:'4. Verification', text:'GET /api/pipeline/audit/{run_id} returns the full chain of custody'},
              ].map(s => (
                <div key={s.label} className="flex gap-2">
                  <i className={`ti ${s.icon} text-teal-500 text-base flex-shrink-0 mt-0.5`}/>
                  <div><div className="font-semibold text-teal-800 mb-0.5">{s.label}</div>{s.text}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Episode trace tab ── */}
      {tab === 'episodes' && (
        <div className="space-y-4">
          <div className="card">
            <div className="section-label">What is the Episode Log?</div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-gray-600 mb-4">
              {[
                {icon:'ti-file-text', color: C.teal, label:'File location',
                  text:'/app/data/runs/{run_id}_episodes.jsonl — append-only, one JSON line per episode'},
                {icon:'ti-list-details', color: C.blue, label:'What each entry contains',
                  text:'episode number, mode, seed, total reward, epsilon, Q-table size, initial/final state, step-by-step action trace with Q-values'},
                {icon:'ti-filter', color: C.purple, label:'Which episodes are traced',
                  text:'First 10 training eps (full), every 20th after that, last 5 training, first 5 eval eps per policy'},
              ].map(s => (
                <div key={s.label} className="flex gap-2.5 p-3 bg-gray-50 rounded-xl">
                  <i className={`ti ${s.icon} text-lg flex-shrink-0 mt-0.5`} style={{color:s.color}}/>
                  <div>
                    <div className="font-semibold text-gray-800 mb-0.5">{s.label}</div>
                    <div>{s.text}</div>
                  </div>
                </div>
              ))}
            </div>

            {currentRun ? (
              <EpisodeViewer runId={currentRun.run_id}/>
            ) : (
              <div className="text-center py-10 text-gray-400">
                <i className="ti ti-timeline text-3xl block mb-2"/>
                <div className="text-sm mb-3">No pipeline run yet.</div>
                <button className="text-teal-600 text-sm hover:underline" onClick={() => navigate('/pipeline')}>
                  Run the pipeline first →
                </button>
              </div>
            )}
          </div>

          {/* JSONL format reference */}
          <div className="card">
            <div className="section-label">JSONL Format Reference</div>
            <pre className="bg-gray-900 text-green-300 rounded-xl p-4 text-xs font-mono overflow-x-auto leading-relaxed">
{`// Training episode with full step trace:
{
  "episode": 3,
  "mode": "train",
  "seed": 10003,
  "total_reward": 0.4821,
  "n_steps": 12,
  "epsilon": 0.9851,
  "q_table_size": 24,
  "dataset_id": "ds-pdc1",
  "content_hash": "c2cccb39792228c2",   // ← lineage anchor
  "initial_state": {
    "bottleneck_score": 0.9823,
    "sla_risk_index": 0.5012,
    "cost_variance_norm": 0.9934,
    "dominant_activity_share": 0.0871,
    "rework_probability": 0.4978,
    "resource_utilisation": 0.4612
  },
  "final_state": { ... },
  "steps": [
    {
      "step": 1,
      "action": "reorder_activities",
      "action_id": 6,
      "reward_delta": 0.0855,
      "state_before": { "bottleneck_score": 0.9823, ... },
      "state_after":  { "bottleneck_score": 0.8923, ... },
      "q_values": {
        "no_op": 0.0,
        "parallelise": 0.0312,
        "reorder_activities": 0.0421,  // ← chosen (highest)
        ...
      }
    },
    ...
  ]
}

// Compact training episode (not in trace set):
{
  "episode": 11,
  "mode": "train_compact",
  "seed": 10011,
  "total_reward": 0.5103,
  "epsilon": 0.9459,
  "q_table_size": 67,
  "content_hash": "c2cccb39792228c2"
}`}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}
