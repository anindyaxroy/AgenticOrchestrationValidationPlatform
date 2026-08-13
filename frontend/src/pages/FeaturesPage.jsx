import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../services/store'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
         BarChart, Bar, Cell, ReferenceLine, RadarChart, Radar,
         PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts'
import RunSwitcher from '../components/RunSwitcher'

const COL  = { learned:'#085041', random:'#185FA5', do_nothing:'#888', greedy:'#D85A30' }
const FEAT_COLORS = ['#085041','#185FA5','#534AB7','#B45309','#7C3AED','#0369A1']

const FEATURE_META = {
  bottleneck_score:        { label:'Bottleneck Score',         time:false, range:'[0,1]',
    formula:'max_a(mean_relative_position(a) × freq_share(a))',
    interpretation:'Higher = one activity dominates late-stage processing. Top activity appearing at 80%+ position with high frequency = structural bottleneck.' },
  sla_risk_index:          { label:'SLA Risk Index',           time:true,  range:'[0,1]',
    formula:'share(pdc:isPos=False) OR share(length > p75)',
    interpretation:'Higher = more cases are at-risk of breaching SLA. [PROXY] Without timestamps this uses conformance labels or long-tail length.' },
  cost_variance_norm:      { label:'Cost Variance (norm)',     time:false, range:'[0,1]',
    formula:'CV(case_cost) = std(cost)/mean(cost), clipped to [0,1]',
    interpretation:'Higher = unpredictable cost distribution. Near 1 = some cases cost dramatically more than average.' },
  dominant_activity_share: { label:'Dominant Activity Share',  time:false, range:'[0,1]',
    formula:'count(top_activity) / total_events',
    interpretation:'Higher = process is dominated by one activity. Low = balanced workload across activities.' },
  rework_probability:      { label:'Rework Probability',       time:false, range:'[0,1]',
    formula:'|{cases: ∃ activity that repeats}| / n_cases',
    interpretation:'Higher = more process loops. 0.5 means half of all cases contain at least one repeated activity.' },
  resource_utilisation:    { label:'Resource Utilisation',     time:true,  range:'[0,1]',
    formula:'mean_trace_length / max_trace_length',
    interpretation:'[PROXY] Without timestamps this is structural load. 1.0 means all traces are as long as the longest — uniform workload.' },
}

const METRIC_DEFS = [
  { key:'Mean Return',         icon:'ti-chart-line',      color:'#085041',
    what:'Average cumulative reward per episode on 100 held-out seeds the agent never trained on.',
    why:'The primary measure of policy quality. Higher = the agent consistently improves process health.' },
  { key:'Std Return',          icon:'ti-arrows-diff',     color:'#185FA5',
    what:'Standard deviation of episode returns across held-out seeds.',
    why:'Measures policy consistency. Low std = reliable; high std = erratic — the agent behaves differently on slight state variations.' },
  { key:"Cohen's d",           icon:'ti-ruler-measure',   color:'#534AB7',
    what:'Effect size: (mean_learned − mean_baseline) / pooled_std.',
    why:'Scale-free comparison. |d| > 0.8 = large effect (conventional threshold). Shows whether the difference is practically meaningful, not just statistically present.' },
  { key:'Win Rate',            icon:'ti-trophy',          color:'#B45309',
    what:'Fraction of held-out episodes where the learned policy return exceeded the baseline return.',
    why:'Intuitive dominance measure. 1.0 = learned policy won every single held-out episode.' },
  { key:'MAE vs Learned',      icon:'ti-calculator',      color:'#7C3AED',
    what:'Mean Absolute Error of baseline returns vs learned returns, per episode.',
    why:'How far off the baseline was, in reward units. Large MAE = baseline is a poor approximation of the learned policy.' },
  { key:'Relative Improvement', icon:'ti-percentage',    color:'#0369A1',
    what:'(mean_learned − mean_baseline) / |mean_baseline| × 100%.',
    why:'Percentage improvement over baseline. Easier to explain to business stakeholders than Cohen\'s d.' },
  { key:'Policy Entropy',      icon:'ti-arrows-shuffle',  color:'#D85A30',
    what:'Mean entropy of the action-value distribution across all Q-table states.',
    why:'Near 0 = deterministic converged policy. High entropy = agent is still uncertain which action to take in many states.' },
  { key:'Convergence Episode', icon:'ti-flag',            color:'#085041',
    what:'Episode at which training reward first reached 90% of its final value.',
    why:'Shows how quickly the agent learned. Early convergence = fast learning; late or absent = slow / insufficient training.' },
  { key:'Q-table Coverage',    icon:'ti-map',             color:'#185FA5',
    what:'States visited as % of theoretical state space (5^6 = 15,625 bins for 6 features × 5 discretisation bins).',
    why:'Low coverage = sparse exploration. The agent has not seen most possible states, so generalization is limited.' },
]

function MetricBadge({value, label, sub, color, icon}) {
  return (
    <div className="bg-white rounded-xl border border-gray-100 p-3 shadow-sm">
      <div className="flex items-center gap-1.5 mb-1">
        <i className={`ti ${icon} text-xs`} style={{color}}/>
        <div className="text-xs text-gray-400 leading-tight">{label}</div>
      </div>
      <div className="text-xl font-bold font-mono" style={{color}}>{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
    </div>
  )
}

export default function FeaturesPage() {
  const { currentRun } = useStore()
  const navigate = useNavigate()
  const [expandedFeat, setExpandedFeat] = useState(null)
  const [expandedMetric, setExpandedMetric] = useState(null)

  if (!currentRun) return (
    <div className="max-w-2xl">
      <div className="card text-center py-10 text-gray-400">
        <i className="ti ti-vector text-3xl block mb-2"/>
        <div className="text-sm mb-3">No pipeline run yet.</div>
        <button className="text-teal-600 text-sm hover:underline" onClick={() => navigate('/pipeline')}>Run the pipeline first →</button>
      </div>
    </div>
  )

  const fv  = currentRun.features
  const rl  = currentRun.rl
  const met = rl.metrics || {}
  const tr  = met.training || {}
  const hld = met.held_out || {}
  const cmp = met.proof?.comparisons || rl.proof?.comparisons || {}
  const actDist = met.action_distribution || {}

  const featNames = Object.keys(FEATURE_META)
  const featVals  = fv.values || {}

  const radarData = featNames.map(k => ({
    feature: FEATURE_META[k].label.replace(' (norm)', '').replace(' Score','').replace(' Index','').replace(' Share','').replace(' Probability',''),
    value:   Math.round((featVals[k] || 0) * 100),
  }))

  const rewardData = rl.reward_curve?.map((v,i) => ({
    ep: i+1,
    reward: +v.toFixed(4),
    smoothed: rl.reward_curve_smoothed?.[i] != null ? +rl.reward_curve_smoothed[i].toFixed(4) : null,
  })) || []
  const proofBars  = Object.entries(cmp).map(([name,c]) => ({
    name, adv: +c.mean_advantage.toFixed(4),
    win: c.win_rate, d: c.cohens_d,
    rel: c.relative_improvement_pct,
    mae: c.mae_vs_learned,
    better: c.learned_better,
  }))

  const actDistData = Object.entries(actDist)
    .map(([k,v]) => ({action: k.replace('_',' '), pct: Math.round(v*100)}))
    .sort((a,b) => b.pct - a.pct)

  return (
    <div className="max-w-5xl space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-medium text-gray-900">Features & Evaluation Metrics</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Source: <span className="font-mono text-teal-700">{currentRun.log_summary.source_filename}</span>
            <span className="ml-2 font-mono text-gray-400">hash: {currentRun.content_hash}</span>
          </p>
        </div>
        <RunSwitcher/>
      </div>

      {/* ── SECTION 1: MDP Feature Vector ── */}
      <div className="card">
        <div className="section-label">MDP State Vector — 6 Features</div>
        <p className="text-xs text-gray-500 mb-4">
          These six values form the state vector fed to the Q-learning agent at each decision step.
          Every value is computed from the uploaded log with a documented formula. Click a feature to see its definition.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Feature bars */}
          <div className="space-y-2">
            {featNames.map((k, i) => {
              const val  = featVals[k] || 0
              const meta = FEATURE_META[k]
              const prov = fv.provenance?.[k] || ''
              const open = expandedFeat === k
              return (
                <div key={k} className={`rounded-xl border transition-all cursor-pointer
                  ${meta.time ? 'border-amber-200' : 'border-gray-100'}
                  ${open ? 'shadow-md' : 'hover:border-gray-200'}`}
                  onClick={() => setExpandedFeat(open ? null : k)}>
                  <div className="flex items-center gap-3 p-3">
                    <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                      style={{background: FEAT_COLORS[i]+'20'}}>
                      {meta.time
                        ? <i className="ti ti-clock text-amber-500 text-xs"/>
                        : <i className="ti ti-calculator text-xs" style={{color: FEAT_COLORS[i]}}/>}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-xs font-semibold text-gray-800">{meta.label}</div>
                        <div className="font-mono text-sm font-bold" style={{color: FEAT_COLORS[i]}}>{val.toFixed(4)}</div>
                      </div>
                      <div className="bg-gray-100 rounded-full h-2 overflow-hidden">
                        <div className="h-full rounded-full transition-all" style={{width:`${Math.min(val,1)*100}%`, background: FEAT_COLORS[i]}}/>
                      </div>
                    </div>
                    <i className={`ti ${open?'ti-chevron-up':'ti-chevron-down'} text-gray-300 text-xs flex-shrink-0`}/>
                  </div>
                  {open && (
                    <div className="border-t border-gray-100 px-3 pb-3 pt-2 space-y-1.5">
                      <div className="text-xs"><span className="font-semibold text-gray-600">Formula: </span>
                        <code className="font-mono bg-gray-100 px-1 rounded text-teal-700">{meta.formula}</code></div>
                      <div className="text-xs text-gray-600"><span className="font-semibold">Range: </span>{meta.range}</div>
                      <div className="text-xs text-gray-600"><span className="font-semibold">Interpretation: </span>{meta.interpretation}</div>
                      {meta.time && <div className="text-xs text-amber-600 bg-amber-50 rounded px-2 py-1">
                        <i className="ti ti-clock-exclamation mr-1"/>Time-dependent proxy — real measure requires timestamps</div>}
                      {prov && <div className="text-xs font-mono text-gray-400 bg-gray-50 rounded px-2 py-1 break-all">
                        Derived: {prov}</div>}
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Radar chart */}
          <div>
            <div className="text-xs text-gray-400 mb-2 text-center">Feature Profile (% of max)</div>
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e5e7eb"/>
                <PolarAngleAxis dataKey="feature" tick={{fontSize:9, fill:'#6b7280'}}/>
                <PolarRadiusAxis domain={[0,100]} tick={{fontSize:8}} tickCount={4}/>
                <Radar dataKey="value" stroke="#085041" fill="#085041" fillOpacity={0.25} strokeWidth={2}/>
                <Tooltip formatter={v => [`${v}%`, 'value']}/>
              </RadarChart>
            </ResponsiveContainer>
            {(!currentRun.log_summary.has_timestamps) && (
              <div className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg p-2 mt-2">
                <i className="ti ti-clock-exclamation mr-1"/>
                <strong>2 features use proxies</strong> (amber border) — no timestamps in this log.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── SECTION 2: Training Dynamics ── */}
      <div className="card">
        <div className="section-label">RL Training Dynamics</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <MetricBadge label="Episodes" value={tr.episodes || rl.episodes}
            sub="training iterations" icon="ti-repeat" color="#185FA5"/>
          <MetricBadge label="Convergence" value={tr.convergence_episode ? `Ep ${tr.convergence_episode}` : '—'}
            sub="90% of final reward" icon="ti-flag" color="#085041"/>
          <MetricBadge label="Policy Entropy" value={tr.policy_entropy ?? '—'}
            sub="0=deterministic" icon="ti-arrows-shuffle" color="#D85A30"/>
          <MetricBadge label="Q-table States" value={tr.q_table_size || rl.q_table_size}
            sub={`${tr.q_table_coverage_pct?.toFixed(3) || '?'}% of state space`} icon="ti-map" color="#534AB7"/>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          <MetricBadge label="First Half Mean Return" value={tr.first_half_mean_return ?? '—'}
            sub="episodes 1 – N/2" icon="ti-chart-line" color="#888"/>
          <MetricBadge label="Second Half Mean Return" value={tr.second_half_mean_return ?? '—'}
            sub="episodes N/2 – N" icon="ti-chart-line-up" color="#085041"/>
          <MetricBadge label="Learning Improvement" value={tr.learning_improvement_pct != null ? `${tr.learning_improvement_pct}%` : '—'}
            sub="2nd half vs 1st half" icon="ti-trending-up"
            color={tr.learning_improvement_pct > 0 ? '#085041' : '#D85A30'}/>
        </div>
        <div className="text-xs text-gray-400 mb-2">
          Training Reward Curve (real Bellman updates — not manufactured)
        </div>
        {rl.reward_curve_smoothed?.length > 0 && (
          <div className="text-xs text-gray-400 mb-2">
            Faint line = raw per-episode return (noisy because each episode is a <em>different real trace</em>,
            not small noise around one point). Bold line = centered moving average
            {tr.smoothing_window ? ` (window = ${tr.smoothing_window} episodes)` : ''}, for a readable trend.
          </div>
        )}
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={rewardData}>
            <XAxis dataKey="ep" tick={{fontSize:10}} label={{value:'Episode', position:'insideBottom', offset:-2, fontSize:10}}/>
            <YAxis tick={{fontSize:10}} domain={['auto','auto']}/>
            <Tooltip formatter={(v,name)=>[v?.toFixed?.(4) ?? v, name === 'smoothed' ? 'moving avg' : 'return']} labelFormatter={l=>`Episode ${l}`}/>
            <Line type="monotone" dataKey="reward" stroke={COL.learned} dot={false} strokeWidth={1} strokeOpacity={0.35}/>
            {rl.reward_curve_smoothed?.length > 0 &&
              <Line type="monotone" dataKey="smoothed" stroke={COL.greedy} dot={false} strokeWidth={2.5}/>}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── SECTION 3: Held-out proof ── */}
      <div className="card">
        <div className="section-label">Proof of Learning — Held-Out Evaluation (100 seeds)</div>
        <p className="text-xs text-gray-500 mb-4">
          After training, the learned policy was evaluated on 100 initial states it <strong>never encountered during training</strong> (disjoint seed space 900,000–900,099 vs training space 10,000–10,N).
          The deltas below are the evidence of learning — not a training metric.
        </p>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <MetricBadge label="Learned Mean Return" value={hld.learned?.mean ?? rl.learned_eval?.mean}
            sub={`± ${hld.learned?.std ?? rl.learned_eval?.std}`} icon="ti-star" color="#085041"/>
          {proofBars.map(b => (
            <MetricBadge key={b.name}
              label={`vs ${b.name}`}
              value={b.better ? `+${b.adv}` : b.adv}
              sub={`win ${(b.win*100).toFixed(0)}%  d=${b.d}`}
              icon={b.better ? 'ti-check' : 'ti-x'}
              color={b.better ? '#085041' : '#D85A30'}/>
          ))}
        </div>

        {/* Full comparison table */}
        <div className="overflow-x-auto mb-4">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="text-left px-3 py-2 font-semibold text-gray-600">Baseline</th>
                <th className="text-right px-3 py-2 font-semibold text-gray-600">Mean Return</th>
                <th className="text-right px-3 py-2 font-semibold text-gray-600">Advantage</th>
                <th className="text-right px-3 py-2 font-semibold text-gray-600">Win Rate</th>
                <th className="text-right px-3 py-2 font-semibold text-gray-600">Cohen's d</th>
                <th className="text-right px-3 py-2 font-semibold text-gray-600">Rel. Improvement</th>
                <th className="text-right px-3 py-2 font-semibold text-gray-600">MAE</th>
                <th className="text-left px-3 py-2 font-semibold text-gray-600">Verdict</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-100 bg-teal-50">
                <td className="px-3 py-2 font-semibold text-teal-800">Learned (Q-learning)</td>
                <td className="px-3 py-2 text-right font-mono font-bold text-teal-700">{hld.learned?.mean ?? rl.learned_eval?.mean}</td>
                <td className="px-3 py-2 text-right text-gray-400">—</td>
                <td className="px-3 py-2 text-right text-gray-400">—</td>
                <td className="px-3 py-2 text-right text-gray-400">—</td>
                <td className="px-3 py-2 text-right text-gray-400">—</td>
                <td className="px-3 py-2 text-right text-gray-400">—</td>
                <td className="px-3 py-2 font-medium text-teal-700">Reference</td>
              </tr>
              {proofBars.map(b => (
                <tr key={b.name} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="px-3 py-2 font-semibold text-gray-700 capitalize">{b.name.replace('_',' ')}</td>
                  <td className="px-3 py-2 text-right font-mono text-gray-600">
                    {(hld.baselines?.[b.name]?.mean ?? rl.baseline_eval?.[b.name]?.mean)?.toFixed(4)}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono font-bold ${b.better?'text-teal-600':'text-red-500'}`}>
                    {b.adv > 0 ? '+' : ''}{b.adv}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-600">{(b.win*100).toFixed(0)}%</td>
                  <td className="px-3 py-2 text-right font-mono text-gray-600">{b.d}</td>
                  <td className={`px-3 py-2 text-right font-mono ${b.rel>0?'text-teal-600':'text-red-500'}`}>
                    {b.rel > 0 ? '+' : ''}{b.rel}%
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-gray-600">{b.mae}</td>
                  <td className={`px-3 py-2 font-medium ${b.better?'text-teal-700':'text-red-600'}`}>
                    <i className={`ti ${b.better?'ti-check':'ti-x'} mr-1`}/>
                    {b.better ? 'Learned wins' : 'Baseline competitive'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-gray-400 mb-2">Mean advantage over baseline (positive = learned better)</div>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={proofBars}>
                <XAxis dataKey="name" tick={{fontSize:10}}/>
                <YAxis tick={{fontSize:10}}/>
                <ReferenceLine y={0} stroke="#ccc"/>
                <Tooltip formatter={v=>[v.toFixed(4),'advantage']}/>
                <Bar dataKey="adv" radius={[3,3,0,0]}>
                  {proofBars.map((b,i)=><Cell key={i} fill={b.better?COL.learned:COL.greedy}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-2">Learned policy action distribution (held-out episodes)</div>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={actDistData} layout="vertical">
                <XAxis type="number" tickFormatter={v=>`${v}%`} tick={{fontSize:9}}/>
                <YAxis type="category" dataKey="action" tick={{fontSize:9}} width={90}/>
                <Tooltip formatter={v=>[`${v}%`,'share']}/>
                <Bar dataKey="pct" fill="#085041" radius={[0,3,3,0]}/>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {proofBars.some(b => !b.better && b.name === 'greedy') && (
          <div className="mt-4 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-3">
            <i className="ti ti-alert-triangle mr-1"/>
            <strong>Research finding:</strong> Greedy one-step baseline is competitive on current dynamics.
            The transition model is near-myopic — actions produce immediate rewards without delayed consequences.
            To justify RL over a heuristic, enrich ACTION_EFFECTS with interaction effects (e.g. parallelising
            blocks later compliance insertion). This is a documented finding, not a failure.
          </div>
        )}
      </div>

      {/* ── SECTION 4: Metric Definitions ── */}
      <div className="card">
        <div className="section-label">Metric Definitions & Why They Matter</div>
        <p className="text-xs text-gray-500 mb-4">
          Click any metric to expand its definition. These are the metrics used to evaluate the RL agent
          and constitute the empirical evidence for your thesis claims.
        </p>
        <div className="space-y-1.5">
          {METRIC_DEFS.map(m => {
            const open = expandedMetric === m.key
            return (
              <div key={m.key} className="rounded-xl border border-gray-100 cursor-pointer hover:border-gray-200 transition-all"
                onClick={() => setExpandedMetric(open ? null : m.key)}>
                <div className="flex items-center gap-3 px-3 py-2.5">
                  <i className={`ti ${m.icon} text-sm flex-shrink-0`} style={{color:m.color}}/>
                  <div className="text-xs font-semibold text-gray-800 flex-1">{m.key}</div>
                  <i className={`ti ${open?'ti-chevron-up':'ti-chevron-down'} text-gray-300 text-xs`}/>
                </div>
                {open && (
                  <div className="border-t border-gray-100 px-3 pb-3 pt-2 space-y-1.5">
                    <div className="text-xs text-gray-600"><strong>What it measures:</strong> {m.what}</div>
                    <div className="text-xs text-gray-600"><strong>Why it matters:</strong> {m.why}</div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className="flex gap-2">
        <button className="btn-primary" onClick={() => navigate('/agent')}>
          <i className="ti ti-robot mr-1"/> Agent Reasoning →
        </button>
      </div>
    </div>
  )
}
