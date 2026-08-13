import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../services/store'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Sankey } from 'recharts'
import RunSwitcher from '../components/RunSwitcher'

const COL = ['#085041','#185FA5','#534AB7','#B45309','#7C3AED','#0369A1','#D85A30','#888']

export default function MiningPage() {
  const { currentRun, selectedDataset } = useStore()
  const navigate = useNavigate()

  if (!currentRun) return (
    <div className="max-w-2xl">
      <div className="card text-center py-10 text-gray-400">
        <i className="ti ti-chart-dots text-3xl block mb-2"/>
        <div className="text-sm mb-3">No pipeline run yet.</div>
        <button className="text-teal-600 text-sm hover:underline" onClick={() => navigate('/pipeline')}>
          Run the pipeline first →
        </button>
      </div>
    </div>
  )

  const m = currentRun.mining
  const s = currentRun.log_summary

  return (
    <div className="max-w-5xl space-y-5">
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-medium text-gray-900">Process Mining Results</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Source: <span className="font-mono text-teal-700">{s.source_filename}</span>
            <span className="ml-2 font-mono text-gray-400">hash: {currentRun.content_hash}</span>
          </p>
        </div>
        <RunSwitcher/>
      </div>

      {/* Dataset characterisation */}
      <div className="card">
        <div className="section-label">Dataset Characterisation (what was loaded)</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          {[
            {label:'Cases',       val: s.n_cases,      sub:'traces in the log'},
            {label:'Events',      val: s.n_events,      sub:'activity executions'},
            {label:'Activities',  val: s.n_activities,  sub:'unique activity types'},
            {label:'Timestamps',  val: s.has_timestamps ? 'Yes' : 'No', sub: s.has_timestamps ? 'real durations available' : 'sequence-based proxies used'},
          ].map(({label,val,sub}) => (
            <div key={label} className="bg-gray-50 rounded-xl p-3">
              <div className="text-xs text-gray-400">{label}</div>
              <div className="text-2xl font-bold text-gray-800 font-mono leading-tight">{val}</div>
              <div className="text-xs text-gray-400 mt-0.5">{sub}</div>
            </div>
          ))}
        </div>
        <div className="text-xs text-gray-400 mb-1">Trace-level attributes detected in this log</div>
        <div className="flex gap-2 flex-wrap">
          {s.trace_attr_keys.map(k => (
            <span key={k} className="font-mono text-xs bg-gray-100 border border-gray-200 rounded px-2 py-0.5 text-gray-600">{k}</span>
          ))}
          {s.trace_attr_keys.length === 0 && <span className="text-xs text-gray-400">None detected</span>}
        </div>
        {!s.has_timestamps && (
          <div className="mt-3 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-2">
            <i className="ti ti-clock-exclamation mr-1"/>
            <strong>No timestamps detected.</strong> Cycle time, waiting time, and throughput cannot be computed from this log.
            Time-dependent features use sequence-based proxies (documented in the Features page).
            For real operational KPIs, use a timestamped log such as BPI Challenge 2017.
          </div>
        )}
      </div>

      {/* Conformance */}
      {m.conformance.has_labels && (
        <div className="card">
          <div className="section-label">Conformance Analysis (label-based)</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-teal-50 rounded-xl p-4 text-center">
              <div className="text-3xl font-bold text-teal-700 font-mono">{m.conformance.positive}</div>
              <div className="text-xs text-teal-600 mt-1">Conforming cases</div>
              <div className="text-xs text-gray-400">(pdc:isPos = true)</div>
              {m.conformance.pos_mean_cost != null && (
                <div className="text-xs text-gray-500 mt-1">mean cost: {m.conformance.pos_mean_cost}</div>
              )}
            </div>
            <div className="bg-red-50 rounded-xl p-4 text-center">
              <div className="text-3xl font-bold text-red-600 font-mono">{m.conformance.negative}</div>
              <div className="text-xs text-red-500 mt-1">Non-conforming cases</div>
              <div className="text-xs text-gray-400">(pdc:isPos = false)</div>
              {m.conformance.neg_mean_cost != null && (
                <div className="text-xs text-gray-500 mt-1">mean cost: {m.conformance.neg_mean_cost}</div>
              )}
            </div>
            <div className="bg-gray-50 rounded-xl p-4 text-center">
              <div className="text-3xl font-bold text-gray-700 font-mono">
                {(m.conformance.conformance_rate * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">Conformance rate</div>
              {m.conformance.neg_mean_cost != null && m.conformance.pos_mean_cost != null && (
                <div className="text-xs text-gray-400 mt-1">
                  Non-conforming costs {(m.conformance.neg_mean_cost / (m.conformance.pos_mean_cost || 1)).toFixed(1)}× more
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Bottlenecks */}
      <div className="card">
        <div className="section-label">
          {m.bottleneck_basis?.startsWith('[REAL]') ? 'Bottleneck Analysis (Real Wait Time)' : 'Sequence Bottleneck Analysis'}
        </div>
        <p className="text-xs text-gray-400 mb-3">
          {m.bottleneck_basis?.startsWith('[REAL]')
            ? <>Bottleneck score = real mean waiting time before an activity × its frequency share, computed from this file's actual timestamps. {m.bottleneck_basis.replace('[REAL] ', '')}</>
            : <>Bottleneck score = mean relative position × frequency share. Activities that appear late AND frequently
                score highest — this file's timestamps weren't usable for a real wait-time measure
                {m.bottleneck_basis ? ` (${m.bottleneck_basis.replace('[PROXY] ', '')})` : ''}.</>}
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div>
            {m.bottlenecks.slice(0,8).map((b,i) => (
              <div key={b.activity} className="flex items-center gap-3 mb-2">
                <div className="font-mono text-sm font-bold w-6 text-center" style={{color: COL[i % COL.length]}}>{b.activity}</div>
                <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                  <div className="h-full rounded-full transition-all" style={{width:`${b.score*100}%`, background: COL[i % COL.length]}}/>
                </div>
                <div className="text-xs font-mono text-gray-600 w-12 text-right">{b.score.toFixed(3)}</div>
                <div className="text-xs text-gray-400 w-20">{b.occurrences} events</div>
              </div>
            ))}
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-2">Mean relative position (how late in traces)</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={m.bottlenecks.slice(0,8)} layout="vertical">
                <XAxis type="number" domain={[0,1]} tick={{fontSize:10}}/>
                <YAxis type="category" dataKey="activity" tick={{fontSize:11, fontFamily:'monospace'}} width={20}/>
                <Tooltip formatter={v=>[v.toFixed(3),'mean position']}/>
                <Bar dataKey="mean_relative_position" radius={[0,3,3,0]}>
                  {m.bottlenecks.slice(0,8).map((b,i) => <Cell key={i} fill={COL[i%COL.length]}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Variants */}
      <div className="card">
        <div className="section-label">Process Variants</div>
        <div className="grid grid-cols-3 gap-3 mb-4">
          {[
            {label:'Distinct variants', val: m.n_variants},
            {label:'Cases', val: s.n_cases},
            {label:'Variability', val: m.n_variants === s.n_cases ? 'Maximum — every trace unique' : `${(m.n_variants/s.n_cases*100).toFixed(0)}% of cases`},
          ].map(({label,val}) => (
            <div key={label} className="bg-gray-50 rounded-xl p-3">
              <div className="text-xs text-gray-400">{label}</div>
              <div className="text-sm font-semibold text-gray-800 font-mono">{val}</div>
            </div>
          ))}
        </div>
        <div className="text-xs text-gray-400 mb-2">Top variants by frequency</div>
        {m.variants?.slice(0,8).map((v,i) => (
          <div key={i} className="flex items-center gap-3 mb-1.5">
            <div className="text-xs font-mono text-gray-400 w-5">V{v.id}</div>
            <div className="flex-1 bg-gray-100 rounded h-2 overflow-hidden">
              <div className="h-full rounded bg-teal-500" style={{width:`${v.frequency_pct}%`}}/>
            </div>
            <div className="text-xs font-mono text-gray-600 w-10 text-right">{v.frequency_pct}%</div>
            <div className="text-xs text-gray-400 w-16">{v.cases} case{v.cases!==1?'s':''}</div>
            <div className="text-xs font-mono text-gray-300 truncate max-w-48 hidden md:block">{v.path}</div>
          </div>
        ))}
      </div>

      {/* Directly-follows */}
      <div className="card">
        <div className="section-label">Top Directly-Follows Transitions</div>
        <p className="text-xs text-gray-400 mb-3">The most frequent activity pairs — these form the backbone of the process model.</p>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {m.directly_follows?.slice(0,9).map((e,i) => (
            <div key={i} className="flex items-center gap-2 bg-gray-50 rounded-lg px-3 py-2">
              <span className="font-mono text-sm font-bold text-gray-700">{e.from}</span>
              <i className="ti ti-arrow-right text-gray-300 text-xs"/>
              <span className="font-mono text-sm font-bold text-gray-700">{e.to}</span>
              <span className="ml-auto font-mono text-xs text-gray-400">{e.count}×</span>
            </div>
          ))}
        </div>
      </div>

      {/* Findings */}
      <div className="card">
        <div className="section-label">Agent-Ready Findings (embedded in knowledge store)</div>
        <p className="text-xs text-gray-400 mb-3">
          These natural-language findings are vector-embedded and retrieved by the reasoning agent at query time.
        </p>
        {m.findings_text?.map((f,i) => (
          <div key={i} className="flex gap-2 mb-2 text-xs text-gray-700 bg-teal-50 rounded-lg p-2.5">
            <i className="ti ti-point-filled text-teal-400 flex-shrink-0 mt-0.5"/>
            <span>{f}</span>
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <button className="btn-primary" onClick={() => navigate('/features')}>
          <i className="ti ti-vector mr-1"/> View Features & RL Proof →
        </button>
      </div>
    </div>
  )
}
