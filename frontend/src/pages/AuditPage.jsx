import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../services/store'
import RunSwitcher from '../components/RunSwitcher'
import FullReportExport from '../components/FullReportExport'

const EVENT_META = {
  run_started:         { icon:'ti-player-play',    color:'#085041', label:'Run started' },
  log_loaded:          { icon:'ti-file-import',    color:'#185FA5', label:'Log loaded & hashed' },
  mining_done:         { icon:'ti-chart-dots',     color:'#534AB7', label:'Process mining complete' },
  features_extracted:  { icon:'ti-vector',         color:'#B45309', label:'Features extracted' },
  findings_embedded:   { icon:'ti-database',       color:'#7C3AED', label:'Findings embedded' },
  rl_trained:          { icon:'ti-brain',          color:'#0369A1', label:'RL training + proof complete' },
  agent_reasoned:      { icon:'ti-robot',          color:'#D85A30', label:'Agent reasoning complete' },
  run_completed:       { icon:'ti-circle-check',   color:'#085041', label:'Run completed' },
}

export default function AuditPage() {
  const { currentRun } = useStore()
  const navigate = useNavigate()

  if (!currentRun) return (
    <div className="max-w-2xl">
      <div className="card text-center py-10 text-gray-400">
        <i className="ti ti-shield-check text-3xl block mb-2"/>
        <div className="text-sm mb-3">No pipeline run yet.</div>
        <button className="text-teal-600 text-sm hover:underline" onClick={() => navigate('/pipeline')}>Run the pipeline first →</button>
      </div>
    </div>
  )

  const log = currentRun.audit_log || []

  return (
    <div className="max-w-4xl space-y-5">
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-medium text-gray-900">Audit & Lineage</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Every event carries the source file's content hash — any output is traceable to the exact bytes uploaded.
          </p>
        </div>
        <RunSwitcher/>
      </div>

      <FullReportExport run={currentRun}/>

      {/* Lineage summary */}
      <div className="card bg-teal-50 border-teal-200">
        <div className="section-label">Lineage Anchors</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            {label:'Run ID',       val: currentRun.run_id,       icon:'ti-fingerprint'},
            {label:'Content Hash', val: currentRun.content_hash, icon:'ti-lock',  sub:'SHA-256 of source file bytes'},
            {label:'Source File',  val: currentRun.log_summary?.source_filename, icon:'ti-file'},
          ].map(({label,val,icon,sub}) => (
            <div key={label} className="bg-white rounded-xl p-3 border border-teal-100">
              <div className="flex items-center gap-1.5 mb-1">
                <i className={`ti ${icon} text-teal-500 text-sm`}/>
                <div className="text-xs text-gray-400">{label}</div>
              </div>
              <div className="font-mono text-xs text-gray-800 break-all">{val}</div>
              {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
            </div>
          ))}
        </div>
        <div className="mt-3 text-xs text-teal-700">
          <i className="ti ti-info-circle mr-1"/>
          Every line in the audit log below carries this content hash. If you upload a different file,
          the hash changes and all derived outputs trace to the new file — proving data lineage.
        </div>
      </div>

      {/* Append-only audit log */}
      <div className="card">
        <div className="section-label">Append-Only Audit Log ({log.length} events)</div>
        <p className="text-xs text-gray-400 mb-4">
          Written sequentially as the pipeline executes. Cannot be retroactively modified.
          Each event records the stage, its key outputs, and the lineage hash.
        </p>
        <div className="space-y-2">
          {log.map(e => {
            const meta = EVENT_META[e.event] || {icon:'ti-point', color:'#888', label:e.event}
            return (
              <div key={e.seq} className="flex gap-3 p-3 rounded-xl border border-gray-100 hover:bg-gray-50 transition-colors">
                <div className="flex flex-col items-center">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{background: meta.color + '15'}}>
                    <i className={`ti ${meta.icon} text-sm`} style={{color: meta.color}}/>
                  </div>
                  {e.seq < log.length && (
                    <div className="w-0.5 bg-gray-100 flex-1 my-1"/>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-semibold text-gray-800">{meta.label}</span>
                    <span className="font-mono text-xs text-gray-400">#{e.seq}</span>
                    <span className="font-mono text-xs text-teal-600 bg-teal-50 px-1.5 py-0.5 rounded">{e.content_hash}</span>
                    <span className="font-mono text-xs text-gray-300">{e.ts}</span>
                  </div>
                  <div className="mt-1 text-xs text-gray-500 font-mono grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-0.5">
                    {Object.entries(e.detail || {}).map(([k,v]) => (
                      <div key={k} className="truncate">
                        <span className="text-gray-400">{k}:</span>{' '}
                        <span className="text-gray-600">{Array.isArray(v) ? v.join(', ') : String(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* How lineage works */}
      <div className="card">
        <div className="section-label">How Lineage Works — Supervisor Evidence</div>
        <div className="space-y-3 text-xs text-gray-600">
          <div className="flex gap-3">
            <i className="ti ti-circle-1 text-teal-500 text-base flex-shrink-0"/>
            <div><strong>File upload</strong> → SHA-256 hash computed from raw bytes → stored as content_hash</div>
          </div>
          <div className="flex gap-3">
            <i className="ti ti-circle-2 text-teal-500 text-base flex-shrink-0"/>
            <div><strong>Every pipeline stage</strong> asserts its input carries the same hash before running (lineage check)</div>
          </div>
          <div className="flex gap-3">
            <i className="ti ti-circle-3 text-teal-500 text-base flex-shrink-0"/>
            <div><strong>Every audit event</strong> appends the hash → the log is a chain of custody for this exact file</div>
          </div>
          <div className="flex gap-3">
            <i className="ti ti-circle-4 text-teal-500 text-base flex-shrink-0"/>
            <div><strong>Swap the file</strong> → different hash → different run_id → all numbers change → new audit log proves it</div>
          </div>
          <div className="flex gap-3">
            <i className="ti ti-circle-5 text-teal-500 text-base flex-shrink-0"/>
            <div><strong>Feature vector</strong>, <strong>mining findings</strong>, <strong>RL reward curve</strong>, and <strong>agent summary</strong>
            all carry the run_id so they can be joined back to this audit log</div>
          </div>
        </div>
      </div>
    </div>
  )
}
