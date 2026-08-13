import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { pipelineApi, datasetsApi } from '../services/api'
import { useStore } from '../services/store'

const STAGES = [
  { key:'load',    label:'Load + Lineage',     icon:'ti-file-import',  color:'#085041',
    detail:'Parses XES/CSV bytes, computes SHA-256 content hash, detects format, extracts trace attributes.' },
  { key:'mine',    label:'Process Mining',      icon:'ti-chart-dots',   color:'#185FA5',
    detail:'Computes real process variants, directly-follows graph, sequence-based bottleneck scores, and conformance rates.' },
  { key:'extract', label:'Feature Extraction',  icon:'ti-vector',       color:'#534AB7',
    detail:'Derives the 6-dim MDP state vector with documented formula and provenance string.' },
  { key:'embed',   label:'Embed Findings',      icon:'ti-database',     color:'#B45309',
    detail:'Natural-language mining findings embedded into the Chroma vector store.' },
  { key:'train',   label:'RL Training + Proof', icon:'ti-brain',        color:'#7C3AED',
    detail:'Trains a tabular Q-learning agent and evaluates against baselines on held-out seeds.' },
  { key:'reason',  label:'Agent Reasoning',     icon:'ti-robot',        color:'#0369A1',
    detail:'5-node LangGraph graph retrieves grounded evidence and calls Claude Sonnet to reason over it.' },
]

const POLL_MS = 1800

export default function PipelinePage() {
  const {
    selectedDataset, selectedDatasetIds, clearDatasetSelection,
    currentBatch, setCurrentBatch, addRunForDataset,
  } = useStore()
  const [datasetsById, setDatasetsById] = useState({})
  const [episodes, setEpisodes] = useState(300)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState(null)
  const pollRef = useRef(null)
  const navigate = useNavigate()

  // Resolve names for whatever is selected, for display purposes
  useEffect(() => {
    datasetsApi.list().then(list => {
      const byId = {}
      list.forEach(d => { byId[d.id] = d })
      setDatasetsById(byId)
    }).catch(() => {})
  }, [])

  const targetIds = selectedDatasetIds.length > 0
    ? selectedDatasetIds
    : (selectedDataset ? [selectedDataset.id] : [])

  const startBatch = async () => {
    if (targetIds.length === 0) return
    setStarting(true); setError(null)
    try {
      const batch = await pipelineApi.runBatch(targetIds, episodes)
      setCurrentBatch(batch)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setStarting(false)
    }
  }

  // Poll the running batch until every item is done
  useEffect(() => {
    if (!currentBatch || currentBatch.status !== 'running') return
    pollRef.current = setInterval(async () => {
      try {
        const fresh = await pipelineApi.batchStatus(currentBatch.batch_id)
        setCurrentBatch(fresh)
        // As each item completes, register its run in the store immediately
        // so other pages can already show it before the whole batch finishes.
        Object.values(fresh.items).forEach(item => {
          if (item.status === 'complete' && item.result) {
            addRunForDataset(item.dataset_id, item.result)
          }
        })
        if (fresh.status === 'complete') clearInterval(pollRef.current)
      } catch (err) {
        clearInterval(pollRef.current)
        setError(err.response?.data?.detail || err.message)
      }
    }, POLL_MS)
    return () => clearInterval(pollRef.current)
  }, [currentBatch?.batch_id, currentBatch?.status])

  const items = currentBatch ? Object.values(currentBatch.items) : []
  const completedCount = items.filter(i => i.status === 'complete').length
  const errorCount = items.filter(i => i.status === 'error').length
  const running = currentBatch?.status === 'running'
  const allDone = currentBatch?.status === 'complete'

  if (targetIds.length === 0 && !currentBatch) return (
    <div className="max-w-2xl">
      <div className="card text-center py-10 text-gray-400">
        <i className="ti ti-database text-3xl block mb-2"/>
        <div className="text-sm mb-3">No datasets selected.</div>
        <button className="text-teal-600 text-sm hover:underline" onClick={() => navigate('/')}>
          Go to Data Ingestion →
        </button>
      </div>
    </div>
  )

  return (
    <div className="max-w-5xl space-y-5">
      <div>
        <h1 className="text-lg font-medium text-gray-900">Run Pipeline</h1>
        <p className="text-xs text-gray-500 mt-0.5">
          {currentBatch
            ? <>Batch <span className="font-mono text-teal-700">{currentBatch.batch_id.slice(0,8)}…</span> · {items.length} file{items.length > 1 ? 's' : ''}</>
            : <>{targetIds.length} file{targetIds.length > 1 ? 's' : ''} queued to process</>}
        </p>
      </div>

      {/* Configuration — only shown before the batch starts */}
      {!currentBatch && (
        <div className="card">
          <div className="section-label">Files in this Batch</div>
          <div className="space-y-1.5 mb-4">
            {targetIds.map(id => (
              <div key={id} className="flex items-center gap-2 text-xs text-gray-700 bg-gray-50 rounded-lg px-3 py-2">
                <i className="ti ti-file-text text-gray-400"/>
                <span className="font-mono">{datasetsById[id]?.name || id.slice(0,16)}</span>
              </div>
            ))}
          </div>

          <div className="section-label">Configuration</div>
          <div className="flex items-end gap-6 flex-wrap">
            <div>
              <label className="text-xs text-gray-500 block mb-1">RL Episodes (applies to every file in this batch)</label>
              <input type="number" value={episodes} min={10} max={2000} step={50}
                onChange={e => setEpisodes(+e.target.value)}
                className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm w-28 font-mono"
                disabled={starting}/>
              <div className="text-xs text-gray-400 mt-0.5">50–300 for demo · 500+ for full training</div>
            </div>
            <div className="flex-1"/>
            <button className="btn-primary h-9 px-5" onClick={startBatch} disabled={starting || targetIds.length === 0}>
              {starting
                ? <><i className="ti ti-loader animate-spin mr-1"/>Starting…</>
                : <><i className="ti ti-player-play mr-1"/>Run Pipeline on {targetIds.length} file{targetIds.length > 1 ? 's' : ''}</>}
            </button>
          </div>
          {error && (
            <div className="mt-3 text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
              <i className="ti ti-alert-circle mr-1"/>{error}
            </div>
          )}
          <div className="mt-3 text-xs text-gray-400">
            <i className="ti ti-info-circle mr-1"/>
            Files are processed one at a time in the background (not in parallel) so RL training and
            the live Claude call for each file get full resources and don't collide. You can watch
            each file's progress below once the batch starts — you don't need to wait for one file
            to finish before seeing the next start.
          </div>
        </div>
      )}

      {/* Batch progress */}
      {currentBatch && (
        <>
          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <div className="section-label mb-0">Batch Progress</div>
              <div className="text-xs text-gray-500 font-mono">
                {completedCount}/{items.length} complete
                {errorCount > 0 && <span className="text-red-500 ml-2">{errorCount} failed</span>}
              </div>
            </div>
            <div className="w-full bg-gray-100 rounded-full h-2 mb-4 overflow-hidden">
              <div className="bg-teal-500 h-2 rounded-full transition-all duration-500"
                style={{width: `${items.length ? (completedCount / items.length) * 100 : 0}%`}}/>
            </div>

            <div className="space-y-2">
              {items.map(item => {
                const ds = datasetsById[item.dataset_id]
                return (
                  <div key={item.dataset_id}
                    className={`rounded-xl border p-3 transition-all
                      ${item.status === 'complete' ? 'border-teal-300 bg-teal-50' : ''}
                      ${item.status === 'running'  ? 'border-blue-300 bg-blue-50' : ''}
                      ${item.status === 'error'    ? 'border-red-300 bg-red-50' : ''}
                      ${item.status === 'queued'   ? 'border-gray-100 bg-white' : ''}`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 min-w-0">
                        {item.status === 'complete' && <i className="ti ti-circle-check text-teal-600"/>}
                        {item.status === 'running'  && <i className="ti ti-loader animate-spin text-blue-500"/>}
                        {item.status === 'error'    && <i className="ti ti-alert-circle text-red-500"/>}
                        {item.status === 'queued'   && <i className="ti ti-clock text-gray-300"/>}
                        <span className="text-xs font-semibold text-gray-800 truncate">
                          {ds?.name || item.dataset_id.slice(0,16)}
                        </span>
                      </div>
                      <span className={`text-xs font-medium capitalize
                        ${item.status === 'complete' ? 'text-teal-700' : ''}
                        ${item.status === 'running'  ? 'text-blue-600' : ''}
                        ${item.status === 'error'    ? 'text-red-600' : ''}
                        ${item.status === 'queued'   ? 'text-gray-400' : ''}`}>
                        {item.status}
                      </span>
                    </div>
                    {item.status === 'error' && (
                      <div className="mt-2 text-xs text-red-600 font-mono">{item.error}</div>
                    )}
                    {item.status === 'complete' && item.result && (
                      <div className="mt-2 flex items-center gap-3 text-xs text-gray-500">
                        <span>{item.result.mining.n_variants} variants</span>
                        <span>{item.result.rl.episodes} episodes</span>
                        <button
                          className="text-teal-600 hover:underline ml-auto"
                          onClick={() => { addRunForDataset(item.dataset_id, item.result); navigate('/mining') }}>
                          View results →
                        </button>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {running && (
            <div className="card bg-blue-50 border-blue-200">
              <div className="flex items-center gap-3">
                <i className="ti ti-loader animate-spin text-blue-500 text-xl"/>
                <div className="text-sm text-blue-800">
                  Processing in the background — this page keeps polling even if you navigate away and come back.
                </div>
              </div>
            </div>
          )}

          {allDone && (
            <div className="card bg-teal-50 border-teal-300">
              <div className="flex items-center gap-3 flex-wrap">
                <i className="ti ti-circle-check text-teal-600 text-2xl"/>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-semibold text-teal-800">
                    Batch complete — {completedCount} of {items.length} file{items.length > 1 ? 's' : ''} processed successfully.
                  </div>
                  <div className="text-xs text-teal-600">Use the switcher on any results page to move between files.</div>
                </div>
                <button className="btn-primary text-xs" onClick={() => { setCurrentBatch(null); clearDatasetSelection(); navigate('/mining') }}>
                  Explore results →
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Stage reference (informational, applies to each file in the batch) */}
      <div className="card">
        <div className="section-label">What each file goes through</div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {STAGES.map(s => (
            <div key={s.key} className="rounded-xl border border-gray-100 p-3">
              <div className="flex items-center gap-2 mb-1.5">
                <i className={`ti ${s.icon}`} style={{color: s.color}}/>
                <div className="font-semibold text-xs text-gray-800">{s.label}</div>
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">{s.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
