import React, { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { datasetsApi } from '../services/api'
import { useStore } from '../services/store'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORS = ['#085041','#185FA5','#534AB7','#B45309','#7C3AED','#0369A1','#D85A30','#888','#059669','#DC2626']

const FMT_LABELS = {
  xes: { label: 'XES Event Log', color: '#185FA5', icon: 'ti-binary-tree' },
  csv: { label: 'CSV Event Log',  color: '#085041', icon: 'ti-table' },
  tsv: { label: 'TSV Event Log',  color: '#534AB7', icon: 'ti-table' },
}

function PreviewPanel({ preview, name }) {
  if (!preview) return (
    <div className="flex items-center justify-center h-40 text-gray-300">
      <div className="text-center"><i className="ti ti-loader animate-spin text-2xl block mb-2"/><div className="text-xs">Loading preview…</div></div>
    </div>
  )
  if (preview.status === 'error') return (
    <div className="text-xs text-red-500 p-3 bg-red-50 rounded-lg">
      <i className="ti ti-alert-circle mr-1"/>Parse error: {preview.error}
    </div>
  )

  const fmt = FMT_LABELS[preview.format] || { label: preview.format?.toUpperCase(), color: '#888', icon: 'ti-file' }

  return (
    <div className="space-y-4">
      {/* Header stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {[
          { label:'Cases',       val: preview.n_cases,      sub:'process instances',         icon:'ti-git-branch' },
          { label:'Events',      val: preview.n_events,     sub:'activity executions',        icon:'ti-activity' },
          { label:'Activities',  val: preview.n_activities, sub:'unique activity types',      icon:'ti-circles-relation' },
          { label:'Timestamps',  val: preview.has_timestamps ? '✓ Yes' : '✗ No',
            sub: preview.has_timestamps ? 'real durations available' : 'sequence-based analysis only',
            icon:'ti-clock', warn: !preview.has_timestamps },
        ].map(({label,val,sub,icon,warn}) => (
          <div key={label} className={`rounded-xl p-3 border ${warn ? 'bg-amber-50 border-amber-200' : 'bg-gray-50 border-gray-100'}`}>
            <div className="flex items-center gap-1.5 mb-1">
              <i className={`ti ${icon} text-xs ${warn?'text-amber-500':'text-gray-400'}`}/>
              <div className="text-xs text-gray-400">{label}</div>
            </div>
            <div className={`text-xl font-bold font-mono leading-tight ${warn?'text-amber-700':'text-gray-800'}`}>{val}</div>
            <div className="text-xs text-gray-400 mt-0.5 leading-tight">{sub}</div>
          </div>
        ))}
      </div>

      {/* Format badge + hash */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full"
          style={{background: fmt.color+'18', color: fmt.color}}>
          <i className={`ti ${fmt.icon}`}/>{fmt.label}
        </span>
        <span className="font-mono text-xs text-gray-400 bg-gray-50 border border-gray-100 px-2 py-0.5 rounded">
          hash: {preview.content_hash}
        </span>
        {preview.trace_attr_keys?.map(k => (
          <span key={k} className="text-xs font-mono bg-teal-50 text-teal-700 border border-teal-100 px-2 py-0.5 rounded">{k}</span>
        ))}
      </div>

      {/* Timestamps warning */}
      {!preview.has_timestamps && (
        <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-2.5">
          <i className="ti ti-clock-exclamation mr-1"/>
          <strong>No timestamps in this file.</strong> Cycle time, waiting time and throughput cannot be computed.
          The pipeline will use sequence-based proxies for time-dependent features.
          For real operational KPIs, use a timestamped log such as BPI Challenge 2017.
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Activity frequency chart */}
        <div>
          <div className="section-label">Top Activity Frequencies</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={preview.top_activities} layout="vertical">
              <XAxis type="number" tick={{fontSize:10}}/>
              <YAxis type="category" dataKey="activity" tick={{fontSize:11, fontFamily:'monospace'}} width={24}/>
              <Tooltip formatter={v => [v, 'occurrences']}/>
              <Bar dataKey="count" radius={[0,3,3,0]}>
                {preview.top_activities?.map((_,i) => <Cell key={i} fill={COLORS[i%COLORS.length]}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Right column: label balance + trace lengths + costs */}
        <div className="space-y-3">
          {/* Label balance */}
          {preview.label_balance && (
            <div>
              <div className="section-label">Label Balance ({preview.label_balance.key})</div>
              <div className="flex gap-2">
                {[
                  {label:'Conforming',    val:preview.label_balance.positive, color:'#085041'},
                  {label:'Non-conforming',val:preview.label_balance.negative, color:'#D85A30'},
                ].map(r => (
                  <div key={r.label} className="flex-1 rounded-xl p-3 text-center"
                    style={{background: r.color+'15', border: `1px solid ${r.color}30`}}>
                    <div className="text-2xl font-bold font-mono" style={{color:r.color}}>{r.val}</div>
                    <div className="text-xs text-gray-500 mt-0.5">{r.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Trace length stats */}
          <div>
            <div className="section-label">Trace Length Distribution</div>
            <div className="grid grid-cols-3 gap-2">
              {[
                {label:'Min', val:preview.min_trace_length},
                {label:'Mean',val:preview.mean_trace_length},
                {label:'Max', val:preview.max_trace_length},
              ].map(s => (
                <div key={s.label} className="bg-gray-50 rounded-lg p-2 text-center">
                  <div className="font-mono text-sm font-bold text-gray-800">{s.val}</div>
                  <div className="text-xs text-gray-400">{s.label} events</div>
                </div>
              ))}
            </div>
          </div>

          {/* Cost stats */}
          {preview.cost_stats && (
            <div>
              <div className="section-label">Cost Attribute ({preview.cost_stats.key})</div>
              <div className="grid grid-cols-3 gap-2">
                {[
                  {label:'Min', val:preview.cost_stats.min},
                  {label:'Mean',val:preview.cost_stats.mean},
                  {label:'Max', val:preview.cost_stats.max},
                ].map(s => (
                  <div key={s.label} className="bg-gray-50 rounded-lg p-2 text-center">
                    <div className="font-mono text-sm font-bold text-gray-800">{s.val}</div>
                    <div className="text-xs text-gray-400">{s.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Activity alphabet */}
      <div>
        <div className="section-label">Activity Alphabet ({preview.n_activities} unique)</div>
        <div className="flex gap-1.5 flex-wrap">
          {preview.activity_alphabet?.map((a,i) => (
            <span key={a} className="font-mono text-xs px-2 py-0.5 rounded-full font-bold"
              style={{background: COLORS[i%COLORS.length]+'20', color: COLORS[i%COLORS.length]}}>
              {a}
            </span>
          ))}
          {preview.n_activities > 30 && (
            <span className="text-xs text-gray-400 py-0.5">+{preview.n_activities - 30} more</span>
          )}
        </div>
      </div>

      {/* Sample traces */}
      <div>
        <div className="section-label">Sample Traces (first 3 cases)</div>
        {preview.sample_traces?.map((t,i) => (
          <div key={i} className="mb-2 bg-gray-50 rounded-xl p-3 border border-gray-100">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-xs font-semibold text-gray-600">Case {t.case_id}</span>
              <span className="text-xs text-gray-400">{t.length} events</span>
              {Object.entries(t.attributes).map(([k,v]) => (
                <span key={k} className="text-xs font-mono bg-white border border-gray-200 rounded px-1.5 py-0.5 text-gray-600">
                  {k.split(':').pop()}={v}
                </span>
              ))}
            </div>
            <div className="flex gap-1 flex-wrap">
              {t.sequence.map((a,j) => (
                <React.Fragment key={j}>
                  <span className="font-mono text-xs font-bold px-1.5 py-0.5 rounded"
                    style={{background: COLORS[j%COLORS.length]+'20', color: COLORS[j%COLORS.length]}}>
                    {a}
                  </span>
                  {j < t.sequence.length - 1 && <i className="ti ti-arrow-right text-gray-200 text-xs self-center"/>}
                </React.Fragment>
              ))}
              {t.length > 12 && <span className="text-xs text-gray-400 self-center">+{t.length-12} more</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function DatasetsPage() {
  const [datasets,   setDatasets]   = useState([])
  const [uploading,  setUploading]  = useState(false)
  const [progress,   setProgress]   = useState(0)
  const [errors,     setErrors]     = useState([])
  const [dragging,   setDragging]   = useState(false)
  const [previews,   setPreviews]   = useState({})  // id -> preview data
  const {
    selectedDataset, setSelectedDataset,
    selectedDatasetIds, toggleDatasetSelection, selectAllDatasets, clearDatasetSelection,
  } = useStore()
  const navigate = useNavigate()

  const load = async () => {
    const ds = await datasetsApi.list().catch(() => [])
    setDatasets(ds)
    // Load previews for all datasets
    ds.forEach(d => {
      if (!previews[d.id] && d.metadata?.preview) {
        setPreviews(p => ({...p, [d.id]: d.metadata.preview}))
      } else if (!previews[d.id]) {
        datasetsApi.preview(d.id)
          .then(pv => setPreviews(p => ({...p, [d.id]: pv})))
          .catch(() => {})
      }
    })
  }

  useEffect(() => { load() }, [])

  const handleFiles = async (files) => {
    if (!files || files.length === 0) return
    setUploading(true); setErrors([])
    try {
      const arr = Array.from(files)
      const results = arr.length === 1
        ? [await datasetsApi.upload(arr[0], setProgress)]
        : await datasetsApi.uploadMany(arr, setProgress)
      const errs = results.filter(r => r.error)
      if (errs.length) setErrors(errs.map(e => `${e.name}: ${e.error}`))
      // Cache previews from upload response
      results.filter(r => !r.error && r.preview).forEach(r => {
        setPreviews(p => ({...p, [r.id]: r.preview}))
      })
      await load()
      // Auto-select last successful upload for preview, and add every
      // successful upload to the multi-select batch set.
      const ok = results.filter(r => r.id)
      if (ok.length) {
        const last = ok[ok.length - 1]
        setSelectedDataset(last)
        selectAllDatasets([...new Set([...selectedDatasetIds, ...ok.map(r => r.id)])])
      }
    } catch(err) {
      setErrors([err.response?.data?.detail || err.message])
    } finally { setUploading(false); setProgress(0) }
  }

  // Drag and drop
  const onDragOver  = e => { e.preventDefault(); setDragging(true) }
  const onDragLeave = e => { e.preventDefault(); setDragging(false) }
  const onDrop      = e => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files) }

  const del = async (id, e) => {
    e.stopPropagation()
    if (!confirm('Delete this dataset?')) return
    await datasetsApi.delete(id).catch(() => {})
    if (selectedDataset?.id === id) setSelectedDataset(null)
    if (selectedDatasetIds.includes(id)) toggleDatasetSelection(id)
    setPreviews(p => { const n = {...p}; delete n[id]; return n })
    load()
  }

  const allIds = datasets.map(d => d.id)
  const allSelected = allIds.length > 0 && allIds.every(id => selectedDatasetIds.includes(id))
  const toggleSelectAll = () => allSelected ? clearDatasetSelection() : selectAllDatasets(allIds)

  const selected = selectedDataset
  const selectedPreview = selected ? previews[selected.id] : null

  return (
    <div className="max-w-6xl space-y-5">
      <div>
        <h1 className="text-lg font-medium text-gray-900">Data Ingestion</h1>
        <p className="text-xs text-gray-500 mt-0.5">
          Upload one or more event logs, check the box on any files you want processed together,
          then run the pipeline across all of them in one batch.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* LEFT: upload + dataset list */}
        <div className="space-y-4">
          {/* Drop zone */}
          <div className="card"
            onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
            <div className="section-label">Upload Event Logs</div>
            <label className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-8 cursor-pointer transition-all
              ${dragging   ? 'border-teal-500 bg-teal-50 scale-[1.01]' : ''}
              ${uploading  ? 'border-blue-300 bg-blue-50' : ''}
              ${!dragging && !uploading ? 'border-gray-200 hover:border-teal-300 hover:bg-gray-50' : ''}`}>
              <i className={`ti ${uploading ? 'ti-loader animate-spin text-blue-400' : dragging ? 'ti-cloud-upload text-teal-500' : 'ti-upload text-gray-400'} text-3xl mb-2`}/>
              <div className="text-sm font-medium text-gray-700">
                {uploading ? `Uploading… ${progress}%` : dragging ? 'Drop files here' : 'Click to browse or drag & drop'}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Supports: <span className="font-mono">.xes  .csv  .tsv  .gz</span> · Multiple files allowed
              </div>
              <input type="file"
                accept=".xes,.csv,.tsv,.gz,.xes.gz,text/csv,text/tab-separated-values,application/x-gzip"
                multiple
                className="hidden"
                onChange={e => handleFiles(e.target.files)}
                disabled={uploading}/>
            </label>
            {errors.length > 0 && (
              <div className="mt-2 space-y-1">
                {errors.map((e,i) => (
                  <div key={i} className="text-xs text-red-500 bg-red-50 border border-red-200 rounded px-2 py-1">
                    <i className="ti ti-alert-circle mr-1"/>{e}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Supported formats */}
          <div className="card">
            <div className="section-label">Supported Formats</div>
            <div className="grid grid-cols-2 gap-2">
              {[
                {ext:'.xes',    desc:'IEEE XES standard event logs (PDC, BPI Challenge)',     icon:'ti-binary-tree',    color:'#185FA5'},
                {ext:'.csv',    desc:'CSV logs with case/activity/timestamp columns',          icon:'ti-table',          color:'#085041'},
                {ext:'.tsv',    desc:'Tab-separated event logs',                              icon:'ti-table',          color:'#534AB7'},
                {ext:'.xes.gz', desc:'Gzip-compressed XES files',                            icon:'ti-file-zip',       color:'#B45309'},
              ].map(f => (
                <div key={f.ext} className="flex items-start gap-2 p-2 rounded-lg bg-gray-50">
                  <i className={`ti ${f.icon} text-base mt-0.5 flex-shrink-0`} style={{color:f.color}}/>
                  <div>
                    <div className="font-mono text-xs font-bold" style={{color:f.color}}>{f.ext}</div>
                    <div className="text-xs text-gray-500 leading-tight">{f.desc}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 text-xs text-gray-400">
              <i className="ti ti-info-circle mr-1"/>
              For CSV files the parser auto-detects columns named{' '}
              <span className="font-mono">case_id / activity / timestamp</span>{' '}
              (and common variations). Column names are case-insensitive.
            </div>
          </div>

          {/* Dataset list */}
          <div className="card">
            <div className="flex items-center justify-between mb-1">
              <div className="section-label mb-0">Loaded Datasets ({datasets.length})</div>
              {datasets.length > 0 && (
                <button onClick={toggleSelectAll} className="text-xs text-teal-600 hover:underline">
                  {allSelected ? 'Clear selection' : 'Select all'}
                </button>
              )}
            </div>

            {selectedDatasetIds.length > 0 && (
              <div className="flex items-center justify-between gap-2 mb-3 mt-2 p-2.5 rounded-lg bg-teal-50 border border-teal-200">
                <div className="text-xs text-teal-800 font-medium">
                  <i className="ti ti-checks mr-1"/>{selectedDatasetIds.length} file{selectedDatasetIds.length > 1 ? 's' : ''} selected
                </div>
                <button
                  className="btn-primary text-xs px-3 py-1.5"
                  onClick={() => navigate('/pipeline')}>
                  <i className="ti ti-player-play mr-1"/>
                  Run Pipeline on {selectedDatasetIds.length} file{selectedDatasetIds.length > 1 ? 's' : ''} →
                </button>
              </div>
            )}

            {datasets.length === 0 ? (
              <div className="text-xs text-gray-400 text-center py-8">
                <i className="ti ti-database text-2xl block mb-2 text-gray-300"/>
                No datasets yet. Upload files above.
              </div>
            ) : (
              <div className="space-y-2">
                {datasets.map(ds => {
                  const pv = previews[ds.id]
                  const isSelected = selectedDataset?.id === ds.id
                  const isChecked = selectedDatasetIds.includes(ds.id)
                  const fmt = pv ? FMT_LABELS[pv.format] : null
                  return (
                    <div key={ds.id}
                      onClick={() => setSelectedDataset(ds)}
                      className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all hover:shadow-sm
                        ${isSelected ? 'border-teal-400 bg-teal-50 shadow-sm' : 'border-gray-100 hover:border-gray-200 bg-white'}`}>
                      <input type="checkbox" checked={isChecked}
                        onClick={e => e.stopPropagation()}
                        onChange={() => toggleDatasetSelection(ds.id)}
                        className="w-4 h-4 accent-teal-600 flex-shrink-0"
                        title="Include in batch pipeline run"/>
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                        style={{background: (fmt?.color || '#888')+'18'}}>
                        <i className={`ti ${fmt?.icon || 'ti-file'} text-sm`}
                          style={{color: fmt?.color || '#888'}}/>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-semibold text-gray-800 truncate">{ds.name}</div>
                        <div className="text-xs text-gray-400 font-mono mt-0.5">
                          {ds.id.slice(0,12)}…
                          {pv?.status === 'ok' && (
                            <span className="ml-2 not-mono">
                              {pv.n_cases} cases · {pv.n_events} events · {pv.n_activities} acts
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        {isSelected && (
                          <span className="text-xs bg-teal-100 text-teal-700 px-2 py-0.5 rounded-full font-medium">previewing</span>
                        )}
                        {pv?.has_timestamps === false && (
                          <i className="ti ti-clock-exclamation text-amber-400 text-sm" title="No timestamps"/>
                        )}
                        <button onClick={e => del(ds.id, e)}
                          className="text-gray-300 hover:text-red-400 transition-colors">
                          <i className="ti ti-trash text-sm"/>
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>

        {/* RIGHT: data preview panel */}
        <div className="card">
          {!selected ? (
            <div className="flex flex-col items-center justify-center h-full min-h-80 text-gray-300">
              <i className="ti ti-eye text-4xl mb-3"/>
              <div className="text-sm font-medium">Select a dataset to preview</div>
              <div className="text-xs mt-1">Click any dataset on the left to explore its contents</div>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <div className="text-sm font-semibold text-gray-900 truncate max-w-xs">{selected.name}</div>
                  <div className="text-xs text-gray-400 font-mono">{selected.id}</div>
                </div>
                <button className="btn-primary text-xs" onClick={() => {
                  if (!selectedDatasetIds.includes(selected.id)) toggleDatasetSelection(selected.id)
                  navigate('/pipeline')
                }}>
                  <i className="ti ti-player-play mr-1"/>Run Pipeline →
                </button>
              </div>
              <PreviewPanel preview={selectedPreview} name={selected.name}/>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
