import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { pipelineApi } from '../services/api'

const COL = ['#085041', '#185FA5', '#534AB7', '#B45309', '#7C3AED', '#0369A1', '#D85A30', '#888']

function shortName(r) {
  return r.source_filename || r.dataset_name || r.run_id?.slice(0, 12) || '—'
}

const COMPLETENESS_LABEL = {
  cache_full: 'full detail',
  db_summary: 'summary only',
  db_legacy:  'legacy (limited)',
}
const COMPLETENESS_COLOR = {
  cache_full: 'text-teal-600 bg-teal-100',
  db_summary: 'text-amber-600 bg-amber-100',
  db_legacy:  'text-gray-500 bg-gray-100',
}

export default function ComparePage() {
  const navigate = useNavigate()
  const [allRuns, setAllRuns] = useState([])
  const [loadingList, setLoadingList] = useState(true)
  const [listError, setListError] = useState(null)
  const [selected, setSelected] = useState([])
  const [comparing, setComparing] = useState(false)
  const [result, setResult] = useState(null)
  const [compareError, setCompareError] = useState(null)
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState(null)

  useEffect(() => {
    pipelineApi.listAllRuns()
      .then(setAllRuns)
      .catch(err => setListError(err.response?.data?.detail || err.message))
      .finally(() => setLoadingList(false))
  }, [])

  const toggle = id => setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id])
  const selectAll = () => setSelected(allRuns.map(r => r.run_id))
  const clearAll = () => { setSelected([]); setResult(null) }

  const runCompare = async () => {
    if (selected.length < 2) return
    setComparing(true); setCompareError(null)
    try {
      const res = await pipelineApi.compare(selected)
      setResult(res)
    } catch (err) {
      setCompareError(err.response?.data?.detail || err.message)
    } finally {
      setComparing(false)
    }
  }

  const exportPdf = async () => {
    if (selected.length < 2) return
    setExporting(true); setExportError(null)
    try {
      await pipelineApi.downloadCompareReport(selected)
    } catch (err) {
      setExportError(err.response?.data?.detail || err.message)
    } finally {
      setExporting(false)
    }
  }

  if (!loadingList && allRuns.length === 0 && !listError) return (
    <div className="max-w-2xl">
      <div className="card text-center py-10 text-gray-400">
        <i className="ti ti-chart-bar text-3xl block mb-2"/>
        <div className="text-sm mb-3">No completed runs yet.</div>
        <button className="text-teal-600 text-sm hover:underline" onClick={() => navigate('/')}>
          Go to Data Ingestion →
        </button>
      </div>
    </div>
  )

  const rows = result?.rows || []
  const missing = result?.missing || []

  const barData = (key, pct = false) => rows
    .filter(r => r[key] !== null && r[key] !== undefined)
    .map(r => ({ name: shortName(r).slice(0, 16), value: pct ? +(r[key] * 100).toFixed(1) : +Number(r[key]).toFixed(4) }))

  const greedyBarData = rows
    .filter(r => r.comparisons?.greedy)
    .map(r => ({
      name: shortName(r).slice(0, 16),
      adv: r.comparisons.greedy.mean_advantage,
      better: r.comparisons.greedy.learned_better,
    }))

  return (
    <div className="max-w-6xl space-y-5">
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-medium text-gray-900">Compare Datasets</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Pick two or more completed runs — across any datasets — to compare mining findings and RL results side by side.
          </p>
        </div>
        {rows.length >= 2 && (
          <button className="btn-primary text-xs h-9 px-4" onClick={exportPdf} disabled={exporting}>
            {exporting
              ? <><i className="ti ti-loader animate-spin mr-1"/>Generating…</>
              : <><i className="ti ti-file-download mr-1"/>Export Comparison PDF</>}
          </button>
        )}
      </div>
      {exportError && (
        <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
          <i className="ti ti-alert-circle mr-1"/>{exportError}
        </div>
      )}

      {/* Run picker */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <div className="section-label mb-0">Select Runs ({selected.length} selected)</div>
          <div className="flex gap-3 text-xs">
            <button className="text-teal-600 hover:underline" onClick={selectAll}>Select all</button>
            <button className="text-gray-500 hover:underline" onClick={clearAll}>Clear</button>
          </div>
        </div>
        {loadingList ? (
          <div className="text-xs text-gray-400 py-6 text-center">
            <i className="ti ti-loader animate-spin mr-1"/>Loading runs…
          </div>
        ) : listError ? (
          <div className="text-xs text-red-600">{listError}</div>
        ) : (
          <div className="space-y-1.5 max-h-80 overflow-y-auto">
            {allRuns.map(r => {
              const checked = selected.includes(r.run_id)
              return (
                <label key={r.run_id}
                  className={`flex items-center gap-3 text-xs rounded-lg px-3 py-2 border cursor-pointer transition-colors
                    ${checked ? 'border-teal-300 bg-teal-50' : 'border-gray-100 hover:bg-gray-50'}`}>
                  <input type="checkbox" checked={checked} onChange={() => toggle(r.run_id)} className="accent-teal-700"/>
                  <i className="ti ti-file-text text-gray-400"/>
                  <span className="font-semibold text-gray-800 truncate flex-1">
                    {r.source_filename || r.run_id.slice(0, 16)}
                  </span>
                  <span className="text-gray-400 font-mono truncate max-w-32">{r.dataset_name}</span>
                  <span className="text-gray-300 font-mono">{r.content_hash?.slice(0, 10)}</span>
                  <span className="text-gray-400">{r.created_at ? new Date(r.created_at).toLocaleDateString() : '—'}</span>
                  {r.cached
                    ? <span className="text-teal-600 bg-teal-100 px-1.5 py-0.5 rounded">full detail</span>
                    : <span className="text-amber-600 bg-amber-100 px-1.5 py-0.5 rounded">summary only</span>}
                </label>
              )
            })}
          </div>
        )}
        <div className="mt-4 flex items-center gap-3">
          <button className="btn-primary text-xs h-9 px-5" onClick={runCompare} disabled={selected.length < 2 || comparing}>
            {comparing
              ? <><i className="ti ti-loader animate-spin mr-1"/>Comparing…</>
              : <><i className="ti ti-chart-bar mr-1"/>Compare Selected ({selected.length})</>}
          </button>
          {selected.length < 2 && <span className="text-xs text-gray-400">Select at least 2 runs</span>}
          {compareError && <span className="text-xs text-red-600">{compareError}</span>}
        </div>
      </div>

      {missing.length > 0 && (
        <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-3">
          <i className="ti ti-alert-triangle mr-1"/>
          {missing.length} selected run{missing.length > 1 ? 's' : ''} could not be found (neither cached
          nor in the database) and {missing.length > 1 ? 'were' : 'was'} skipped from the comparison.
        </div>
      )}

      {rows.length > 0 && (
        <>
          <div className="card overflow-x-auto">
            <div className="section-label">Summary</div>
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="text-left px-3 py-2 font-semibold text-gray-600">Dataset / File</th>
                  <th className="text-right px-3 py-2 font-semibold text-gray-600">Cases</th>
                  <th className="text-right px-3 py-2 font-semibold text-gray-600">Variants</th>
                  <th className="text-right px-3 py-2 font-semibold text-gray-600">Conformance</th>
                  <th className="text-right px-3 py-2 font-semibold text-gray-600">Bottleneck</th>
                  <th className="text-right px-3 py-2 font-semibold text-gray-600">Rework Prob.</th>
                  <th className="text-right px-3 py-2 font-semibold text-gray-600">Learned Mean</th>
                  <th className="text-right px-3 py-2 font-semibold text-gray-600">vs Greedy</th>
                  <th className="text-left px-3 py-2 font-semibold text-gray-600">Data</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(r => {
                  const greedy = r.comparisons?.greedy
                  return (
                    <tr key={r.run_id} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="px-3 py-2">
                        <div className="font-semibold text-gray-800">{shortName(r)}</div>
                        <div className="text-gray-400 font-mono">{r.dataset_name}</div>
                      </td>
                      <td className="px-3 py-2 text-right font-mono">{r.n_cases ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono">{r.n_variants ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono">
                        {r.conformance_rate != null ? `${(r.conformance_rate * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="px-3 py-2 text-right font-mono">{r.bottleneck_score ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono">{r.rework_probability ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono font-bold text-teal-700">
                        {r.learned_mean != null ? r.learned_mean.toFixed(4) : '—'}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${
                        greedy ? (greedy.learned_better ? 'text-teal-600' : 'text-red-500') : 'text-gray-400'}`}>
                        {greedy ? `${greedy.relative_improvement_pct > 0 ? '+' : ''}${greedy.relative_improvement_pct}%` : '—'}
                      </td>
                      <td className="px-3 py-2">
                        <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${COMPLETENESS_COLOR[r.data_completeness] || 'text-gray-400 bg-gray-100'}`}>
                          {COMPLETENESS_LABEL[r.data_completeness] || r.data_completeness}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="card">
              <div className="section-label">Process Variants</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={barData('n_variants')}>
                  <XAxis dataKey="name" tick={{ fontSize: 9 }}/>
                  <YAxis tick={{ fontSize: 9 }}/>
                  <Tooltip/>
                  <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                    {barData('n_variants').map((_, i) => <Cell key={i} fill={COL[i % COL.length]}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card">
              <div className="section-label">Conformance Rate</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={barData('conformance_rate', true)}>
                  <XAxis dataKey="name" tick={{ fontSize: 9 }}/>
                  <YAxis tick={{ fontSize: 9 }} tickFormatter={v => `${v}%`}/>
                  <Tooltip formatter={v => [`${v}%`, 'conformance']}/>
                  <Bar dataKey="value" fill="#185FA5" radius={[3, 3, 0, 0]}/>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card">
              <div className="section-label">Bottleneck Score</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={barData('bottleneck_score')}>
                  <XAxis dataKey="name" tick={{ fontSize: 9 }}/>
                  <YAxis tick={{ fontSize: 9 }}/>
                  <Tooltip/>
                  <Bar dataKey="value" fill="#B45309" radius={[3, 3, 0, 0]}/>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card">
              <div className="section-label">Learned Policy — Advantage vs Greedy Baseline</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={greedyBarData}>
                  <XAxis dataKey="name" tick={{ fontSize: 9 }}/>
                  <YAxis tick={{ fontSize: 9 }}/>
                  <Tooltip formatter={v => [v.toFixed(4), 'advantage']}/>
                  <Bar dataKey="adv" radius={[3, 3, 0, 0]}>
                    {greedyBarData.map((b, i) => <Cell key={i} fill={b.better ? '#085041' : '#D85A30'}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="text-xs text-gray-400">
            <i className="ti ti-info-circle mr-1"/>
            Rows tagged <span className="font-mono">summary only</span> or <span className="font-mono">legacy (limited)</span> were
            reconstructed from the persisted database record because the full in-memory run cache had been recycled
            (e.g. a server restart) — open that run's results pages once to refresh full detail before exporting.
          </div>
        </>
      )}
    </div>
  )
}
