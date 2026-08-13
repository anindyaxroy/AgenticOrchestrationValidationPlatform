import React, { useState } from 'react'
import { pipelineApi } from '../services/api'

/* Single export point for a run's full report — combines every screen
   (dataset characterisation, mining, features, RL proof, agent reasoning,
   audit trail) into one document, offered as PDF or HTML. Lives only on
   the Audit & Lineage page, since that's the natural "this run in full"
   vantage point — every other page already links back here. */
export default function FullReportExport({ run }) {
  const [busy, setBusy] = useState(null) // 'pdf' | 'html' | null
  const [error, setError] = useState(null)

  if (!run) return null

  const download = async (fmt) => {
    setBusy(fmt); setError(null)
    try {
      await pipelineApi.downloadRunReport(run.run_id, fmt, `bpmn_report_${run.content_hash}.${fmt}`)
    } catch (err) {
      setError(
        err.response?.status === 404
          ? "This run's full detail isn't in the server's live cache (e.g. after a restart). Re-run the pipeline for this file to regenerate it."
          : (err.response?.data?.detail || err.message)
      )
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="card border-teal-200">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div className="flex-1 min-w-64">
          <div className="section-label mb-1">Export Full Report</div>
          <p className="text-xs text-gray-500">
            One document combining every screen for this run — dataset characterisation, process mining
            findings, the MDP feature vector, RL training &amp; proof, agent reasoning, and this audit log —
            all anchored to run <span className="font-mono text-gray-600">{run.run_id}</span>.
          </p>
        </div>
        <div className="flex gap-2 flex-shrink-0">
          <button className="btn-outline text-xs h-9 px-4" onClick={() => download('pdf')} disabled={!!busy}>
            {busy === 'pdf'
              ? <><i className="ti ti-loader animate-spin mr-1"/>Generating…</>
              : <><i className="ti ti-file-type-pdf mr-1"/>Download PDF</>}
          </button>
          <button className="btn-outline text-xs h-9 px-4" onClick={() => download('html')} disabled={!!busy}>
            {busy === 'html'
              ? <><i className="ti ti-loader animate-spin mr-1"/>Generating…</>
              : <><i className="ti ti-file-type-html mr-1"/>Download HTML</>}
          </button>
        </div>
      </div>
      {error && (
        <div className="mt-3 text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
          <i className="ti ti-alert-circle mr-1"/>{error}
        </div>
      )}
    </div>
  )
}
