import React from 'react'
import { useStore } from '../services/store'

/* Dropdown letting the user flip "currentRun" between every dataset that has
   a completed run in this session — needed now that a batch can process
   several files at once instead of one at a time. Renders nothing if there's
   only zero or one processed file, so it stays invisible for the common
   single-file case. */
export default function RunSwitcher() {
  const { currentRun, runs, runsByDataset, focusDataset } = useStore()
  const entries = Object.entries(runsByDataset) // [datasetId, runId][]

  if (entries.length <= 1) return null

  return (
    <div className="flex items-center gap-1.5">
      <i className="ti ti-switch-horizontal text-gray-400 text-sm"/>
      <select
        className="text-xs border border-gray-200 rounded-lg px-2 py-1 font-mono bg-white hover:border-teal-300 focus:outline-none focus:border-teal-400"
        value={currentRun?.run_id || ''}
        onChange={e => {
          const runId = e.target.value
          const [datasetId] = entries.find(([, rid]) => rid === runId) || []
          if (datasetId) focusDataset(datasetId)
        }}>
        {entries.map(([datasetId, runId]) => {
          const run = runs[runId]
          const name = run?.log_summary?.source_filename || datasetId.slice(0, 12)
          return <option key={runId} value={runId}>{name}</option>
        })}
      </select>
      <span className="text-xs text-gray-400">({entries.length} files processed)</span>
    </div>
  )
}
