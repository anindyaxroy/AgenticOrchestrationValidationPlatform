import { create } from 'zustand'

export const useStore = create((set, get) => ({
  // Legacy single-select — kept because several result pages (Mining, Features,
  // Flow, Agent, Audit) still key off "the dataset currently in focus".
  selectedDataset: null,
  setSelectedDataset: ds => set({ selectedDataset: ds }),

  // Multi-select for batch upload/processing on the Data Ingestion page.
  selectedDatasetIds: [],
  toggleDatasetSelection: id => set(s => ({
    selectedDatasetIds: s.selectedDatasetIds.includes(id)
      ? s.selectedDatasetIds.filter(x => x !== id)
      : [...s.selectedDatasetIds, id],
  })),
  setSelectedDatasetIds: ids => set({ selectedDatasetIds: ids }),
  selectAllDatasets: ids => set({ selectedDatasetIds: ids }),
  clearDatasetSelection: () => set({ selectedDatasetIds: [] }),

  // One run "in focus" — persists across page navigation and drives the
  // Mining / Features / Flow / Agent / Audit pages.
  currentRun: null,
  setCurrentRun: run => set({ currentRun: run }),

  // All completed runs keyed by run_id.
  runs: {},
  addRun: run => set(s => ({ runs: { ...s.runs, [run.run_id]: run }, currentRun: run })),

  // Latest run_id per dataset_id — lets a "switch file" control jump straight
  // to that file's own results without re-running anything.
  runsByDataset: {},
  addRunForDataset: (datasetId, run) => set(s => ({
    runs: { ...s.runs, [run.run_id]: run },
    runsByDataset: { ...s.runsByDataset, [datasetId]: run.run_id },
    currentRun: run,
  })),

  // Switch "current run in focus" to whichever run belongs to this dataset,
  // without touching any other state (used by the RunSwitcher control).
  focusDataset: (datasetId) => set(s => {
    const runId = s.runsByDataset[datasetId]
    if (!runId || !s.runs[runId]) return {}
    return { currentRun: s.runs[runId] }
  }),

  // Batch pipeline execution tracking (survives navigation while a batch runs).
  currentBatch: null,
  setCurrentBatch: batch => set({ currentBatch: batch }),
}))
