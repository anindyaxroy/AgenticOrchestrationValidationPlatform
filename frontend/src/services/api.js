import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 300000 })

api.interceptors.response.use(
  r => r,
  err => { console.error('API error:', err.response?.data?.detail || err.message); return Promise.reject(err) }
)

export const datasetsApi = {
  list:        ()         => api.get('/datasets/').then(r => r.data),
  get:         id         => api.get(`/datasets/${id}`).then(r => r.data),
  preview:     id         => api.get(`/datasets/${id}/preview`).then(r => r.data),

  // Single file (legacy)
  upload: (file, onProgress) => {
    const fd = new FormData(); fd.append('file', file)
    return api.post('/datasets/upload', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: e => onProgress?.(Math.round(e.loaded / e.total * 100)),
    }).then(r => r.data)
  },

  // Multiple files
  uploadMany: (files, onProgress) => {
    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    return api.post('/datasets/upload-many', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: e => onProgress?.(Math.round(e.loaded / e.total * 100)),
    }).then(r => r.data)
  },

  delete: id => api.delete(`/datasets/${id}`).then(r => r.data),
}

// Triggers a browser download for a PDF blob response.
function _downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename
  document.body.appendChild(a); a.click(); a.remove()
  window.URL.revokeObjectURL(url)
}

export const pipelineApi = {
  run:       (id, episodes = 300) => api.post(`/pipeline/run/${id}?episodes=${episodes}`).then(r => r.data),
  runBatch:  (datasetIds, episodes = 300) =>
               api.post('/pipeline/run-batch', { dataset_ids: datasetIds, episodes }).then(r => r.data),
  batchStatus: batchId            => api.get(`/pipeline/batch/${batchId}`).then(r => r.data),
  listRuns:  id                   => api.get(`/pipeline/runs/${id}`).then(r => r.data),
  listAllRuns: ()                 => api.get('/pipeline/runs').then(r => r.data),
  result:    runId                => api.get(`/pipeline/result/${runId}`).then(r => r.data),
  audit:     runId                => api.get(`/pipeline/audit/${runId}`).then(r => r.data),
  ask:       (id, question)       => api.post(`/pipeline/ask/${id}`, { question }).then(r => r.data),
  episodes:  (runId, mode='all', limit=50) => api.get(`/pipeline/episodes/${runId}?mode=${mode}&limit=${limit}`).then(r => r.data),

  // Cross-dataset comparison
  compare:   runIds               => api.post('/pipeline/compare', { run_ids: runIds }).then(r => r.data),

  // Full-run report export — 'pdf' (default) or 'html'. Both resolve after
  // triggering the browser download.
  downloadRunReport: async (runId, fmt = 'pdf', filenameHint) => {
    const r = await api.get(`/pipeline/report/${runId}`, { params: { fmt }, responseType: 'blob' })
    _downloadBlob(r.data, filenameHint || `bpmn_report_${runId.slice(0,12)}.${fmt}`)
  },
  downloadCompareReport: async (runIds) => {
    const r = await api.post('/pipeline/report/compare', { run_ids: runIds }, { responseType: 'blob' })
    _downloadBlob(r.data, 'bpmn_comparison_report.pdf')
  },
}

export default api
