import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { pipelineApi } from '../services/api'
import { useStore } from '../services/store'
import RunSwitcher from '../components/RunSwitcher'

/* ─── Self-contained Markdown renderer ─────────────────────────────────────
   No npm packages. Handles: ## h2, ### h3, **bold**, `code`,
   - bullets, | tables |, > blockquotes, ``` code blocks, --- hr            */
function parseInline(text) {
  const bold  = text.split(/(\*\*[^*]+\*\*)/g)
  return bold.map((seg, i) => {
    if (seg.startsWith('**') && seg.endsWith('**')) {
      return <strong key={i} className="font-semibold text-gray-900">{seg.slice(2,-2)}</strong>
    }
    const ticks = seg.split(/(`[^`]+`)/g)
    return ticks.map((t, j) =>
      t.startsWith('`') && t.endsWith('`')
        ? <code key={j} className="font-mono text-xs bg-gray-100 text-teal-700 px-1.5 py-0.5 rounded">{t.slice(1,-1)}</code>
        : t
    )
  })
}

function MD({ content }) {
  if (!content) return null
  const lines = content.split('\n')
  const out   = []
  let i = 0
  while (i < lines.length) {
    const L = lines[i]

    if (L.startsWith('## ')) {
      out.push(
        <h2 key={i} className="text-base font-bold text-gray-900 mt-6 mb-2 pb-1.5 border-b-2 border-teal-100 flex items-center gap-2">
          <span className="w-1 h-5 rounded-full flex-shrink-0 bg-teal-700"/>
          {parseInline(L.slice(3))}
        </h2>
      )
      i++; continue
    }

    if (L.startsWith('### ')) {
      out.push(
        <h3 key={i} className="text-sm font-semibold text-gray-800 mt-4 mb-1.5">
          {parseInline(L.slice(4))}
        </h3>
      )
      i++; continue
    }

    if (L.startsWith('|')) {
      const rows = []
      while (i < lines.length && lines[i].startsWith('|')) {
        if (!lines[i].match(/^\|[\s\-:|]+\|$/)) rows.push(lines[i])
        i++
      }
      const cells = r => r.split('|').filter((_,idx,a) => idx>0 && idx<a.length-1)
      out.push(
        <div key={`t${i}`} className="overflow-x-auto my-4">
          <table className="w-full text-xs border-collapse">
            <thead className="bg-gray-50">
              <tr>{cells(rows[0]).map((c,ci)=>(
                <th key={ci} className="text-left px-3 py-2 font-semibold text-gray-700 border border-gray-200">{parseInline(c.trim())}</th>
              ))}</tr>
            </thead>
            <tbody>{rows.slice(1).map((r,ri)=>(
              <tr key={ri} className="hover:bg-gray-50">
                {cells(r).map((c,ci)=>(
                  <td key={ci} className="px-3 py-2 text-gray-600 border border-gray-200">{parseInline(c.trim())}</td>
                ))}
              </tr>
            ))}</tbody>
          </table>
        </div>
      )
      continue
    }

    if (L.startsWith('> ')) {
      out.push(
        <blockquote key={i} className="border-l-4 border-amber-400 bg-amber-50 pl-3 py-2 my-2 rounded-r-lg text-xs text-amber-800 italic">
          {parseInline(L.slice(2))}
        </blockquote>
      )
      i++; continue
    }

    if (L.match(/^[\s]*[-*] /)) {
      const indent = L.match(/^(\s*)/)[1].length
      out.push(
        <div key={i} className="flex gap-2 mb-1.5 text-sm text-gray-700" style={{paddingLeft: indent*8}}>
          <span className="text-teal-500 flex-shrink-0 mt-0.5 font-bold">•</span>
          <span>{parseInline(L.replace(/^[\s]*[-*] /,''))}</span>
        </div>
      )
      i++; continue
    }

    if (L.startsWith('```')) {
      i++
      const code = []
      while (i < lines.length && !lines[i].startsWith('```')) { code.push(lines[i]); i++ }
      out.push(
        <pre key={`c${i}`} className="bg-gray-900 text-green-300 rounded-xl p-4 text-xs font-mono overflow-x-auto my-3 leading-relaxed">
          <code>{code.join('\n')}</code>
        </pre>
      )
      i++; continue
    }

    if (L.match(/^---+$/)) { out.push(<hr key={i} className="border-gray-200 my-4"/>); i++; continue }
    if (L.trim() === '')   { i++; continue }

    out.push(
      <p key={i} className="text-sm text-gray-700 leading-relaxed mb-3">{parseInline(L)}</p>
    )
    i++
  }
  return <div className="space-y-0.5">{out}</div>
}

/* ─── Node metadata ─────────────────────────────────────────────────────── */
const NODE_META = {
  init_node:      { label:'1. Init',      icon:'ti-list-check',      color:'#085041',
    desc:'Plans execution steps and sets up the reasoning pipeline.' },
  retrieve_node:  { label:'2. Retrieve',  icon:'ti-database-search', color:'#185FA5',
    desc:'Queries Chroma vector store for mining findings semantically relevant to the question.' },
  rl_node:        { label:'3. RL Proof',  icon:'ti-brain',           color:'#534AB7',
    desc:'Loads the learned-policy vs baseline comparison on held-out seeds.' },
  reason_node:    { label:'4. Reason',    icon:'ti-message-dots',    color:'#B45309',
    desc:'Calls Claude Sonnet with grounded evidence. Instructed to cite only retrieved findings.' },
  summarise_node: { label:'5. Summarise', icon:'ti-report',          color:'#7C3AED',
    desc:'Produces a business-reader summary from the detailed analysis.' },
}

const PRESET_Q = [
  'What is the main bottleneck activity and what evidence supports this finding?',
  'Did the RL agent demonstrably learn or is a greedy heuristic sufficient? Cite the metrics.',
  'What are the conformance issues, their cost impact, and what do they mean for the process?',
  'What are the key limitations of this analysis and what data would improve it?',
  'Which actions did the learned policy prefer and what does that tell us about the process?',
]

/* ─── Main page ─────────────────────────────────────────────────────────── */
export default function AgentPage() {
  const { currentRun, selectedDataset } = useStore()
  const [question, setQuestion] = useState('')
  const [asking,   setAsking]   = useState(false)
  const [response, setResponse] = useState(null)
  const [error,    setError]    = useState(null)
  const [tab,      setTab]      = useState('summary')
  const navigate = useNavigate()

  if (!currentRun) return (
    <div className="max-w-2xl">
      <div className="card text-center py-10 text-gray-400">
        <i className="ti ti-robot text-3xl block mb-2"/>
        <div className="text-sm mb-3">No pipeline run yet.</div>
        <button className="text-teal-600 text-sm hover:underline" onClick={() => navigate('/pipeline')}>
          Run the pipeline first →
        </button>
      </div>
    </div>
  )

  const ag        = currentRun.agent
  const nodeAudit = ag.node_audit || []
  const executed  = nodeAudit.map(a => a.node)

  const askAgent = async () => {
    if (!question.trim() || !selectedDataset) return
    setAsking(true); setError(null); setResponse(null)
    try {
      const r = await pipelineApi.ask(selectedDataset.id, question)
      setResponse(r); setTab('ask')
    } catch(err) {
      setError(err.response?.data?.detail || err.message)
    } finally { setAsking(false) }
  }

  return (
    <div className="max-w-5xl space-y-5">
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-medium text-gray-900">Agent Reasoning</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            LangGraph 5-node agent ·{' '}
            <span className="font-mono text-teal-700">{currentRun.log_summary.source_filename}</span>
            <span className="ml-2 font-mono text-gray-400">run: {currentRun.run_id}</span>
          </p>
        </div>
        <RunSwitcher/>
      </div>

      {/* Node execution trace */}
      <div className="card">
        <div className="section-label">LangGraph Execution Trace</div>
        <div className="flex items-center gap-1 flex-wrap">
          {Object.entries(NODE_META).map(([key, meta], i) => {
            const ran = executed.includes(key)
            const nd  = nodeAudit.find(a => a.node === key)
            return (
              <React.Fragment key={key}>
                <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition-all
                  ${ran ? 'border-teal-300 bg-teal-50' : 'border-gray-100 bg-gray-50'}`}>
                  <i className={`ti ${meta.icon} text-sm`} style={{color: ran ? meta.color : '#ccc'}}/>
                  <span className={ran ? 'text-gray-800 font-medium' : 'text-gray-400'}>{meta.label}</span>
                  {ran && nd?.llm_live !== undefined && (
                    <span className={`font-mono text-xs px-1 rounded ${nd.llm_live ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                      {nd.llm_live ? 'live' : 'template'}
                    </span>
                  )}
                  {ran && nd?.n_hits !== undefined && (
                    <span className="font-mono text-xs text-blue-600">{nd.n_hits} hits</span>
                  )}
                </div>
                {i < 4 && <i className="ti ti-arrow-right text-gray-200 text-xs"/>}
              </React.Fragment>
            )
          })}
        </div>
        {nodeAudit.some(a => a.llm_live === false) && (
          <div className="mt-2 text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg p-2">
            <i className="ti ti-key mr-1"/>
            Agent ran in <strong>grounded-template mode</strong> — ANTHROPIC_API_KEY not detected.
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200">
        {[
          {id:'summary',   label:'Executive Summary',  icon:'ti-report'},
          {id:'reasoning', label:'Full Analysis',      icon:'ti-microscope'},
          {id:'ask',       label:'Ask the Agent',      icon:'ti-message-question'},
        ].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium border-b-2 transition-colors -mb-px
              ${tab===t.id ? 'border-teal-600 text-teal-700' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>
            <i className={`ti ${t.icon}`}/>{t.label}
          </button>
        ))}
      </div>

      {/* Summary */}
      {tab === 'summary' && (
        <div className="card">
          <MD content={ag.summary}/>
        </div>
      )}

      {/* Full reasoning */}
      {tab === 'reasoning' && (
        <div className="card">
          <MD content={ag.reasoning}/>
        </div>
      )}

      {/* Ask */}
      {tab === 'ask' && (
        <div className="space-y-4">
          <div className="card">
            <div className="section-label">Preset Questions</div>
            <div className="space-y-1.5">
              {PRESET_Q.map(q => (
                <button key={q} onClick={() => setQuestion(q)}
                  className={`w-full text-left text-xs px-3 py-2 rounded-lg border transition-colors
                    ${question===q ? 'border-teal-400 bg-teal-50 text-teal-800' : 'border-gray-200 text-gray-600 hover:border-teal-200 hover:bg-gray-50'}`}>
                  <i className="ti ti-message-question mr-1.5 text-gray-400"/>"{q}"
                </button>
              ))}
            </div>
            <div className="flex gap-2 mt-3">
              <input value={question} onChange={e => setQuestion(e.target.value)}
                onKeyDown={e => e.key==='Enter' && askAgent()}
                placeholder="Or type your own question…"
                className="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-teal-400"/>
              <button className="btn-primary px-4" onClick={askAgent}
                disabled={asking || !question.trim()}>
                {asking ? <i className="ti ti-loader animate-spin"/> : <i className="ti ti-send"/>}
              </button>
            </div>
            {error && <div className="mt-2 text-xs text-red-500">{error}</div>}
          </div>

          {response && (
            <div className="card">
              <div className="text-xs text-gray-400 mb-3">
                Nodes: {response.node_audit?.map(a=>a.node).join(' → ')}
              </div>
              <MD content={response.summary || response.reasoning}/>
              {response.reasoning && response.reasoning !== response.summary && (
                <details className="mt-3">
                  <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600">
                    Show full detailed analysis →
                  </summary>
                  <div className="mt-2 border-t border-gray-100 pt-3">
                    <MD content={response.reasoning}/>
                  </div>
                </details>
              )}
            </div>
          )}
        </div>
      )}

      <div className="flex gap-2">
        <button className="btn-primary" onClick={() => navigate('/audit')}>
          <i className="ti ti-shield-check mr-1"/> Audit & Lineage →
        </button>
      </div>
    </div>
  )
}
