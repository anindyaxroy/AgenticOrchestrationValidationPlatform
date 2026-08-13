import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import DatasetsPage  from './pages/DatasetsPage'
import PipelinePage  from './pages/PipelinePage'
import MiningPage    from './pages/MiningPage'
import FeaturesPage  from './pages/FeaturesPage'
import AgentPage     from './pages/AgentPage'
import AuditPage     from './pages/AuditPage'
import FlowPage     from './pages/FlowPage'
import ComparePage  from './pages/ComparePage'
import { useStore }  from './services/store'

const NAV = [
  { divider: 'data' },
  { path:'/',          label:'Data Ingestion',     icon:'ti-database',      step:'01', end:true },
  { divider: 'pipeline' },
  { path:'/pipeline',  label:'Run Pipeline',        icon:'ti-player-play',   step:'02' },
  { divider: 'analysis' },
  { path:'/mining',    label:'Process Mining',      icon:'ti-chart-dots',    step:'03' },
  { path:'/features',  label:'Features & RL Proof', icon:'ti-vector',        step:'04' },
  { divider: 'agent' },
  { path:'/agent',     label:'Agent Reasoning',     icon:'ti-robot',         step:'05' },
  { divider: 'compare' },
  { path:'/compare',   label:'Compare Datasets',    icon:'ti-chart-bar',     step:'08' },
  { divider: 'deep dive' },
  { path:'/flow',     label:'Architecture',       icon:'ti-sitemap',       step:'07' },
  { divider: 'audit' },
  { path:'/audit',     label:'Audit & Lineage',     icon:'ti-shield-check',  step:'06' },
]

export default function App() {
  const { selectedDataset, currentRun } = useStore()
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-5 py-2.5 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs font-bold bg-teal-800 text-teal-50 px-2 py-1 rounded">BPMN·AI</span>
          <div>
            <div className="text-sm font-medium text-gray-900">Agentic Orchestration Validation Platform</div>
            <div className="text-xs text-gray-400">Anindya Roy · UvA MBA AI · 2026</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {selectedDataset && (
            <span className="font-mono text-xs text-teal-800 bg-teal-50 border border-teal-200 px-3 py-1 rounded-full truncate max-w-52">
              <i className="ti ti-database mr-1"/>{selectedDataset.name}
            </span>
          )}
          {currentRun && (
            <span className="font-mono text-xs text-purple-800 bg-purple-50 border border-purple-200 px-3 py-1 rounded-full">
              <i className="ti ti-shield-check mr-1"/>{currentRun.content_hash}
            </span>
          )}
        </div>
      </header>

      <div className="flex flex-1">
        <nav className="w-52 bg-white border-r border-gray-200 p-2 flex-shrink-0 overflow-y-auto">
          {NAV.map((item, i) => {
            if (item.divider) return (
              <div key={i} className="mx-3 mt-3 mb-1 border-t border-gray-100 pt-2">
                <div className="text-xs text-gray-400 font-mono px-1">{item.divider}</div>
              </div>
            )
            return (
              <NavLink key={item.path} to={item.path} end={item.end}
                className={({isActive}) =>
                  `flex items-center gap-2.5 px-3 py-2 rounded-lg mb-0.5 transition-colors
                   ${isActive ? 'bg-teal-800 text-teal-50' : 'text-gray-600 hover:bg-gray-100'}`
                }>
                {({isActive}) => (<>
                  <i className={`ti ${item.icon} text-sm flex-shrink-0`}/>
                  <div>
                    <div className="text-xs font-medium leading-tight">{item.label}</div>
                    <div className={`font-mono text-xs ${isActive?'text-teal-300':'text-gray-400'}`}>step {item.step}</div>
                  </div>
                </>)}
              </NavLink>
            )
          })}
          <div className="mt-3 pt-3 border-t border-gray-100">
            <a href="/api/docs" target="_blank" rel="noreferrer"
               className="flex items-center gap-2 px-3 py-1.5 text-xs text-gray-500 hover:text-gray-900 hover:bg-gray-50 rounded-lg">
              <i className="ti ti-api text-xs"/> API docs
            </a>
            <a href="/api/info" target="_blank" rel="noreferrer"
               className="flex items-center gap-2 px-3 py-1.5 text-xs text-gray-500 hover:text-gray-900 hover:bg-gray-50 rounded-lg">
              <i className="ti ti-info-circle text-xs"/> Platform info
            </a>
          </div>
        </nav>

        <main className="flex-1 p-5 overflow-auto">
          <Routes>
            <Route path="/"         element={<DatasetsPage />} />
            <Route path="/pipeline" element={<PipelinePage />} />
            <Route path="/mining"   element={<MiningPage />} />
            <Route path="/features" element={<FeaturesPage />} />
            <Route path="/agent"    element={<AgentPage />} />
            <Route path="/compare"  element={<ComparePage />} />
            <Route path="/audit"    element={<AuditPage />} />
            <Route path="/flow"     element={<FlowPage />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
