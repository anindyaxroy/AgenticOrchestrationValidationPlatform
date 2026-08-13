"""
reasoning_agent.py — LangGraph 5-node agent producing Markdown output.

DEFINITIVE LANGGRAPH FIX: Single opaque 'data' key in TypedDict so node names
(init_node, retrieve_node, rl_node, reason_node, summarise_node) can NEVER
clash with state keys (only key is 'data').

Output format: Markdown — headings, bullet lists, bold metrics, tables.
"""
from __future__ import annotations
import os, json
from typing import Optional, TypedDict
from langgraph.graph import StateGraph, END
from .knowledge_store import KnowledgeStore


class AgentState(TypedDict):
    data: dict   # single opaque key — zero clash risk with any node name


class LLM:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        self.key    = os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        if self.key:
            try:
                from langchain_anthropic import ChatAnthropic
                self._client = ChatAnthropic(model=model, max_tokens=2000, temperature=0.2)
            except Exception:
                self._client = None

    @property
    def live(self) -> bool:
        return self._client is not None

    def complete(self, system: str, user: str) -> str:
        if self._client is not None:
            from langchain_core.messages import SystemMessage, HumanMessage
            resp = self._client.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            return resp.content if isinstance(resp.content, str) else str(resp.content)
        # Grounded Markdown fallback
        return _grounded_markdown(user)


def _grounded_markdown(evidence: str) -> str:
    """Produce structured Markdown from raw evidence when no API key is present."""
    return (
        "## Analysis (Grounded Template Mode)\n\n"
        "> **Note:** `ANTHROPIC_API_KEY` is not set. "
        "This analysis is synthesised directly from retrieved evidence without an LLM. "
        "Set the key for full Claude Sonnet reasoning.\n\n"
        "### Evidence Retrieved\n\n"
        + "\n".join(f"- {line.strip()}" for line in evidence.split("\n")
                    if line.strip() and not line.startswith("QUESTION") and not line.startswith("MINING") and not line.startswith("RL"))
        + "\n\n### Interpretation\n\n"
        "The evidence above was retrieved from the Chroma knowledge store and the RL proof harness. "
        "Run with a live API key to receive a full analytical interpretation.\n"
    )


def make_graph(store: KnowledgeStore, rl_evidence_provider,
               llm: Optional[LLM] = None):
    llm = llm or LLM()

    def init_node(state: AgentState) -> AgentState:
        d = dict(state.get("data") or {})
        audit = list(d.get("audit") or [])
        audit.append({"node": "init_node", "question": d.get("question")})
        d["exec_plan"] = ["retrieve", "load_rl", "reason", "summarise"]
        d["audit"] = audit
        return {"data": d}

    def retrieve_node(state: AgentState) -> AgentState:
        d = dict(state.get("data") or {})
        hits = store.query(d.get("question", ""), dataset_id=d.get("dataset_id"), k=6)
        audit = list(d.get("audit") or [])
        audit.append({"node": "retrieve_node", "n_hits": len(hits),
                       "top_dist": round(hits[0]["distance"], 3) if hits else None})
        d["retrieved"] = hits
        d["audit"] = audit
        return {"data": d}

    def rl_node(state: AgentState) -> AgentState:
        d = dict(state.get("data") or {})
        ev = rl_evidence_provider(d.get("dataset_id"))
        beats_random = (ev.get("comparisons", {}).get("random", {}).get("learned_better")
                        if ev else None)
        audit = list(d.get("audit") or [])
        audit.append({"node": "rl_node", "has_proof": bool(ev),
                       "beats_random": beats_random})
        d["rl_evidence"] = ev
        d["audit"] = audit
        return {"data": d}

    def reason_node(state: AgentState) -> AgentState:
        d = dict(state.get("data") or {})
        findings = "\n".join(
            f"- {h['finding']}" for h in (d.get("retrieved") or [])
        )
        rl  = d.get("rl_evidence") or {}
        cmp = json.dumps(rl.get("comparisons", {}), indent=2)

        system = """You are a process-mining and reinforcement-learning analyst writing a thesis research report.

CRITICAL RULES:
1. Output ONLY valid Markdown — use ## headings, **bold**, bullet lists, and tables.
2. Cite ONLY numbers from the evidence provided — never invent metrics.
3. Flag time-dependent features (sla_risk_index, resource_utilisation) as proxies when no timestamps exist.
4. State clearly whether the RL agent learned (beat baselines on held-out data) or not, and why.
5. Use a Markdown table to compare baselines.
6. End with a ## Limitations section.

OUTPUT STRUCTURE:
## Process Characterisation
## Bottleneck Analysis  
## Conformance & Risk Profile
## RL Learning Evidence
### Baseline Comparison Table
## Key Findings
## Limitations"""

        user = f"""QUESTION: {d.get('question')}

RETRIEVED MINING FINDINGS (from Chroma knowledge store):
{findings}

RL LEARNING PROOF (held-out evaluation, never-seen seeds):
{cmp}

Write the full Markdown analysis following the structure in the system prompt."""

        out = llm.complete(system, user)
        audit = list(d.get("audit") or [])
        audit.append({"node": "reason_node", "llm_live": llm.live, "chars": len(out)})
        d["reasoning"] = out
        d["audit"] = audit
        return {"data": d}

    def summarise_node(state: AgentState) -> AgentState:
        d = dict(state.get("data") or {})
        system = """Summarise in Markdown for a business stakeholder.
Use this exact structure:
## Executive Summary
3-4 sentences covering: what the process looks like, main risk, whether the AI agent learned, key limitation.

## Recommended Next Steps
3 bullet points.

No fabricated numbers. Bold the most important insight."""

        out = llm.complete(system, f"Full analysis:\n{d.get('reasoning', '')}")
        audit = list(d.get("audit") or [])
        audit.append({"node": "summarise_node", "chars": len(out)})
        d["summary"] = out
        d["audit"] = audit
        return {"data": d}

    g = StateGraph(AgentState)
    g.add_node("init_node",      init_node)
    g.add_node("retrieve_node",  retrieve_node)
    g.add_node("rl_node",        rl_node)
    g.add_node("reason_node",    reason_node)
    g.add_node("summarise_node", summarise_node)
    g.set_entry_point("init_node")
    g.add_edge("init_node",      "retrieve_node")
    g.add_edge("retrieve_node",  "rl_node")
    g.add_edge("rl_node",        "reason_node")
    g.add_edge("reason_node",    "summarise_node")
    g.add_edge("summarise_node", END)
    return g.compile()
