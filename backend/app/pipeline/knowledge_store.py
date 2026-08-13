"""
knowledge_store.py — Embed REAL process-mining findings so the agent can
retrieve and reason over them (supervisor point 7).

Why Chroma, not Milvus: at thesis scale (hundreds of cases, dozens of findings
per dataset) Milvus is operational overkill — it needs its own server, etcd, and
object storage. Chroma gives the same retrieval semantics (vector similarity over
embedded findings) in-process, so the whole pipeline stays runnable for a defense
demo. The interface below is deliberately thin: swapping in Milvus later means
re-implementing add()/query() against the Milvus client and nothing else changes.

Lineage: every embedded finding is tagged with dataset_id + content_hash, so a
retrieved finding can be traced to the exact source file it came from.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

import math
import re

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class HashingEmbeddingFunction(EmbeddingFunction):
    """
    Deterministic, dependency-free bag-of-words hashing embedding.

    Used so the knowledge store is fully runnable offline (no model download).
    In a production/thesis environment with internet access, swap this for
    chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
    (all-MiniLM-L6-v2) — the KnowledgeStore interface is unchanged. The hashing
    embedding gives real semantic-ish retrieval via term overlap in a fixed
    vector space, which is sufficient to demonstrate retrieval mechanics.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def name(self) -> str:
        return "hashing-bow-256"

    def __call__(self, input: Documents) -> Embeddings:
        vecs = []
        for doc in input:
            v = [0.0] * self.dim
            tokens = re.findall(r"[a-z0-9]+", doc.lower())
            for tok in tokens:
                h = int(__import__("hashlib").md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs


class KnowledgeStore:
    def __init__(self, persist_dir: Optional[str] = None, collection: str = "pm_findings",
                 embedding_function=None):
        # Offline-safe by default. Pass a real SentenceTransformer EF in prod.
        self.client = (chromadb.PersistentClient(path=persist_dir)
                       if persist_dir else chromadb.EphemeralClient())
        self.ef = embedding_function or HashingEmbeddingFunction()
        self.col = self.client.get_or_create_collection(
            name=collection, embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_findings(self, dataset_id: str, content_hash: str,
                     findings: list[str], kind: str = "mining") -> int:
        ids, metas = [], []
        for i, f in enumerate(findings):
            fid = hashlib.sha256(f"{content_hash}:{kind}:{i}:{f}".encode()).hexdigest()[:24]
            ids.append(fid)
            metas.append({
                "dataset_id": dataset_id,
                "content_hash": content_hash[:16],
                "kind": kind,
                "ordinal": i,
            })
        # upsert so re-running the same dataset doesn't duplicate
        self.col.upsert(ids=ids, documents=findings, metadatas=metas)
        return len(findings)

    def query(self, question: str, dataset_id: Optional[str] = None,
              k: int = 5) -> list[dict[str, Any]]:
        where = {"dataset_id": dataset_id} if dataset_id else None
        res = self.col.query(query_texts=[question], n_results=k, where=where)
        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for d, m, dist in zip(docs, metas, dists):
            out.append({"finding": d, "metadata": m, "distance": round(float(dist), 4)})
        return out

    def count(self, dataset_id: Optional[str] = None) -> int:
        if dataset_id:
            return self.col.get(where={"dataset_id": dataset_id}).get("ids", []).__len__()
        return self.col.count()
