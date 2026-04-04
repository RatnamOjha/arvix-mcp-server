"""
VectorStore — local vector database backed by numpy arrays + JSON metadata.
Supports cosine similarity search with optional paper filtering.

For production, swap this out for Qdrant or ChromaDB with minimal changes
to the add_paper() and search() interfaces.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

STORE_FILE = "vectors.pkl"
META_FILE = "papers.json"


class VectorStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.store_path = data_dir / STORE_FILE
        self.meta_path = data_dir / META_FILE

        # In-memory state
        self.chunks: list[str] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, D)
        self.chunk_metadata: list[dict] = []  # per-chunk paper info
        self.papers: dict[str, dict] = {}  # arxiv_id -> metadata

        self._load()

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_paper(
        self,
        arxiv_id: str,
        chunks: list[str],
        embeddings: np.ndarray,
        metadata: dict,
    ) -> None:
        """Add all chunks from a paper. Idempotent — replaces existing data."""
        # Remove old chunks if re-indexing
        if arxiv_id in self.papers:
            self._remove_paper_chunks(arxiv_id)

        self.papers[arxiv_id] = metadata

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            self.chunks.append(chunk)
            self.chunk_metadata.append({
                "arxiv_id": arxiv_id,
                "chunk_index": i,
                "title": metadata.get("title", ""),
            })
            if self.embeddings is None:
                self.embeddings = emb.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])

        self._save()

    def _remove_paper_chunks(self, arxiv_id: str) -> None:
        keep = [i for i, m in enumerate(self.chunk_metadata) if m["arxiv_id"] != arxiv_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.chunk_metadata = [self.chunk_metadata[i] for i in keep]
        if self.embeddings is not None and len(keep) > 0:
            self.embeddings = self.embeddings[keep]
        elif len(keep) == 0:
            self.embeddings = None

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 6,
        arxiv_id_filter: Optional[str] = None,
    ) -> list[dict]:
        """Cosine similarity search. Returns top_k chunks with scores."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Filter by paper if requested
        if arxiv_id_filter:
            indices = [i for i, m in enumerate(self.chunk_metadata) if m["arxiv_id"] == arxiv_id_filter]
            if not indices:
                return []
            emb_subset = self.embeddings[indices]
            scores = emb_subset @ query_embedding
            top_local = np.argsort(scores)[::-1][:top_k]
            top_global = [indices[i] for i in top_local]
            top_scores = scores[top_local]
        else:
            scores = self.embeddings @ query_embedding
            top_global = np.argsort(scores)[::-1][:top_k]
            top_scores = scores[top_global]

        results = []
        for idx, score in zip(top_global, top_scores):
            meta = self.chunk_metadata[idx]
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "arxiv_id": meta["arxiv_id"],
                "chunk_index": meta["chunk_index"],
                "title": meta["title"],
            })
        return results

    def list_papers(self) -> list[dict]:
        return list(self.papers.values())

    def get_paper_chunks(self, arxiv_id: str) -> list[str]:
        return [
            self.chunks[i]
            for i, m in enumerate(self.chunk_metadata)
            if m["arxiv_id"] == arxiv_id
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        payload = {
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "chunk_metadata": self.chunk_metadata,
        }
        with open(self.store_path, "wb") as f:
            pickle.dump(payload, f)

        with open(self.meta_path, "w") as f:
            json.dump(self.papers, f, indent=2, default=str)

        logger.info(f"Vector store saved: {len(self.chunks)} chunks, {len(self.papers)} papers")

    def _load(self) -> None:
        if self.store_path.exists():
            with open(self.store_path, "rb") as f:
                payload = pickle.load(f)
            self.chunks = payload["chunks"]
            self.embeddings = payload["embeddings"]
            self.chunk_metadata = payload["chunk_metadata"]
            logger.info(f"Loaded {len(self.chunks)} chunks from disk")

        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self.papers = json.load(f)
            logger.info(f"Loaded {len(self.papers)} paper records")
