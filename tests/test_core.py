"""
Tests for ArXiv MCP Server components.
Run with: pytest tests/ -v --asyncio-mode=auto
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.vector_store import VectorStore
from src.reading_list import ReadingList
from src.embedder import Embedder


# ── VectorStore ────────────────────────────────────────────────────────────────

class TestVectorStore:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.store = VectorStore(data_dir=Path(self.tmp))

    def _fake_embeddings(self, n: int, dim: int = 384) -> np.ndarray:
        rng = np.random.default_rng(42)
        embs = rng.random((n, dim)).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

    def test_add_and_search(self):
        chunks = ["Deep learning is great", "Transformers use attention", "BERT is pretrained"]
        embs = self._fake_embeddings(3)
        metadata = {"arxiv_id": "1234.5678", "title": "Test Paper", "authors": ["Alice"]}
        self.store.add_paper("1234.5678", chunks, embs, metadata)

        assert len(self.store.chunks) == 3
        assert "1234.5678" in self.store.papers

        query_emb = self._fake_embeddings(1)[0]
        results = self.store.search(query_emb, top_k=2)
        assert len(results) == 2
        assert all("chunk" in r for r in results)
        assert all("score" in r for r in results)

    def test_idempotent_add(self):
        chunks = ["Chunk A", "Chunk B"]
        embs = self._fake_embeddings(2)
        meta = {"arxiv_id": "abc.123", "title": "Paper", "authors": []}
        self.store.add_paper("abc.123", chunks, embs, meta)
        self.store.add_paper("abc.123", ["New chunk"], self._fake_embeddings(1), meta)
        # Should only have 1 chunk (re-indexed)
        assert len(self.store.chunks) == 1

    def test_persistence(self):
        chunks = ["Hello world"]
        embs = self._fake_embeddings(1)
        meta = {"arxiv_id": "save.test", "title": "Saved Paper", "authors": []}
        self.store.add_paper("save.test", chunks, embs, meta)

        # Reload from disk
        store2 = VectorStore(data_dir=Path(self.tmp))
        assert len(store2.chunks) == 1
        assert "save.test" in store2.papers

    def test_list_papers(self):
        for i in range(3):
            meta = {"arxiv_id": f"paper.{i}", "title": f"Paper {i}", "authors": []}
            self.store.add_paper(f"paper.{i}", ["chunk"], self._fake_embeddings(1), meta)
        papers = self.store.list_papers()
        assert len(papers) == 3


# ── ReadingList ────────────────────────────────────────────────────────────────

class TestReadingList:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.rl = ReadingList(data_dir=Path(self.tmp))

    def _paper(self, arxiv_id: str) -> dict:
        return {"arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "authors": ["Bob"]}

    def test_add_and_get(self):
        self.rl.add(self._paper("2301.0001"))
        papers = self.rl.get_all()
        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "2301.0001"
        assert papers[0]["read"] is False

    def test_mark_read(self):
        self.rl.add(self._paper("2301.0002"))
        result = self.rl.mark_read("2301.0002")
        assert result is True
        assert self.rl.get_all()[0]["read"] is True

    def test_remove(self):
        self.rl.add(self._paper("2301.0003"))
        self.rl.remove("2301.0003")
        assert self.rl.get_all() == []

    def test_note(self):
        self.rl.add(self._paper("2301.0004"))
        self.rl.add_note("2301.0004", "Very relevant to my thesis")
        assert self.rl.get_all()[0]["note"] == "Very relevant to my thesis"

    def test_persistence(self):
        self.rl.add(self._paper("2301.0005"))
        rl2 = ReadingList(data_dir=Path(self.tmp))
        assert len(rl2.get_all()) == 1


# ── Embedder (mocked) ─────────────────────────────────────────────────────────

class TestEmbedder:
    def test_encode_shape(self):
        """Test that encode returns correct shape."""
        embedder = Embedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        embedder._model = mock_model

        result = embedder.encode(["a", "b", "c"])
        assert result.shape == (3, 384)

    def test_encode_query_shape(self):
        embedder = Embedder()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        embedder._model = mock_model

        result = embedder.encode_query("test query")
        assert result.shape == (384,)