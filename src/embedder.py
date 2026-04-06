"""
Embedder — wraps sentence-transformers for encoding text chunks.
Uses all-MiniLM-L6-v2 by default (fast, 384-dim, great for retrieval).
"""

import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy load

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded.")
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into embedding vectors."""
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,  # cosine similarity via dot product
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        return self.encode([query])[0]

    async def embed_and_store(
        self,
        chunks: list[str],
        metadata: dict,
        vector_store,
    ) -> None:
        """Embed all chunks and persist them in the vector store."""
        loop = asyncio.get_event_loop()

        logger.info(f"Embedding {len(chunks)} chunks for {metadata['arxiv_id']}...")

        def _encode():
            return self.encode(chunks)

        embeddings = await loop.run_in_executor(None, _encode)

        vector_store.add_paper(
            arxiv_id=metadata["arxiv_id"],
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata,
        )
        logger.info(f"Stored {len(chunks)} chunks for {metadata['arxiv_id']}")
