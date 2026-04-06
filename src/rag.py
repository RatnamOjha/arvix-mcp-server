"""
RAG Engine — retrieval-augmented generation over indexed papers.
Retrieves relevant chunks, reranks with a cross-encoder, formats context.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
        self._reranker = None  # lazy load

    def _load_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Cross-encoder reranker loaded.")
            except Exception as e:
                logger.warning(f"Reranker unavailable, skipping: {e}")
                self._reranker = False
        return self._reranker if self._reranker is not False else None

    async def query(self, question: str, top_k: int = 6) -> dict:
        """
        Full RAG pipeline:
        1. Embed the question
        2. Retrieve top-k chunks by cosine similarity
        3. Rerank with cross-encoder
        4. Return structured answer with citations
        """
        if not self.vector_store.chunks:
            return {
                "answer": "Your library is empty. Use fetch_paper to index papers first.",
                "sources": [],
            }

        # Step 1: Embed question
        loop = asyncio.get_event_loop()
        query_emb = await loop.run_in_executor(
            None, self.embedder.encode_query, question
        )

        # Step 2: Retrieve candidate chunks (fetch 2x top_k for reranking)
        candidates = self.vector_store.search(query_emb, top_k=top_k * 2)

        if not candidates:
            return {"answer": "No relevant content found.", "sources": []}

        # Step 3: Rerank
        reranker = await loop.run_in_executor(None, self._load_reranker)
        if reranker:
            pairs = [(question, c["chunk"]) for c in candidates]
            scores = await loop.run_in_executor(None, reranker.predict, pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            candidates = [c for c, _ in ranked[:top_k]]

        # Step 4: Build context + response
        context_parts = []
        sources = []
        seen_papers = set()

        for i, chunk in enumerate(candidates):
            context_parts.append(
                f"[Source {i+1}] '{chunk['title']}' (arxiv:{chunk['arxiv_id']})\n{chunk['chunk']}"
            )
            if chunk["arxiv_id"] not in seen_papers:
                seen_papers.add(chunk["arxiv_id"])
                paper = self.vector_store.papers.get(chunk["arxiv_id"], {})
                sources.append({
                    "arxiv_id": chunk["arxiv_id"],
                    "title": chunk["title"],
                    "authors": paper.get("authors", []),
                    "relevance_score": round(chunk["score"], 4),
                })

        context = "\n\n---\n\n".join(context_parts)

        return {
            "question": question,
            "context": context,
            "sources": sources,
            "instruction": (
                "Use the context above to answer the question. "
                "Cite sources as [Source N] inline. "
                "If the context doesn't contain the answer, say so."
            ),
        }

    async def summarize_paper(self, arxiv_id: str) -> dict:
        """Retrieve all chunks for a paper and structure a summary request."""
        if arxiv_id not in self.vector_store.papers:
            return {"error": f"Paper {arxiv_id} not in your library. Fetch it first."}

        metadata = self.vector_store.papers[arxiv_id]
        chunks = self.vector_store.get_paper_chunks(arxiv_id)

        # Take first ~10 chunks (intro + methods) and last ~3 (conclusion)
        sample_chunks = chunks[:10] + chunks[-3:] if len(chunks) > 13 else chunks
        context = "\n\n".join(sample_chunks[:2000])  # safe context limit

        return {
            "arxiv_id": arxiv_id,
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "abstract": metadata.get("abstract"),
            "context": context,
            "instruction": (
                "Summarize this paper covering: (1) Problem & motivation, "
                "(2) Key method/contribution, (3) Main results, (4) Limitations, "
                "(5) Why it matters. Use the abstract and extracted text."
            ),
        }
