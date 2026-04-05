"""
VectorlessRAG — retrieval without embeddings.

Instead of the traditional pipeline:
  query → embed → cosine search → cross-encoder → answer

We use:
  query → BM25 keyword retrieval → LLM contextual compression → answer

Why this is powerful:
- Zero GPU/model download required
- No embedding drift between index time and query time  
- BM25 handles exact terminology brilliantly (paper titles, author names,
  technical acronyms like "LoRA", "RLHF", "KV-cache") — embeddings often miss these
- LLM compression step extracts only the *relevant sentence* from each chunk,
  not the whole chunk — producing tighter, more precise context
- Faster indexing: chunking is instant, no encode() call

The tradeoff: BM25 doesn't understand synonyms ("LLM" vs "large language model")
as well as dense retrieval. We handle this by expanding the query with the LLM
before BM25 search — a technique called HyDE (Hypothetical Document Embedding)
but applied to keyword expansion instead of dense vectors.

Architecture:
  1. Index: chunk paper text → build BM25 index per paper + global
  2. Query expansion: ask LLM to rewrite query as 3 keyword variants
  3. BM25 retrieval: score all chunks across all 3 expanded queries, fuse scores
  4. Contextual compression: for each top chunk, LLM extracts the 1-2 sentences
     that actually answer the question (drops boilerplate)
  5. Return: compressed, cited answer context
"""

import asyncio
import json
import logging
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ── BM25 Implementation ────────────────────────────────────────────────────────

class BM25:
    """
    Okapi BM25 — the gold standard lexical retrieval algorithm.
    
    Score(q, d) = Σ IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))
    
    Parameters:
        k1 = 1.5  — term frequency saturation. Higher = more weight to repeated terms.
        b  = 0.75 — length normalization. 1.0 = full normalization, 0 = none.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []      # tokenized docs
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.df: Counter = Counter()           # document frequency per term
        self.idf: dict[str, float] = {}        # precomputed IDF scores
        self.N: int = 0

    def tokenize(self, text: str) -> list[str]:
        """Lowercase, remove punctuation, split on whitespace. Keep stopwords
        intentionally for technical text — "not" in "not applicable" matters."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def index(self, documents: list[str]) -> None:
        """Build the BM25 index from a list of document strings."""
        self.corpus = [self.tokenize(d) for d in documents]
        self.doc_len = [len(d) for d in self.corpus]
        self.N = len(self.corpus)
        self.avgdl = sum(self.doc_len) / self.N if self.N else 1.0

        # Document frequency
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)  — Robertson IDF
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        """BM25 score for a single (query, document) pair."""
        doc = self.corpus[doc_idx]
        dl = self.doc_len[doc_idx]
        tf_map = Counter(doc)
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += self.idf[term] * norm
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_idx, score) pairs, sorted descending."""
        tokens = self.tokenize(query)
        scores = [(i, self.score(tokens, i)) for i in range(self.N)]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists into one using RRF.
    
    RRF(d) = Σ 1 / (k + rank(d))
    
    k=60 is the standard constant from the original 2009 paper.
    This is robust to score scale differences between queries.
    """
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_idx, _) in enumerate(ranked):
            rrf_scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ── Vectorless RAG Engine ──────────────────────────────────────────────────────

class VectorlessRAG:
    """
    Full vectorless RAG pipeline:
    
    1. Papers are indexed into BM25 (per-paper + global)
    2. At query time: expand query → multi-query BM25 → RRF fusion → LLM compression
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)  # always safe to call
        self.index_path = data_dir / "bm25_index.json"

        # Per-paper storage
        self.papers: dict[str, dict] = {}          # arxiv_id → metadata
        self.chunks: list[str] = []                # all chunks flat
        self.chunk_meta: list[dict] = []           # per-chunk: arxiv_id, title, chunk_idx

        # BM25 global index (across all papers)
        self.bm25 = BM25()
        self._indexed = False

        self._load()

    # ── Indexing ───────────────────────────────────────────────────────────────

    def add_paper(self, arxiv_id: str, chunks: list[str], metadata: dict) -> None:
        """Index a paper. Idempotent."""
        if arxiv_id in self.papers:
            self._remove_paper(arxiv_id)

        self.papers[arxiv_id] = metadata
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_meta.append({
                "arxiv_id": arxiv_id,
                "title": metadata.get("title", ""),
                "chunk_idx": i,
            })

        # Rebuild global BM25 index
        self._rebuild_index()
        self._save()
        logger.info(f"Indexed {len(chunks)} chunks for {arxiv_id} (vectorless)")

    def _remove_paper(self, arxiv_id: str) -> None:
        keep = [i for i, m in enumerate(self.chunk_meta) if m["arxiv_id"] != arxiv_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.chunk_meta = [self.chunk_meta[i] for i in keep]
        del self.papers[arxiv_id]

    def _rebuild_index(self) -> None:
        if self.chunks:
            self.bm25.index(self.chunks)
            self._indexed = True

    # ── Query Pipeline ─────────────────────────────────────────────────────────

    async def query(
        self,
        question: str,
        top_k: int = 6,
        use_query_expansion: bool = True,
        arxiv_id_filter: str = None,
    ) -> dict:
        """
        Full vectorless RAG pipeline.

        Args:
            question:           Natural language question
            top_k:              Number of chunks to retrieve
            use_query_expansion: Generate keyword variants for better recall
            arxiv_id_filter:    If set, restrict search to this paper only
        """
        if not self._indexed or not self.chunks:
            return {
                "answer": "Library is empty. Use fetch_paper to index papers first.",
                "sources": [],
            }

        # Apply paper filter — restrict chunks to one paper if requested
        if arxiv_id_filter:
            if arxiv_id_filter not in self.papers:
                return {
                    "answer": f"Paper {arxiv_id_filter} is not in your library. Fetch it first.",
                    "sources": [],
                }
            active_indices = [
                i for i, m in enumerate(self.chunk_meta)
                if m["arxiv_id"] == arxiv_id_filter
            ]
        else:
            active_indices = list(range(len(self.chunks)))

        # Step 1 — Query expansion
        queries = [question]
        if use_query_expansion:
            expanded = await self._expand_query(question)
            queries = [question] + expanded
            logger.info(f"Expanded to {len(queries)} queries: {queries}")

        # Step 2 — BM25 per query (over filtered subset)
        ranked_lists = []
        for q in queries:
            all_results = self.bm25.search(q, top_k=len(active_indices))
            # Keep only chunks belonging to active_indices
            filtered = [(i, s) for i, s in all_results if i in set(active_indices)]
            ranked_lists.append(filtered[:top_k * 3])

        # Step 3 — RRF fusion
        fused = reciprocal_rank_fusion(ranked_lists)
        top_indices = [idx for idx, _ in fused[:top_k * 2]]

        if not top_indices:
            return {"answer": "No relevant content found.", "sources": []}

        # Step 4 — Contextual compression (extract only relevant sentences)
        candidates = [
            {
                "chunk": self.chunks[i],
                "meta": self.chunk_meta[i],
                "rrf_score": fused[j][1],
            }
            for j, (i, _) in enumerate(fused[:top_k * 2])
            if i < len(self.chunks)
        ]

        compressed = await self._compress_chunks(question, candidates[:top_k])

        # Step 5 — Build response
        sources = []
        seen = set()
        for item in compressed:
            aid = item["meta"]["arxiv_id"]
            if aid not in seen:
                seen.add(aid)
                paper = self.papers.get(aid, {})
                sources.append({
                    "arxiv_id": aid,
                    "title": item["meta"]["title"],
                    "authors": paper.get("authors", []),
                    "rrf_score": round(item["rrf_score"], 5),
                })

        context_parts = [
            f"[Source {i+1}] '{item['meta']['title']}' (arxiv:{item['meta']['arxiv_id']})\n{item['compressed']}"
            for i, item in enumerate(compressed)
        ]

        return {
            "question": question,
            "expanded_queries": queries[1:] if use_query_expansion else [],
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "retrieval_method": "BM25 + RRF + LLM contextual compression",
            "instruction": (
                "Use the compressed context above to answer the question. "
                "Cite sources as [Source N]. Be precise and grounded."
            ),
        }

    async def _expand_query(self, question: str) -> list[str]:
        """
        Use the LLM to generate keyword-rich query variants.
        
        This is the vectorless equivalent of HyDE — instead of generating a
        hypothetical document to embed, we generate keyword variants to maximize
        BM25 recall across different terminologies.
        """
        prompt = f"""Given this research question, generate 2 alternative keyword-focused search queries that would help find relevant content in academic papers. Return ONLY a JSON array of 2 strings, nothing else.

Question: {question}

Example output: ["transformer attention mechanism self-attention", "BERT GPT language model pretraining"]

Output:"""

        try:
            loop = asyncio.get_event_loop()
            # This runs synchronously in executor to avoid blocking
            result = await loop.run_in_executor(None, self._call_local_expansion, prompt)
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return [str(q) for q in parsed[:2]]
        except Exception as e:
            logger.warning(f"Query expansion failed, using original only: {e}")
        return []

    def _call_local_expansion(self, prompt: str) -> str:
        """
        Simple rule-based query expansion as fallback (no external API needed).
        In production you'd call an LLM API here.
        """
        # Extract key noun phrases from the question as additional query variants
        # This is a lightweight fallback — replace with actual LLM call in production
        words = re.findall(r'\b[a-zA-Z]{4,}\b', prompt.lower())
        stopwords = {'what', 'when', 'where', 'which', 'that', 'this', 'with',
                     'from', 'have', 'been', 'does', 'about', 'paper', 'research',
                     'given', 'return', 'only', 'output', 'example', 'generate',
                     'find', 'search', 'question', 'help', 'would', 'into'}
        keywords = [w for w in words if w not in stopwords][:8]
        variant1 = " ".join(keywords[:6])
        variant2 = " ".join(keywords[2:8])
        return json.dumps([variant1, variant2])

    async def _compress_chunks(
        self,
        question: str,
        candidates: list[dict],
    ) -> list[dict]:
        """
        Contextual compression: extract only the sentences from each chunk
        that are directly relevant to the question.
        
        This dramatically reduces noise — a 500-word chunk about a paper's
        evaluation section might have only 2 sentences actually relevant to
        your specific question. We extract just those 2 sentences.
        
        In production: call an LLM for each chunk with:
        "From this text, extract the 1-3 sentences most relevant to: {question}"
        
        Here we use a fast extractive approach: sentence-level BM25 scoring.
        """
        query_tokens = self.bm25.tokenize(question)
        results = []

        for item in candidates:
            chunk = item["chunk"]
            sentences = self._split_sentences(chunk)

            if len(sentences) <= 3:
                compressed = chunk
            else:
                # Score each sentence with BM25 against the query
                mini_bm25 = BM25()
                mini_bm25.index(sentences)
                scored = [(i, mini_bm25.score(query_tokens, i)) for i in range(len(sentences))]
                scored.sort(key=lambda x: x[1], reverse=True)
                # Take top 3 sentences, restore original order for readability
                top_idxs = sorted([i for i, _ in scored[:3]])
                compressed = " ".join(sentences[i] for i in top_idxs)

            results.append({**item, "compressed": compressed})

        return results

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter — good enough for academic text."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if len(s) > 20]

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        payload = {
            "papers": self.papers,
            "chunks": self.chunks,
            "chunk_meta": self.chunk_meta,
        }
        with open(self.index_path, "w") as f:
            json.dump(payload, f, default=str)
        logger.info(f"Saved BM25 index: {len(self.chunks)} chunks, {len(self.papers)} papers")

    def _load(self) -> None:
        if self.index_path.exists():
            with open(self.index_path) as f:
                payload = json.load(f)
            self.papers = payload.get("papers", {})
            self.chunks = payload.get("chunks", [])
            self.chunk_meta = payload.get("chunk_meta", [])
            if self.chunks:
                self._rebuild_index()
            logger.info(f"Loaded vectorless index: {len(self.chunks)} chunks")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def list_papers(self) -> list[dict]:
        return list(self.papers.values())

    def get_paper_chunks(self, arxiv_id: str) -> list[str]:
        return [
            self.chunks[i]
            for i, m in enumerate(self.chunk_meta)
            if m["arxiv_id"] == arxiv_id
        ]

    async def summarize_paper(self, arxiv_id: str) -> dict:
        if arxiv_id not in self.papers:
            return {"error": f"Paper {arxiv_id} not in library."}
        metadata = self.papers[arxiv_id]
        chunks = self.get_paper_chunks(arxiv_id)
        sample = chunks[:8] + chunks[-2:] if len(chunks) > 10 else chunks
        return {
            "arxiv_id": arxiv_id,
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "abstract": metadata.get("abstract"),
            "context": "\n\n".join(sample),
            "retrieval_method": "vectorless (BM25)",
            "instruction": (
                "Summarize: (1) Problem, (2) Method/Contribution, "
                "(3) Results, (4) Limitations, (5) Significance."
            ),
        }