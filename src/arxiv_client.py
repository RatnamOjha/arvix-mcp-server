"""
ArXiv API client — search and fetch paper metadata + PDFs.
"""
import asyncio
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import arxiv
import httpx
import pdfplumber

logger = logging.getLogger(__name__)

# ArXiv asks for max 1 req/3s. We wait between retries on 429.
_RATE_LIMIT_WAIT = 10   # seconds to wait on first 429
_MAX_RETRIES = 4


class ArxivClient:
    def __init__(self):
        self.client = arxiv.Client(
            page_size=20,
            delay_seconds=3.0,   # increased from 1.0 — respects ArXiv's rate limit
            num_retries=3,
        )

    async def search(
        self,
        query: str,
        max_results: int = 8,
        category: Optional[str] = None,
    ) -> list[dict]:
        if category:
            query = f"cat:{category} AND ({query})"
        search = arxiv.Search(
            query=query,
            max_results=min(max_results, 20),
            sort_by=arxiv.SortCriterion.Relevance,
        )
        loop = asyncio.get_event_loop()

        def _fetch():
            papers = []
            for paper in self.client.results(search):
                papers.append({
                    "arxiv_id": paper.entry_id.split("/abs/")[-1],
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors[:5]],
                    "abstract": paper.summary[:800],
                    "published": paper.published.isoformat() if paper.published else None,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                })
            return papers

        return await self._with_retry(loop, _fetch)

    async def fetch_and_index_vectorless(self, arxiv_id: str, rag) -> dict:
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        search = arxiv.Search(id_list=[arxiv_id])
        loop = asyncio.get_event_loop()

        def _get_meta():
            results = list(self.client.results(search))
            if not results:
                raise ValueError(f"Paper {arxiv_id} not found on ArXiv")
            return results[0]

        paper = await self._with_retry(loop, _get_meta)

        metadata = {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "authors": [a.name for a in paper.authors[:5]],
            "abstract": paper.summary,
            "published": paper.published.isoformat() if paper.published else None,
            "categories": paper.categories,
            "pdf_url": paper.pdf_url,
        }

        text = await self._extract_pdf_text(paper.pdf_url)
        chunks = self._chunk_text(text)
        rag.add_paper(arxiv_id=arxiv_id, chunks=chunks, metadata=metadata)

        return {
            "metadata": metadata,
            "chunks_indexed": len(chunks),
            "retrieval_method": "vectorless (BM25 + RRF)",
            "status": "success",
            "message": f"'{paper.title}' indexed with {len(chunks)} chunks. No embeddings used.",
        }

    async def _with_retry(self, loop, fn):
        """
        Run a blocking ArXiv API call in an executor with exponential backoff
        on HTTP 429 (rate limit) responses.
        """
        wait = _RATE_LIMIT_WAIT
        for attempt in range(_MAX_RETRIES):
            try:
                return await loop.run_in_executor(None, fn)
            except arxiv.HTTPError as e:
                if e.status == 429:
                    if attempt < _MAX_RETRIES - 1:
                        logger.warning(
                            f"ArXiv rate limit hit (429). "
                            f"Waiting {wait}s before retry {attempt + 1}/{_MAX_RETRIES - 1}..."
                        )
                        print(f"\n  ⚠  ArXiv rate limit hit. Waiting {wait}s and retrying... "
                              f"({attempt + 1}/{_MAX_RETRIES - 1})")
                        await asyncio.sleep(wait)
                        wait *= 2   # exponential backoff: 10s → 20s → 40s
                    else:
                        raise RuntimeError(
                            f"ArXiv rate limit (HTTP 429) persists after {_MAX_RETRIES} retries. "
                            f"Please wait a minute and try again.\n"
                            f"Tip: avoid running multiple fetch/search commands in quick succession."
                        ) from e
                else:
                    raise

    async def _extract_pdf_text(self, pdf_url: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(resp.content)
            tmp_path = Path(f.name)
        try:
            with pdfplumber.open(tmp_path) as pdf:
                pages = [p.extract_text() for p in pdf.pages[:40] if p.extract_text()]
            return "\n\n".join(pages)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> list[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i: i + chunk_size])
            if len(chunk.strip()) > 100:
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks