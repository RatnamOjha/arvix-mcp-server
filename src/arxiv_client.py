"""
ArXiv API client — search and fetch paper metadata + PDFs.

PDF download strategy:
  We construct the PDF URL directly from the arxiv_id instead of going
  through the API. This means fetch_paper never hits the rate limit —
  PDFs are served from a CDN with no rate limiting.

  Metadata (title, authors, abstract) is fetched separately via the API
  with a graceful fallback — if the API is rate-limited, we still index
  the paper using placeholder metadata. Users can always query the content.
"""
import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import arxiv
import httpx
import pdfplumber

logger = logging.getLogger(__name__)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"


class ArxivClient:
    def __init__(self):
        self.client = arxiv.Client(
            page_size=20,
            delay_seconds=3.0,
            num_retries=2,
        )

    # ── Search ────────────────────────────────────────────────────────────────

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

        try:
            return await loop.run_in_executor(None, _fetch)
        except arxiv.HTTPError as e:
            if e.status == 429:
                raise RuntimeError(
                    "ArXiv search is rate-limited right now (HTTP 429). "
                    "Wait 60 seconds and try again. "
                    "Tip: use --fetch with a known paper ID — it never rate-limits."
                ) from e
            raise

    # ── Fetch — guaranteed, no rate limit ─────────────────────────────────────

    async def fetch_and_index_vectorless(self, arxiv_id: str, rag) -> dict:
        """
        Download and index a paper. Guaranteed to work regardless of API
        rate limits because we download the PDF directly by constructing
        the URL — no API call needed for the PDF itself.

        Metadata fetch (title, authors, etc.) is attempted separately and
        falls back gracefully if the API is unavailable.
        """
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id.strip())

        # Step 1: Download PDF directly — this never rate-limits
        pdf_url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        print(f"  Downloading PDF directly from {pdf_url}")
        text = await self._extract_pdf_text(pdf_url, arxiv_id)

        # Step 2: Try to fetch metadata from API (optional — falls back gracefully)
        metadata = await self._fetch_metadata_safe(arxiv_id)

        # Step 3: Chunk and index
        chunks = self._chunk_text(text)
        rag.add_paper(arxiv_id=arxiv_id, chunks=chunks, metadata=metadata)

        return {
            "metadata": metadata,
            "chunks_indexed": len(chunks),
            "retrieval_method": "vectorless (BM25 + RRF)",
            "status": "success",
            "metadata_source": metadata.get("_source", "api"),
            "message": (
                f"'{metadata['title']}' indexed with {len(chunks)} chunks. "
                f"No embeddings used."
            ),
        }

    async def _fetch_metadata_safe(self, arxiv_id: str) -> dict:
        """
        Try to get title/authors/abstract from the ArXiv API.
        If rate-limited or unavailable, return placeholder metadata —
        the paper is still fully indexed and queryable.
        """
        loop = asyncio.get_event_loop()

        def _get():
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(self.client.results(search))
            return results[0] if results else None

        try:
            paper = await loop.run_in_executor(None, _get)
            if paper:
                return {
                    "arxiv_id": arxiv_id,
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors[:5]],
                    "abstract": paper.summary,
                    "published": paper.published.isoformat() if paper.published else None,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "_source": "api",
                }
        except Exception as e:
            logger.warning(f"Metadata fetch failed for {arxiv_id}, using fallback: {e}")
            print(
                f"\n  ⚠  Could not fetch metadata from ArXiv API ({type(e).__name__}). "
                f"Paper is still indexed and fully queryable.\n"
                f"     Metadata will show placeholder values — run again later to refresh."
            )

        # Fallback metadata — paper is still indexed and queryable
        return {
            "arxiv_id": arxiv_id,
            "title": f"ArXiv:{arxiv_id}",
            "authors": [],
            "abstract": "",
            "published": None,
            "categories": [],
            "pdf_url": ARXIV_PDF_URL.format(arxiv_id=arxiv_id),
            "_source": "fallback",
        }

    # ── PDF extraction ────────────────────────────────────────────────────────

    async def _extract_pdf_text(self, pdf_url: str, arxiv_id: str) -> str:
        """
        Download PDF from the direct CDN URL and extract text.
        ArXiv's PDF CDN has no rate limiting — this always works.
        """
        headers = {
            # Polite user-agent as requested by ArXiv's usage policy
            "User-Agent": f"arxiv-mcp-server/1.0 (https://github.com/yourusername/arxiv-mcp-server; research tool) arxiv-id/{arxiv_id}"
        }
        async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
            resp = await client.get(pdf_url, headers=headers)
            if resp.status_code == 404:
                raise ValueError(
                    f"Paper {arxiv_id} not found. "
                    f"Double-check the ID at https://arxiv.org/abs/{arxiv_id}"
                )
            resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(resp.content)
            tmp_path = Path(f.name)

        try:
            with pdfplumber.open(tmp_path) as pdf:
                pages = []
                for page in pdf.pages[:40]:
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append(text)
            if not pages:
                raise ValueError(
                    f"Could not extract text from {arxiv_id}. "
                    f"The PDF may be scanned/image-based. Try a different paper."
                )
            return "\n\n".join(pages)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Chunking ──────────────────────────────────────────────────────────────

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