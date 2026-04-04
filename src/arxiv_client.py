import asyncio, logging, re, tempfile
from pathlib import Path
from typing import Optional
import arxiv, httpx, pdfplumber

logger = logging.getLogger(__name__)

class ArxivClient:
    def __init__(self):
        self.client = arxiv.Client(page_size=20, delay_seconds=1.0, num_retries=3)

    async def search(self, query: str, max_results: int = 8, category: Optional[str] = None) -> list[dict]:
        if category:
            query = f"cat:{category} AND ({query})"
        search = arxiv.Search(query=query, max_results=min(max_results, 20), sort_by=arxiv.SortCriterion.Relevance)
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
        return await loop.run_in_executor(None, _fetch)

    async def fetch_and_index_vectorless(self, arxiv_id: str, rag) -> dict:
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        search = arxiv.Search(id_list=[arxiv_id])
        loop = asyncio.get_event_loop()
        def _get_meta():
            results = list(self.client.results(search))
            if not results:
                raise ValueError(f"Paper {arxiv_id} not found")
            return results[0]
        paper = await loop.run_in_executor(None, _get_meta)
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

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i: i + chunk_size])
            if len(chunk.strip()) > 100:
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks
