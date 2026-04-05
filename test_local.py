#!/usr/bin/env python3
"""
test_local.py — test the ArXiv MCP server locally without Claude Pro.

Usage:
    python test_local.py                  # run all tests
    python test_local.py --search         # only test ArXiv search
    python test_local.py --fetch          # fetch + index a paper
    python test_local.py --query          # query your library
    python test_local.py --arxiv-id 2005.11401  # fetch a specific paper
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Make sure src/ is importable regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path.home() / ".arxiv-mcp"
DEFAULT_PAPER = "2005.11401"   # Original RAG paper — Lewis et al. 2020


# ── Helpers ───────────────────────────────────────────────────────────────────

def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def info(msg: str) -> None:
    print(f"     {msg}")


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_search(query: str = "retrieval augmented generation", n: int = 3):
    header(f"ArXiv Search — '{query}'")
    from src.arxiv_client import ArxivClient
    client = ArxivClient()
    results = await client.search(query, max_results=n)
    ok(f"Found {len(results)} papers")
    for r in results:
        info(f"[{r['arxiv_id']}] {r['title'][:70]}")
        info(f"       Authors: {', '.join(r['authors'][:2])}")
        info(f"       Published: {r['published'][:10] if r['published'] else 'unknown'}")
        print()
    return results


async def test_fetch(arxiv_id: str = DEFAULT_PAPER):
    header(f"Fetch & Index Paper — arxiv:{arxiv_id}")
    from src.arxiv_client import ArxivClient
    from src.vectorless_rag import VectorlessRAG

    client = ArxivClient()
    rag = VectorlessRAG(data_dir=DATA_DIR)

    print(f"  Downloading PDF and building BM25 index...")
    print(f"  (this takes 10-20 seconds)")
    try:
        result = await client.fetch_and_index_vectorless(arxiv_id, rag)
    except RuntimeError as e:
        print(f"\n  ✗  {e}")
        return None
    except Exception as e:
        print(f"\n  ✗  Unexpected error: {e}")
        return None

    ok(result["message"])
    ok(f"Retrieval method: {result['retrieval_method']}")
    info(f"Title:   {result['metadata']['title']}")
    info(f"Authors: {', '.join(result['metadata']['authors'][:3])}")
    info(f"Chunks:  {result['chunks_indexed']}")
    return result


async def test_query(question: str = "what retrieval method does RAG use?"):
    header(f"Vectorless RAG Query")
    from src.vectorless_rag import VectorlessRAG

    rag = VectorlessRAG(data_dir=DATA_DIR)

    if not rag.papers:
        print("  ! Library is empty — run with --fetch first")
        print("    python test_local.py --fetch")
        return

    print(f"  Question: {question}\n")
    result = await rag.query(question, top_k=4)

    ok("Query expanded into BM25 variants:")
    for q in result.get("expanded_queries", []):
        info(f"  › {q}")

    print()
    ok("Sources retrieved:")
    for s in result["sources"]:
        info(f"  [{s['arxiv_id']}] {s['title'][:60]}")
        info(f"           RRF score: {s['rrf_score']}")

    print()
    ok("Compressed context (what Claude would receive):")
    ctx = result["context"][:600]
    for line in ctx.split("\n"):
        info(line)
    if len(result["context"]) > 600:
        info("  ... (truncated)")

    print()
    ok(f"Retrieval method: {result['retrieval_method']}")


async def test_library():
    header("Library Contents")
    from src.vectorless_rag import VectorlessRAG
    from src.reading_list import ReadingList

    rag = VectorlessRAG(data_dir=DATA_DIR)
    rl = ReadingList(data_dir=DATA_DIR)

    papers = rag.list_papers()
    ok(f"{len(papers)} paper(s) indexed in BM25 store")
    for p in papers:
        info(f"  [{p['arxiv_id']}] {p['title'][:60]}")

    reading = rl.get_all()
    print()
    ok(f"{len(reading)} paper(s) in reading list")
    for r in reading:
        status = "read" if r.get("read") else "unread"
        info(f"  [{r['arxiv_id']}] {r['title'][:55]}  ({status})")


async def run_all(arxiv_id: str = DEFAULT_PAPER):
    print("\n ArXiv MCP Server — Local Test Suite")
    print(" No Claude Pro required\n")

    await test_search()
    await test_fetch(arxiv_id)
    await test_query()
    await test_library()

    header("All tests complete")
    print("  Your library is saved at:", DATA_DIR)
    print("  Run again anytime — indexed papers persist between runs.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ArXiv MCP server locally")
    parser.add_argument("--search", action="store_true", help="Test ArXiv search only")
    parser.add_argument("--fetch", action="store_true", help="Fetch and index a paper")
    parser.add_argument("--query", action="store_true", help="Query your library")
    parser.add_argument("--library", action="store_true", help="Show library contents")
    parser.add_argument("--arxiv-id", default=DEFAULT_PAPER, help="ArXiv ID to fetch")
    parser.add_argument("--question", default="what retrieval method does RAG use?", help="Question to ask")
    args = parser.parse_args()

    if args.search:
        asyncio.run(test_search())
    elif args.fetch:
        asyncio.run(test_fetch(args.arxiv_id))
    elif args.query:
        asyncio.run(test_query(args.question))
    elif args.library:
        asyncio.run(test_library())
    else:
        asyncio.run(run_all(args.arxiv_id))