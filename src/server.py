"""
ArXiv Research Intelligence MCP Server
Vectorless RAG — BM25 + RRF + LLM contextual compression.
No embeddings, no GPU, no model downloads for retrieval.
"""

import asyncio
import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .arxiv_client import ArxivClient
from .vectorless_rag import VectorlessRAG
from .reading_list import ReadingList

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path.home() / ".arxiv-mcp"
DATA_DIR.mkdir(parents=True, exist_ok=True)

server = Server("arxiv-research")

arxiv_client = ArxivClient()
rag = VectorlessRAG(data_dir=DATA_DIR)
reading_list = ReadingList(data_dir=DATA_DIR)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_papers",
            description=(
                "Search ArXiv for research papers by natural language query. "
                "Returns ranked results with titles, authors, abstracts, and IDs. "
                "Use this to discover papers before fetching them."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query e.g. 'speculative decoding LLMs'",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 8,
                        "description": "Number of results to return (max 20)",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional ArXiv category filter e.g. cs.LG, cs.CL, stat.ML",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="fetch_paper",
            description=(
                "Download and index a paper by ArXiv ID. "
                "Downloads the PDF directly (no rate limiting), extracts text, "
                "chunks it, and builds a BM25 index. No embeddings needed. "
                "After fetching, use query_library to ask questions about it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "ArXiv paper ID e.g. '2305.10601' or full URL https://arxiv.org/abs/2305.10601",
                    },
                    "add_to_reading_list": {
                        "type": "boolean",
                        "default": True,
                        "description": "Save to reading list for later reference",
                    },
                },
                "required": ["arxiv_id"],
            },
        ),
        Tool(
            name="query_library",
            description=(
                "Ask a research question over your indexed paper library using vectorless RAG. "
                "Pipeline: query expansion → multi-query BM25 → RRF fusion → contextual compression → cited answer. "
                "Use arxiv_id_filter to restrict the search to one specific paper, "
                "or leave it empty to search across your entire library."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Research question to answer from your library",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 6,
                        "description": "Number of chunks to retrieve",
                    },
                    "use_query_expansion": {
                        "type": "boolean",
                        "default": True,
                        "description": "Generate keyword variants to improve BM25 recall",
                    },
                    "arxiv_id_filter": {
                        "type": "string",
                        "description": (
                            "Optional. Restrict search to one specific paper e.g. '2507.07171'. "
                            "Leave empty to search all papers in the library."
                        ),
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="list_library",
            description=(
                "List all papers currently indexed in your local BM25 library. "
                "Shows titles, authors, categories, and chunk counts."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_reading_list",
            description=(
                "Get your saved reading list with metadata, timestamps, "
                "read/unread status, and personal notes."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="summarize_paper",
            description=(
                "Generate a structured summary of a specific paper in your library. "
                "Covers: problem, method/contribution, results, limitations, significance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "ArXiv ID of the paper to summarize",
                    },
                },
                "required": ["arxiv_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_papers":
            results = await arxiv_client.search(
                query=arguments["query"],
                max_results=arguments.get("max_results", 8),
                category=arguments.get("category"),
            )
            return [TextContent(type="text", text=json.dumps(results, indent=2))]

        elif name == "fetch_paper":
            arxiv_id = (
                arguments["arxiv_id"]
                .strip()
                .replace("https://arxiv.org/abs/", "")
                .replace("http://arxiv.org/abs/", "")
            )
            result = await arxiv_client.fetch_and_index_vectorless(
                arxiv_id=arxiv_id, rag=rag
            )
            if arguments.get("add_to_reading_list", True):
                reading_list.add(result["metadata"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "query_library":
            answer = await rag.query(
                question=arguments["question"],
                top_k=arguments.get("top_k", 6),
                use_query_expansion=arguments.get("use_query_expansion", True),
                arxiv_id_filter=arguments.get("arxiv_id_filter") or None,
            )
            return [TextContent(type="text", text=json.dumps(answer, indent=2))]

        elif name == "list_library":
            papers = rag.list_papers()
            return [TextContent(type="text", text=json.dumps(papers, indent=2))]

        elif name == "get_reading_list":
            papers = reading_list.get_all()
            return [TextContent(type="text", text=json.dumps(papers, indent=2))]

        elif name == "summarize_paper":
            summary = await rag.summarize_paper(arguments["arxiv_id"])
            return [TextContent(type="text", text=json.dumps(summary, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())