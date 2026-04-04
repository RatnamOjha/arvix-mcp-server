"""
ArXiv Research Intelligence MCP Server
Now using Vectorless RAG — BM25 + RRF + LLM contextual compression.
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
DATA_DIR.mkdir(exist_ok=True)

server = Server("arxiv-research")

arxiv = ArxivClient()
rag = VectorlessRAG(data_dir=DATA_DIR)
reading_list = ReadingList(data_dir=DATA_DIR)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_papers",
            description="Search ArXiv for research papers. Returns ranked metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 8},
                    "category": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="fetch_paper",
            description="Download and index a paper via vectorless BM25 indexing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "add_to_reading_list": {"type": "boolean", "default": True},
                },
                "required": ["arxiv_id"],
            },
        ),
        Tool(
            name="query_library",
            description="BM25 + RRF + contextual compression RAG query. No embeddings needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {"type": "integer", "default": 6},
                    "use_query_expansion": {"type": "boolean", "default": True},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="list_library",
            description="List all indexed papers.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_reading_list",
            description="Get your saved reading list.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="summarize_paper",
            description="Structured summary of a paper in your library.",
            inputSchema={
                "type": "object",
                "properties": {"arxiv_id": {"type": "string"}},
                "required": ["arxiv_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_papers":
            results = await arxiv.search(
                query=arguments["query"],
                max_results=arguments.get("max_results", 8),
                category=arguments.get("category"),
            )
            return [TextContent(type="text", text=json.dumps(results, indent=2))]

        elif name == "fetch_paper":
            arxiv_id = arguments["arxiv_id"].strip().replace("https://arxiv.org/abs/", "")
            result = await arxiv.fetch_and_index_vectorless(arxiv_id=arxiv_id, rag=rag)
            if arguments.get("add_to_reading_list", True):
                reading_list.add(result["metadata"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "query_library":
            answer = await rag.query(
                question=arguments["question"],
                top_k=arguments.get("top_k", 6),
                use_query_expansion=arguments.get("use_query_expansion", True),
            )
            return [TextContent(type="text", text=json.dumps(answer, indent=2))]

        elif name == "list_library":
            return [TextContent(type="text", text=json.dumps(rag.list_papers(), indent=2))]

        elif name == "get_reading_list":
            return [TextContent(type="text", text=json.dumps(reading_list.get_all(), indent=2))]

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
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
