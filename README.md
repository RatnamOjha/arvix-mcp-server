# ArXiv Research Intelligence MCP Server

> Semantic search, full RAG, and cross-encoder reranking over ArXiv — as an MCP server for Claude.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What this is

An MCP (Model Context Protocol) server that gives Claude a full ML-powered research assistant:

- **Semantic search** over ArXiv via the live API
- **PDF ingestion** — download, extract, chunk, and embed any paper
- **RAG queries** — ask research questions, get cited answers from your library
- **Cross-encoder reranking** — precision retrieval using ms-marco-MiniLM
- **Reading list** — persistent JSON-backed saved papers with notes

## Architecture

```
Claude (MCP client)
        │  tool calls via stdio
        ▼
  MCP Server (FastMCP / Python)
  ├── search_papers   → ArXiv REST API
  ├── fetch_paper     → PDF download → pdfplumber → chunker → embedder → VectorStore
  ├── query_library   → embed query → cosine search → cross-encoder → cited context
  ├── summarize_paper → chunk retrieval → structured prompt
  ├── list_library    → VectorStore metadata
  └── get_reading_list→ JSON ReadingList
        │
        ├── Embedder (sentence-transformers/all-MiniLM-L6-v2, 384-dim)
        ├── VectorStore (numpy + pickle, swappable for Qdrant/ChromaDB)
        ├── CrossEncoder (ms-marco-MiniLM-L-6-v2)
        └── ReadingList (JSON persistence)
```

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/arxiv-mcp-server
cd arxiv-mcp-server
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

> First run downloads ~90MB of embedding models automatically.

### 2. Add to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "arxiv-research": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/absolute/path/to/arxiv-mcp-server"
    }
  }
}
```

### 3. Restart Claude Desktop

The arxiv-research tools will appear in Claude's tool panel.

### 4. Try it

```
"Search for papers on speculative decoding from the last year"
"Fetch paper 2305.10601 and add it to my library"
"What does my library say about KV cache optimization?"
"Summarize paper 2305.10601"
"Show me my reading list"
```

## MCP Tools

| Tool | Args | Description |
|------|------|-------------|
| `search_papers` | `query`, `max_results?`, `category?` | Live ArXiv search |
| `fetch_paper` | `arxiv_id`, `add_to_reading_list?` | Download + index full paper |
| `query_library` | `question`, `top_k?` | RAG over your indexed papers |
| `summarize_paper` | `arxiv_id` | Structured paper summary |
| `list_library` | — | All indexed papers |
| `get_reading_list` | — | Saved reading list |

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| MCP protocol | `mcp` | stdio server framework |
| ArXiv API | `arxiv` | Paper search + metadata |
| Embedding | `sentence-transformers` | all-MiniLM-L6-v2, 384-dim |
| Reranking | `sentence-transformers` | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF extraction | `pdfplumber` | Text extraction from PDFs |
| Vector ops | `numpy` | Cosine similarity search |
| HTTP client | `httpx` | Async PDF downloads |
| Persistence | pickle + JSON | Local vector + reading list store |

## Upgrading to a Production Vector DB

The `VectorStore` class has a simple interface — swap it out for Qdrant in two steps:

```python
# Install: pip install qdrant-client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Replace numpy store with Qdrant — same add_paper() / search() interface
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v --asyncio-mode=auto
ruff check src/
mypy src/
```

## Data storage

All data is stored at `~/.arxiv-mcp/`:
- `vectors.pkl` — chunk embeddings
- `papers.json` — paper metadata index
- `reading_list.json` — reading list

## License

MIT
