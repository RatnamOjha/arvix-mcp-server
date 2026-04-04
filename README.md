# ArXiv Research Intelligence — MCP Server

> Semantic paper search, vectorless RAG, and BM25 retrieval over ArXiv — as an MCP server for Claude.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What changed in v2 — Vectorless RAG

The original version used `sentence-transformers` to embed every chunk into a 384-dimensional vector, then retrieved by cosine similarity. **v2 replaces this entirely** with a vectorless pipeline:

| | v1 (vector RAG) | v2 (vectorless RAG) |
|---|---|---|
| Indexing | encode every chunk with neural model | pure text, instant |
| Retrieval | cosine similarity over embeddings | Okapi BM25 |
| Multi-query | single query only | 3 variants + RRF fusion |
| Compression | cross-encoder reranker | sentence-level BM25 |
| GPU required | yes (or slow CPU) | no |
| Model download | ~90MB on first run | none |
| Handles exact terms (LoRA, RLHF) | sometimes misses | always catches |

### How vectorless RAG works

```
Your question
     │
     ▼
Query expansion ── LLM rewrites into 3 keyword variants
     │
     ▼
BM25 retrieval ── each variant scored against all chunks independently
     │
     ▼
RRF fusion ── Reciprocal Rank Fusion combines 3 ranked lists
              score = Σ 1/(60 + rank)  — robust, no score normalization needed
     │
     ▼
Contextual compression ── sentence-level BM25 extracts only the
                          2-3 sentences per chunk that answer the question
     │
     ▼
Cited context returned to Claude
```

---

## No Claude Pro? No problem

You can use the full pipeline — search, fetch, index, and query — directly from your terminal without Claude Desktop at all.

### Run tests locally

```bash
# Clone and install
git clone https://github.com/yourusername/arxiv-mcp-server
cd arxiv-mcp-server
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run the full test suite (no Claude needed)
python test_local.py

# Or test individual parts
python test_local.py --search                        # live ArXiv search
python test_local.py --fetch --arxiv-id 2005.11401  # fetch the RAG paper
python test_local.py --query --question "what is BM25?"
python test_local.py --library                       # see what's indexed
```

What `test_local.py` shows you:
- **Search** — live ArXiv results for any query
- **Fetch** — PDF download → text extraction → BM25 indexing (no embeddings)
- **Query** — expanded queries, RRF-fused sources, compressed context
- **Library** — all indexed papers and your reading list

Everything persists between runs at `~/.arxiv-mcp/`.

---

## With Claude Pro — MCP integration

### Install

```bash
git clone https://github.com/yourusername/arxiv-mcp-server
cd arxiv-mcp-server
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arxiv-research": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "src.server"],
      "cwd": "/absolute/path/to/arxiv-mcp-server"
    }
  }
}
```

> Use the full absolute path to your `.venv` Python to avoid conflicts with Anaconda or system Python.

Restart Claude Desktop. The 🔨 hammer icon will show 6 new tools.

### Example prompts

```
Search for recent papers on speculative decoding
Fetch paper 2305.10601 and add it to my library
What does my library say about KV cache optimization?
Summarize paper 2305.10601
Show me my reading list
```

---

## MCP Tools

| Tool | Args | Description |
|------|------|-------------|
| `search_papers` | `query`, `max_results?`, `category?` | Live ArXiv search |
| `fetch_paper` | `arxiv_id`, `add_to_reading_list?` | Download + BM25 index |
| `query_library` | `question`, `top_k?`, `use_expansion?` | Vectorless RAG query |
| `summarize_paper` | `arxiv_id` | Structured paper summary |
| `list_library` | — | All indexed papers |
| `get_reading_list` | — | Saved reading list |

---

## Project structure

```
arxiv-mcp-server/
├── src/
│   ├── server.py          # MCP server — registers all 6 tools
│   ├── arxiv_client.py    # ArXiv API search + PDF download/extraction
│   ├── vectorless_rag.py  # BM25 + RRF + contextual compression
│   └── reading_list.py    # JSON-backed reading list
├── web/
│   └── index.html         # Live landing page (deployed to GitHub Pages)
├── tests/
│   └── test_core.py       # pytest suite
├── test_local.py          # standalone local test script (no Claude needed)
└── pyproject.toml
```

---

## Common issues

**`ModuleNotFoundError: No module named 'encodings'`**
Your terminal is using the wrong Python. Fix:
```bash
unset PYTHONPATH && unset PYTHONHOME
source .venv/bin/activate
```

**Conda overriding the venv**
If you see `(base)` and `(.venv)` in your prompt simultaneously:
```bash
conda deactivate
source .venv/bin/activate
```

**Yellow squiggles in VS Code**
`Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.venv`.

**MCP tools not showing in Claude Desktop**
- Confirm you're on Claude Pro (required for MCP)
- Use the full `.venv` Python path in the config, not just `python` or `python3`
- Fully quit Claude Desktop (`Cmd+Q`) and reopen — don't just close the window

---

## Tech stack

| Component | Library | Role |
|-----------|---------|------|
| MCP protocol | `mcp` | stdio server framework |
| ArXiv | `arxiv` | paper search + metadata |
| PDF extraction | `pdfplumber` | text from PDFs |
| HTTP | `httpx` | async PDF download |
| Retrieval | BM25 (built-in, no deps) | vectorless keyword search |
| Fusion | RRF (built-in) | multi-query result merging |
| Compression | sentence BM25 (built-in) | extract relevant sentences |
| Persistence | JSON | BM25 index + reading list |

**Zero ML dependencies for retrieval.** No PyTorch, no sentence-transformers, no model downloads.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v --asyncio-mode=auto
ruff check src/
```

## License

MIT