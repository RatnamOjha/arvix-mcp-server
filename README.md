# ArXiv Research Intelligence — MCP Server

> Vectorless RAG over ArXiv — BM25 + RRF + contextual compression — as an MCP server for Claude.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What this is

An MCP server that gives Claude live tools to search ArXiv, download and index full papers, and answer research questions from your personal paper library — all without embeddings, GPU, or model downloads.

---

## Vectorless RAG — why and how

Most RAG pipelines embed every text chunk with a neural model, store vectors, and retrieve by cosine similarity. This works but has real costs: GPU or slow CPU inference, ~90MB model downloads, embedding drift, and poor handling of exact technical terms like `LoRA`, `RLHF`, `KV-cache`.

**v2 replaces the embedding step entirely with Okapi BM25** — the same algorithm that powers Elasticsearch and academic search engines. Zero model downloads, instant indexing, and it handles exact terminology perfectly.

### The full pipeline

```
Your question
     │
     ▼
Query expansion ── generates keyword variants to maximize BM25 recall
     │
     ▼
Multi-query BM25 ── each variant scored against all chunks independently
     │
     ▼
RRF fusion ── Reciprocal Rank Fusion merges ranked lists
              score = Σ 1/(60 + rank) — no score normalization needed
     │
     ▼
Contextual compression ── sentence-level BM25 extracts only the 2–3
                          sentences per chunk that answer the question
     │
     ▼
Cited context returned to Claude
```

### v1 vs v2 comparison

| | v1 (vector RAG) | v2 (vectorless RAG) |
|---|---|---|
| Indexing | encode every chunk with neural model | pure text, instant |
| Retrieval | cosine similarity over embeddings | Okapi BM25 |
| Multi-query | single query only | 3 variants + RRF fusion |
| Compression | cross-encoder reranker | sentence-level BM25 |
| GPU required | yes (or slow CPU) | no |
| Model download | ~90MB on first run | none |
| Exact terms (LoRA, RLHF) | sometimes misses | always catches |

---

## No Claude Pro? No problem

The entire ML pipeline runs from your terminal with `test_local.py`. Claude Pro is only needed for the Claude Desktop chat interface.

### Setup

```bash
git clone https://github.com/RatnamOjha/arvix-mcp-server
cd arvix-mcp-server
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Run tests

```bash
python3 test_local.py                          # full end-to-end demo
python3 test_local.py --search                 # live ArXiv search
python3 test_local.py --fetch --arxiv-id 2005.11401   # fetch a paper
python3 test_local.py --query --question "what is BM25?"
python3 test_local.py --library                # see what's indexed
```

---

## Test with your own paper

### Step 1 — Find your paper ID

```
https://arxiv.org/abs/2301.07041
                   ^^^^^^^^^^^^^ this is your paper ID
```

### Step 2 — Fetch and index it

```bash
python3 test_local.py --fetch --arxiv-id 2301.07041
```

Downloads the PDF directly (no API, no rate limiting), extracts text using pymupdf, chunks and indexes into BM25.

### Step 3 — Query it

```bash
# Query the whole library
python3 test_local.py --query --question "what problem does this paper solve?"

# Query one specific paper only
python3 test_local.py --query --paper 2301.07041 --question "what are the limitations?"

# See full untruncated output
python3 test_local.py --query --paper 2301.07041 --question "what methods do they use?" --full
```

### Step 4 — Build a multi-paper library

```bash
python3 test_local.py --fetch --arxiv-id 2301.07041
python3 test_local.py --fetch --arxiv-id 2305.10601
python3 test_local.py --fetch --arxiv-id 2005.11401

# Cross-paper query
python3 test_local.py --query --question "how do these papers approach retrieval differently?"
```

Everything persists at `~/.arxiv-mcp/` between runs.

---

## With Claude Pro — MCP integration

### Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arxiv-research": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "src.server"],
      "cwd": "/absolute/path/to/arvix-mcp-server"
    }
  }
}
```

> Use the **full absolute path** to your `.venv` Python — avoids conflicts with Anaconda or system Python.

Restart Claude Desktop (Cmd+Q, then reopen). The 🔨 hammer icon will show 6 new tools.

### Example prompts

```
Search for recent papers on speculative decoding
Fetch paper 2305.10601 and add it to my library
What does my library say about KV cache optimization?
What does paper 2507.07171 say about the evaluation? (uses arxiv_id_filter automatically)
Summarize paper 2305.10601
Show me my reading list
```

---

## MCP Tools

| Tool | Args | Description |
|------|------|-------------|
| `search_papers` | `query`, `max_results?`, `category?` | Live ArXiv search |
| `fetch_paper` | `arxiv_id`, `add_to_reading_list?` | Download + BM25 index |
| `query_library` | `question`, `top_k?`, `use_expansion?`, `arxiv_id_filter?` | Vectorless RAG query |
| `summarize_paper` | `arxiv_id` | Structured paper summary |
| `list_library` | — | All indexed papers |
| `get_reading_list` | — | Saved reading list |

The `arxiv_id_filter` parameter on `query_library` restricts search to one specific paper — Claude uses this automatically when you mention a paper ID in your question.

---

## Known edge cases and how they're handled

### ArXiv API rate limiting (HTTP 429)
The API allows ~1 request/3 seconds. If you run multiple searches quickly you'll hit a 429.

**Fix:** PDF download never rate-limits — we go directly to `https://arxiv.org/pdf/{id}` which is a CDN with no limit. Metadata (title, authors) is fetched separately and falls back gracefully if the API is unavailable. The paper is still fully indexed and queryable even with placeholder metadata.

### Garbled text from LaTeX PDFs
Physics and math papers use LaTeX-rendered fonts that pdfplumber misreads, producing output like `b e a h A b r ξ o a s α v t`.

**Fix:** We use **pymupdf** (fitz) as the primary extractor — it reconstructs reading order from glyph positions and handles multi-column layouts correctly. A post-processing step filters lines where average token length < 1.8 characters (the statistical signature of scrambled LaTeX fonts). pdfplumber remains as a fallback if pymupdf is not installed.

### Anaconda overriding the venv Python
If you see `(base)` and `(.venv)` in your prompt simultaneously, conda is winning.

**Fix:**
```bash
conda deactivate
source .venv/bin/activate
which python3  # must show .venv path
```

### Yellow squiggles in VS Code
VS Code is using the wrong interpreter.

**Fix:** `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.venv`.

### Wrong python being used for test scripts
Always use the full venv path:
```bash
/path/to/project/.venv/bin/python3 test_local.py --fetch --arxiv-id 2005.11401
```

---

## Project structure

```
arxiv-mcp-server/
├── src/
│   ├── server.py           # MCP server — registers all 6 tools
│   ├── arxiv_client.py     # ArXiv search + direct PDF download/extraction
│   ├── vectorless_rag.py   # BM25 + RRF + contextual compression
│   └── reading_list.py     # JSON-backed reading list
├── web/
│   └── index.html          # Live landing page (GitHub Pages)
├── tests/
│   └── test_core.py        # pytest suite (runs on every push via CI)
├── test_local.py           # Standalone demo — no Claude needed
└── pyproject.toml
```

---

## Tech stack

| Component | Library | Role |
|-----------|---------|------|
| MCP protocol | `mcp` | stdio server framework |
| ArXiv | `arxiv` | paper search + metadata |
| PDF download | `httpx` | direct CDN download, no rate limit |
| PDF extraction | `pymupdf` (primary) | handles LaTeX, multi-column |
| PDF extraction | `pdfplumber` (fallback) | general extraction |
| Retrieval | BM25 (built-in) | vectorless keyword search |
| Fusion | RRF (built-in) | multi-query result merging |
| Compression | sentence BM25 (built-in) | extract relevant sentences |
| Persistence | JSON | BM25 index + reading list |

**Zero ML dependencies for retrieval.** No PyTorch, no sentence-transformers, no model downloads for the RAG pipeline.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v --asyncio-mode=auto
ruff check src/
```

## Roadmap

- [ ] Qdrant/ChromaDB backend option for larger libraries
- [ ] LLM-powered query expansion (replaces rule-based fallback)
- [ ] Multi-user support with shared paper libraries
- [ ] Semantic Scholar + PubMed as additional sources
- [ ] Citation graph traversal — fetch papers that cite or are cited by a paper

## License

MIT — built by [Ratnam Ojha](https://github.com/RatnamOjha)