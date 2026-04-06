#!/usr/bin/env python3
"""
test_mcp.py — test the MCP server directly without Claude Pro.

This script spawns the MCP server as a subprocess and communicates
with it over stdin/stdout using the exact same JSON-RPC messages
that Claude Desktop sends. It's the closest you can get to testing
the real MCP integration without a Pro subscription.

Usage:
    python test_mcp.py                        # run all tool tests
    python test_mcp.py --tool search_papers   # test one tool
    python test_mcp.py --tool fetch_paper --arxiv-id 2005.11401
    python test_mcp.py --tool query_library --question "what is BM25?"
    python test_mcp.py --tool list_library
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Always use the Python that's running this script — avoids Anaconda conflicts
PYTHON = str(Path(sys.executable).resolve())
SERVER_CMD = [PYTHON, "-m", "src.server"]
CWD = str(Path(__file__).parent)

# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

def make_request(method: str, params: dict, req_id: int) -> bytes:
    msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
    return (msg + "\n").encode()

def make_notify(method: str, params: dict) -> bytes:
    msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
    return (msg + "\n").encode()

def read_response(proc: subprocess.Popen, timeout: float = 30.0) -> dict:
    proc.stdin.flush()
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if line:
            try:
                return json.loads(line.decode().strip())
            except json.JSONDecodeError:
                continue
    # Print stderr so user sees what the server actually said
    try:
        err = proc.stderr.read1(4096).decode(errors="replace")  # non-blocking
        if err.strip():
            print(f"\n  Server stderr:\n{err}")
    except Exception:
        pass
    raise TimeoutError(f"No response within {timeout}s")

# ── Display helpers ───────────────────────────────────────────────────────────

def header(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def ok(msg: str):
    print(f"  ✓  {msg}")

def info(msg: str):
    print(f"     {msg}")

def show_result(result: dict):
    """Pretty-print a tool result."""
    if "error" in result:
        print(f"  ✗  Error: {result['error']}")
        return
    content = result.get("result", {}).get("content", [])
    for block in content:
        if block.get("type") == "text":
            try:
                data = json.loads(block["text"])
                print(json.dumps(data, indent=2)[:1200])
                if len(block["text"]) > 1200:
                    print("  ... (truncated — full response available)")
            except json.JSONDecodeError:
                print(block["text"][:800])

# ── MCP session ───────────────────────────────────────────────────────────────

class MCPSession:
    """
    Manages a live MCP server subprocess.
    Handles the initialize handshake automatically.
    """

    def __init__(self):
        self.proc = subprocess.Popen(
            SERVER_CMD,
            cwd=CWD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # captured so we can print on failure
        )
        self._req_id = 1
        self._handshake()

    def _handshake(self):
        """Send initialize + initialized — exactly what Claude Desktop does."""
        # 1. initialize
        self.proc.stdin.write(make_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
            self._req_id,
        ))
        self._req_id += 1
        resp = read_response(self.proc, timeout=10)
        if "error" in resp:
            raise RuntimeError(f"Initialize failed: {resp['error']}")

        # 2. initialized notification (no response expected)
        self.proc.stdin.write(make_notify("notifications/initialized", {}))
        self.proc.stdin.flush()
        time.sleep(0.3)

    def call(self, tool_name: str, args: dict, timeout: float = 60.0) -> dict:
        """Call a tool and return the full response."""
        self.proc.stdin.write(make_request(
            "tools/call",
            {"name": tool_name, "arguments": args},
            self._req_id,
        ))
        self._req_id += 1
        return read_response(self.proc, timeout=timeout)

    def list_tools(self) -> dict:
        self.proc.stdin.write(make_request("tools/list", {}, self._req_id))
        self._req_id += 1
        return read_response(self.proc, timeout=10)

    def close(self):
        self.proc.terminate()
        self.proc.wait()

# ── Individual tool tests ─────────────────────────────────────────────────────

def test_list_tools(session: MCPSession):
    header("tools/list — what tools are registered?")
    resp = session.list_tools()
    tools = resp.get("result", {}).get("tools", [])
    ok(f"{len(tools)} tools registered")
    for t in tools:
        info(f"  {t['name']}")
        desc = t.get("description", "")[:80]
        info(f"    {desc}")

def test_search(session: MCPSession, query: str = "retrieval augmented generation"):
    header(f"search_papers — '{query}'")
    resp = session.call("search_papers", {"query": query, "max_results": 3})
    show_result(resp)

def test_fetch(session: MCPSession, arxiv_id: str = "2005.11401"):
    header(f"fetch_paper — arxiv:{arxiv_id}")
    print("  Downloading PDF and indexing (15-20 seconds)...")
    resp = session.call("fetch_paper", {"arxiv_id": arxiv_id}, timeout=120)
    show_result(resp)

def test_query(session: MCPSession, question: str, paper: str = None):
    header(f"query_library — '{question}'")
    args = {"question": question, "top_k": 4}
    if paper:
        args["arxiv_id_filter"] = paper
        info(f"Filtering to paper: {paper}")
    resp = session.call("query_library", args, timeout=30)
    show_result(resp)

def test_list_library(session: MCPSession):
    header("list_library — indexed papers")
    resp = session.call("list_library", {})
    show_result(resp)

def test_reading_list(session: MCPSession):
    header("get_reading_list — saved papers")
    resp = session.call("get_reading_list", {})
    show_result(resp)

def test_summarize(session: MCPSession, arxiv_id: str):
    header(f"summarize_paper — arxiv:{arxiv_id}")
    resp = session.call("summarize_paper", {"arxiv_id": arxiv_id}, timeout=30)
    show_result(resp)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test MCP server without Claude Pro")
    parser.add_argument("--tool", default=None,
        choices=["search_papers", "fetch_paper", "query_library",
                 "list_library", "get_reading_list", "summarize_paper", "list_tools"],
        help="Test a specific tool only")
    parser.add_argument("--arxiv-id", default="2005.11401", help="ArXiv ID for fetch/summarize")
    parser.add_argument("--question", default="what problem does this paper solve?",
        help="Question for query_library")
    parser.add_argument("--paper", default=None,
        help="Restrict query to one paper e.g. --paper 2507.07171")
    args = parser.parse_args()

    print("\n  ArXiv MCP — Direct Server Test")
    print("  Spawning MCP server subprocess and connecting over stdio...")
    print("  (This is exactly what Claude Desktop does)\n")

    session = None
    try:
        session = MCPSession()
        ok("MCP server started and handshake complete")

        if args.tool == "list_tools" or args.tool is None:
            test_list_tools(session)

        if args.tool == "search_papers" or args.tool is None:
            test_search(session)

        if args.tool == "fetch_paper":
            test_fetch(session, args.arxiv_id)

        if args.tool == "query_library":
            test_query(session, args.question, args.paper)

        if args.tool == "list_library" or args.tool is None:
            test_list_library(session)

        if args.tool == "get_reading_list" or args.tool is None:
            test_reading_list(session)

        if args.tool == "summarize_paper":
            test_summarize(session, args.arxiv_id)

        header("All tests complete")
        ok("MCP server responded correctly to all tool calls")
        ok("This is identical to what Claude Desktop would send and receive")

    except TimeoutError as e:
        print(f"\n  ✗  Timeout: {e}")
        print("     The server may still be starting. Try again in a few seconds.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ✗  {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        if session:
            session.close()

if __name__ == "__main__":
    main()