"""
Localist MCP Server
====================
Standalone MCP (Model Context Protocol) service on port 8003.

Exposes file_op capabilities as three MCP tools — read_file, write_file,
append_file — plus fetch_url (Phase 2: ports the retired standalone Fetcher
microservice's /extract path in-process), web_search (Phase 3: ports the
LangSearch integration in-process, no runtime.infer() hallucination
fallback), news_search (NewsAPI.org /v2/everything — first-tier provider
for news-shaped queries; MCPToolDispatcher falls back to web_search on a
miss, see news-query-routing plan §4), generate_chart (renders a
bar/line/pie chart from structured data server-side via matplotlib), and
github_search/github_read/github_release (public-repo GitHub REST reads —
search, file/README/directory content, and release notes, read-only, no
archive/CLI capability; see mcp_server/github.py), and hacker_news_search
(on-demand Hacker News story search via Algolia's HN Search API — the
chat-callable counterpart to backend/hacker_news.py's Live Feed panel
data source; see mcp_server/hacker_news.py), ollama_web_search
(Ollama Cloud's web search API — a second provider underneath the
Planner-facing web_search tool; the dispatcher decides fallback/primary
orchestration between it and Brave/LangSearch, see
mcp_server/ollama_web_search.py), and ollama_web_fetch (Ollama Cloud's web
fetch API — a fallback-only tier underneath the Planner-facing url_fetch
tool when the in-process fetch_url extraction fails; no primary mode, no
Planner visibility, see ollama-web-search-mcp-tool-scoping.md §5 and
mcp_server/ollama_web_fetch.py), and ocr_extract (local text extraction from
uploaded images — including HEIC — and PDFs via Apple's Vision framework
and PyMuPDF, entirely independent of whichever chat inference backend is
active; never planner-routed, called directly by backend/main.py's
POST /chat/files at upload time — see mcp_server/ocr.py and
docs/architecture/22-local-ocr-service.md) — over SSE transport, using
the official `mcp` Python SDK's FastMCP. See backend/mcp_tool_dispatcher.py
for the dispatch seam.

Endpoints
---------
  GET  /health   — {"status": "ok"}
  GET  /sse       — MCP SSE stream (mounted from FastMCP)
  POST /messages/ — MCP message endpoint (mounted from FastMCP)

Configuration
-------------
  LOCALIST_MCP_PROJECT_ROOT   Sandbox root for file_op tools. Defaults to
                               backend/ (parent of this package) — see
                               mcp_server/file_ops.py.
  LOCALIST_LOG_LEVEL           Root log level (default INFO).
  SEARCH_PROVIDER               Which web_search provider is active:
                               "langsearch" (default) or "brave". See
                               mcp_server/web_search.py.
  LANGSEARCH_API_KEY           Required for web_search when
                               SEARCH_PROVIDER=langsearch — see
                               mcp_server/web_search.py. Loaded from
                               backend/.env below, same as backend/main.py —
                               this is a separate process, so it does not
                               inherit backend/main.py's own load_dotenv().
  BRAVE_API_KEY                Required for web_search when
                               SEARCH_PROVIDER=brave — see
                               mcp_server/web_search.py.
  NEWSAPI_API_KEY               Required for news_search — see
                               mcp_server/news_search.py. Free Developer
                               tier only (100 req/day, dev/test use only);
                               a paid tier is required if this code is
                               ever deployed off a single local machine.
  GITHUB_TOKEN                 Optional for github_search / github_read /
                               github_release — see mcp_server/github.py.
                               All three are public-repo GitHub REST reads;
                               the token is used opportunistically if
                               present. Rate limits differ by endpoint:
                               github_read/github_release (core REST
                               bucket) get 60 req/hr unauthenticated vs
                               5000 req/hr authenticated; github_search
                               (Search API's separate, stricter bucket)
                               gets 10 req/min unauthenticated vs 30
                               req/min authenticated.
                               Also read by backend/github_watch.py in the
                               main backend process, where it is required
                               (not optional) — GET /user/subscriptions
                               needs an authenticated identity. A classic
                               PAT with no scopes selected is sufficient
                               for all of this (public data only) — a
                               fine-grained PAT cannot call
                               GET /user/subscriptions at all (no
                               corresponding account permission exists).
  (no key needed for hacker_news_search — Algolia's HN Search API is
   public and unauthenticated; see mcp_server/hacker_news.py.)
  LOCALIST_MCP_UPLOAD_ROOT      Sandbox root for ocr_extract's temp uploaded
                               images/PDFs. Leave blank to default to
                               backend/ (parent of this package), same
                               convention as LOCALIST_MCP_PROJECT_ROOT — see
                               mcp_server/ocr.py.
  LOCALIST_OCR_MAX_PDF_PAGES     Page cap for ocr_extract's rasterize+OCR
                               fallback path (scanned PDFs only — a real
                               text layer is read directly regardless of
                               page count). Default 20 if unset/invalid.
  (no key needed for ocr_extract's Apple Silicon path — Vision framework is
   local and requires no account/network.)
  LOCALIST_OLLAMA_VISION_MODEL   Vision-capable Ollama chat model
                               (e.g. "llava", "qwen2.5vl") for ocr_extract's
                               image OCR path on non-Apple-Silicon platforms
                               (mcp_server/ocr_ollama.py). No default — image
                               OCR fails with a clear config error if unset
                               on such a platform. PDFs are unaffected and
                               still require Apple Silicon regardless of
                               this setting.
  LOCALIST_OLLAMA_URL           Local Ollama daemon base URL for the above.
                               Default http://localhost:11434 — same
                               variable name/meaning as backend/main.py's
                               own Settings.ollama_url, read independently
                               here since this is a separate process.
  OLLAMA_API_KEY                Required for ollama_web_search and
                               ollama_web_fetch — see
                               mcp_server/ollama_web_search.py and
                               mcp_server/ollama_web_fetch.py. Loaded the
                               same way as the other provider keys above
                               (backend/.env via load_dotenv() below). This
                               is a distinct credential from Ollama Cloud
                               chat/embed (ollama_runtime_client.py), which
                               authenticates via the local Ollama daemon's
                               own signed-in session and never sends an
                               API key over HTTP. Provisioned from an
                               ollama.com account (2026-07-31) and
                               confirmed working live against
                               /api/web_search; /api/web_fetch uses the
                               same key but has not itself been
                               live-verified yet.

Start
-----
    uvicorn mcp_server.main:app --host 127.0.0.1 --port 8003

Or from backend/ with venv activated:
    python -m uvicorn mcp_server.main:app --host 127.0.0.1 --port 8003 --reload
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
# backend/src/localist/mcp_server/main.py -> backend/.env (4 parents up)
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP

from . import (
    chart as _chart,
    file_ops,
    github as _github,
    hacker_news as _hacker_news,
    news_search as _news_search,
    ocr as _ocr,
    ocr_ollama as _ocr_ollama,
    ollama_web_fetch as _ollama_web_fetch,
    ollama_web_search as _ollama_web_search,
    url_fetch as _url_fetch,
    web_search as _web_search,
)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = os.environ.get("LOCALIST_LOG_LEVEL", "INFO").upper(),
    format = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("localist-mcp")


# ── MCP tools ────────────────────────────────────────────────────────────────

mcp = FastMCP(name="localist-mcp")


@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file. path is resolved relative to project_root and sandboxed."""
    return file_ops.read_file(path)


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a UTF-8 text file. path is resolved relative to project_root and sandboxed."""
    return file_ops.write_file(path, content)


@mcp.tool()
def append_file(path: str, content: str, turn_id: str | None = None) -> str:
    """Append content to a UTF-8 text file. path is resolved relative to project_root and sandboxed."""
    return file_ops.append_file(path, content, turn_id)


@mcp.tool()
async def fetch_url(url: str, timeout: float = 10.0) -> dict:
    """Fetch a URL and extract clean article text (title, author, date, cleaned_text, word_count)."""
    return await _url_fetch.fetch_url(url, timeout)


@mcp.tool()
async def web_search(query: str) -> dict:
    """Run one web search query via the configured search provider (SEARCH_PROVIDER=langsearch|brave)."""
    return await _web_search.web_search(query)


@mcp.tool()
async def news_search(query: str, url: str | None = None) -> dict:
    """Run one news search query via NewsAPI.org (/v2/everything, sorted by publish date). Pass url to pin the result to one already-known article."""
    return await _news_search.news_search(query, url)


@mcp.tool()
async def ollama_web_search(query: str) -> dict:
    """Run one web search query via Ollama Cloud's web search API (https://ollama.com/api/web_search)."""
    return await _ollama_web_search.ollama_web_search(query)


@mcp.tool()
async def ollama_web_fetch(url: str) -> dict:
    """Fetch a URL via Ollama Cloud's web fetch API (https://ollama.com/api/web_fetch), normalized to fetch_url's own response shape."""
    return await _ollama_web_fetch.ollama_web_fetch(url)


@mcp.tool()
def generate_chart(chart_type: str, labels: list[str], datasets: list[dict], title: str = "") -> dict:
    """Render a bar/line/pie chart from structured data and save it as a PNG. Returns summary, png_path, and chart_config."""
    return _chart.generate_chart(chart_type, labels, datasets, title)


@mcp.tool()
async def github_search(query: str, kind: str = "repositories") -> dict:
    """Search public GitHub repositories or code (kind='repositories'|'code') via the GitHub Search API."""
    return await _github.github_search(query, kind)


@mcp.tool()
async def github_read(owner: str, repo: str, path: str | None = None) -> dict:
    """Read a public repo's README (path omitted), a file's contents, or a directory listing (path set)."""
    return await _github.github_read(owner, repo, path)


@mcp.tool()
async def github_release(owner: str, repo: str, tag: str | None = None) -> dict:
    """Fetch one release's notes — the latest release (tag omitted) or a specific tagged release."""
    return await _github.github_release(owner, repo, tag)


@mcp.tool()
async def hacker_news_search(query: str, url: str | None = None) -> dict:
    """Search Hacker News stories via Algolia's HN Search API. Pass url to pin the result to one already-known story."""
    return await _hacker_news.hacker_news_search(query, url)


@mcp.tool()
def ocr_extract(path: str, mime_type: str, max_pdf_pages: int | None = None) -> str:
    """Extract text from an uploaded image (incl. HEIC) or PDF, entirely locally. Apple Silicon: Vision framework + PyMuPDF. Other platforms: images only, via a configured Ollama vision model (LOCALIST_OLLAMA_VISION_MODEL) — PDFs still require Apple Silicon. path is resolved relative to upload_root and sandboxed."""
    if _ocr._is_apple_silicon():
        return _ocr.extract_text(path, mime_type, max_pdf_pages)
    if mime_type.startswith("image/"):
        return _ocr_ollama.OllamaVisionOCRProvider().extract_text(path, mime_type, max_pdf_pages)
    return _ocr.extract_text(path, mime_type, max_pdf_pages)  # PDFs: unchanged rejection


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Localist MCP Server starting on port 8003 — project_root=%s upload_root=%s",
        file_ops.get_project_root(), _ocr.get_upload_root(),
    )
    yield
    logger.info("Localist MCP Server shutting down.")


app = FastAPI(
    title       = "Localist MCP Server",
    description = "MCP tool server for Localist — file_op tools (read_file/write_file/append_file), fetch_url, web_search, news_search, generate_chart, github_search/github_read/github_release, hacker_news_search, ollama_web_search, ollama_web_fetch, and ocr_extract.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://127.0.0.1:5173",
                         "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Mount the MCP SSE app at root — exposes GET /sse and POST /messages/.
# Registered after /health so the explicit route takes precedence over the mount.
app.mount("/", mcp.sse_app())


# ── Dev entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mcp_server.main:app",
        host      = "127.0.0.1",
        port      = 8003,
        reload    = True,
        log_level = "info",
    )
