"""
Localist MCP Server — ollama_web_search tool implementation
========================================================
Ollama Cloud's web search API (https://ollama.com/api/web_search) as a
second provider underneath the existing web_search tool — see
ollama-web-search-mcp-tool-scoping.md §4. Structured directly off
_web_search_brave (mcp_server/web_search.py): same auth-loading, error
handling, and result-formatting shape, adapted for Ollama's request/
response contract.

This function carries no fallback logic and no awareness of Brave or
LangSearch — orchestration between providers (config-driven primary
selection, fallback-on-empty) lives entirely in mcp_tool_dispatcher.py, not
here. Failures raise ValueError with the same "ERROR: ..." shape the other
provider functions use, so that dispatcher's existing exception handling
works against this function unmodified.

The env var is read lazily inside ollama_web_search() rather than cached at
import time, same reasoning as web_search.py: this process does not
inherit backend/main.py's own load_dotenv() call (see mcp_server/main.py's
module docstring).
"""

from __future__ import annotations

import logging
import os

import httpx

from . import search_format

logger = logging.getLogger(__name__)

_OLLAMA_WEB_SEARCH_ENDPOINT: str = "https://ollama.com/api/web_search"
_OLLAMA_MAX_RESULTS: int = 5
_OLLAMA_SUMMARY_CHARS: int = 700


async def ollama_web_search(query: str) -> dict:
    """
    Run one web_search query via Ollama Cloud's web search API.

    Raises
    ------
    ValueError
        "ERROR: OLLAMA_API_KEY not configured" if the key is unset/empty
        (no inference fallback — same policy as web_search.py), or
        "ERROR: ollama_web_search failed — <exc>" on any network/HTTP/
        parsing error.
    """
    api_key = os.environ.get("OLLAMA_API_KEY", "")

    if not api_key:
        raise ValueError("ERROR: OLLAMA_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "query":       query,
        "max_results": _OLLAMA_MAX_RESULTS,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(_OLLAMA_WEB_SEARCH_ENDPOINT, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("ollama_web_search: Ollama Cloud failed for query=%r: %s", query, exc)
        raise ValueError(f"ERROR: ollama_web_search failed — {exc}") from exc

    results = data.get("results", [])

    if not results:
        return {"query": query, "result_text": "No results found.", "result_count": 0}

    search_results: list[search_format.SearchResult] = []
    for result in results[:_OLLAMA_MAX_RESULTS]:
        title   = result.get("title", "").strip()
        content = result.get("content", "").strip()
        url     = result.get("url", "").strip()
        search_results.append(search_format.SearchResult(title=title, summary=content, url=url))

    result_text = search_format.format_results(
        search_results, per_result_budget=_OLLAMA_SUMMARY_CHARS
    )
    logger.info(
        "ollama_web_search: Ollama Cloud complete for query=%r results=%d result_chars=%d.",
        query, len(results), len(result_text),
    )
    return {"query": query, "result_text": result_text, "result_count": len(results)}
