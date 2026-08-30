"""
Localist MCP Server — ollama_web_fetch tool implementation
========================================================
Ollama Cloud's web fetch API (https://ollama.com/api/web_fetch) as a
fallback tier underneath the existing fetch_url tool — see
ollama-web-search-mcp-tool-scoping.md §5 (Track B). Structured directly off
fetch_url() (mcp_server/url_fetch.py) and ollama_web_search.py: same
auth-loading and error-handling shape, adapted for Ollama's web_fetch
request/response contract.

Response normalization: Ollama's response is {"title", "content",
"links": [...]}, while fetch_url()'s success path returns {"url", "title",
"author", "date_published", "cleaned_text", "word_count",
"fetch_duration_ms"}. This function maps Ollama's response into that exact
same shape — content -> cleaned_text, word_count computed from it,
author/date_published/fetch_duration_ms defaulted since Ollama doesn't
supply them, links dropped since fetch_url's shape has no equivalent field
— so mcp_tool_dispatcher.py's fallback orchestration can format both
tiers' results with the identical code path; nothing downstream needs to
branch on which tier answered.

This function carries no fallback logic and no awareness of the
fetch_url/fetcher tier — orchestration lives entirely in
mcp_tool_dispatcher.py, not here. Failures raise ValueError with the same
"ERROR: ..." shape the other provider functions use, so the dispatcher's
existing exception handling works against this function unmodified.

The env var is read lazily inside ollama_web_fetch() rather than cached at
import time, same reasoning as ollama_web_search.py: this process does not
inherit backend/main.py's own load_dotenv() call.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_OLLAMA_WEB_FETCH_ENDPOINT: str = "https://ollama.com/api/web_fetch"


async def ollama_web_fetch(url: str) -> dict:
    """
    Fetch one URL via Ollama Cloud's web fetch API, normalized into
    fetch_url()'s own response shape (see module docstring).

    Raises
    ------
    ValueError
        "ERROR: OLLAMA_API_KEY not configured" if the key is unset/empty,
        or "ERROR: ollama_web_fetch failed — <exc>" on any network/HTTP/
        parsing error.
    """
    api_key = os.environ.get("OLLAMA_API_KEY", "")

    if not api_key:
        raise ValueError("ERROR: OLLAMA_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {"url": url}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(_OLLAMA_WEB_FETCH_ENDPOINT, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("ollama_web_fetch: Ollama Cloud failed for url=%r: %s", url, exc)
        raise ValueError(f"ERROR: ollama_web_fetch failed — {exc}") from exc

    content = (data.get("content") or "").strip()

    logger.info(
        "ollama_web_fetch: Ollama Cloud complete for url=%r content_chars=%d.",
        url, len(content),
    )
    return {
        "url":               url,
        "title":             data.get("title", ""),
        "author":            "",
        "date_published":    "",
        "cleaned_text":      content,
        "word_count":        len(content.split()),
        "fetch_duration_ms": 0.0,
    }
