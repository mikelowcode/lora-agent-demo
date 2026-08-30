"""
Localist MCP Server — web_search tool implementation
========================================================
Ports the real-LangSearch branch of ToolDispatcher._execute_single_search
verbatim (payload, endpoint, response parsing, truncation, formatting) —
same proven-working request/response contract, no redesign.

Phase 3 locked decision: the legacy fallback that called runtime.infer() to
generate plausible-sounding bullet points when LANGSEARCH_API_KEY was unset
is removed entirely — model-hallucinated content indistinguishable from a
real search result to every downstream consumer. Missing API key raises a
clean error here; nothing calls inference on this path. Confirmed via grep
that _WEB_SEARCH_FALLBACK_SYSTEM (tool_dispatcher.py) has no other callers,
so nothing else depends on the removed behaviour.

Async (httpx) rather than sync (requests) — brings this tool in line with
fetch_url's async style from Phase 2 and removes a sync HTTP call that was
previously blocking inside what may be an async context. The
request/response contract itself is unchanged.

Provider switch: a second provider (Brave) was added alongside LangSearch.
Exactly one provider is active per process, selected by the
SEARCH_PROVIDER env var (default "langsearch"). The env var is read lazily
inside web_search() rather than cached at import time, since this process
does not inherit backend/main.py's own load_dotenv() call (see
mcp_server/main.py's module docstring) — reading at call time keeps this
correct regardless of when/whether .env has been loaded.
"""

from __future__ import annotations

import logging
import os

import httpx

from . import search_format

logger = logging.getLogger(__name__)

_LANGSEARCH_ENDPOINT: str = "https://api.langsearch.com/v1/web-search"
_LANGSEARCH_COUNT: int = 3
_LANGSEARCH_SUMMARY_CHARS: int = 700

_BRAVE_ENDPOINT: str = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_COUNT: int = 3
_BRAVE_SUMMARY_CHARS: int = 700


async def _web_search_langsearch(query: str) -> dict:
    """
    Run one web_search query via the LangSearch API.

    Raises
    ------
    ValueError
        "ERROR: LANGSEARCH_API_KEY not configured" if the key is unset/empty
        (no inference fallback — see module docstring), or
        "ERROR: web_search failed — <exc>" on any network/HTTP/parsing error.
    """
    api_key = os.environ.get("LANGSEARCH_API_KEY", "")

    if not api_key:
        raise ValueError("ERROR: LANGSEARCH_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "query":     query,
        "summary":   True,
        "count":     _LANGSEARCH_COUNT,
        "freshness": "noLimit",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(_LANGSEARCH_ENDPOINT, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("web_search: LangSearch failed for query=%r: %s", query, exc)
        raise ValueError(f"ERROR: web_search failed — {exc}") from exc

    pages = data.get("data", {}).get("webPages", {}).get("value", [])

    if not pages:
        return {"query": query, "result_text": "No results found.", "result_count": 0}

    results: list[search_format.SearchResult] = []
    for page in pages[:_LANGSEARCH_COUNT]:
        name    = page.get("name", "").strip()
        snippet = page.get("snippet", "").strip()
        url     = page.get("displayUrl", page.get("url", "")).strip()
        # Prefer summary over snippet when available
        body    = page.get("summary") or snippet
        results.append(search_format.SearchResult(title=name, summary=body, url=url))

    result_text = search_format.format_results(
        results, per_result_budget=_LANGSEARCH_SUMMARY_CHARS
    )
    logger.info(
        "web_search: LangSearch complete for query=%r results=%d result_chars=%d.",
        query, len(pages), len(result_text),
    )
    return {"query": query, "result_text": result_text, "result_count": len(pages)}


async def _web_search_brave(query: str) -> dict:
    """
    Run one web_search query via the Brave Search API.

    Raises
    ------
    ValueError
        "ERROR: BRAVE_API_KEY not configured" if the key is unset/empty
        (no inference fallback — see module docstring), or
        "ERROR: web_search failed — <exc>" on any network/HTTP/parsing error.
    """
    api_key = os.environ.get("BRAVE_API_KEY", "")

    if not api_key:
        raise ValueError("ERROR: BRAVE_API_KEY not configured")

    headers = {
        "X-Subscription-Token": api_key,
        "Accept":               "application/json",
    }
    params = {
        "q":     query,
        "count": _BRAVE_COUNT,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_BRAVE_ENDPOINT, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("web_search: Brave failed for query=%r: %s", query, exc)
        raise ValueError(f"ERROR: web_search failed — {exc}") from exc

    results = data.get("web", {}).get("results", [])

    if not results:
        return {"query": query, "result_text": "No results found.", "result_count": 0}

    search_results: list[search_format.SearchResult] = []
    for result in results[:_BRAVE_COUNT]:
        title = result.get("title", "").strip()
        body  = result.get("description", "").strip()
        url   = result.get("url", "").strip()
        search_results.append(search_format.SearchResult(title=title, summary=body, url=url))

    result_text = search_format.format_results(
        search_results, per_result_budget=_BRAVE_SUMMARY_CHARS
    )
    logger.info(
        "web_search: Brave complete for query=%r results=%d result_chars=%d.",
        query, len(results), len(result_text),
    )
    return {"query": query, "result_text": result_text, "result_count": len(results)}


async def web_search(query: str) -> dict:
    """
    Run one web_search query via the configured search provider.

    Dispatches to _web_search_langsearch or _web_search_brave based on the
    SEARCH_PROVIDER env var (default "langsearch"), read lazily on every
    call — see module docstring.

    Raises
    ------
    ValueError
        Provider-specific missing-API-key or request-failure errors (see
        _web_search_langsearch / _web_search_brave), or
        "ERROR: unknown SEARCH_PROVIDER <value>" if SEARCH_PROVIDER is set
        to anything other than "langsearch" or "brave" — no silent
        fallback between providers.
    """
    provider = os.environ.get("SEARCH_PROVIDER", "langsearch").lower()

    if provider == "langsearch":
        return await _web_search_langsearch(query)
    if provider == "brave":
        return await _web_search_brave(query)

    raise ValueError(f"ERROR: unknown SEARCH_PROVIDER {provider!r}")
