"""
Localist MCP Server — news_search tool implementation
========================================================
NewsAPI.org (/v2/everything) is the first-tier provider for news-shaped
queries ("latest news on X", "headlines about Y") — a purpose-built news
index with publish dates and source attribution, unlike the generic
web_search tool (LangSearch/Brave), which has no concept of article
freshness. See news-query-routing plan, §4.1.

Auth is via the X-Api-Key header rather than the ?apiKey= query param —
functionally equivalent, but keeps the key out of any request logging that
captures URLs (same rationale as web_search's choice of Authorization
header for LangSearch).

Free Developer-tier terms (100 req/day, 24h publish delay, last-month-only
article window, explicitly not licensed for production) are a natural fit
here since Localist runs single-user on localhost and never leaves it —
see NEWSAPI_API_KEY's .env.example comment.

The env var is read lazily inside news_search() rather than cached at
import time, same reasoning as web_search.py: this process does not
inherit backend/main.py's own load_dotenv() call.
"""

from __future__ import annotations

import logging
import os

import httpx

from . import search_format

logger = logging.getLogger(__name__)

_NEWSAPI_ENDPOINT: str = "https://newsapi.org/v2/everything"
_NEWSAPI_PAGE_SIZE: int = 5
_NEWSAPI_SUMMARY_CHARS: int = 600
_NEWSAPI_CONTENT_CHARS: int = 500


async def news_search(query: str, url: str | None = None) -> dict:
    """
    Run one news_search query via NewsAPI's /v2/everything endpoint.

    Parameters
    ----------
    query : the search text (usually a headline or topic).
    url   : optional — when a caller already knows the exact article (e.g.
            the user clicked a specific story in the Live Feed panel), pass
            its URL to pin the result to that one article rather than
            trusting the query text to find it again among near-duplicate
            coverage. If none of the returned articles match, falls back to
            the normal unfiltered top-5 behavior — an older clicked article
            may simply be outside NewsAPI dev-tier's last-month window.

    Returns
    -------
    dict with keys:
      query        : the input query, echoed back
      result_text  : formatted bullet list, or "" when there are no
                     results (caller decides what "no results" means —
                     see is_miss below)
      result_count : number of articles returned
      is_miss      : True when NewsAPI itself reports no usable result
                     (status != "ok", or totalResults == 0) — the precise
                     "fall through to Brave" trigger condition from §4.1.
                     False for a genuine transport/HTTP error, which
                     raises instead (indistinguishable from any other
                     tool-call failure, same as web_search).

    Raises
    ------
    ValueError
        "ERROR: NEWSAPI_API_KEY not configured" if the key is unset/empty,
        or "ERROR: news_search failed — <exc>" on any network/HTTP/
        parsing error (transport failure, not a zero-result response —
        those two are deliberately distinct, see is_miss above).
    """
    api_key = os.environ.get("NEWSAPI_API_KEY", "")

    if not api_key:
        raise ValueError("ERROR: NEWSAPI_API_KEY not configured")

    headers = {
        "X-Api-Key": api_key,
    }
    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": _NEWSAPI_PAGE_SIZE,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_NEWSAPI_ENDPOINT, headers=headers, params=params)
            data = resp.json()
    except Exception as exc:
        logger.warning("news_search: NewsAPI failed for query=%r: %s", query, exc)
        raise ValueError(f"ERROR: news_search failed — {exc}") from exc

    if data.get("status") != "ok" or data.get("totalResults", 0) == 0:
        logger.info(
            "news_search: NewsAPI miss for query=%r status=%s totalResults=%s.",
            query, data.get("status"), data.get("totalResults"),
        )
        return {"query": query, "result_text": "", "result_count": 0, "is_miss": True}

    articles = data.get("articles", [])

    # Pin to the one already-known article when a caller (e.g. the Live Feed
    # panel's "Ask about this") supplies its URL — otherwise a query built
    # from a headline can resolve to several near-duplicate stories and the
    # model would have no way to tell which one the user actually clicked.
    pinned = False
    if url:
        matched = [a for a in articles if (a.get("url") or "").strip() == url]
        if matched:
            articles = matched
            pinned = True

    results: list[search_format.SearchResult] = []
    for article in articles[:_NEWSAPI_PAGE_SIZE]:
        title        = article.get("title", "").strip()
        description  = (article.get("description") or "").strip()
        source_name  = article.get("source", {}).get("name", "").strip()
        published_at = article.get("publishedAt", "").strip().split("T", 1)[0]
        article_url  = article.get("url", "").strip()

        # `content` is NewsAPI's own short body snippet (also truncated on
        # the dev tier) — surfaced only in the pinned single-article case
        # so the multi-result formatting other callers rely on is unchanged.
        extra = None
        if pinned:
            content = (article.get("content") or "").strip()
            if content:
                extra = search_format.truncate_summary(content, _NEWSAPI_CONTENT_CHARS)

        results.append(search_format.SearchResult(
            title=title,
            summary=description,
            url=article_url,
            source=source_name or None,
            published_at=published_at or None,
            extra=extra,
        ))

    result_text = search_format.format_results(
        results, per_result_budget=_NEWSAPI_SUMMARY_CHARS
    )
    logger.info(
        "news_search: NewsAPI complete for query=%r results=%d result_chars=%d.",
        query, len(articles), len(result_text),
    )
    return {"query": query, "result_text": result_text, "result_count": len(articles), "is_miss": False}
