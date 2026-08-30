"""
Localist MCP Server — hacker_news_search tool implementation
========================================================
On-demand Hacker News story search for chat, via Algolia's HN Search API
(hn.algolia.com/api/v1/search) — the actual full-text search interface.
HN's own Firebase API (used by backend/hacker_news.py's Live Feed panel
data source) has no search endpoint at all, only ranked-id listing and
single-item lookup, so this module is an independent implementation
rather than a thin wrapper around that one — same deliberate cross-process
duplication convention already documented for news_search.py/news_brief.py
and github.py's github_release/backend's github_watch.py pairs: this is
the on-demand, any-query chat counterpart to the Live Feed's fixed top-10
snapshot.

Comment text (fetch_top_comments/_clean_comment_text) matters here for a
correctness reason, not just completeness: the Search API's hits carry
only a comment *count* (num_comments), never comment bodies. Before this
was added, a pinned query would hand the model a bare number and nothing
else — live testing showed the model then fabricated plausible-sounding
"one commenter noted..." paraphrases with no grounding at all. Comments
are therefore only fetched for the pinned single-story case (see
hacker_news_search's docstring) so the model has real text to draw from
whenever it's in a position to claim there is any.

Public, unauthenticated, no key required — same as backend/hacker_news.py.
"""

from __future__ import annotations

import html
import logging
import re

import httpx

from . import search_format

logger = logging.getLogger(__name__)

_HN_ALGOLIA_SEARCH_ENDPOINT: str = "https://hn.algolia.com/api/v1/search"
_HN_ALGOLIA_ITEM_ENDPOINT:   str = "https://hn.algolia.com/api/v1/items/{}"
_HN_ITEM_URL_BASE:           str = "https://news.ycombinator.com/item?id="
_HN_SEARCH_COUNT:            int = 5
_HN_SUMMARY_CHARS:           int = 500
_HN_TOP_COMMENT_COUNT:       int = 3
_HN_COMMENT_CHARS:           int = 300

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_comment_text(raw: str) -> str:
    """
    Algolia comment text is HTML (paragraph tags, links, entity-escaped
    quotes/angle-brackets) — strip tags and unescape entities to get plain
    text suitable for a chat answer. Deliberately not full HTML parsing
    (no readability-lxml here, unlike url_fetch's article extraction) —
    comment bodies are short and simply-structured enough that a tag strip
    is sufficient.
    """
    text = _HTML_TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    return " ".join(text.split())


async def fetch_top_comments(object_id: str, count: int = _HN_TOP_COMMENT_COUNT) -> list[dict]:
    """
    Fetch a story's direct top-level comments (not nested replies) via
    Algolia's item API — the only Algolia endpoint that returns comment
    bodies; the Search API used by hacker_news_search() below only ever
    returns story metadata (title, points, comment *count*), never comment
    text. Deleted/flagged comments (null text or author) are skipped
    rather than surfaced as empty entries.

    Returns [{author, text}, ...], in the order Algolia returns them
    (unsorted by score — the item API doesn't expose per-comment points).

    Raises on any transport/HTTP failure — hacker_news_search() below
    contains this into a never-raise contract itself (a comment-fetch
    failure degrades to no comments, not a failed search).
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(_HN_ALGOLIA_ITEM_ENDPOINT.format(object_id))
        resp.raise_for_status()
        data = resp.json()

    comments: list[dict] = []
    for child in data.get("children") or []:
        text = child.get("text")
        author = child.get("author")
        if not text or not author:
            continue
        comments.append({"author": author, "text": _clean_comment_text(text)})
        if len(comments) >= count:
            break
    return comments


async def hacker_news_search(query: str, url: str | None = None) -> dict:
    """
    Run one search query via Algolia's HN Search API, restricted to
    stories (tags=story) — comments/polls/jobs are out of scope here.

    Parameters
    ----------
    query : the search text, passed through to Algolia's `query` param
            unchanged.
    url   : optional — when a caller already knows the exact story (e.g.
            the user clicked a specific story in the Live Feed panel), pass
            its article url to pin the result to that one story rather than
            trusting the query text to find it again among similarly-titled
            submissions. If none of the returned hits match, falls back to
            the normal unfiltered top-5 behavior — same caller-supplied-pin
            convention as news_search()'s own `url` parameter.

    When `url` pins to exactly one story, that story's top
    _HN_TOP_COMMENT_COUNT direct comments (real text, via
    fetch_top_comments — see this module's docstring for why: the Search
    API alone only ever gives a bare comment *count*, and leaving the
    model with just that number and nothing else is what led it to
    fabricate commentary in live testing) are fetched and appended to the
    result as an `extra` block. Not done for an unpinned multi-result
    search — that would be up to _HN_SEARCH_COUNT extra HTTP calls for
    mostly-irrelevant noise when the caller is just browsing several
    stories, not asking about one.

    Returns
    -------
    dict with keys: query, result_text, result_count, is_miss — same shape
    as github_search()'s/news_search()'s return, formatted via the shared
    search_format module so the `[{url}]` bracket-line convention (used by
    MCPToolDispatcher's fetch_url enrichment chaining) is preserved. Each
    result's url is the external article link when the story has one, or
    the HN discussion page itself for a self-post (Ask HN/Show HN) — same
    fallback backend/hacker_news.py's Live Feed uses.

    Raises
    ------
    ValueError
        "ERROR: hacker_news_search failed — <exc>" on any network/HTTP/
        parsing error. A failure to fetch the pinned story's comments does
        NOT raise — it degrades to no comment block, same never-raise
        contract as every other per-item enrichment in this codebase (e.g.
        github_watch.py's per-repo release lookup).
    """
    params = {"query": query, "tags": "story", "hitsPerPage": _HN_SEARCH_COUNT}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_HN_ALGOLIA_SEARCH_ENDPOINT, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("hacker_news_search: failed for query=%r: %s", query, exc)
        raise ValueError(f"ERROR: hacker_news_search failed — {exc}") from exc

    hits = data.get("hits", [])
    if not hits:
        return {"query": query, "result_text": "", "result_count": 0, "is_miss": True}

    def _article_url(hit: dict) -> str:
        object_id = hit.get("objectID", "")
        hn_url = f"{_HN_ITEM_URL_BASE}{object_id}" if object_id else ""
        return hit.get("url") or hn_url

    # Pin to the one already-known story when a caller (e.g. the Live Feed
    # panel's "Ask about this") supplies its url — otherwise a query built
    # from a title can resolve to several similarly-titled submissions and
    # the model would have no way to tell which one the user actually
    # clicked.
    pinned = False
    if url:
        matched = [h for h in hits if _article_url(h) == url]
        if matched:
            hits = matched
            pinned = True

    results: list[search_format.SearchResult] = []
    for hit in hits[:_HN_SEARCH_COUNT]:
        points = hit.get("points")
        num_comments = hit.get("num_comments")
        extra_bits = []
        if points is not None:
            extra_bits.append(f"{points} points")
        if num_comments is not None:
            extra_bits.append(f"{num_comments} comments")
        extra = " · ".join(extra_bits) if extra_bits else None

        if pinned:
            object_id = hit.get("objectID", "")
            comments: list[dict] = []
            if object_id:
                try:
                    comments = await fetch_top_comments(object_id)
                except Exception as exc:
                    logger.warning(
                        "hacker_news_search: comment fetch failed for objectID=%r: %s",
                        object_id, exc,
                    )
            if comments:
                comment_block = "\n".join(
                    f"• {c['author']}: {search_format.truncate_summary(c['text'], _HN_COMMENT_CHARS)}"
                    for c in comments
                )
                comment_block = f"Top comments:\n{comment_block}"
                extra = f"{extra}\n{comment_block}" if extra else comment_block

        results.append(search_format.SearchResult(
            title   = hit.get("title") or "",
            summary = hit.get("story_text") or "",
            url     = _article_url(hit),
            source  = "Hacker News",
            extra   = extra,
        ))

    result_text = search_format.format_results(results, per_result_budget=_HN_SUMMARY_CHARS)
    logger.info(
        "hacker_news_search: complete for query=%r results=%d result_chars=%d.",
        query, len(hits), len(result_text),
    )
    return {"query": query, "result_text": result_text, "result_count": len(hits), "is_miss": False}
