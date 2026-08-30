"""
Hacker News Top Stories — Live Feed panel data source
========================================================
Fetches the "Hacker News" section of the Previews panel's Live Feed: the
current top-ranked stories via HN's public Firebase API. The structural
twin of github_watch.py/news_brief.py — same "lives in the main backend
process, not mcp_server/" rationale (no chat/tool-dispatch involvement
here either — no Planner routing, no MCPToolDispatcher), same per-item
failure containment. Unlike github_watch.py, no token/env var is read at
all — the Firebase API is public and unauthenticated, no key required.

Zero inference cost end to end — no runtime.infer() call anywhere in this
module, same as news_brief.py/github_watch.py.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_HN_API_BASE:      str = "https://hacker-news.firebaseio.com/v0"
_HN_ITEM_URL_BASE: str = "https://news.ycombinator.com/item?id="
_TOP_STORIES_COUNT: int = 10

# Sentinel key for a whole-feed failure entry (see build_top_stories) —
# distinguishes "the feed itself failed" from a normal story whose id
# happens to collide with it (HN item ids are numeric, this key never is).
_TOP_STORIES_ERROR_KEY: str = "_error"


def _error_entry(message: str) -> dict[str, Any]:
    return {
        "key":    _TOP_STORIES_ERROR_KEY,
        "title":  "Hacker News",
        "url":    "",
        "hn_url": "",
        "score":  None,
        "by":     None,
        "error":  message,
    }


async def fetch_top_story_ids() -> list[int]:
    """
    Fetch the current top-story ranking (up to 500 ids, ranked, per HN's
    API contract).

    Raises on any transport/HTTP failure — build_top_stories() below is
    the only caller and contains this into a never-raise contract itself;
    kept raising here so it stays independently testable/reusable, same
    split as github_watch.fetch_watched_repos/fetch_latest_release.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{_HN_API_BASE}/topstories.json")
        resp.raise_for_status()
        return resp.json()


async def fetch_item(item_id: int) -> dict[str, Any] | None:
    """
    Fetch one item's fields. Returns None when the item has been deleted
    or never existed — HN's API responds with a literal JSON `null` body
    for a missing id rather than a 404, so that's the "not found" signal
    here, not an HTTP error status.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{_HN_API_BASE}/item/{item_id}.json")
        resp.raise_for_status()
        return resp.json()


async def build_top_stories(count: int = _TOP_STORIES_COUNT) -> list[dict[str, Any]]:
    """
    Fetch the top `count` stories, each carrying both its external
    article URL (what a reader actually wants to click through to) and
    its HN discussion URL. Self-posts (Ask HN/Show HN, no external `url`
    field) fall back to the HN item page as the "article" link too, so
    every entry still resolves to a real hyperlink.

    Never raises — a failure to list top-story ids degrades to a
    single-entry feed carrying the error message (same never-raise
    contract github_watch.build_watch_feed() documents for itself).
    Per-item failure containment for the item fetch only — a single
    story's lookup failing degrades that one row, not the whole feed. A
    deleted/nonexistent item (fetch_item returns None) is skipped
    entirely rather than surfaced as an error row.

    Returns [{key, title, url, hn_url, score, by, error}, ...], in HN's
    own top-story rank order.
    """
    try:
        story_ids = await fetch_top_story_ids()
    except Exception as exc:
        logger.warning("hacker_news: failed to list top stories — %s", exc)
        return [_error_entry(str(exc))]

    feed: list[dict[str, Any]] = []
    for item_id in story_ids[:count]:
        hn_url = f"{_HN_ITEM_URL_BASE}{item_id}"
        try:
            item = await fetch_item(item_id)
        except Exception as exc:
            logger.warning("hacker_news: item lookup failed for id=%r — %s", item_id, exc)
            feed.append({
                "key":    str(item_id),
                "title":  "",
                "url":    "",
                "hn_url": hn_url,
                "score":  None,
                "by":     None,
                "error":  str(exc),
            })
            continue

        if item is None:
            continue

        feed.append({
            "key":    str(item_id),
            "title":  item.get("title", ""),
            "url":    item.get("url") or hn_url,
            "hn_url": hn_url,
            "score":  item.get("score"),
            "by":     item.get("by"),
            "error":  None,
        })

    return feed
