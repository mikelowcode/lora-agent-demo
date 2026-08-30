"""
Daily News Brief — NewsAPI call builders
===========================================
Fetches and formats the sections behind the Daily News Brief feature
(docs/daily-news-brief-plan.md): Top Stories, Local (keyword-
approximated), and the 9-member special-interest topic pool (§3).

Lives in the main backend process, not mcp_server/ — this feature has no
chat/tool-dispatch involvement (no Planner routing, no MCPToolDispatcher),
so routing it through localist-mcp's MCP/SSE transport would be
unnecessary indirection; it's a dedicated REST feature, same tier as
memory_manager.py's other direct callers. NEWSAPI_API_KEY is read lazily
on every call, same reasoning mcp_server/news_search.py documents for its
own key reads — this does duplicate a small amount of NewsAPI
request-building logic already there, deliberately: matches this
codebase's established cross-process-duplication convention (see that
module's docstring, and file_ops.py's _MAX_FILE_READ_CHARS /
chart.py's validate_chart_arguments() for the same reasoning applied
elsewhere).

Zero inference cost end to end — no runtime.infer() call anywhere in this
module (docs/daily-news-brief-plan.md §6).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TOP_HEADLINES_ENDPOINT: str = "https://newsapi.org/v2/top-headlines"
_EVERYTHING_ENDPOINT:    str = "https://newsapi.org/v2/everything"
_SECTION_PAGE_SIZE:      int = 5

# Special-interest topic pool (docs/daily-news-brief-plan.md §3). Each entry
# maps a topic key to the NewsAPI endpoint/params that back it — "top_headlines"
# entries are reliable native categories; "everything" entries are keyword-
# approximated, same caveat as the Local section, for topics NewsAPI has no
# native category for. This is the domain constant news_preferences.topics_json
# entries are validated against (main.py's PUT /news/preferences) —
# memory_manager.py's set_news_preferences() deliberately doesn't know this
# pool, only that exactly 3 entries are required.
NEWS_TOPIC_POOL: dict[str, dict[str, Any]] = {
    "finance":       {"endpoint": "top_headlines", "params": {"category": "business"}},
    "technology":    {"endpoint": "top_headlines", "params": {"category": "technology"}},
    "science":       {"endpoint": "top_headlines", "params": {"category": "science"}},
    "health":        {"endpoint": "top_headlines", "params": {"category": "health"}},
    "sports":        {"endpoint": "top_headlines", "params": {"category": "sports"}},
    "entertainment": {"endpoint": "top_headlines", "params": {"category": "entertainment"}},
    "video_games":   {"endpoint": "everything",    "params": {"q": '"video games"'}},
    "politics":      {"endpoint": "everything",    "params": {"q": "politics"}},
    "crypto":        {"endpoint": "everything",    "params": {"q": "cryptocurrency OR crypto"}},
}

# Display labels — shared by the preferences endpoint (so the frontend topic
# picker never drifts from the pool above) and format_brief_markdown() below.
NEWS_TOPIC_LABELS: dict[str, str] = {
    "finance":       "Finance",
    "technology":    "Technology",
    "science":       "Science",
    "health":        "Health",
    "sports":        "Sports",
    "entertainment": "Entertainment",
    "video_games":   "Video Games",
    "politics":      "Politics",
    "crypto":        "Crypto",
}


async def _fetch_top_headlines(params: dict[str, Any], api_key: str) -> dict:
    request_params = {**params, "pageSize": _SECTION_PAGE_SIZE}
    headers = {"X-Api-Key": api_key}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(_TOP_HEADLINES_ENDPOINT, headers=headers, params=request_params)
        return resp.json()


async def _fetch_everything(params: dict[str, Any], api_key: str) -> dict:
    request_params = {**params, "pageSize": _SECTION_PAGE_SIZE, "sortBy": "publishedAt", "language": "en"}
    headers = {"X-Api-Key": api_key}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(_EVERYTHING_ENDPOINT, headers=headers, params=request_params)
        return resp.json()


def _format_articles(data: dict) -> list[dict[str, str]]:
    """
    Normalize a NewsAPI response's `articles` array into plain fields for
    direct rendering — deliberately NOT the formatted-bullet-string shape
    mcp_server/news_search.py returns for prompt-slot consumption (§14.9).
    That tool feeds an LLM prompt slot; this feeds real JSON fields for a
    chat-turn markdown message and (via collect_sources()) a sources list.
    """
    articles = data.get("articles", []) or []
    return [
        {
            "title":        a.get("title") or "",
            "description":  a.get("description") or "",
            "source":       (a.get("source") or {}).get("name") or "",
            "published_at": a.get("publishedAt") or "",
            "url":          a.get("url") or "",
        }
        for a in articles
    ]


async def fetch_section(
    key:    str,
    params: dict[str, Any],
    *,
    use_everything: bool = False,
) -> dict[str, Any]:
    """
    Fetch one brief section (World/National/Local, or one topic-pool entry).

    Returns {key, articles, error} — `error` is None on success, a short
    human-readable string on failure. Never raises: a missing API key or a
    NewsAPI-side failure degrades this one section only — per-section
    failure containment, docs/daily-news-brief-plan.md §6 — the caller
    (build_brief() below) must not let one bad section fail the whole brief.
    """
    api_key = os.environ.get("NEWSAPI_API_KEY", "")
    if not api_key:
        return {"key": key, "articles": [], "error": "NEWSAPI_API_KEY not configured"}

    try:
        data = (
            await _fetch_everything(params, api_key)
            if use_everything
            else await _fetch_top_headlines(params, api_key)
        )
    except Exception as exc:
        logger.warning("news_brief: section %r failed — %s", key, exc)
        return {"key": key, "articles": [], "error": str(exc)}

    if data.get("status") != "ok":
        message = data.get("message", "unknown NewsAPI error")
        logger.warning("news_brief: section %r NewsAPI error — %s", key, message)
        return {"key": key, "articles": [], "error": message}

    return {"key": key, "articles": _format_articles(data), "error": None}


async def build_brief(
    home_country: str,
    local_query:  str | None,
    topics:       list[str],
) -> list[dict[str, Any]]:
    """
    Fetch every section for one brief: Top Stories, Local (only when
    local_query is set — no point calling NewsAPI with an empty/meaningless
    query), then the 3 selected topics in the order given.

    Top Stories used to be two separate calls, "World" (`category=general`,
    no country) and "National" (`country=home_country`, no category) —
    NewsAPI's top-headlines endpoint returns near-identical results for
    those two param shapes in practice, so the brief was showing the same
    headlines twice under different labels. Collapsed into one call
    (2026-07-24), still anchored to the user's home country.

    Runs sequentially, not concurrently — at most 5 calls, well within a
    single request's reasonable latency budget, and keeps NewsAPI request
    pacing simple (docs/daily-news-brief-plan.md §7's rate-limit budget).

    Returns a list of section dicts: {key, label, articles, error}, in
    fixed display order (Top Stories, Local, then topics).
    """
    sections: list[dict[str, Any]] = []

    top_stories = await fetch_section("top_stories", {"country": home_country})
    sections.append({**top_stories, "label": "Top Stories"})

    if local_query:
        local = await fetch_section("local", {"q": local_query}, use_everything=True)
        sections.append({**local, "label": "Local"})

    for topic_key in topics:
        topic = NEWS_TOPIC_POOL.get(topic_key)
        if topic is None:
            logger.warning("news_brief: unknown topic key %r — skipping.", topic_key)
            continue
        section = await fetch_section(
            topic_key,
            topic["params"],
            use_everything=(topic["endpoint"] == "everything"),
        )
        sections.append({**section, "label": NEWS_TOPIC_LABELS.get(topic_key, topic_key)})

    return sections


def format_brief_markdown(sections: list[dict[str, Any]]) -> str:
    """
    Format fetched sections into the markdown message inserted as the
    assistant chat_turn (docs/daily-news-brief-plan.md §6). Pure string
    formatting — no inference call, by design.
    """
    lines: list[str] = []
    for section in sections:
        lines.append(f"## {section['label']}")
        if section.get("error"):
            lines.append(f"_Unavailable: {section['error']}_")
        elif not section["articles"]:
            lines.append("_No articles found._")
        else:
            for article in section["articles"]:
                lines.append(
                    f"- **{article['title']}** — {article['source']} "
                    f"({article['published_at']})"
                )
                if article["description"]:
                    lines.append(f"  {article['description']}")
                lines.append(f"  {article['url']}")
        lines.append("")
    return "\n".join(lines).strip()


def collect_sources(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flatten every article URL across all sections into the same
    {"path", "type", "name"} shape ControllerAgent already populates
    `sources` with elsewhere (controller_agent.py's
    _build_conversational_result) — same provenance convention a normal
    grounded answer uses, not a special case for this feature. `type:
    "web"` is a new value (existing values are "wiki"/"raw"/"session") —
    ChatPanel.svelte's source-badge rendering is extended to link out for it.
    """
    sources: list[dict[str, Any]] = []
    for section in sections:
        for article in section.get("articles", []):
            if article.get("url"):
                sources.append({
                    "path": article["url"],
                    "type": "web",
                    "name": article.get("title") or article["url"],
                })
    return sources
