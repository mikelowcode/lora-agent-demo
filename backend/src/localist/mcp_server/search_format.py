"""
Localist MCP Server — shared search-result formatting
========================================================
web_search.py (LangSearch, Brave) and news_search.py (NewsAPI) each return a
list of provider-shaped results that need to become one deterministic
"Title — Source — Summary" bullet-list string before they're handed back as
`result_text`. This module is that one shared formatter, used by all three
so the three providers don't each hand-roll a slightly different bullet
shape.

The `[{url}]` line is load-bearing: mcp_tool_dispatcher.py's
`_extract_first_url()` / `_URL_RE` depends on every result URL appearing on
its own line wrapped in literal brackets (used to chain a follow-up
`fetch_url` call onto a search result). Preserve that line shape exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class SearchResult:
    title:        str
    summary:      str
    url:          str
    source:       str | None = None   # explicit publisher name; derived from url if falsy
    published_at: str | None = None   # folded into the header line when present
    extra:        str | None = None   # e.g. NewsAPI pinned-article content, appended after summary


def derive_source_from_url(url: str) -> str:
    """
    Best-effort site name for a result with no explicit publisher field
    (Brave, LangSearch both only supply a URL). Handles both full URLs
    (Brave's `url`, scheme included) and bare-domain strings (LangSearch's
    `displayUrl`, e.g. "example.com/omlx", no scheme) via the "//" trick,
    which makes urlparse treat the leading segment as netloc either way.
    """
    parsed = urlparse(url if "://" in url else f"//{url}")
    netloc = parsed.netloc or url
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def truncate_summary(summary: str, budget: int) -> str:
    """
    Word-boundary cut at `budget` chars. Appends a single "…" iff this call
    actually shortened the text — a bare per-summary truncation signal,
    deliberately distinct from PromptBuilder's own "… [truncated]" marker
    (that one means "the whole result got cut, other content may be
    missing entirely"; this one just means "this summary was shortened").
    """
    if len(summary) <= budget:
        return summary
    cut = summary[:budget].rsplit(" ", 1)[0]
    return cut + "…"


def format_results(results: list[SearchResult], *, per_result_budget: int) -> str:
    """
    Render `results` as a "\\n\\n"-joined bullet list:

        • {title} — {source}[ — {published_at}]
          {summary, truncated to per_result_budget}
          [{url}]
        [  {extra}, only when set  ]

    Returns "" for an empty list — callers keep owning their own
    "No results found." sentinel string rather than this module hardcoding it.
    """
    if not results:
        return ""

    blocks: list[str] = []
    for r in results:
        source = r.source or derive_source_from_url(r.url)
        header = f"{r.title} — {source}"
        if r.published_at:
            header += f" — {r.published_at}"

        summary = truncate_summary(r.summary, per_result_budget)
        block = f"• {header}\n  {summary}\n  [{r.url}]"
        if r.extra:
            block += f"\n  {r.extra}"
        blocks.append(block)

    return "\n\n".join(blocks)
