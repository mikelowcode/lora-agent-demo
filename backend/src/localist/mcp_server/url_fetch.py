"""
Localist MCP Server — url_fetch tool implementation
=======================================================
Ports the retired standalone Fetcher microservice's /extract path
(backend/fetcher/client.py's fetch() + backend/fetcher/extractor.py's
extract()) verbatim, in-process — no HTTP hop to a separate service.

Only /extract is ported. /fetch (raw HTML) and /api (JSON passthrough) are
not — nothing in the codebase calls them, url_fetch only ever needs
"readable content," and porting unused capability speculatively is out of
scope for this phase.

Error taxonomy mirrors fetcher/models.py's ErrorResponse.error_code exactly
(connection_error | timeout | http_client_error | http_server_error |
extraction_failed), carried as a "ERROR: <code> — <message> (<detail>)"
string raised from fetch_url() so the MCP protocol layer's isError path
picks it up — see mcp_tool_dispatcher.py's _normalize_mcp_error_text(),
reused unchanged from the Phase 1 follow-up, which strips FastMCP's own
"Error executing tool <name>: " wrapper back down to this exact shape.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import httpx
import lxml.html
from readability import Document

logger = logging.getLogger(__name__)

# Browser-like UA reduces bot-blocking on common sites (ported from fetcher/client.py).
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_BASE_HEADERS = {
    "User-Agent": _DEFAULT_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}

# Timeout clamp — same bounds as fetcher/models.py's ExtractRequest validator.
_MIN_TIMEOUT: float = 1.0
_MAX_TIMEOUT: float = 30.0


# ---------------------------------------------------------------------------
# Ported from fetcher/client.py — async HTTP GET
# ---------------------------------------------------------------------------

@dataclass
class RawResponse:
    url:               str
    status_code:       int
    content_type:      str
    content:           bytes
    headers:           dict[str, str]
    fetch_duration_ms: float


async def _fetch(url: str, timeout: float = 10.0) -> RawResponse:
    """
    Perform an async HTTP GET. Returns RawResponse.

    Raises
    ------
    httpx.TimeoutException        — caller maps to error_code="timeout"
    httpx.ConnectError            — caller maps to error_code="connection_error"
    httpx.HTTPStatusError         — caller maps to http_client_error / http_server_error
    httpx.RequestError            — caller maps to error_code="connection_error"
    """
    t0 = time.perf_counter()

    logger.debug("fetch_url: fetch() → %s  timeout=%.1fs", url, timeout)

    async with httpx.AsyncClient(
        follow_redirects = True,
        timeout          = httpx.Timeout(timeout),
    ) as client:
        response = await client.get(url, headers=_BASE_HEADERS)
        response.raise_for_status()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    content_type = response.headers.get("content-type", "")

    logger.info(
        "fetch_url: fetch() ← %s  status=%d  content_type=%r  bytes=%d  duration=%.0fms",
        url, response.status_code, content_type, len(response.content), elapsed_ms,
    )

    return RawResponse(
        url               = str(response.url),
        status_code       = response.status_code,
        content_type      = content_type,
        content           = response.content,
        headers           = dict(response.headers),
        fetch_duration_ms = elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Ported from fetcher/extractor.py — readability extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractedContent:
    title:          str
    author:         str
    date_published: str
    cleaned_text:   str
    word_count:     int


def _extract(html: bytes, url: str = "") -> ExtractedContent:
    """
    Extract clean article content from raw HTML bytes.

    Raises
    ------
    ValueError
        If readability produces empty content (e.g. login walls, paywalls).
    """
    html_str = html.decode("utf-8", errors="replace")
    doc    = Document(html_str, url=url)
    title  = (doc.title() or "").strip()
    summary_html = doc.summary(html_partial=True)

    if not summary_html:
        raise ValueError("readability returned empty content — possible paywall or login wall")

    root         = lxml.html.fromstring(summary_html)
    cleaned_text = _clean_text(root.text_content())

    if not cleaned_text:
        raise ValueError("extraction produced empty text after tag stripping")

    try:
        full_doc = lxml.html.fromstring(html)
        author   = _extract_meta(full_doc, ["author", "article:author"])
        date     = _extract_meta(full_doc, [
            "article:published_time", "datePublished", "pubdate", "date",
        ])
    except Exception:
        author = ""
        date   = ""

    word_count = len(cleaned_text.split())

    logger.debug(
        "fetch_url: extract() ← title=%r  words=%d  author=%r  date=%r",
        title, word_count, author, date,
    )

    return ExtractedContent(
        title          = title,
        author         = author,
        date_published = date,
        cleaned_text   = cleaned_text,
        word_count     = word_count,
    )


def _clean_text(raw: str) -> str:
    """Normalise whitespace and strip control characters."""
    text = re.sub(r"\s+", " ", raw)
    return text.strip()


def _extract_meta(doc: "lxml.html.HtmlElement", names: list[str]) -> str:
    """Extract the first matching meta tag content — checks name= and property=."""
    for name in names:
        for attr in ("name", "property"):
            nodes = doc.xpath(f'//meta[@{attr}="{name}"]/@content')
            if nodes:
                return str(nodes[0]).strip()
    return ""


# ---------------------------------------------------------------------------
# fetch_url — the MCP tool entry point
# ---------------------------------------------------------------------------

class FetchUrlError(ValueError):
    """Raised on any fetch/extraction failure — str(self) always starts with 'ERROR:'."""

    def __init__(self, error_code: str, message: str, detail: str = ""):
        self.error_code = error_code
        self.message    = message
        self.detail     = detail
        text = f"ERROR: {error_code} — {message}"
        if detail:
            text += f" ({detail})"
        super().__init__(text)


def _clamp_timeout(v: float) -> float:
    return max(_MIN_TIMEOUT, min(v, _MAX_TIMEOUT))


async def fetch_url(url: str, timeout: float = 10.0) -> dict:
    """
    Fetch a URL and extract clean article content — the /extract path,
    ported in-process. Raises FetchUrlError (a ValueError) on any failure,
    carrying the fetcher/models.py-style error_code in its message.
    """
    clamped = _clamp_timeout(timeout)

    try:
        raw = await _fetch(url, clamped)
    except httpx.TimeoutException as exc:
        raise FetchUrlError("timeout", "Request timed out.", str(exc)) from exc
    except httpx.ConnectError as exc:
        raise FetchUrlError("connection_error", "Could not connect to host.", str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        code = "http_client_error" if exc.response.status_code < 500 else "http_server_error"
        raise FetchUrlError(
            code, f"Target returned HTTP {exc.response.status_code}.", str(exc)
        ) from exc
    except Exception as exc:
        raise FetchUrlError("connection_error", "HTTP request failed.", str(exc)) from exc

    try:
        content = _extract(raw.content, raw.url)
    except ValueError as exc:
        raise FetchUrlError("extraction_failed", "Content extraction failed.", str(exc)) from exc

    return {
        "url":               raw.url,
        "title":             content.title,
        "author":            content.author,
        "date_published":    content.date_published,
        "cleaned_text":      content.cleaned_text,
        "word_count":        content.word_count,
        "fetch_duration_ms": raw.fetch_duration_ms,
    }
