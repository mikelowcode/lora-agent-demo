"""
Localist MCP Server — github tool implementations (github_search, github_read,
github_release)
========================================================
Public-repo GitHub REST reads for on-demand crawling during chat — the
GitHub counterpart to news_search.py. Both tools are read-only (GET-only,
nothing ever written back to GitHub) and never touch archive/zip endpoints
or shell out to `git`/`gh` — no download, no CLI capability, by design.

GITHUB_TOKEN is optional here (unlike NEWSAPI_API_KEY/LANGSEARCH_API_KEY,
which hard-fail when unset) — both endpoints below are public data, and
the token is used opportunistically when present rather than required,
since these tools have no need to act as a specific identity (contrast
backend/github_watch.py's watch-feed fetch, which does need one and
hard-fails without a token). Read lazily on every call, same reasoning as
every other mcp_server tool: this process does not inherit
backend/main.py's own load_dotenv() call.

Rate limits differ by endpoint, confirmed live via response headers —
don't assume one figure covers both tools:
  - github_read (Contents API) uses GitHub's general "core" REST bucket:
    60 req/hr unauthenticated, 5000 req/hr authenticated.
  - github_search (Search API) uses a separate, much stricter bucket:
    10 req/min unauthenticated, 30 req/min authenticated — not the core
    figure above. The token still helps (3x), just far less dramatically.
"""

from __future__ import annotations

import logging
import os

import httpx

from . import search_format

logger = logging.getLogger(__name__)

_GITHUB_API_BASE: str = "https://api.github.com"
_GITHUB_API_VERSION: str = "2022-11-28"
_GITHUB_USER_AGENT: str = "localist-mcp"

_GITHUB_SEARCH_COUNT: int = 5
_GITHUB_SUMMARY_CHARS: int = 500
_GITHUB_CONTENT_CHARS: int = 4000
_GITHUB_RELEASE_BODY_CHARS: int = 3000

_VALID_SEARCH_KINDS = frozenset({"repositories", "code"})


def _headers(*, accept: str = "application/vnd.github+json") -> dict[str, str]:
    """
    Shared header set for every GitHub REST call. A User-Agent is mandatory
    — GitHub's API 403s any request without one. Authorization is added
    only when GITHUB_TOKEN is set (see module docstring).
    """
    headers = {
        "Accept":                accept,
        "X-GitHub-Api-Version":  _GITHUB_API_VERSION,
        "User-Agent":            _GITHUB_USER_AGENT,
    }
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def github_search(query: str, kind: str = "repositories") -> dict:
    """
    Run one search query via GitHub's Search API (/search/repositories or
    /search/code).

    Parameters
    ----------
    query : the search text, passed through to GitHub's `q` search-qualifier
            syntax unchanged (e.g. "topic:mcp language:python").
    kind  : "repositories" (default) or "code".

    Returns
    -------
    dict with keys: query, result_text, result_count, is_miss — same shape
    as news_search.news_search()'s return, formatted via the shared
    search_format module so the `[{url}]` bracket-line convention (used by
    MCPToolDispatcher's fetch_url enrichment chaining) is preserved.

    Raises
    ------
    ValueError
        "ERROR: unknown github_search kind ..." for an invalid kind, or
        "ERROR: github_search failed — <exc>" on any network/HTTP/parsing
        error (rate-limited, invalid query syntax, etc.).
    """
    if kind not in _VALID_SEARCH_KINDS:
        raise ValueError(
            f"ERROR: unknown github_search kind {kind!r} — must be one of {sorted(_VALID_SEARCH_KINDS)}"
        )

    params = {"q": query, "per_page": _GITHUB_SEARCH_COUNT}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{_GITHUB_API_BASE}/search/{kind}", headers=_headers(), params=params
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("github_search: failed for query=%r kind=%r: %s", query, kind, exc)
        raise ValueError(f"ERROR: github_search failed — {exc}") from exc

    items = data.get("items", [])
    if not items:
        return {"query": query, "result_text": "", "result_count": 0, "is_miss": True}

    results: list[search_format.SearchResult] = []
    for item in items[:_GITHUB_SEARCH_COUNT]:
        if kind == "repositories":
            results.append(search_format.SearchResult(
                title   = item.get("full_name", ""),
                summary = item.get("description") or "",
                url     = item.get("html_url", ""),
                source  = "GitHub",
                extra   = f"⭐ {item.get('stargazers_count', 0)}",
            ))
        else:  # code
            repo_name = item.get("repository", {}).get("full_name", "")
            results.append(search_format.SearchResult(
                title   = item.get("path", ""),
                summary = f"in {repo_name}" if repo_name else "",
                url     = item.get("html_url", ""),
                source  = "GitHub",
            ))

    result_text = search_format.format_results(results, per_result_budget=_GITHUB_SUMMARY_CHARS)
    logger.info(
        "github_search: complete for query=%r kind=%r results=%d result_chars=%d.",
        query, kind, len(items), len(result_text),
    )
    return {"query": query, "result_text": result_text, "result_count": len(items), "is_miss": False}


async def github_read(owner: str, repo: str, path: str | None = None) -> dict:
    """
    Read a public repo's README (path=None), a specific file's contents
    (path points at a file), or a directory listing (path points at a
    directory) — all via GitHub's Contents API, one endpoint covering all
    three cases.

    Parameters
    ----------
    owner : repo owner/org login.
    repo  : repo name.
    path  : file or directory path relative to the repo root. Omit (or pass
            None) to read the repo's README instead.

    Returns
    -------
    dict with keys: owner, repo, path, kind ("file" | "directory"), content
    (raw text, or a name-per-line directory listing), truncated (bool —
    file/README content is capped at _GITHUB_CONTENT_CHARS; a directory
    listing is never truncated).

    Raises
    ------
    ValueError
        "ERROR: github_read — <owner>/<repo>[/<path>] not found" on a 404,
        or "ERROR: github_read failed — <exc>" on any other network/HTTP/
        parsing error.
    """
    raw_headers = _headers(accept="application/vnd.github.raw+json")
    endpoint = (
        f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
        if path
        else f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(endpoint, headers=raw_headers)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            target = f"{owner}/{repo}" + (f"/{path}" if path else " (no README)")
            raise ValueError(f"ERROR: github_read — {target} not found") from exc
        logger.warning("github_read: failed for %s/%s path=%r: %s", owner, repo, path, exc)
        raise ValueError(f"ERROR: github_read failed — {exc}") from exc
    except Exception as exc:
        logger.warning("github_read: failed for %s/%s path=%r: %s", owner, repo, path, exc)
        raise ValueError(f"ERROR: github_read failed — {exc}") from exc

    # GitHub only honors the raw media type for file blobs — a directory
    # path always comes back as a JSON array regardless of Accept, so
    # Content-Type is what actually distinguishes the two cases here.
    if "application/json" in resp.headers.get("content-type", ""):
        try:
            entries = resp.json()
        except Exception as exc:
            raise ValueError(f"ERROR: github_read failed — could not parse directory listing ({exc})") from exc

        if not isinstance(entries, list):
            raise ValueError(f"ERROR: github_read — {path or '/'} in {owner}/{repo} is not a file or directory")

        listing = "\n".join(
            f"{'📁' if entry.get('type') == 'dir' else '📄'} {entry.get('name', '')}"
            for entry in entries
        )
        logger.info(
            "github_read: directory listing for %s/%s path=%r entries=%d.",
            owner, repo, path, len(entries),
        )
        return {
            "owner": owner, "repo": repo, "path": path,
            "kind": "directory", "content": listing, "truncated": False,
        }

    text = resp.text
    truncated = len(text) > _GITHUB_CONTENT_CHARS
    if truncated:
        text = text[:_GITHUB_CONTENT_CHARS] + "…"

    logger.info(
        "github_read: complete for %s/%s path=%r chars=%d truncated=%s.",
        owner, repo, path, len(text), truncated,
    )
    return {
        "owner": owner, "repo": repo, "path": path,
        "kind": "file", "content": text, "truncated": truncated,
    }


async def github_release(owner: str, repo: str, tag: str | None = None) -> dict:
    """
    Fetch one release's notes — the latest release (tag=None) or a
    specific tagged release — via GitHub's Releases API. A different
    endpoint from github_read's Contents API (releases aren't repo
    content) and from backend/github_watch.py's own latest-release lookup
    (that one is scoped to the Watch Feed's single-repo-at-a-time,
    latest-only use; this tool is the on-demand, any-repo, any-tag chat
    counterpart — same deliberate cross-process duplication convention as
    news_search.py/news_brief.py, see module docstring).

    Parameters
    ----------
    owner : repo owner/org login.
    repo  : repo name.
    tag   : exact git tag name (e.g. "v0.5.3"). Omit (or pass None) for
            the latest release. If a caller passes a bare version like
            "0.5.3" and that exact tag doesn't exist, one retry is made
            with a toggled "v" prefix (added if missing, stripped if
            present) before giving up — real repos are inconsistent about
            the "v" prefix and a caller asking about "0.5.3" has no way to
            know which convention a given repo uses.

    Returns
    -------
    dict with keys: owner, repo, tag_name, name, body (release notes
    markdown, truncated to _GITHUB_RELEASE_BODY_CHARS), published_at,
    html_url, truncated (bool).

    Raises
    ------
    ValueError
        "ERROR: github_release — <owner>/<repo> has no releases" when
        tag=None and the repo has never cut a release, "ERROR:
        github_release — <owner>/<repo> tag '<tag>' not found" when a
        specific tag doesn't exist (after the v-prefix retry), or
        "ERROR: github_release failed — <exc>" on any other network/HTTP/
        parsing error.
    """
    endpoint = (
        f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/releases/latest"
        if not tag
        else f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/releases/tags/{tag}"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(endpoint, headers=_headers())
    except Exception as exc:
        logger.warning("github_release: failed for %s/%s tag=%r: %s", owner, repo, tag, exc)
        raise ValueError(f"ERROR: github_release failed — {exc}") from exc

    if resp.status_code == 404 and tag:
        alt_tag = tag[1:] if tag.lower().startswith("v") else f"v{tag}"
        alt_endpoint = f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/releases/tags/{alt_tag}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(alt_endpoint, headers=_headers())
        except Exception as exc:
            logger.warning(
                "github_release: v-prefix retry failed for %s/%s tag=%r: %s",
                owner, repo, alt_tag, exc,
            )
            raise ValueError(f"ERROR: github_release failed — {exc}") from exc

    if resp.status_code == 404:
        if tag:
            raise ValueError(f"ERROR: github_release — {owner}/{repo} tag {tag!r} not found")
        raise ValueError(f"ERROR: github_release — {owner}/{repo} has no releases")

    try:
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("github_release: failed for %s/%s tag=%r: %s", owner, repo, tag, exc)
        raise ValueError(f"ERROR: github_release failed — {exc}") from exc

    body = (data.get("body") or "").strip()
    truncated = len(body) > _GITHUB_RELEASE_BODY_CHARS
    if truncated:
        body = body[:_GITHUB_RELEASE_BODY_CHARS] + "…"

    logger.info(
        "github_release: complete for %s/%s tag=%r resolved_tag=%r chars=%d truncated=%s.",
        owner, repo, tag, data.get("tag_name"), len(body), truncated,
    )
    return {
        "owner":        owner,
        "repo":         repo,
        "tag_name":     data.get("tag_name", ""),
        "name":         data.get("name") or data.get("tag_name", ""),
        "body":         body,
        "published_at": data.get("published_at", ""),
        "html_url":     data.get("html_url", ""),
        "truncated":    truncated,
    }
