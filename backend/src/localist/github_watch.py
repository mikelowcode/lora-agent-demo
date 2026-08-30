"""
GitHub Watch Feed — watched-repo release polling
========================================================
Fetches the "GitHub" section of the Previews panel's Live Feed: the
authenticated user's watched repos (GET /user/subscriptions) paired with
each one's latest release. The structural twin of news_brief.py — same
"lives in the main backend process, not mcp_server/" rationale (no chat/
tool-dispatch involvement here either — no Planner routing, no
MCPToolDispatcher), same per-item failure containment, same lazy env-var
read.

GITHUB_TOKEN is required here (unlike mcp_server/github.py's
github_search/github_read, which treat it as optional) — GET
/user/subscriptions needs an authenticated identity to know *whose*
watched repos to return; there is no equivalent anonymous call. See
GITHUB_TOKEN's .env.example comment.

Releases only (no commits/issues/PRs) — smallest, cheapest live-update
surface, matching this feature's locked scope.

Zero inference cost end to end — no runtime.infer() call anywhere in this
module, same as news_brief.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_GITHUB_API_BASE:       str = "https://api.github.com"
_GITHUB_API_VERSION:    str = "2022-11-28"
_GITHUB_USER_AGENT:     str = "localist-github-watch"
_SUBSCRIPTIONS_PER_PAGE: int = 100

# Sentinel key for a whole-feed failure entry (see build_watch_feed) —
# distinguishes "GitHub Watch Feed itself failed" from a normal repo whose
# full_name happens to collide with it (full_name always contains "/",
# this key never does).
_WATCH_FEED_ERROR_KEY: str = "_error"


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization":       f"Bearer {token}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": _GITHUB_API_VERSION,
        "User-Agent":           _GITHUB_USER_AGENT,
    }


def _error_entry(message: str) -> dict[str, Any]:
    return {
        "key":            _WATCH_FEED_ERROR_KEY,
        "label":          "GitHub Watch Feed",
        "repo_url":       "",
        "latest_release": None,
        "error":          message,
    }


async def fetch_watched_repos(token: str) -> list[dict[str, Any]]:
    """
    Fetch the authenticated user's watched repos (GET /user/subscriptions,
    first _SUBSCRIPTIONS_PER_PAGE — pagination beyond one page is out of
    scope, same "single-user, small watch list" assumption the rest of
    this feature makes).

    Raises on any transport/HTTP failure — build_watch_feed() below is the
    only caller and contains this into a never-raise contract itself; kept
    raising here so it stays independently testable/reusable.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{_GITHUB_API_BASE}/user/subscriptions",
            headers=_headers(token),
            params={"per_page": _SUBSCRIPTIONS_PER_PAGE},
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_latest_release(token: str, full_name: str) -> dict[str, Any] | None:
    """
    Fetch one repo's latest release. Returns None when the repo has no
    releases (a 404 here means exactly that — most repos never cut a
    release — not an error) rather than raising, so a repo with no
    releases doesn't degrade the whole feed the way a network/auth
    failure should.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{_GITHUB_API_BASE}/repos/{full_name}/releases/latest",
            headers=_headers(token),
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return {
            "tag_name":     data.get("tag_name", ""),
            "name":         data.get("name") or data.get("tag_name", ""),
            "published_at": data.get("published_at", ""),
            "html_url":     data.get("html_url", ""),
        }


async def build_watch_feed(
    pinned_full_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch the full watch feed: every watched repo, each paired with its
    latest release (or None), plus any pinned repos not already watched.

    Never raises — a missing token or a failure to even list watched repos
    degrades to a single-entry feed carrying the error message (same
    never-raise contract news_brief.fetch_section documents for its own
    per-section failures; main.py's POST /github/watch/refresh has no
    try/except around this call, same as its news_brief.build_brief()
    counterpart). Per-repo failure containment for the release lookup only
    — a single repo's release lookup failing (rate limit mid-loop,
    transient error) degrades that one row, not the whole feed. Neither
    the no-token nor the failed-subscriptions-lookup case attempts pinned
    repos — both mean the token itself is unusable, which pinned repos'
    release lookups need just as much as watched repos' do.

    Parameters
    ----------
    pinned_full_names :
        "owner/repo" slugs to track independently of GitHub's Watch
        relationship (see mm.get_pinned_github_repos()). Any entry that
        matches an already-watched repo (case-insensitively) is skipped —
        the watched entry wins and is not duplicated.

    Returns [{key, label, repo_url, latest_release, error, source}, ...]
    — one entry per watched repo (source="watched"), in the order GitHub
    returns them, followed by one entry per pinned repo not already
    watched (source="pinned").
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return [_error_entry("GITHUB_TOKEN not configured")]

    try:
        repos = await fetch_watched_repos(token)
    except Exception as exc:
        logger.warning("github_watch: failed to list watched repos — %s", exc)
        return [_error_entry(str(exc))]

    feed: list[dict[str, Any]] = []
    watched_names: set[str] = set()
    for repo in repos:
        full_name = repo.get("full_name", "")
        if not full_name:
            continue

        try:
            latest_release = await fetch_latest_release(token, full_name)
            error = None
        except Exception as exc:
            logger.warning("github_watch: release lookup failed for %r — %s", full_name, exc)
            latest_release = None
            error = str(exc)

        watched_names.add(full_name.lower())
        feed.append({
            "key":            full_name,
            "label":          full_name,
            "repo_url":       f"https://github.com/{full_name}/releases",
            "latest_release": latest_release,
            "error":          error,
            "source":         "watched",
        })

    for full_name in (pinned_full_names or []):
        if full_name.lower() in watched_names:
            continue

        try:
            latest_release = await fetch_latest_release(token, full_name)
            error = None
        except Exception as exc:
            logger.warning("github_watch: release lookup failed for pinned %r — %s", full_name, exc)
            latest_release = None
            error = str(exc)

        feed.append({
            "key":            full_name,
            "label":          full_name,
            "repo_url":       f"https://github.com/{full_name}/releases",
            "latest_release": latest_release,
            "error":          error,
            "source":         "pinned",
        })

    return feed
