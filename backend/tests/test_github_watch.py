"""
Tests for github_watch.py — GitHub Watch Feed fetch/build logic.

Mirrors mcp_server/github.py's own test conventions (test_mcp_server.py):
httpx.AsyncClient.get mocked per-test, no real network calls.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import github_watch


def _subscriptions_response(repos: list[dict], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", f"{github_watch._GITHUB_API_BASE}/user/subscriptions")
    return httpx.Response(status_code, json=repos, request=request)


def _release_response(data: dict | None, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", f"{github_watch._GITHUB_API_BASE}/repos/o/r/releases/latest")
    return httpx.Response(status_code, json=data or {}, request=request)


class TestFetchWatchedRepos:
    def test_success_returns_parsed_list(self):
        repos = [{"full_name": "o/r", "html_url": "https://github.com/o/r"}]
        response = _subscriptions_response(repos)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github_watch.fetch_watched_repos("tok"))
        assert result == repos

    def test_http_error_raises(self):
        response = httpx.Response(
            401, json={"message": "Bad credentials"},
            request=httpx.Request("GET", f"{github_watch._GITHUB_API_BASE}/user/subscriptions"),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(github_watch.fetch_watched_repos("bad-token"))


class TestFetchLatestRelease:
    def test_success_returns_release_fields(self):
        data = {
            "tag_name": "v1.0.0", "name": "v1.0.0", "published_at": "2026-07-01T00:00:00Z",
            "html_url": "https://github.com/o/r/releases/tag/v1.0.0",
        }
        response = _release_response(data)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github_watch.fetch_latest_release("tok", "o/r"))
        assert result == {
            "tag_name":     "v1.0.0",
            "name":         "v1.0.0",
            "published_at": "2026-07-01T00:00:00Z",
            "html_url":     "https://github.com/o/r/releases/tag/v1.0.0",
        }

    def test_no_releases_returns_none(self):
        response = _release_response(None, status_code=404)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(github_watch.fetch_latest_release("tok", "o/r"))
        assert result is None

    def test_other_error_raises(self):
        response = _release_response(None, status_code=500)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(github_watch.fetch_latest_release("tok", "o/r"))


class TestBuildWatchFeed:
    def test_missing_token_returns_single_error_entry(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = asyncio.run(github_watch.build_watch_feed())
        assert len(result) == 1
        assert result[0]["error"] == "GITHUB_TOKEN not configured"

    def test_subscriptions_failure_returns_single_error_entry(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        with patch.object(
            github_watch, "fetch_watched_repos",
            AsyncMock(side_effect=httpx.ConnectError("refused")),
        ):
            result = asyncio.run(github_watch.build_watch_feed())
        assert len(result) == 1
        assert "refused" in result[0]["error"]

    def test_success_builds_one_entry_per_repo(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        repos = [
            {"full_name": "o/has-release", "html_url": "https://github.com/o/has-release"},
            {"full_name": "o/no-release", "html_url": "https://github.com/o/no-release"},
        ]

        async def fake_release(token, full_name):
            if full_name == "o/has-release":
                return {"tag_name": "v1", "name": "v1", "published_at": "x", "html_url": "y"}
            return None

        with patch.object(github_watch, "fetch_watched_repos", AsyncMock(return_value=repos)), \
             patch.object(github_watch, "fetch_latest_release", fake_release):
            result = asyncio.run(github_watch.build_watch_feed())

        assert len(result) == 2
        assert result[0]["key"] == "o/has-release"
        assert result[0]["latest_release"]["tag_name"] == "v1"
        assert result[0]["error"] is None
        assert result[1]["key"] == "o/no-release"
        assert result[1]["latest_release"] is None

    def test_per_repo_release_failure_contained(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        repos = [{"full_name": "o/r", "html_url": "https://github.com/o/r"}]

        with patch.object(github_watch, "fetch_watched_repos", AsyncMock(return_value=repos)), \
             patch.object(
                 github_watch, "fetch_latest_release",
                 AsyncMock(side_effect=RuntimeError("boom")),
             ):
            result = asyncio.run(github_watch.build_watch_feed())

        assert len(result) == 1
        assert result[0]["latest_release"] is None
        assert "boom" in result[0]["error"]

    def test_repo_without_full_name_skipped(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        repos = [{"html_url": "https://github.com/o/r"}]  # no full_name

        with patch.object(github_watch, "fetch_watched_repos", AsyncMock(return_value=repos)):
            result = asyncio.run(github_watch.build_watch_feed())

        assert result == []

    def test_watched_repos_get_watched_source(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        repos = [{"full_name": "o/r", "html_url": "https://github.com/o/r"}]

        with patch.object(github_watch, "fetch_watched_repos", AsyncMock(return_value=repos)), \
             patch.object(github_watch, "fetch_latest_release", AsyncMock(return_value=None)):
            result = asyncio.run(github_watch.build_watch_feed())

        assert result[0]["source"] == "watched"

    def test_pinned_repo_not_in_watched_appears_with_pinned_source(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "tok")

        with patch.object(github_watch, "fetch_watched_repos", AsyncMock(return_value=[])), \
             patch.object(github_watch, "fetch_latest_release", AsyncMock(return_value=None)):
            result = asyncio.run(
                github_watch.build_watch_feed(pinned_full_names=["ollama/ollama"])
            )

        assert len(result) == 1
        assert result[0] == {
            "key":            "ollama/ollama",
            "label":          "ollama/ollama",
            "repo_url":       "https://github.com/ollama/ollama/releases",
            "latest_release": None,
            "error":          None,
            "source":         "pinned",
        }

    def test_pinned_repo_matching_watched_is_not_duplicated(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        watched = [{"full_name": "Ollama/Ollama", "html_url": "https://github.com/Ollama/Ollama"}]

        with patch.object(github_watch, "fetch_watched_repos", AsyncMock(return_value=watched)), \
             patch.object(github_watch, "fetch_latest_release", AsyncMock(return_value=None)):
            result = asyncio.run(
                github_watch.build_watch_feed(pinned_full_names=["ollama/ollama"])
            )

        assert len(result) == 1
        assert result[0]["key"] == "Ollama/Ollama"
        assert result[0]["source"] == "watched"

    def test_pinned_repo_release_failure_contained(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "tok")
        watched = [{"full_name": "o/watched", "html_url": "https://github.com/o/watched"}]

        async def fake_release(token, full_name):
            if full_name == "o/pinned-broken":
                raise RuntimeError("boom")
            return None

        with patch.object(github_watch, "fetch_watched_repos", AsyncMock(return_value=watched)), \
             patch.object(github_watch, "fetch_latest_release", fake_release):
            result = asyncio.run(
                github_watch.build_watch_feed(
                    pinned_full_names=["o/pinned-broken", "o/pinned-ok"]
                )
            )

        assert len(result) == 3
        by_key = {r["key"]: r for r in result}
        assert by_key["o/watched"]["source"] == "watched"
        assert by_key["o/pinned-broken"]["source"] == "pinned"
        assert "boom" in by_key["o/pinned-broken"]["error"]
        assert by_key["o/pinned-ok"]["error"] is None

    def test_missing_token_short_circuits_even_with_pinned_repos(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = asyncio.run(
            github_watch.build_watch_feed(pinned_full_names=["ollama/ollama"])
        )
        assert len(result) == 1
        assert result[0]["error"] == "GITHUB_TOKEN not configured"
