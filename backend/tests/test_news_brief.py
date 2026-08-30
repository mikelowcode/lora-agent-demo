"""
Tests for backend/news_brief.py — the Daily News Brief's NewsAPI call
builders (docs/daily-news-brief-plan.md).

Async functions are exercised directly via asyncio.run(), same convention
test_mcp_tool_dispatcher.py already uses (no pytest-asyncio dependency).
httpx.AsyncClient is monkeypatched at the module level — no real network
connection is ever opened.
"""

import asyncio
from unittest.mock import patch

import pytest

from localist import news_brief


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _FakeAsyncClient:
    """Records every (endpoint, params) call it receives, keyed by call
    order, and returns the next queued response — good enough for
    build_brief()'s sequential multi-call flow."""

    calls: list[tuple[str, dict]] = []
    responses: list[dict] = []
    _index = 0

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, headers=None, params=None):
        type(self).calls.append((url, params))
        data = type(self).responses[type(self)._index]
        type(self)._index += 1
        return _FakeResponse(data)


def _patched_client(responses: list[dict]):
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = responses
    _FakeAsyncClient._index = 0
    return patch.object(news_brief.httpx, "AsyncClient", _FakeAsyncClient)


def _article(title="Title", url="https://example.com/a"):
    return {
        "title": title,
        "description": "desc",
        "source": {"name": "Example"},
        "publishedAt": "2026-07-22T00:00:00Z",
        "url": url,
    }


class TestFetchSection:
    def test_missing_api_key_returns_error_without_http_call(self, monkeypatch):
        monkeypatch.delenv("NEWSAPI_API_KEY", raising=False)
        with _patched_client([]):
            result = asyncio.run(news_brief.fetch_section("world", {"category": "general"}))

        assert result["articles"] == []
        assert "NEWSAPI_API_KEY not configured" in result["error"]
        assert _FakeAsyncClient.calls == []

    def test_success_returns_formatted_articles(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        data = {"status": "ok", "totalResults": 1, "articles": [_article()]}
        with _patched_client([data]):
            result = asyncio.run(news_brief.fetch_section("world", {"category": "general"}))

        assert result["error"] is None
        assert len(result["articles"]) == 1
        assert result["articles"][0]["title"] == "Title"
        assert result["articles"][0]["source"] == "Example"
        assert result["articles"][0]["url"] == "https://example.com/a"

    def test_newsapi_error_status_returns_error(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        data = {"status": "error", "code": "rateLimited", "message": "You have exceeded your rate limit."}
        with _patched_client([data]):
            result = asyncio.run(news_brief.fetch_section("world", {"category": "general"}))

        assert result["articles"] == []
        assert "rate limit" in result["error"]

    def test_transport_exception_returns_error_not_raise(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")

        class _RaisingClient(_FakeAsyncClient):
            async def get(self, url, headers=None, params=None):
                raise ConnectionError("boom")

        with patch.object(news_brief.httpx, "AsyncClient", _RaisingClient):
            result = asyncio.run(news_brief.fetch_section("world", {"category": "general"}))

        assert result["articles"] == []
        assert "boom" in result["error"]

    def test_everything_endpoint_used_when_flagged(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        data = {"status": "ok", "totalResults": 0, "articles": []}
        with _patched_client([data]):
            asyncio.run(
                news_brief.fetch_section("local", {"q": "Seattle"}, use_everything=True)
            )

        url, params = _FakeAsyncClient.calls[0]
        assert url == news_brief._EVERYTHING_ENDPOINT
        assert params["q"] == "Seattle"
        assert params["sortBy"] == "publishedAt"


class TestBuildBrief:
    def test_top_stories_always_included(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        ok = {"status": "ok", "totalResults": 1, "articles": [_article()]}
        with _patched_client([ok]):
            sections = asyncio.run(news_brief.build_brief("us", None, []))

        keys = [s["key"] for s in sections]
        assert keys == ["top_stories"]

    def test_top_stories_anchored_to_home_country(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        ok = {"status": "ok", "totalResults": 1, "articles": [_article()]}
        with _patched_client([ok]):
            asyncio.run(news_brief.build_brief("gb", None, []))

        _, params = _FakeAsyncClient.calls[0]
        assert params["country"] == "gb"
        assert "category" not in params

    def test_local_omitted_when_no_query(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        ok = {"status": "ok", "totalResults": 1, "articles": [_article()]}
        with _patched_client([ok, ok]):
            sections = asyncio.run(news_brief.build_brief("us", "", ["finance"]))
        # local_query is falsy ("") -> Local section omitted, but a topic still fires
        keys = [s["key"] for s in sections]
        assert "local" not in keys

    def test_local_included_when_query_set(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        ok = {"status": "ok", "totalResults": 1, "articles": [_article()]}
        with _patched_client([ok, ok]):
            sections = asyncio.run(news_brief.build_brief("us", "Seattle", []))

        keys = [s["key"] for s in sections]
        assert keys == ["top_stories", "local"]

    def test_one_topic_failing_does_not_fail_the_others(self, monkeypatch):
        """Per-section failure containment (docs/daily-news-brief-plan.md §6):
        one bad topic call degrades to an error section, the rest still succeed."""
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        ok = {"status": "ok", "totalResults": 1, "articles": [_article()]}
        error = {"status": "error", "code": "rateLimited", "message": "rate limited"}
        # top_stories, then 2 topics: finance (ok), technology (error)
        with _patched_client([ok, ok, error]):
            sections = asyncio.run(
                news_brief.build_brief("us", None, ["finance", "technology"])
            )

        by_key = {s["key"]: s for s in sections}
        assert by_key["finance"]["error"] is None
        assert len(by_key["finance"]["articles"]) == 1
        assert by_key["technology"]["error"] == "rate limited"
        assert by_key["technology"]["articles"] == []

    def test_unknown_topic_key_skipped(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        ok = {"status": "ok", "totalResults": 1, "articles": [_article()]}
        with _patched_client([ok]):
            sections = asyncio.run(news_brief.build_brief("us", None, ["not-a-real-topic"]))

        keys = [s["key"] for s in sections]
        assert keys == ["top_stories"]

    def test_labels_attached(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
        ok = {"status": "ok", "totalResults": 1, "articles": [_article()]}
        with _patched_client([ok, ok]):
            sections = asyncio.run(news_brief.build_brief("us", None, ["finance"]))

        by_key = {s["key"]: s for s in sections}
        assert by_key["top_stories"]["label"] == "Top Stories"
        assert by_key["finance"]["label"] == "Finance"


class TestFormatBriefMarkdown:
    def test_success_section_lists_articles(self):
        sections = [{
            "key": "world", "label": "World", "error": None,
            "articles": [{
                "title": "Big Story", "description": "Something happened.",
                "source": "Reuters", "published_at": "2026-07-22T00:00:00Z",
                "url": "https://example.com/story",
            }],
        }]
        md = news_brief.format_brief_markdown(sections)
        assert "## World" in md
        assert "Big Story" in md
        assert "Reuters" in md
        assert "https://example.com/story" in md

    def test_error_section_shows_unavailable(self):
        sections = [{"key": "finance", "label": "Finance", "error": "rate limited", "articles": []}]
        md = news_brief.format_brief_markdown(sections)
        assert "Finance" in md
        assert "Unavailable" in md
        assert "rate limited" in md

    def test_empty_section_shows_no_articles(self):
        sections = [{"key": "finance", "label": "Finance", "error": None, "articles": []}]
        md = news_brief.format_brief_markdown(sections)
        assert "No articles found" in md


class TestCollectSources:
    def test_flattens_urls_across_sections(self):
        sections = [
            {"key": "world", "label": "World", "error": None, "articles": [
                {"title": "A", "url": "https://example.com/a", "description": "", "source": "", "published_at": ""},
            ]},
            {"key": "finance", "label": "Finance", "error": None, "articles": [
                {"title": "B", "url": "https://example.com/b", "description": "", "source": "", "published_at": ""},
            ]},
        ]
        sources = news_brief.collect_sources(sections)

        assert len(sources) == 2
        assert all(s["type"] == "web" for s in sources)
        assert {s["path"] for s in sources} == {"https://example.com/a", "https://example.com/b"}

    def test_articles_without_url_skipped(self):
        sections = [{"key": "world", "label": "World", "error": None, "articles": [
            {"title": "A", "url": "", "description": "", "source": "", "published_at": ""},
        ]}]
        assert news_brief.collect_sources(sections) == []
