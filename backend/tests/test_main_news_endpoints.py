"""
Tests for the Daily News Brief endpoints in main.py
(docs/daily-news-brief-plan.md §5/§6):

  GET  /news/preferences
  PUT  /news/preferences
  GET  /news/brief/preview
  POST /news/brief/open

Follows the same TestClient + real-temp-file-MemoryManager pattern as
test_main_memory_reembed.py. news_brief.build_brief() is mocked
(AsyncMock) at every call site that would otherwise reach real NewsAPI —
this is a pure wiring/persistence test, not a NewsAPI integration test
(that's news_brief.py's own test_news_brief.py, plus live verification).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager

_FAKE_SECTIONS = [
    {
        "key": "top_stories", "label": "Top Stories", "error": None,
        "articles": [{
            "title": "Big Story", "description": "desc", "source": "Reuters",
            "published_at": "2026-07-22T00:00:00Z", "url": "https://example.com/a",
        }],
    },
    {"key": "local", "label": "Local", "error": None, "articles": []},
]


@pytest.fixture()
def client(tmp_path):
    prev_memory = main._state.memory_manager
    mm = MemoryManager(db_path=tmp_path / "news_endpoints.db")
    main._state.memory_manager = mm
    yield TestClient(main.app), mm
    main._state.memory_manager = prev_memory


class TestGetPreferences:
    def test_returns_defaults_when_unset(self, client):
        test_client, _ = client
        resp = test_client.get("/news/preferences")

        assert resp.status_code == 200
        body = resp.json()
        assert body["home_country"] == "us"
        assert body["local_query"] is None
        assert body["topics"] == []
        assert "finance" in body["topic_pool"]

    def test_returns_stored_preferences(self, client):
        test_client, mm = client
        mm.set_news_preferences("gb", "London", ["finance", "technology", "sports"])

        resp = test_client.get("/news/preferences")

        body = resp.json()
        assert body["home_country"] == "gb"
        assert body["local_query"] == "London"
        assert body["topics"] == ["finance", "technology", "sports"]


class TestPutPreferences:
    def test_valid_request_roundtrips(self, client):
        test_client, _ = client
        resp = test_client.put("/news/preferences", json={
            "home_country": "us", "local_query": "Seattle",
            "topics": ["finance", "technology", "sports"],
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["home_country"] == "us"
        assert body["local_query"] == "Seattle"
        assert body["topics"] == ["finance", "technology", "sports"]

    def test_wrong_topic_count_rejected(self, client):
        test_client, _ = client
        resp = test_client.put("/news/preferences", json={
            "home_country": "us", "local_query": None,
            "topics": ["finance", "technology"],
        })
        assert resp.status_code == 422

    def test_unknown_topic_key_rejected(self, client):
        test_client, _ = client
        resp = test_client.put("/news/preferences", json={
            "home_country": "us", "local_query": None,
            "topics": ["finance", "technology", "not-a-real-topic"],
        })
        assert resp.status_code == 422

    def test_home_country_lowercased(self, client):
        test_client, _ = client
        resp = test_client.put("/news/preferences", json={
            "home_country": "US", "local_query": None,
            "topics": ["finance", "technology", "sports"],
        })
        assert resp.json()["home_country"] == "us"


class TestBriefPreview:
    def test_unavailable_when_no_cache(self, client):
        test_client, _ = client
        with patch.object(main.news_brief, "build_brief", new=AsyncMock()) as mock_build:
            resp = test_client.get("/news/brief/preview")

        assert resp.status_code == 200
        assert resp.json()["available"] is False
        mock_build.assert_not_called()

    def test_unavailable_when_cache_is_from_a_previous_day(self, client):
        test_client, mm = client
        mm.set_news_brief_cache("2000-01-01", _FAKE_SECTIONS, "conv-old")

        with patch.object(main.news_brief, "build_brief", new=AsyncMock()) as mock_build:
            resp = test_client.get("/news/brief/preview")

        assert resp.json()["available"] is False
        mock_build.assert_not_called()

    def test_available_when_cache_matches_today(self, client):
        test_client, mm = client
        today = main._today_str()
        mm.set_news_brief_cache(today, _FAKE_SECTIONS, "conv-today")

        with patch.object(main.news_brief, "build_brief", new=AsyncMock()) as mock_build:
            resp = test_client.get("/news/brief/preview")

        body = resp.json()
        assert body["available"] is True
        assert body["brief_date"] == today
        assert body["sections"][0]["key"] == "top_stories"
        mock_build.assert_not_called()


class TestBriefOpen:
    """
    POST /news/brief/open only refreshes news_brief_cache for the Live Feed
    panel. It used to also open a synthetic chat_turns conversation and
    seed conversation_log with the full brief markdown — removed
    2026-07-24 once the per-article "Ask about this" button made dumping
    the whole, unscoped brief into a chat turn on every refresh redundant
    noise. These tests are the regression guard: a refresh must leave both
    chat_turns and conversation_log untouched.
    """

    def test_generates_and_caches_without_touching_chat(self, client):
        test_client, mm = client
        mm.set_news_preferences("us", None, ["finance", "technology", "sports"])

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ) as mock_build:
            resp = test_client.post("/news/brief/open")

        assert resp.status_code == 200
        assert resp.json() == {"success": True}
        mock_build.assert_awaited_once_with("us", None, ["finance", "technology", "sports"])

        cache = mm.get_news_brief_cache()
        assert cache["brief_date"] == main._today_str()
        assert cache["content"] == _FAKE_SECTIONS

        _, total = mm.get_chat_turns(limit=10)
        assert total == 0

    def test_repeat_call_same_day_always_regenerates(self, client):
        """
        A previous revision reopened a same-day cached conversation instead
        of regenerating. That meant pressing a link literally labeled
        "Daily News Brief Refresh" could silently show stale articles
        instead of doing what the label says — confirmed live, 2026-07-22.
        Every call must still hit NewsAPI again.
        """
        test_client, mm = client
        mm.set_news_preferences("us", None, ["finance", "technology", "sports"])

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ):
            test_client.post("/news/brief/open")

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ) as mock_build:
            test_client.post("/news/brief/open")

        mock_build.assert_awaited_once()

    def test_stale_cache_also_regenerates(self, client):
        test_client, mm = client
        mm.set_news_brief_cache("2000-01-01", _FAKE_SECTIONS, None)

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ) as mock_build:
            resp = test_client.post("/news/brief/open")

        assert resp.status_code == 200
        mock_build.assert_awaited_once()
        assert mm.get_news_brief_cache()["brief_date"] == main._today_str()

    def test_does_not_seed_conversation_log(self, client):
        test_client, mm = client
        mm.set_news_preferences("us", None, ["finance", "technology", "sports"])

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ):
            test_client.post("/news/brief/open")

        assert mm.get_context_window(task_id="global") == []
