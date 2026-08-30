"""
Tests for the Hacker News Live Feed endpoints in main.py:

  GET  /hacker-news/top/preview
  POST /hacker-news/top/refresh

Follows the same TestClient + real-temp-file-MemoryManager pattern as
test_main_github_watch_endpoints.py. hacker_news.build_top_stories() is
mocked (AsyncMock) at every call site that would otherwise reach real
Hacker News — this is a pure wiring/persistence test, not a Hacker News
integration test (that's hacker_news.py's own test_hacker_news.py, plus
live verification).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager

_FAKE_STORIES = [
    {
        "key": "111", "title": "Some story",
        "url": "https://example.com/a", "hn_url": "https://news.ycombinator.com/item?id=111",
        "score": 42, "by": "alice", "error": None,
    },
]


@pytest.fixture()
def client(tmp_path):
    prev_memory = main._state.memory_manager
    mm = MemoryManager(db_path=tmp_path / "hacker_news_endpoints.db")
    main._state.memory_manager = mm
    yield TestClient(main.app), mm
    main._state.memory_manager = prev_memory


class TestTopPreview:
    def test_unavailable_when_no_cache(self, client):
        test_client, _ = client
        with patch.object(main.hacker_news, "build_top_stories", new=AsyncMock()) as mock_build:
            resp = test_client.get("/hacker-news/top/preview")

        assert resp.status_code == 200
        assert resp.json()["available"] is False
        mock_build.assert_not_called()

    def test_available_when_cache_present(self, client):
        test_client, mm = client
        mm.set_hacker_news_cache(_FAKE_STORIES)

        with patch.object(main.hacker_news, "build_top_stories", new=AsyncMock()) as mock_build:
            resp = test_client.get("/hacker-news/top/preview")

        body = resp.json()
        assert body["available"] is True
        assert body["stories"][0]["key"] == "111"
        assert body["generated_at"] > 0
        mock_build.assert_not_called()


class TestTopRefresh:
    def test_generates_and_caches(self, client):
        test_client, mm = client

        with patch.object(
            main.hacker_news, "build_top_stories", new=AsyncMock(return_value=_FAKE_STORIES)
        ) as mock_build:
            resp = test_client.post("/hacker-news/top/refresh")

        assert resp.status_code == 200
        assert resp.json() == {"success": True}
        mock_build.assert_awaited_once()

        cache = mm.get_hacker_news_cache()
        assert cache["content"] == _FAKE_STORIES

    def test_repeat_call_always_regenerates(self, client):
        test_client, mm = client

        with patch.object(
            main.hacker_news, "build_top_stories", new=AsyncMock(return_value=_FAKE_STORIES)
        ):
            test_client.post("/hacker-news/top/refresh")

        with patch.object(
            main.hacker_news, "build_top_stories", new=AsyncMock(return_value=_FAKE_STORIES)
        ) as mock_build:
            test_client.post("/hacker-news/top/refresh")

        mock_build.assert_awaited_once()
