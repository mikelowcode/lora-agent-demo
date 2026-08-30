"""
Tests for hacker_news.py — Hacker News Top Stories fetch/build logic.

Mirrors github_watch.py's own test conventions (test_github_watch.py):
httpx.AsyncClient.get mocked per-test, no real network calls.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from localist import hacker_news


def _top_stories_response(ids: list[int], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", f"{hacker_news._HN_API_BASE}/topstories.json")
    return httpx.Response(status_code, json=ids, request=request)


def _item_response(data: dict | None, item_id: int = 1, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", f"{hacker_news._HN_API_BASE}/item/{item_id}.json")
    if data is None:
        # HN's API responds with a literal JSON `null` body for a missing
        # item id — httpx.Response(json=None) would instead send an empty
        # body, so build it from raw content to match the real contract.
        return httpx.Response(status_code, content=b"null", request=request)
    return httpx.Response(status_code, json=data, request=request)


class TestFetchTopStoryIds:
    def test_success_returns_parsed_list(self):
        ids = [111, 222, 333]
        response = _top_stories_response(ids)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.fetch_top_story_ids())
        assert result == ids

    def test_http_error_raises(self):
        response = httpx.Response(
            500, json={"error": "boom"},
            request=httpx.Request("GET", f"{hacker_news._HN_API_BASE}/topstories.json"),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(hacker_news.fetch_top_story_ids())


class TestFetchItem:
    def test_success_returns_item_fields(self):
        data = {"id": 1, "title": "Some story", "url": "https://example.com/a", "score": 42, "by": "alice"}
        response = _item_response(data, item_id=1)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.fetch_item(1))
        assert result == data

    def test_deleted_item_returns_none(self):
        response = _item_response(None, item_id=1)
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            result = asyncio.run(hacker_news.fetch_item(1))
        assert result is None

    def test_http_error_raises(self):
        response = httpx.Response(
            500, json={"error": "boom"},
            request=httpx.Request("GET", f"{hacker_news._HN_API_BASE}/item/1.json"),
        )
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(hacker_news.fetch_item(1))


class TestBuildTopStories:
    def test_top_story_ids_failure_returns_single_error_entry(self):
        with patch.object(
            hacker_news, "fetch_top_story_ids",
            AsyncMock(side_effect=httpx.ConnectError("refused")),
        ):
            result = asyncio.run(hacker_news.build_top_stories())
        assert len(result) == 1
        assert "refused" in result[0]["error"]
        assert result[0]["key"] == hacker_news._TOP_STORIES_ERROR_KEY

    def test_success_builds_one_entry_per_story(self):
        ids = [1, 2]

        async def fake_item(item_id):
            if item_id == 1:
                return {"title": "External link story", "url": "https://example.com/a", "score": 10, "by": "alice"}
            return {"title": "Ask HN: something", "score": 5, "by": "bob"}  # no url — self-post

        with patch.object(hacker_news, "fetch_top_story_ids", AsyncMock(return_value=ids)), \
             patch.object(hacker_news, "fetch_item", fake_item):
            result = asyncio.run(hacker_news.build_top_stories())

        assert len(result) == 2
        assert result[0]["key"] == "1"
        assert result[0]["url"] == "https://example.com/a"
        assert result[0]["error"] is None
        assert result[1]["key"] == "2"
        # self-post falls back to the HN discussion url as its "article" link
        assert result[1]["url"] == result[1]["hn_url"]

    def test_per_item_failure_contained(self):
        ids = [1]
        with patch.object(hacker_news, "fetch_top_story_ids", AsyncMock(return_value=ids)), \
             patch.object(hacker_news, "fetch_item", AsyncMock(side_effect=RuntimeError("boom"))):
            result = asyncio.run(hacker_news.build_top_stories())

        assert len(result) == 1
        assert result[0]["key"] == "1"
        assert "boom" in result[0]["error"]

    def test_deleted_item_skipped(self):
        ids = [1, 2]

        async def fake_item(item_id):
            if item_id == 1:
                return None  # deleted/nonexistent
            return {"title": "Still here", "url": "https://example.com/b", "score": 1, "by": "carol"}

        with patch.object(hacker_news, "fetch_top_story_ids", AsyncMock(return_value=ids)), \
             patch.object(hacker_news, "fetch_item", fake_item):
            result = asyncio.run(hacker_news.build_top_stories())

        assert len(result) == 1
        assert result[0]["key"] == "2"

    def test_respects_count_argument(self):
        ids = list(range(1, 21))

        async def fake_item(item_id):
            return {"title": f"story {item_id}", "url": f"https://example.com/{item_id}", "score": 1, "by": "x"}

        with patch.object(hacker_news, "fetch_top_story_ids", AsyncMock(return_value=ids)), \
             patch.object(hacker_news, "fetch_item", fake_item):
            result = asyncio.run(hacker_news.build_top_stories(count=3))

        assert len(result) == 3
        assert [r["key"] for r in result] == ["1", "2", "3"]
