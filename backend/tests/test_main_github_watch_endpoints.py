"""
Tests for the GitHub Watch Feed endpoints in main.py:

  GET  /github/watch/preview
  POST /github/watch/refresh
  GET  /github/watch/pinned-repos
  PUT  /github/watch/pinned-repos

Follows the same TestClient + real-temp-file-MemoryManager pattern as
test_main_news_endpoints.py. github_watch.build_watch_feed() is mocked
(AsyncMock) at every call site that would otherwise reach real GitHub —
this is a pure wiring/persistence test, not a GitHub integration test
(that's github_watch.py's own test_github_watch.py, plus live verification).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager

_FAKE_REPOS = [
    {
        "key": "anthropics/claude-code", "label": "anthropics/claude-code",
        "repo_url": "https://github.com/anthropics/claude-code",
        "latest_release": {"tag_name": "v1.0.0"}, "error": None,
    },
]


@pytest.fixture()
def client(tmp_path):
    prev_memory = main._state.memory_manager
    mm = MemoryManager(db_path=tmp_path / "github_watch_endpoints.db")
    main._state.memory_manager = mm
    yield TestClient(main.app), mm
    main._state.memory_manager = prev_memory


class TestWatchPreview:
    def test_unavailable_when_no_cache(self, client):
        test_client, _ = client
        with patch.object(main.github_watch, "build_watch_feed", new=AsyncMock()) as mock_build:
            resp = test_client.get("/github/watch/preview")

        assert resp.status_code == 200
        assert resp.json()["available"] is False
        mock_build.assert_not_called()

    def test_available_when_cache_present(self, client):
        test_client, mm = client
        mm.set_github_watch_cache(_FAKE_REPOS)

        with patch.object(main.github_watch, "build_watch_feed", new=AsyncMock()) as mock_build:
            resp = test_client.get("/github/watch/preview")

        body = resp.json()
        assert body["available"] is True
        assert body["repos"][0]["key"] == "anthropics/claude-code"
        assert body["generated_at"] > 0
        mock_build.assert_not_called()


class TestWatchRefresh:
    def test_generates_and_caches(self, client):
        test_client, mm = client

        with patch.object(
            main.github_watch, "build_watch_feed", new=AsyncMock(return_value=_FAKE_REPOS)
        ) as mock_build:
            resp = test_client.post("/github/watch/refresh")

        assert resp.status_code == 200
        assert resp.json() == {"success": True}
        mock_build.assert_awaited_once()

        cache = mm.get_github_watch_cache()
        assert cache["content"] == _FAKE_REPOS

    def test_repeat_call_always_regenerates(self, client):
        test_client, mm = client

        with patch.object(
            main.github_watch, "build_watch_feed", new=AsyncMock(return_value=_FAKE_REPOS)
        ):
            test_client.post("/github/watch/refresh")

        with patch.object(
            main.github_watch, "build_watch_feed", new=AsyncMock(return_value=_FAKE_REPOS)
        ) as mock_build:
            test_client.post("/github/watch/refresh")

        mock_build.assert_awaited_once()

    def test_passes_stored_pinned_repos_to_build_watch_feed(self, client):
        test_client, mm = client
        mm.set_pinned_github_repos(["ollama/ollama", "ml-explore/mlx"])

        with patch.object(
            main.github_watch, "build_watch_feed", new=AsyncMock(return_value=_FAKE_REPOS)
        ) as mock_build:
            test_client.post("/github/watch/refresh")

        mock_build.assert_awaited_once_with(
            pinned_full_names=["ollama/ollama", "ml-explore/mlx"]
        )


class TestPinnedRepos:
    def test_get_returns_empty_list_by_default(self, client):
        test_client, _ = client
        resp = test_client.get("/github/watch/pinned-repos")

        assert resp.status_code == 200
        assert resp.json() == {"repos": []}

    def test_put_happy_path_persists_and_echoes(self, client):
        test_client, mm = client
        resp = test_client.put(
            "/github/watch/pinned-repos",
            json={"repos": ["ollama/ollama", "ml-explore/mlx"]},
        )

        assert resp.status_code == 200
        assert resp.json() == {"repos": ["ollama/ollama", "ml-explore/mlx"]}
        assert mm.get_pinned_github_repos() == ["ollama/ollama", "ml-explore/mlx"]

    def test_get_reflects_previously_put_list(self, client):
        test_client, _ = client
        test_client.put("/github/watch/pinned-repos", json={"repos": ["ollama/ollama"]})

        resp = test_client.get("/github/watch/pinned-repos")
        assert resp.json() == {"repos": ["ollama/ollama"]}

    def test_put_rejects_malformed_slug(self, client):
        test_client, _ = client
        resp = test_client.put(
            "/github/watch/pinned-repos", json={"repos": ["not-a-slug"]}
        )
        assert resp.status_code == 422

    def test_put_rejects_slug_with_extra_slash(self, client):
        test_client, _ = client
        resp = test_client.put(
            "/github/watch/pinned-repos", json={"repos": ["o/r/extra"]}
        )
        assert resp.status_code == 422

    def test_put_rejects_over_max_count(self, client):
        test_client, _ = client
        too_many = [f"owner/repo{i}" for i in range(21)]
        resp = test_client.put("/github/watch/pinned-repos", json={"repos": too_many})
        assert resp.status_code == 422

    def test_put_rejects_case_insensitive_duplicates(self, client):
        test_client, _ = client
        resp = test_client.put(
            "/github/watch/pinned-repos",
            json={"repos": ["Ollama/Ollama", "ollama/ollama"]},
        )
        assert resp.status_code == 422

    def test_put_does_not_trigger_a_refresh(self, client):
        test_client, _ = client
        with patch.object(
            main.github_watch, "build_watch_feed", new=AsyncMock()
        ) as mock_build:
            test_client.put("/github/watch/pinned-repos", json={"repos": ["ollama/ollama"]})

        mock_build.assert_not_called()
