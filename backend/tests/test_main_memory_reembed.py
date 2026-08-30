"""
Tests for POST /memory/reembed (main.py) — the manual corpus re-embed
endpoint backing MemoryManager.reembed_corpus() (docs/architecture/
16-runtime-backend-layer.md §16.4).

Follows the same TestClient + real-temp-file-MemoryManager pattern as
test_main_memory_episodes.py — the FastAPI lifespan is never triggered;
_state.memory_manager is swapped in per-test instead.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager, _EMBEDDING_DIM


@pytest.fixture()
def client(tmp_path):
    prev_memory = main._state.memory_manager
    yield TestClient(main.app), tmp_path
    main._state.memory_manager = prev_memory


def _embed_fn():
    return MagicMock(side_effect=lambda text: [0.1] * _EMBEDDING_DIM)


class TestReembedEndpoint:
    def test_reembeds_all_documents_and_returns_counts(self, client):
        test_client, tmp_path = client
        embed_fn = _embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "reembed.db", embed_fn=embed_fn,
            embedding_model_name="mlx-community/embeddinggemma-300m-4bit",
        )
        mm.index_document(path=tmp_path / "a.md", doc_type="wiki", content="alpha", embed=True)
        mm.index_document(path=tmp_path / "b.md", doc_type="wiki", content="beta", embed=True)
        main._state.memory_manager = mm

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 200
        body = resp.json()
        assert body == {
            "reembedded": 2,
            "total": 2,
            "model": "mlx-community/embeddinggemma-300m-4bit",
        }

    def test_returns_503_when_memory_manager_not_initialised(self, client):
        test_client, _ = client
        main._state.memory_manager = None

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 503

    def test_returns_409_when_no_embedding_source_configured(self, client):
        test_client, tmp_path = client
        main._state.memory_manager = MemoryManager(db_path=tmp_path / "keyword_only.db")

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 409


class TestMemoryStatsCorpusStale:
    """GET /memory/stats exposes MemoryManager._corpus_stale, and it toggles
    False -> True -> False around a genuine mismatch / reembed_corpus() call
    (docs/architecture/16-runtime-backend-layer.md §16.4)."""

    def test_corpus_stale_false_by_default(self, client):
        test_client, tmp_path = client
        embed_fn = _embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "stats.db", embed_fn=embed_fn,
            embedding_model_name="mlx-community/embeddinggemma-300m-4bit",
        )
        main._state.memory_manager = mm

        resp = test_client.get("/memory/stats")

        assert resp.status_code == 200
        assert resp.json()["corpus_stale"] is False

    def test_corpus_stale_true_when_flagged(self, client):
        test_client, tmp_path = client
        embed_fn = _embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "stats.db", embed_fn=embed_fn,
            embedding_model_name="mlx-community/embeddinggemma-300m-4bit",
        )
        mm._corpus_stale = True
        main._state.memory_manager = mm

        resp = test_client.get("/memory/stats")

        assert resp.status_code == 200
        assert resp.json()["corpus_stale"] is True

    def test_reembed_clears_corpus_stale(self, client):
        test_client, tmp_path = client
        embed_fn = _embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "stats.db", embed_fn=embed_fn,
            embedding_model_name="mlx-community/embeddinggemma-300m-4bit",
        )
        mm.index_document(path=tmp_path / "a.md", doc_type="wiki", content="alpha", embed=True)
        mm._corpus_stale = True
        main._state.memory_manager = mm

        assert test_client.get("/memory/stats").json()["corpus_stale"] is True

        resp = test_client.post("/memory/reembed")
        assert resp.status_code == 200

        assert test_client.get("/memory/stats").json()["corpus_stale"] is False

    def test_returns_false_when_memory_manager_not_initialised(self, client):
        test_client, _ = client
        main._state.memory_manager = None

        resp = test_client.get("/memory/stats")

        assert resp.status_code == 200
        assert resp.json()["corpus_stale"] is False
        assert resp.json()["available"] is False
