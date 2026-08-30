"""
Tests for the FastAPI layer of chat_turns semantic search: GET /chat/history
?mode=semantic, POST /memory/reembed-chat-turns, and GET /memory/stats'
chat_turns_stale field. Episode Browsing UI plan, Phase 1
(episode-browsing-ui-plan.md).

Follows the same TestClient + real-temp-file-MemoryManager pattern as
test_main_memory_reembed.py.
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


class TestReembedChatTurnsEndpoint:
    def test_reembeds_all_turns_and_returns_counts(self, client):
        test_client, tmp_path = client
        embed_fn = _embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "reembed.db", embed_fn=embed_fn,
            embedding_model_name="mlx-community/embeddinggemma-300m-4bit",
        )
        mm.add_chat_turn(task_id="t1", role="user", content="alpha", conversation_id="c1")
        mm.add_chat_turn(task_id="t2", role="user", content="beta", conversation_id="c1")
        main._state.memory_manager = mm

        resp = test_client.post("/memory/reembed-chat-turns")

        assert resp.status_code == 200
        assert resp.json() == {
            "reembedded": 2,
            "total": 2,
            "model": "mlx-community/embeddinggemma-300m-4bit",
        }

    def test_returns_503_when_memory_manager_not_initialised(self, client):
        test_client, _ = client
        main._state.memory_manager = None

        resp = test_client.post("/memory/reembed-chat-turns")

        assert resp.status_code == 503

    def test_returns_409_when_no_embedding_source_configured(self, client):
        test_client, tmp_path = client
        main._state.memory_manager = MemoryManager(db_path=tmp_path / "keyword_only.db")

        resp = test_client.post("/memory/reembed-chat-turns")

        assert resp.status_code == 409


class TestMemoryStatsChatTurnsStale:
    def test_chat_turns_stale_false_by_default(self, client):
        test_client, tmp_path = client
        mm = MemoryManager(
            db_path=tmp_path / "stats.db", embed_fn=_embed_fn(),
            embedding_model_name="mlx-community/embeddinggemma-300m-4bit",
        )
        main._state.memory_manager = mm

        resp = test_client.get("/memory/stats")

        assert resp.status_code == 200
        assert resp.json()["chat_turns_stale"] is False

    def test_chat_turns_stale_true_when_flagged(self, client):
        test_client, tmp_path = client
        mm = MemoryManager(
            db_path=tmp_path / "stats.db", embed_fn=_embed_fn(),
            embedding_model_name="mlx-community/embeddinggemma-300m-4bit",
        )
        mm._chat_turns_stale = True
        main._state.memory_manager = mm

        resp = test_client.get("/memory/stats")

        assert resp.json()["chat_turns_stale"] is True

    def test_reembed_chat_turns_clears_flag(self, client):
        test_client, tmp_path = client
        mm = MemoryManager(
            db_path=tmp_path / "stats.db", embed_fn=_embed_fn(),
            embedding_model_name="mlx-community/embeddinggemma-300m-4bit",
        )
        mm.add_chat_turn(task_id="t", role="user", content="hi", conversation_id="c")
        mm._chat_turns_stale = True
        main._state.memory_manager = mm

        assert test_client.get("/memory/stats").json()["chat_turns_stale"] is True

        resp = test_client.post("/memory/reembed-chat-turns")
        assert resp.status_code == 200

        assert test_client.get("/memory/stats").json()["chat_turns_stale"] is False

    def test_returns_false_when_memory_manager_not_initialised(self, client):
        test_client, _ = client
        main._state.memory_manager = None

        resp = test_client.get("/memory/stats")

        assert resp.status_code == 200
        assert resp.json()["chat_turns_stale"] is False
        assert resp.json()["available"] is False


class TestChatHistorySemanticMode:
    def test_semantic_mode_returns_scored_results(self, client):
        test_client, tmp_path = client
        embed_fn = MagicMock(side_effect=lambda text: (
            [1.0] + [0.0] * (_EMBEDDING_DIM - 1) if "zebra" in text.lower()
            else [0.0] * _EMBEDDING_DIM
        ))
        mm = MemoryManager(
            db_path=tmp_path / "semantic.db", embed_fn=embed_fn,
            embedding_model_name="mlx-community/embeddinggemma-300m-4bit",
        )
        mm.add_chat_turn(task_id="t1", role="user", content="zebra facts", conversation_id="c")
        mm.add_chat_turn(task_id="t2", role="user", content="unrelated content", conversation_id="c")
        main._state.memory_manager = mm

        resp = test_client.get("/chat/history", params={"q": "zebra", "mode": "semantic", "min_score": 0.5})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["turns"][0]["content"] == "zebra facts"
        assert body["turns"][0]["score"] is not None

    def test_keyword_mode_score_is_none(self, client):
        test_client, tmp_path = client
        main._state.memory_manager = MemoryManager(db_path=tmp_path / "keyword.db")
        main._state.memory_manager.add_chat_turn(
            task_id="t", role="user", content="hello world", conversation_id="c",
        )

        resp = test_client.get("/chat/history", params={"q": "hello"})

        assert resp.status_code == 200
        assert resp.json()["turns"][0]["score"] is None

    def test_semantic_mode_falls_back_when_no_embed_fn(self, client):
        test_client, tmp_path = client
        main._state.memory_manager = MemoryManager(db_path=tmp_path / "keyword_only.db")
        main._state.memory_manager.add_chat_turn(
            task_id="t", role="user", content="zebra facts", conversation_id="c",
        )

        resp = test_client.get("/chat/history", params={"q": "zebra", "mode": "semantic"})

        assert resp.status_code == 200
        assert resp.json()["total"] == 1
