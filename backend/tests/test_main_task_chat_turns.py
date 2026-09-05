"""
Tests for the chat_turns write path wired into POST /task (main.py), and
the GET/PUT /settings/retention endpoints (independent chat_turns/episode
retention presets as of §20.12 — previously one shared preset).

Covers:
  - A successful task persists exactly one user row and one assistant row.
  - A failed task (unhandled exception in controller.handle_task) persists
    a user row but no assistant row — a failed task has no real answer.
  - A memory_manager write failure (add_chat_turn raising) is swallowed and
    never breaks the endpoint's normal successful response.
  - GET /settings/retention returns eviction_preset=null,
    episode_eviction_preset="forever" before any PUT; PUT persists either
    or both presets independently and returns the current values; PUT with
    an invalid preset (either field) is rejected with 422 at the
    request-validation layer.
  - GET /chat/history lists chat_turns (paginated, optionally full-text
    filtered via ?q=).

The FastAPI lifespan (real runtime / embedding model / controller wiring)
is never triggered here — TestClient only runs lifespan when used as a
context manager, and this suite instantiates it directly. _state.controller
and _state.memory_manager are set up per-test instead.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager


@pytest.fixture()
def client(tmp_path):
    """
    TestClient against main.app with a mocked controller and a real,
    temp-file-backed MemoryManager so chat_turns rows can be inspected.

    Restores the previous _state.controller / _state.memory_manager after
    the test so this suite doesn't leak state into other test modules.
    """
    prev_controller = main._state.controller
    prev_memory     = main._state.memory_manager

    main._state.memory_manager = MemoryManager(db_path=tmp_path / "main_chat_turns.db")
    main._state.controller     = MagicMock()

    yield TestClient(main.app)

    main._state.controller     = prev_controller
    main._state.memory_manager = prev_memory


def _chat_turns(memory_manager: MemoryManager) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(memory_manager._db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM chat_turns ORDER BY id").fetchall()
    conn.close()
    return rows


class TestPostTaskChatTurns:

    def test_successful_task_persists_user_and_assistant_rows(self, client):
        main._state.controller.handle_task.return_value = {
            "task_id":  "t1",
            "status":   "complete",
            "answer":   "the answer",
            "sources":  [{"name": "doc"}],
            "metadata": {"agent": "wiki"},
        }

        resp = client.post(
            "/task",
            json={"task_id": "t1", "instruction": "do something", "conversation_id": "c1"},
        )
        assert resp.status_code == 200

        rows = _chat_turns(main._state.memory_manager)
        assert len(rows) == 2
        assert rows[0]["role"]    == "user"
        assert rows[0]["content"] == "do something"
        assert rows[1]["role"]    == "assistant"
        assert rows[1]["content"] == "the answer"

    def test_failed_task_persists_user_row_only(self, client):
        main._state.controller.handle_task.side_effect = RuntimeError("boom")

        resp = client.post(
            "/task",
            json={"task_id": "t2", "instruction": "trigger a failure", "conversation_id": "c2"},
        )
        assert resp.status_code == 500

        rows = _chat_turns(main._state.memory_manager)
        assert len(rows) == 1
        assert rows[0]["role"]    == "user"
        assert rows[0]["content"] == "trigger a failure"

    def test_memory_write_failure_does_not_break_response(self, client):
        main._state.controller.handle_task.return_value = {
            "task_id":  "t3",
            "status":   "complete",
            "answer":   "still works",
            "sources":  [],
            "metadata": {},
        }

        with patch.object(
            main._state.memory_manager, "add_chat_turn", side_effect=RuntimeError("db down"),
        ):
            resp = client.post(
                "/task",
                json={"task_id": "t3", "instruction": "trigger a memory failure", "conversation_id": "c3"},
            )

        assert resp.status_code == 200
        assert resp.json()["answer"] == "still works"


class TestRetentionSettingsEndpoints:
    """
    eviction_preset (chat_turns) and episode_eviction_preset (episodes) are
    independent as of §20.12 — episode_eviction_preset defaults to the
    concrete value "forever" (never null) and is untouched by a PUT that
    only names eviction_preset, and vice versa.
    """

    def test_get_returns_defaults_before_any_put(self, client):
        resp = client.get("/settings/retention")
        assert resp.status_code == 200
        assert resp.json() == {"eviction_preset": None, "episode_eviction_preset": "forever"}

    def test_put_valid_chat_preset_returns_200_leaves_episode_preset_default(self, client):
        resp = client.put("/settings/retention", json={"eviction_preset": "30d"})
        assert resp.status_code == 200
        assert resp.json() == {"eviction_preset": "30d", "episode_eviction_preset": "forever"}

    def test_put_invalid_chat_preset_returns_422(self, client):
        resp = client.put("/settings/retention", json={"eviction_preset": "60d"})
        assert resp.status_code == 422

    def test_put_invalid_episode_preset_returns_422(self, client):
        resp = client.put("/settings/retention", json={"episode_eviction_preset": "60d"})
        assert resp.status_code == 422

    def test_get_after_put_reflects_new_value(self, client):
        put_resp = client.put("/settings/retention", json={"eviction_preset": "forever"})
        assert put_resp.status_code == 200

        get_resp = client.get("/settings/retention")
        assert get_resp.status_code == 200
        assert get_resp.json() == {"eviction_preset": "forever", "episode_eviction_preset": "forever"}

    def test_put_episode_preset_alone_leaves_chat_preset_untouched(self, client):
        resp = client.put("/settings/retention", json={"episode_eviction_preset": "90d"})
        assert resp.status_code == 200
        assert resp.json() == {"eviction_preset": None, "episode_eviction_preset": "90d"}

    def test_put_both_presets_in_one_request(self, client):
        resp = client.put(
            "/settings/retention",
            json={"eviction_preset": "7d", "episode_eviction_preset": "90d"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"eviction_preset": "7d", "episode_eviction_preset": "90d"}

    def test_put_second_call_omitting_episode_preset_does_not_reset_it(self, client):
        client.put("/settings/retention", json={"episode_eviction_preset": "90d"})
        resp = client.put("/settings/retention", json={"eviction_preset": "7d"})
        assert resp.status_code == 200
        assert resp.json() == {"eviction_preset": "7d", "episode_eviction_preset": "90d"}


class TestChatHistoryEndpoint:

    def test_no_rows_returns_empty_turns_and_zero_total(self, client):
        resp = client.get("/chat/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["turns"]  == []
        assert body["total"]  == 0
        assert body["offset"] == 0
        assert body["limit"]  == 50

    def test_paginated_results_without_q(self, client):
        for i in range(3):
            main._state.memory_manager.add_chat_turn(
                task_id="t", role="user", content=f"turn {i}", conversation_id="conv-1",
            )

        resp = client.get("/chat/history", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        body = resp.json()

        assert body["total"]  == 3
        assert body["offset"] == 0
        assert body["limit"]  == 2
        assert len(body["turns"]) == 2
        assert body["turns"][0]["content"] == "turn 2"   # newest first

    def test_q_filters_to_matching_subset(self, client):
        main._state.memory_manager.add_chat_turn(task_id="t", role="user", content="tell me about zebras", conversation_id="conv-1")
        main._state.memory_manager.add_chat_turn(task_id="t", role="assistant", content="zebras are striped", conversation_id="conv-1")
        main._state.memory_manager.add_chat_turn(task_id="t", role="user", content="capital of France", conversation_id="conv-1")

        resp = client.get("/chat/history", params={"q": "zebras"})
        assert resp.status_code == 200
        body = resp.json()

        assert body["total"] == 2
        assert len(body["turns"]) == 2
        assert all("zebra" in t["content"].lower() for t in body["turns"])

    def test_out_of_range_offset_returns_empty_turns_correct_total(self, client):
        for i in range(3):
            main._state.memory_manager.add_chat_turn(
                task_id="t", role="user", content=f"turn {i}", conversation_id="conv-1",
            )

        resp = client.get("/chat/history", params={"offset": 100})
        assert resp.status_code == 200
        body = resp.json()

        assert body["turns"] == []
        assert body["total"] == 3
