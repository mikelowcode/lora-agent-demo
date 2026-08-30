"""
Tests for the episodic memory REST endpoints (main.py):
GET /memory/episodes, GET /memory/episodes/related,
POST /memory/episodes/{id}/approve, POST /memory/episodes/{id}/reject.

Follows the same TestClient + real-temp-file-MemoryManager pattern as
tests/test_main_task_chat_turns.py — the FastAPI lifespan is never
triggered; _state.memory_manager is swapped in per-test instead.
"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager, EpisodicMemoryWriter


@pytest.fixture()
def client(tmp_path):
    """
    TestClient against main.app with a real, temp-file-backed MemoryManager
    so pending/active/retracted episode rows can be seeded and inspected
    directly. Restores the previous _state.memory_manager afterward so this
    suite doesn't leak state into other test modules.
    """
    prev_memory = main._state.memory_manager

    main._state.memory_manager = MemoryManager(db_path=tmp_path / "main_episodes.db")

    yield TestClient(main.app)

    main._state.memory_manager = prev_memory


def _status(memory_manager: MemoryManager, episode_id: int) -> str:
    conn = sqlite3.connect(str(memory_manager._db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT status FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    conn.close()
    return row["status"]


def _insert_pending(memory_manager: MemoryManager) -> int:
    writer = EpisodicMemoryWriter(db_path=memory_manager._db_path)
    row_id = writer.insert(
        episode_type   = "project_fact",
        subject        = "staged fact",
        content        = "The user mentioned something offhand.",
        source         = "model_extracted",
        confidence     = 0.7,
        initial_status = "pending",
    )
    assert row_id is not None
    return row_id


def _graph_node_type(memory_manager: MemoryManager, doc_path: str) -> str | None:
    conn = sqlite3.connect(str(memory_manager._db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT node_type FROM graph_nodes WHERE doc_path = ?", (doc_path,)
    ).fetchone()
    conn.close()
    return row["node_type"] if row is not None else None


class TestApproveEndpoint:

    def test_approve_pending_row_returns_updated_true_and_activates(self, client):
        episode_id = _insert_pending(main._state.memory_manager)

        resp = client.post(f"/memory/episodes/{episode_id}/approve")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": episode_id, "status": "active", "updated": True}
        assert _status(main._state.memory_manager, episode_id) == "active"

    def test_approve_nonexistent_id_returns_updated_false(self, client):
        resp = client.post("/memory/episodes/999999/approve")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": 999999, "status": "active", "updated": False}

    def test_approve_already_active_row_returns_updated_false(self, client):
        # Ordinary explicit insert — active from the start, never pending.
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        row_id = writer.insert("preference", "x", "y.", "explicit")

        resp = client.post(f"/memory/episodes/{row_id}/approve")
        assert resp.status_code == 200
        assert resp.json()["updated"] is False
        assert _status(main._state.memory_manager, row_id) == "active"

    def test_approve_without_controller_still_succeeds_no_graph_node(self, client):
        # main._state.controller is never set by the `client` fixture (no
        # FastAPI lifespan triggered in this suite) — approve() must
        # degrade gracefully rather than raise when there's no controller
        # to retrigger the Phase B graph hook on.
        assert main._state.controller is None
        episode_id = _insert_pending(main._state.memory_manager)

        resp = client.post(f"/memory/episodes/{episode_id}/approve")

        assert resp.status_code == 200
        assert resp.json()["updated"] is True
        assert _graph_node_type(
            main._state.memory_manager, f"episode://{episode_id}",
        ) is None


class TestApproveEndpointGraphHook:
    """
    Approving a pending episode must retrigger
    ControllerAgent._write_episode_graph_node() (memory-graph-inference-plan
    §8.9) — the implicit-extraction hook in _execute_plan() deliberately
    skips graph representation for a still-pending episode, so approval is
    the only remaining trigger point.
    """

    @pytest.fixture()
    def controller(self, client):
        from unittest.mock import MagicMock

        from localist.controller_agent import ControllerAgent

        prev_controller = main._state.controller
        main._state.controller = ControllerAgent(
            runtime        = MagicMock(),
            agents         = [],
            memory_manager = main._state.memory_manager,
        )
        yield main._state.controller
        main._state.controller = prev_controller

    def test_approve_creates_episode_graph_node(self, client, controller):
        episode_id = _insert_pending(main._state.memory_manager)

        resp = client.post(f"/memory/episodes/{episode_id}/approve")

        assert resp.status_code == 200
        assert resp.json()["updated"] is True
        assert _graph_node_type(
            main._state.memory_manager, f"episode://{episode_id}",
        ) == "episode"

    def test_approve_nonexistent_id_does_not_call_graph_hook(self, client, controller):
        # count == 0 (nothing approved) must short-circuit before the hook
        # ever runs — no graph node for an id that was never approved.
        resp = client.post("/memory/episodes/999999/approve")

        assert resp.status_code == 200
        assert resp.json()["updated"] is False
        assert _graph_node_type(
            main._state.memory_manager, "episode://999999",
        ) is None


class TestRejectEndpoint:

    def test_reject_pending_row_returns_updated_true_and_retracts(self, client):
        episode_id = _insert_pending(main._state.memory_manager)

        resp = client.post(f"/memory/episodes/{episode_id}/reject")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": episode_id, "status": "retracted", "updated": True}
        assert _status(main._state.memory_manager, episode_id) == "retracted"

    def test_reject_nonexistent_id_returns_updated_false(self, client):
        resp = client.post("/memory/episodes/999999/reject")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": 999999, "status": "retracted", "updated": False}

    def test_reject_already_retracted_row_returns_updated_false(self, client):
        episode_id = _insert_pending(main._state.memory_manager)
        first = client.post(f"/memory/episodes/{episode_id}/reject")
        assert first.json()["updated"] is True

        second = client.post(f"/memory/episodes/{episode_id}/reject")
        assert second.status_code == 200
        assert second.json()["updated"] is False


class TestReactivateEndpoint:

    def test_reactivate_retracted_row_returns_updated_true_and_activates(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        episode_id = writer.insert("preference", "theme", "Prefers dark mode.", "explicit")
        writer.retract_by_id(episode_id)

        resp = client.post(f"/memory/episodes/{episode_id}/reactivate")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": episode_id, "status": "active", "updated": True}
        assert _status(main._state.memory_manager, episode_id) == "active"

    def test_reactivate_nonexistent_id_returns_updated_false(self, client):
        resp = client.post("/memory/episodes/999999/reactivate")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": 999999, "status": "active", "updated": False}

    def test_reactivate_already_active_row_returns_updated_false(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        row_id = writer.insert("preference", "x", "y.", "explicit")

        resp = client.post(f"/memory/episodes/{row_id}/reactivate")
        assert resp.status_code == 200
        assert resp.json()["updated"] is False
        assert _status(main._state.memory_manager, row_id) == "active"

    def test_reactivate_without_controller_still_succeeds_no_graph_node(self, client):
        assert main._state.controller is None
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        episode_id = writer.insert("preference", "theme", "Prefers dark mode.", "explicit")
        writer.retract_by_id(episode_id)

        resp = client.post(f"/memory/episodes/{episode_id}/reactivate")

        assert resp.status_code == 200
        assert resp.json()["updated"] is True
        assert _graph_node_type(
            main._state.memory_manager, f"episode://{episode_id}",
        ) is None


class TestReactivateEndpointGraphHook:
    """
    Reactivating a retracted episode must retrigger
    ControllerAgent._write_episode_graph_node(), same as approve() —
    upsert_graph_node_for_episode() is idempotent, so this is a correct
    no-op-ish refresh for a row that already had a graph node (was active
    before retraction) and the correct first write for one that never did
    (was rejected while still pending).
    """

    @pytest.fixture()
    def controller(self, client):
        from unittest.mock import MagicMock

        from localist.controller_agent import ControllerAgent

        prev_controller = main._state.controller
        main._state.controller = ControllerAgent(
            runtime        = MagicMock(),
            agents         = [],
            memory_manager = main._state.memory_manager,
        )
        yield main._state.controller
        main._state.controller = prev_controller

    def test_reactivate_creates_episode_graph_node(self, client, controller):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        episode_id = writer.insert("preference", "theme", "Prefers dark mode.", "explicit")
        writer.retract_by_id(episode_id)

        resp = client.post(f"/memory/episodes/{episode_id}/reactivate")

        assert resp.status_code == 200
        assert resp.json()["updated"] is True
        assert _graph_node_type(
            main._state.memory_manager, f"episode://{episode_id}",
        ) == "episode"

    def test_reactivate_nonexistent_id_does_not_call_graph_hook(self, client, controller):
        resp = client.post("/memory/episodes/999999/reactivate")

        assert resp.status_code == 200
        assert resp.json()["updated"] is False
        assert _graph_node_type(
            main._state.memory_manager, "episode://999999",
        ) is None


class TestGetEpisodesTotalCount:

    def test_total_reflects_full_count_not_capped_by_limit(self, client):
        # Regression guard: total used to be len(rows) (i.e. capped by
        # `limit`), which made status=pending&limit=1 — the pending-count
        # badge's query shape — always report 0 or 1 no matter how many
        # pending episodes actually existed. total must now come from
        # MemoryManager.count_episodes(), independent of limit/offset.
        for _ in range(3):
            _insert_pending(main._state.memory_manager)

        resp = client.get("/memory/episodes", params={"status": "pending", "limit": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["episodes"]) == 1   # page is still limited...
        assert body["total"] == 3           # ...but total is not

    def test_total_zero_when_nothing_matches(self, client):
        resp = client.get("/memory/episodes", params={"status": "pending", "limit": 1})
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestGetEpisodesTaskIdFilter:
    """
    task_id filtering (episode-browsing-ui-plan.md Phase 6) backs the
    Episode Browsing UI's per-turn "related memory" overlay.
    """

    def test_filters_to_matching_task_id_only(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        writer.insert("preference", "a", "A.", "explicit", task_id="task-1")
        writer.insert("decision", "b", "B.", "explicit", task_id="task-2")

        resp = client.get("/memory/episodes", params={"task_id": "task-1"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["episodes"][0]["task_id"] == "task-1"

    def test_no_task_id_returns_all(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        writer.insert("preference", "a", "A.", "explicit", task_id="task-1")
        writer.insert("decision", "b", "B.", "explicit", task_id="task-2")

        resp = client.get("/memory/episodes")

        assert resp.json()["total"] == 2


class TestRelatedEpisodesEndpoint:
    """
    GET /memory/episodes/related — semantic-similarity replacement for the
    old task_id exact-match query backing EpisodeAnnotations.svelte's
    "Related Memory" panel. No embed_fn is configured on the `client`
    fixture's MemoryManager, so these exercise the keyword (Jaccard)
    fallback path — the common case for a fresh local install before
    embeddings are backfilled.
    """

    def test_returns_semantically_similar_episode(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        writer.insert(
            "decision", "memory backend", "SQLite committed as the memory backend.",
            "explicit", task_id="other-task",
        )

        resp = client.get(
            "/memory/episodes/related",
            params={"content": "sqlite memory backend", "task_id": "this-task"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["episodes"]) == 1
        assert body["episodes"][0]["subject"] == "memory backend"

    def test_excludes_episodes_from_same_task_id(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        writer.insert(
            "decision", "memory backend", "SQLite memory backend.",
            "explicit", task_id="this-task",
        )
        writer.insert(
            "decision", "memory backend v2", "SQLite memory backend.",
            "explicit", task_id="other-task",
        )

        resp = client.get(
            "/memory/episodes/related",
            params={"content": "sqlite memory backend", "task_id": "this-task"},
        )

        subjects = [ep["subject"] for ep in resp.json()["episodes"]]
        assert "memory backend" not in subjects
        assert "memory backend v2" in subjects

    def test_no_task_id_keeps_all_matches(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        writer.insert(
            "decision", "memory backend", "SQLite committed as the memory backend.",
            "explicit", task_id="this-task",
        )

        resp = client.get(
            "/memory/episodes/related",
            params={"content": "sqlite memory backend"},
        )

        subjects = [ep["subject"] for ep in resp.json()["episodes"]]
        assert "memory backend" in subjects

    def test_unrelated_query_still_returned_without_embeddings(self, client):
        # The `client` fixture's MemoryManager has no embed_fn, so this
        # episode is BM25-scored — by_similarity() skips its min_score
        # floor entirely for BM25-scored episodes (raw BM25 is unbounded,
        # so no floor is meaningful) and trusts ranking instead, the same
        # "no floor in keyword-only mode" contract query_corpus() already
        # has. With only one candidate in the pool, it's still returned
        # even for a completely unrelated query.
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        writer.insert("decision", "memory backend", "SQLite committed.", "explicit")

        resp = client.get(
            "/memory/episodes/related",
            params={"content": "completely unrelated query about weather"},
        )

        assert resp.status_code == 200
        assert len(resp.json()["episodes"]) == 1

    def test_no_memory_manager_returns_empty_list(self, client):
        prev = main._state.memory_manager
        main._state.memory_manager = None
        try:
            resp = client.get(
                "/memory/episodes/related", params={"content": "anything"},
            )
            assert resp.status_code == 200
            assert resp.json()["episodes"] == []
        finally:
            main._state.memory_manager = prev

    def test_respects_limit(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        for i in range(4):
            writer.insert(
                "decision", f"sqlite memory backend {i}", "SQLite committed as the memory backend.",
                "explicit",
            )

        resp = client.get(
            "/memory/episodes/related",
            params={"content": "sqlite memory backend", "limit": 2},
        )

        assert len(resp.json()["episodes"]) <= 2
