"""
GET /health's gate_tiers / active_embedding_model_name fields
(PLAN_semantic_gating_calibration.md §6) — the Settings UI's trust badge
reads these on every page load/poll, not just transiently after a switch or
reembed response. Computed via planner.resolve_gate_tiers(), the exact
function Planner.__init__ uses, so the two can never disagree.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager
from localist.planner import _TUNED_EMBEDDING_MODEL, _GATE_NAMES


@pytest.fixture()
def client(tmp_path):
    prev_settings = main._state.settings
    prev_runtime = main._state.runtime
    prev_memory = main._state.memory_manager
    prev_embedding_engine = main._state.embedding_engine
    prev_active_model_name = main._state.active_embedding_model_name

    main._state.settings = main.Settings(
        runtime_backend="ollama", chat_model=None, chat_model_omlx=None,
        chat_model_ollama=None, chat_model_foundry=None, embedding_model="",
        foundry_url=None, omlx_url="http://localhost:8000",
        ollama_url="http://localhost:11434", request_timeout=30.0,
        stream_timeout=60.0, episodic_write_approval=False,
    )
    main._state.embedding_engine = None
    fake_runtime = MagicMock(name="fake-runtime")
    fake_runtime.health_check.return_value = {
        "reachable": True, "base_url": "http://localhost:11434",
        "models": [], "chat_model_found": True, "embed_model_found": False,
        "error": None,
    }
    main._state.runtime = fake_runtime
    main._state.memory_manager = MemoryManager(db_path=tmp_path / "health_gate_tiers.db")

    yield TestClient(main.app)

    main._state.settings = prev_settings
    main._state.runtime = prev_runtime
    main._state.memory_manager = prev_memory
    main._state.embedding_engine = prev_embedding_engine
    main._state.active_embedding_model_name = prev_active_model_name


class TestHealthGateTiers:
    def test_tuned_model_reports_all_gates_tuned(self, client):
        main._state.active_embedding_model_name = _TUNED_EMBEDDING_MODEL

        resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["active_embedding_model_name"] == _TUNED_EMBEDDING_MODEL
        assert body["gate_tiers"] == {name: "tuned" for name in _GATE_NAMES}

    def test_none_active_model_reports_all_gates_tuned(self, client):
        main._state.active_embedding_model_name = None

        resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["active_embedding_model_name"] is None
        assert body["gate_tiers"] == {name: "tuned" for name in _GATE_NAMES}

    def test_validated_model_reports_validated_tiers(self, client):
        main._state.active_embedding_model_name = "nomic-embed-text:latest"

        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["gate_tiers"] == {name: "validated" for name in _GATE_NAMES}

    def test_calibrated_model_reports_mixed_tiers_from_persisted_row(self, client):
        main._state.memory_manager.set_calibrated_thresholds(
            "mystery-model:latest", {"lookup_request": 0.6, "research_intent": 0.55},
        )
        main._state.active_embedding_model_name = "mystery-model:latest"

        resp = client.get("/health")

        assert resp.status_code == 200
        gate_tiers = resp.json()["gate_tiers"]
        assert gate_tiers["lookup_request"] == "auto-calibrated"
        assert gate_tiers["research_intent"] == "auto-calibrated"
        assert gate_tiers["explicit_search_action"] == "disabled"
        assert gate_tiers["episodic_relevance"] == "disabled"

    def test_never_calibrated_unvalidated_model_reports_all_disabled(self, client):
        main._state.active_embedding_model_name = "totally-unknown-model:latest"

        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["gate_tiers"] == {name: "disabled" for name in _GATE_NAMES}
