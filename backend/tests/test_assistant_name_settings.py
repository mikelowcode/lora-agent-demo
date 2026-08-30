"""
Tests for the configurable assistant name (Settings tab):
  - MemoryManager.get_assistant_name() / set_assistant_name() — Phase 1
    persistence layer, one-row assistant_settings table.
  - GET/PUT /settings/assistant-name (main.py) — the HTTP surface, mirroring
    TestRetentionSettingsEndpoints in test_main_task_chat_turns.py.
  - PUT invalidating ControllerAgent's persona cache (Phase 2) so a name
    change takes effect on the very next request.
  - ConversationalAgent's legacy (non-prebuilt) path resolving the
    configured name into its own system message.

Default value ("Localist") must be returned before any PUT, never None —
unlike retention_preset, callers of get_assistant_name() should never have
to null-check.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.conversational_agent import ConversationalAgent
from localist.controller_agent import SubTask
from localist.memory_manager import MemoryManager


@pytest.fixture()
def mm(tmp_path) -> MemoryManager:
    return MemoryManager(db_path=tmp_path / "assistant_name.db")


class TestMemoryManagerAssistantName:

    def test_default_before_any_set(self, mm):
        assert mm.get_assistant_name() == "Localist"

    def test_set_then_get_round_trips(self, mm):
        mm.set_assistant_name("Percy")
        assert mm.get_assistant_name() == "Percy"

    def test_set_strips_whitespace(self, mm):
        mm.set_assistant_name("  Percy  ")
        assert mm.get_assistant_name() == "Percy"

    def test_set_empty_string_raises(self, mm):
        with pytest.raises(ValueError):
            mm.set_assistant_name("")

    def test_set_whitespace_only_raises(self, mm):
        with pytest.raises(ValueError):
            mm.set_assistant_name("   ")

    def test_set_over_length_cap_raises(self, mm):
        with pytest.raises(ValueError):
            mm.set_assistant_name("x" * 61)

    def test_set_is_idempotent_upsert(self, mm):
        mm.set_assistant_name("Percy")
        mm.set_assistant_name("Ada")
        assert mm.get_assistant_name() == "Ada"

    def test_persists_across_fresh_instance_same_db(self, mm, tmp_path):
        mm.set_assistant_name("Percy")
        reopened = MemoryManager(db_path=tmp_path / "assistant_name.db")
        assert reopened.get_assistant_name() == "Percy"


@pytest.fixture()
def client(tmp_path):
    """
    TestClient against main.app with a mocked controller and a real,
    temp-file-backed MemoryManager — mirrors the `client` fixture in
    test_main_task_chat_turns.py.
    """
    prev_controller = main._state.controller
    prev_memory     = main._state.memory_manager

    main._state.memory_manager = MemoryManager(db_path=tmp_path / "main_assistant_name.db")
    main._state.controller     = MagicMock()

    yield TestClient(main.app)

    main._state.controller     = prev_controller
    main._state.memory_manager = prev_memory


class TestAssistantNameEndpoints:

    def test_get_returns_default_before_any_put(self, client):
        resp = client.get("/settings/assistant-name")
        assert resp.status_code == 200
        assert resp.json() == {"assistant_name": "Localist"}

    def test_put_valid_name_returns_200_with_correct_body(self, client):
        resp = client.put("/settings/assistant-name", json={"assistant_name": "Percy"})
        assert resp.status_code == 200
        assert resp.json() == {"assistant_name": "Percy"}

    def test_put_empty_name_returns_400(self, client):
        resp = client.put("/settings/assistant-name", json={"assistant_name": "   "})
        assert resp.status_code == 400

    def test_put_over_length_cap_returns_400(self, client):
        resp = client.put("/settings/assistant-name", json={"assistant_name": "x" * 61})
        assert resp.status_code == 400

    def test_get_after_put_reflects_new_value(self, client):
        put_resp = client.put("/settings/assistant-name", json={"assistant_name": "Ada"})
        assert put_resp.status_code == 200

        get_resp = client.get("/settings/assistant-name")
        assert get_resp.status_code == 200
        assert get_resp.json() == {"assistant_name": "Ada"}

    def test_put_invalidates_controller_persona_cache(self, client):
        """A successful PUT must invalidate the cache so the change is live
        on the very next request, not just after the persona's own natural
        refresh."""
        resp = client.put("/settings/assistant-name", json={"assistant_name": "Ada"})
        assert resp.status_code == 200
        main._state.controller.invalidate_persona_cache.assert_called_once()

    def test_put_rejected_name_does_not_invalidate_cache(self, client):
        resp = client.put("/settings/assistant-name", json={"assistant_name": "   "})
        assert resp.status_code == 400
        main._state.controller.invalidate_persona_cache.assert_not_called()


class TestConversationalAgentLegacyPathAssistantName:
    """
    ConversationalAgent's non-prebuilt path (reached when no controller has
    pre-assembled a prompt) resolves the configured assistant name itself,
    independent of ControllerAgent's persona-cache path.
    """

    def _make_subtask(self, instruction: str = "What's the weather?") -> SubTask:
        return SubTask(
            subtask_id  = "test-subtask-0",
            agent_name  = "conversational_agent",
            instruction = instruction,
            context     = {},
        )

    def test_configured_name_flows_into_system_message(self):
        rt = MagicMock()
        rt.infer.return_value = "An answer."
        mm = MagicMock()
        mm.get_assistant_name.return_value = "Percy"
        mm.query_corpus.return_value = []

        agent = ConversationalAgent(runtime=rt, memory_manager=mm)
        agent.run(self._make_subtask())

        _, kwargs = rt.infer.call_args
        assert "You are Percy," in kwargs["system"]

    def test_no_memory_manager_falls_back_to_default_name(self):
        rt = MagicMock()
        rt.infer.return_value = "An answer."

        agent = ConversationalAgent(runtime=rt, memory_manager=None)
        agent.run(self._make_subtask())

        _, kwargs = rt.infer.call_args
        assert "You are Localist," in kwargs["system"]
