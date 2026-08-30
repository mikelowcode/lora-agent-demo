"""
Phase 7 integration tests — Localist full pipeline.

Covers:
  7.1  — Full conversational pipeline: corpus hit → fetch_rag=True → answer
  7.1b — Direct answer path: Priority 6 fallback, no corpus match
  7.3  — Working memory 300-token ceiling drops oldest turns
  7.4  — Persona document injected into system prompt via _load_persona()
  7.6  — system_prompt and user_prompt logged at DEBUG after assembly

Tests that require corpus queries use a real SQLite MemoryManager via
tmp_path (matching the Phase 4 convention). Tests that need precise
query_corpus control use a MagicMock memory manager. test_7_3 exercises
PromptBuilder directly without ControllerAgent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localist.controller_agent import (
    ControllerAgent,
    TaskStatus,
    AgentResult,
    SubTask,
)
from localist.memory_manager import MemoryManager
from localist.prompt_builder import PromptBuilder, Turn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    MemoryManager(db_path=path)   # initialise schema
    return path


@pytest.fixture()
def mm(db_path: Path) -> MemoryManager:
    return MemoryManager(db_path=db_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_runtime(infer_return: str = "Test answer.", embed_return=None):
    rt = MagicMock()
    rt.infer.return_value = infer_return
    rt.embed.return_value = embed_return or ([0.0] * 768)
    return rt


def make_conv_agent(answer: str = "Test answer."):
    """Conversational agent mock that captures the SubTask it receives."""
    received: list[SubTask] = []

    agent = MagicMock()
    agent.name = "conversational_agent"
    agent.can_handle.return_value = True

    def run(subtask):
        received.append(subtask)
        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = "conversational_agent",
            status     = TaskStatus.COMPLETE,
            output     = {"answer": answer, "sources": [], "grounded": False},
        )

    agent.run.side_effect = run
    agent._received = received
    return agent


@dataclass
class MockDoc:
    """Lightweight stand-in for MemoryManager DocumentResult."""
    path:                str
    content:             str
    relevance_score:     float
    doc_type:            str  = "wiki"
    name:                str  = "mock"
    scored_by_embedding: bool = True


# ---------------------------------------------------------------------------
# TestFullPipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_7_1_full_pipeline_conversational(self, mm, db_path):
        """Full pipeline: explicit wiki keyword → P4 routing → fetch_rag=True → answer."""
        mm.index_document(
            path     = db_path.parent / "phase7_doc.md",
            doc_type = "wiki",
            content  = "check the wiki LORA research assistant agentic",
        )

        rt   = make_runtime(infer_return="Paris.")
        conv = make_conv_agent(answer="Paris.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "check the wiki for LORA research assistant"})

        assert result["status"] == "complete"
        assert result["answer"] == "Paris."
        routing = conv._received[0].context["_routing"]
        assert routing["fetch_rag"] is True

    def test_7_1_full_pipeline_direct_answer(self, mm):
        """Priority 6 fallback: empty corpus → no RAG, no episodic."""
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent(answer="no")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "What is 2+2?"})

        assert result["status"] == "complete"
        routing = conv._received[0].context["_routing"]
        assert routing["fetch_rag"]      is False
        assert routing["fetch_episodic"] is False

    def test_7_3_working_memory_ceiling(self):
        """300-token ceiling drops oldest turns; Turn -8 must not survive."""
        # 8 turns × ~176 chars formatted each = ~1408 chars > 1200-char ceiling.
        # PromptBuilder drops Turn -8 then Turn -7 before the total fits.
        turns = [
            Turn(
                role    = "user" if i % 2 == 0 else "assistant",
                content = "word " * 32,   # 160 chars per turn
            )
            for i in range(8)
        ]

        total_content_chars = sum(len(t.content) for t in turns)
        assert total_content_chars > 1200   # sanity: raw content already over ceiling

        _, user_prompt = PromptBuilder().build(
            instruction      = "Follow-up question.",
            current_datetime = datetime(2026, 7, 17, 10, 10, 0, tzinfo=timezone.utc),
            working_memory   = turns,
        )

        assert "Turn -8" not in user_prompt
        assert 0 < len(user_prompt.encode()) < 5000

    def test_7_4_persona_injected_into_system_prompt(self):
        """_load_persona() fetches the persona doc and injects it into system_prompt."""
        persona_doc = MockDoc(
            path            = "wiki/persona.md",
            content         = "Localist is a local-first research assistant.",
            relevance_score = 0.9,
        )
        other_doc = MockDoc(
            path            = "wiki/localist-build-order.md",
            content         = "Build order content for Localist.",
            relevance_score = 0.6,
        )

        def query_side_effect(query, **kwargs):
            # _load_persona fetch: "assistant persona identity research"
            if "persona" in query.lower():
                return [persona_doc]
            # Planner P4 + Step 4 RAG fetch: any other instruction query
            return [other_doc]

        mm_mock = MagicMock()
        mm_mock.db_path = None                          # disable episodic hooks
        mm_mock.get_assistant_name.return_value = "Localist"
        mm_mock.query_corpus.side_effect = query_side_effect
        mm_mock.get_context_window.return_value = []

        rt   = make_runtime(infer_return="Paris.")
        conv = make_conv_agent(answer="Paris.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)

        # Explicit wiki keyword → P4 fires → fetch_rag=True → slot 4 populated.
        ctrl.handle_task({"instruction": "check the wiki for Localist research assistant"})

        system_prompt = conv._received[0].context["_prebuilt_system"]
        user_prompt   = conv._received[0].context["_prebuilt_prompt"]

        # Persona content must appear in system_prompt, not as a RAG source
        assert "Localist is a local-first research assistant." in system_prompt
        assert "Source: wiki/persona.md" not in user_prompt

        # The normal RAG source still appears in the user prompt
        assert "Source: wiki/localist-build-order.md" in user_prompt

    def test_7_4b_persona_filtered_from_rag_slot(self, caplog):
        """persona.md is filtered from RAG results; it appears only in system_prompt."""
        persona_doc = MockDoc(
            path            = "wiki/persona.md",
            content         = "# Persona\nTest persona content.",
            relevance_score = 0.9,
        )

        mm_mock = MagicMock()
        mm_mock.db_path = None                          # disable episodic hooks
        mm_mock.get_assistant_name.return_value = "Localist"
        mm_mock.query_corpus.return_value = [persona_doc]
        mm_mock.get_context_window.return_value = []

        rt   = make_runtime(infer_return="I am Localist.")
        conv = make_conv_agent(answer="I am Localist.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)

        with caplog.at_level(logging.DEBUG, logger="localist.controller_agent"):
            result = ctrl.handle_task({"instruction": "check the wiki for Localist research assistant"})

        assert result["status"] == "complete"
        messages = [r.getMessage() for r in caplog.records]
        user_prompt = next(
            m.split("assembled user_prompt:\n", 1)[1]
            for m in messages
            if "assembled user_prompt:" in m
        )
        # Persona doc must not appear in the user message (it's in system_prompt via _load_persona)
        assert user_prompt.count("wiki/persona.md") == 0

        # Persona content is in the system prompt
        system_prompt = conv._received[0].context["_prebuilt_system"]
        assert "# Persona" in system_prompt

    def test_7_6_prompt_logging_emitted(self, mm, caplog):
        """assembled system_prompt and user_prompt are logged at DEBUG."""
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        with caplog.at_level(logging.DEBUG, logger="localist.controller_agent"):
            ctrl.handle_task({"instruction": "What is 2+2?"})

        messages = [r.getMessage() for r in caplog.records]
        assert any("_execute_plan: assembled system_prompt:" in m for m in messages)
        assert any("_execute_plan: assembled user_prompt:" in m for m in messages)
