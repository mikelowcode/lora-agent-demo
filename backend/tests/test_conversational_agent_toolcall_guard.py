"""
Tests for the fabricated tool-call detection backstop in ConversationalAgent.

Covers §8.8 Open Item 11: the model has been observed emitting malformed
tool-call-shaped strings as its entire output on turns where tools=[].
The guard in conversational_agent.py detects these and substitutes a fixed
fallback message before the result is returned to any caller.

Test structure
--------------
1. Unit tests for _is_fabricated_toolcall() — all 7 observed reference
   strings (verbatim), plus negative-control cases.
2. Integration tests for the prebuilt-prompt path and legacy RAG path:
   fabricated input → fallback substitution with grounded=False, sources=[].
3. Negative-control integration tests: ordinary prose passes through unmodified.
"""

from unittest.mock import MagicMock

import pytest

from localist.conversational_agent import (
    ConversationalAgent,
    _is_fabricated_toolcall,
    _SEARCH_UNAVAILABLE_FALLBACK,
)
from localist.controller_agent import AgentResult, SubTask, TaskStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# All seven verbatim reference strings from live occurrences (§8.8 Open Item 11).
_REFERENCE_FABRICATIONS = [
    '<|toolcall>call:web search{queries:[<|"|>Apple price hike MacBook Neo iPad<|"|>,<|"|>MacBook Neo price change<|"|>,<|"|>iPad price increase<|"|>]}<toolcall|>',
    '<|toolcall>call:websearch{query: "next-generation in-house Microsoft AI models Build 2026"}<tool_call|>',
    '<|toolcall>call:websearch{query:<|"|>Microsoft next-generation in-house AI models<|"|>}<tool_call|>',
    '<|toolcall>call:web search{query: "Apple MacBook Neo price hike and iPad price hike"}<tool_call|>',
    '<|toolcall>call:websearch{query: "Microsoft next-generation in-house AI models"}<tool_call|>',
    '<|toolcall>call:websearch{query: "when was microsoft\'s first formal investment in openai"}<tool_call|>',
    '<|tool_call>call:web_search{query:<|"|>LangSmith Engine<|"|>}<tool_call|>',
]

_NEGATIVE_CONTROLS = [
    "I'll use this tool to help you with that.",
    "Let's call this Phase 2 of the project.",
    "",
    "Based on the web search, here is a summary...",
    "You can call the web_search tool if needed.",
]


def _make_runtime(infer_return: str = "Normal answer.") -> MagicMock:
    rt = MagicMock()
    rt.infer.return_value = infer_return
    return rt


def _make_subtask(
    instruction: str = "Can you look up something?",
    context: dict | None = None,
    subtask_id: str = "test-subtask-0",
) -> SubTask:
    return SubTask(
        subtask_id  = subtask_id,
        agent_name  = "conversational_agent",
        instruction = instruction,
        context     = context or {},
    )


def _prebuilt_subtask(infer_return: str, fetch_rag: bool = False) -> SubTask:
    """SubTask that exercises the prebuilt-prompt path."""
    return _make_subtask(
        context={
            "_prebuilt_prompt": "assembled 6-slot prompt goes here",
            "_prebuilt_system": "system prompt",
            "_routing": {"fetch_rag": fetch_rag},
        },
    )


# ---------------------------------------------------------------------------
# 1. Unit tests — _is_fabricated_toolcall()
# ---------------------------------------------------------------------------

class TestIsFabricatedToolcall:

    @pytest.mark.parametrize("fabrication", _REFERENCE_FABRICATIONS)
    def test_all_seven_reference_strings_match(self, fabrication: str):
        """Every verbatim live-observed fabrication string must return True."""
        assert _is_fabricated_toolcall(fabrication) is True, (
            f"Pattern did not match reference string: {fabrication!r}"
        )

    @pytest.mark.parametrize("safe_text", _NEGATIVE_CONTROLS)
    def test_negative_controls_do_not_match(self, safe_text: str):
        """Ordinary prose — including text that mentions 'tool', 'call', or 'web search'
        in natural language — must return False."""
        assert _is_fabricated_toolcall(safe_text) is False, (
            f"Pattern incorrectly matched safe text: {safe_text!r}"
        )

    def test_empty_string_does_not_match(self):
        assert _is_fabricated_toolcall("") is False

    def test_partial_opening_only_does_not_match(self):
        """An opening fragment with no closing bracket must not match."""
        assert _is_fabricated_toolcall("<|toolcall>call:web search{query: ...}") is False

    def test_partial_closing_only_does_not_match(self):
        """A closing fragment with no opening must not match."""
        assert _is_fabricated_toolcall("call:websearch{query: ...}<tool_call|>") is False

    def test_call_web_anchor_is_required(self):
        """Without the 'call:web' middle anchor the pattern must not match,
        even with valid-looking outer brackets."""
        assert _is_fabricated_toolcall("<|toolcall>whatever<tool_call|>") is False


# ---------------------------------------------------------------------------
# 2. Integration tests — prebuilt-prompt path
# ---------------------------------------------------------------------------

class TestPrebuiltPathGuard:

    @pytest.mark.parametrize("fabrication", _REFERENCE_FABRICATIONS)
    def test_fabricated_output_is_replaced_with_fallback(self, fabrication: str):
        """Fabricated tool-call from runtime.infer() → fallback substituted."""
        rt = _make_runtime(infer_return=fabrication)
        agent = ConversationalAgent(runtime=rt)
        subtask = _prebuilt_subtask(infer_return=fabrication)

        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == _SEARCH_UNAVAILABLE_FALLBACK
        assert result.output["grounded"] is False
        assert result.output["sources"] == []

    def test_fabrication_overrides_fetch_rag_true(self):
        """guard forces grounded=False even when _routing.fetch_rag is True."""
        fabrication = _REFERENCE_FABRICATIONS[0]
        rt = _make_runtime(infer_return=fabrication)
        agent = ConversationalAgent(runtime=rt)
        subtask = _prebuilt_subtask(infer_return=fabrication, fetch_rag=True)

        result = agent.run(subtask)

        assert result.output["answer"] == _SEARCH_UNAVAILABLE_FALLBACK
        assert result.output["grounded"] is False
        assert result.output["sources"] == []

    def test_normal_output_passes_through_unmodified(self):
        """Ordinary grounded prose must not be altered by the guard."""
        normal_answer = "Apple's MacBook Neo was announced in March 2026 with a 15% price increase."
        rt = _make_runtime(infer_return=normal_answer)
        agent = ConversationalAgent(runtime=rt)
        subtask = _prebuilt_subtask(infer_return=normal_answer, fetch_rag=True)

        result = agent.run(subtask)

        assert result.output["answer"] == normal_answer
        assert result.output["grounded"] is True
        assert result.status == TaskStatus.COMPLETE

    def test_negative_control_text_passes_through(self):
        """Prose that mentions 'tool' and 'call' in ordinary usage is not blocked."""
        safe_answer = "You can call the web_search tool if needed."
        rt = _make_runtime(infer_return=safe_answer)
        agent = ConversationalAgent(runtime=rt)
        subtask = _prebuilt_subtask(infer_return=safe_answer)

        result = agent.run(subtask)

        assert result.output["answer"] == safe_answer


# ---------------------------------------------------------------------------
# 3. Integration tests — legacy RAG path
# ---------------------------------------------------------------------------

class TestLegacyRagPathGuard:

    @pytest.mark.parametrize("fabrication", _REFERENCE_FABRICATIONS)
    def test_fabricated_output_is_replaced_with_fallback(self, fabrication: str):
        """Fabricated tool-call from runtime.infer() on the legacy path → fallback."""
        rt = _make_runtime(infer_return=fabrication)
        # No memory_manager — legacy path with grounded=False
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask()

        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == _SEARCH_UNAVAILABLE_FALLBACK
        assert result.output["grounded"] is False
        assert result.output["sources"] == []

    def test_fabrication_overrides_grounded_true_from_corpus(self):
        """Guard forces grounded=False even when corpus retrieval would have set it True."""
        fabrication = _REFERENCE_FABRICATIONS[1]
        rt = _make_runtime(infer_return=fabrication)

        # Build a memory_manager that returns a corpus result (would set grounded=True).
        doc = MagicMock()
        doc.path = "/wiki/apple-products.md"
        doc.content = "Apple raised MacBook Neo pricing..."
        doc.doc_type = "wiki"
        doc.relevance_score = 0.85

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]

        agent = ConversationalAgent(runtime=rt, memory_manager=mm)
        subtask = _make_subtask()

        result = agent.run(subtask)

        assert result.output["answer"] == _SEARCH_UNAVAILABLE_FALLBACK
        assert result.output["grounded"] is False
        assert result.output["sources"] == []

    def test_normal_output_passes_through_unmodified(self):
        """Ordinary answer on the legacy path is not touched by the guard."""
        normal_answer = "2 + 2 equals 4."
        rt = _make_runtime(infer_return=normal_answer)
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask(instruction="What is 2 + 2?")

        result = agent.run(subtask)

        assert result.output["answer"] == normal_answer
        assert result.status == TaskStatus.COMPLETE

    def test_grounded_normal_output_preserves_sources(self):
        """Normal grounded answer: sources list and grounded=True must be preserved."""
        normal_answer = "Here is what the wiki says about Localist."
        rt = _make_runtime(infer_return=normal_answer)

        doc = MagicMock()
        doc.path = "/wiki/how-localist-works.md"
        doc.content = "Localist is a local AI assistant..."
        doc.doc_type = "wiki"
        doc.relevance_score = 0.90

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]

        agent = ConversationalAgent(runtime=rt, memory_manager=mm)
        subtask = _make_subtask(instruction="Tell me about Localist.")

        result = agent.run(subtask)

        assert result.output["answer"] == normal_answer
        assert result.output["grounded"] is True
        assert "/wiki/how-localist-works.md" in result.output["sources"]
