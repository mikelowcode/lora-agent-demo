"""
Tests for the empty-completion floor in ConversationalAgent's legacy RAG
path (the non-prebuilt path, reached when no controller has pre-assembled a
prompt via PromptBuilder).

Background: Ollama can return a well-formed but zero-content stream
("done": true, no content in between) — confirmed live for queries the
model has no tool grounding for and can't verify from training alone (e.g.
a specific recent/future date). TaskStatus.COMPLETE must never carry an
empty answer.

The prebuilt-prompt path (the one ControllerAgent._execute_plan() actually
uses in production) is deliberately NOT guarded here — that path is only
ever reached via the controller, which owns a smarter forced-web_search
retry before falling back (see
ControllerAgent._dispatch_conversational_with_empty_guard() and its tests
in test_controller_phase4.py::TestEmptyCompletionGuard). Guarding it a
second time here, with a silent substitution, would mask the empty signal
the controller's retry logic depends on. This file covers only the legacy
path, which has no such wrapper and therefore needs its own floor.
"""

from unittest.mock import MagicMock

from localist.conversational_agent import ConversationalAgent, _EMPTY_RESPONSE_FALLBACK
from localist.controller_agent import SubTask, TaskStatus


def _make_runtime(infer_return: str = "Normal answer.") -> MagicMock:
    rt = MagicMock()
    rt.infer.return_value = infer_return
    return rt


def _make_subtask(
    instruction: str = "Can you look up something?",
    context: dict | None = None,
) -> SubTask:
    return SubTask(
        subtask_id  = "test-subtask-0",
        agent_name  = "conversational_agent",
        instruction = instruction,
        context     = context or {},
    )


class TestLegacyRagPathEmptyGuard:

    def test_empty_answer_replaced_with_fallback(self):
        rt = _make_runtime(infer_return="")
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask()

        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == _EMPTY_RESPONSE_FALLBACK
        assert result.output["answer"].strip() != ""
        assert result.output["grounded"] is False
        assert result.output["sources"] == []

    def test_whitespace_only_answer_replaced_with_fallback(self):
        """A stream that yields only whitespace chunks is also empty in substance."""
        rt = _make_runtime(infer_return="   \n\t  ")
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask()

        result = agent.run(subtask)

        assert result.output["answer"] == _EMPTY_RESPONSE_FALLBACK

    def test_empty_answer_overrides_grounded_true_from_corpus(self):
        """Guard forces grounded=False even when corpus retrieval would have set it True."""
        rt = _make_runtime(infer_return="")

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

        assert result.output["answer"] == _EMPTY_RESPONSE_FALLBACK
        assert result.output["grounded"] is False
        assert result.output["sources"] == []

    def test_normal_output_passes_through_unmodified(self):
        normal_answer = "2 + 2 equals 4."
        rt = _make_runtime(infer_return=normal_answer)
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask(instruction="What is 2 + 2?")

        result = agent.run(subtask)

        assert result.output["answer"] == normal_answer
        assert result.status == TaskStatus.COMPLETE

    def test_fallback_string_itself_is_never_empty(self):
        """The actual invariant this whole guard exists to protect."""
        assert _EMPTY_RESPONSE_FALLBACK.strip() != ""


class TestPrebuiltPathNotGuardedHere:
    """
    Documents the deliberate asymmetry: the prebuilt-prompt path returns
    whatever the runtime gave it, even if empty — ControllerAgent is the
    only thing that guards this path (see test_controller_phase4.py).
    """

    def test_prebuilt_path_returns_empty_answer_unmodified(self):
        rt = _make_runtime(infer_return="")
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask(context={
            "_prebuilt_prompt": "assembled prompt",
            "_prebuilt_system": "system prompt",
            "_routing": {"fetch_rag": False},
        })

        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == ""
