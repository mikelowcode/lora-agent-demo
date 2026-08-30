"""
Tests for session-file source propagation in ConversationalAgent.run().

Covers a gap identified after the session-files source-badge fix: two
places in conversational_agent.py fold attached session files into the
`sources` list as `session://{filename}` sentinel strings.

1. Prebuilt-prompt passthrough path — sources come straight from
   context["_prebuilt_sources"], which controller_agent.py populates
   before dispatch (rag_sources + session:// entries combined).
2. Non-prebuilt corpus-retrieval path — session files are appended to
   the corpus-derived `sources` list just before the agent returns.
"""

from unittest.mock import MagicMock, patch

from localist.conversational_agent import ConversationalAgent
from localist.controller_agent import SubTask, TaskStatus


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


def _fake_session_file(filename: str) -> MagicMock:
    sf = MagicMock()
    sf.filename = filename
    return sf


class TestPrebuiltPathSources:

    def test_prebuilt_path_passes_through_prebuilt_sources(self):
        """_prebuilt_sources (as assembled by controller_agent._execute_plan)
        must be returned unchanged in AgentResult.output['sources']."""
        prebuilt_sources = [
            "/wiki/how-localist-works.md",
            "session://notes.md",
        ]
        rt = _make_runtime(infer_return="Normal answer.")
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask(
            context={
                "_prebuilt_prompt":  "assembled 6-slot prompt goes here",
                "_prebuilt_system":  "system prompt",
                "_prebuilt_sources": prebuilt_sources,
                "_routing": {"fetch_rag": True},
            },
        )

        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert result.output["sources"] == prebuilt_sources


class TestNonPrebuiltPathSessionFiles:

    def test_non_prebuilt_path_appends_session_files_to_sources(self):
        """Session files must be appended alongside RAG-derived sources
        on the legacy corpus-retrieval path."""
        doc = MagicMock()
        doc.path = "/wiki/how-localist-works.md"
        doc.content = "Localist is a local AI assistant..."
        doc.doc_type = "wiki"
        doc.relevance_score = 0.90

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]

        rt = _make_runtime(infer_return="Here is what the wiki says.")
        agent = ConversationalAgent(runtime=rt, memory_manager=mm)
        subtask = _make_subtask(instruction="Tell me about Localist.")

        with patch(
            "localist.conversational_agent._session_files.get_files",
            return_value=[
                _fake_session_file("notes.md"),
                _fake_session_file("todo.txt"),
            ],
        ):
            result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert "/wiki/how-localist-works.md" in result.output["sources"]
        assert "session://notes.md" in result.output["sources"]
        assert "session://todo.txt" in result.output["sources"]
