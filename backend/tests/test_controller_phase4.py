"""
Phase 4 integration tests — Localist ControllerAgent execution pipeline.

Covers:
  - Direct answer path: P6 fallback → PromptBuilder slot 2 only
  - RAG path: P4 corpus hit → fetch_rag=True → rag_sources in prompt
  - Ingest path: P1 → wiki_agent dispatched, Synthesizer called
  - Episodic write path: P2 → EpisodicMemoryWriter called
  - Episodic retrieval path: fetch_episodic=True → bullets in prompt
  - Prebuilt prompt passthrough: ConversationalAgent uses _prebuilt_prompt
  - wiki_doc wiring: _load_persona / _load_user_profile frontmatter handling
  - Working memory: prior turns appear in slot 3
  - Routing metadata: _routing key present in SubTask context
  - Tool stub: tools_to_call logged but not executed
  - Fallback: unregistered agent falls back to conversational_agent

Each test uses a real SQLite DB via tmp_path for paths that exercise
EpisodicMemoryWriter/Reader. Tests that don't need the DB use MagicMock.

Note on RAG test design: MemoryManager.query_corpus() uses Jaccard keyword
overlap scoring. The Planner's Priority 4 threshold is 0.4. Document content
and query strings are chosen to achieve Jaccard >= 0.6 so corpus hits are
reliable regardless of embed availability.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from localist.controller_agent import (
    ControllerAgent,
    Task,
    TaskStatus,
    AgentResult,
    SubTask,
    _memory_key,
    _extract_file_op_content,
    _file_op_confirmation_line,
    _strip_false_tool_attribution,
)
from localist.memory_manager import (
    MemoryManager,
    EpisodicMemoryWriter,
    EpisodicMemoryReader,
    EpisodeRecord,
    GraphEdgeResult,
)
from localist.episodic_extractor import ExtractionResult
from localist.planner import RoutingPlan
from localist.prompt_builder import PromptBuilder, WorkingMemoryState, ToolResult as _ToolResult
from localist.conversational_agent import _EMPTY_RESPONSE_FALLBACK
from localist.wiki_doc import load_wiki_doc, ParsedWikiDoc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    MemoryManager(db_path=path)   # initialises schema v2
    return path


@pytest.fixture()
def mm(db_path: Path) -> MemoryManager:
    return MemoryManager(db_path=db_path)


def make_runtime(infer_return: str = "Test answer.", embed_return=None):
    rt = MagicMock()
    rt.infer.return_value = infer_return
    rt.embed.return_value = embed_return or ([0.0] * 768)
    return rt


def make_conv_agent(answer: str = "Test answer."):
    """Conversational agent that captures the SubTask it receives."""
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


def make_conv_agent_with_answers(answers: list[str]):
    """
    Conversational agent that returns each string in `answers` in turn (the
    last one repeats if called more times than len(answers)), capturing
    every SubTask it receives. Used to simulate an empty first completion
    followed by a real one on retry.
    """
    received: list[SubTask] = []
    call_count = {"n": 0}

    agent = MagicMock()
    agent.name = "conversational_agent"
    agent.can_handle.return_value = True

    def run(subtask):
        received.append(subtask)
        idx = min(call_count["n"], len(answers) - 1)
        call_count["n"] += 1
        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = "conversational_agent",
            status     = TaskStatus.COMPLETE,
            output     = {"answer": answers[idx], "sources": [], "grounded": False},
        )
    agent.run.side_effect = run
    agent._received = received
    return agent


def make_wiki_agent():
    agent = MagicMock()
    agent.name = "wiki_agent"
    agent.can_handle.return_value = True
    agent.run.return_value = AgentResult(
        subtask_id = "wiki-0",
        agent_name = "wiki_agent",
        status     = TaskStatus.COMPLETE,
        output     = {"new_pages": [], "applied": False},
    )
    return agent


# ---------------------------------------------------------------------------
# Path 1 — Direct answer (Priority 6 fallback)
# ---------------------------------------------------------------------------

class TestDirectAnswerPath:

    def test_completes_successfully(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Direct answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "What is 2+2?"})

        assert result["status"] == "complete"
        assert result["answer"] == "Direct answer."

    def test_prebuilt_prompt_passed_to_agent(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is 2+2?"})

        subtask = conv._received[0]
        assert "_prebuilt_prompt" in subtask.context
        assert "[INSTRUCTION]" in subtask.context["_prebuilt_prompt"]

    def test_routing_metadata_in_context(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is 2+2?"})

        routing = conv._received[0].context["_routing"]
        assert routing["fetch_rag"]      is False
        assert routing["fetch_episodic"] is False
        assert routing["tools_to_call"]  == []
        assert routing["write_episode"]  is False

    def test_no_rag_sources_in_prompt(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is 2+2?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" not in prompt


class TestCurrentDatetimeSlot:
    """
    [CURRENT DATETIME] is computed fresh at each _execute_plan() call site
    (controller_agent.py, immediately before PromptBuilder.build()), never
    memoized — unlike _load_persona()'s intentional session-lifetime cache.
    """

    def test_current_datetime_slot_present_in_prompt(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is 2+2?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CURRENT DATETIME]" in prompt
        assert prompt.index("[CURRENT DATETIME]") < prompt.index("[INSTRUCTION]")

    def test_two_turns_with_advancing_clock_render_different_timestamps(self, mm):
        """
        A mocked clock advancing between two build() calls must produce two
        different [CURRENT DATETIME] slot strings — proof the value is read
        fresh per turn, not cached across the ControllerAgent's process
        lifetime the way _load_persona() deliberately is.
        """
        from localist import controller_agent as controller_agent_module
        from datetime import datetime, timezone

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        t1 = datetime(2026, 7, 17, 10, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 7, 17, 11, 45, 0, tzinfo=timezone.utc)

        with patch.object(controller_agent_module, "datetime") as mock_dt:
            mock_dt.now.return_value.astimezone.return_value = t1
            ctrl.handle_task({"instruction": "First turn."})
            prompt_1 = conv._received[0].context["_prebuilt_prompt"]

            mock_dt.now.return_value.astimezone.return_value = t2
            ctrl.handle_task({"instruction": "Second turn."})
            prompt_2 = conv._received[1].context["_prebuilt_prompt"]

        slot_1 = prompt_1.split("\n\n")[0]
        slot_2 = prompt_2.split("\n\n")[0]
        assert slot_1.startswith("[CURRENT DATETIME]")
        assert slot_2.startswith("[CURRENT DATETIME]")
        assert slot_1 != slot_2
        assert "10:10:00" in slot_1
        assert "11:45:00" in slot_2


class TestEmptyCompletionGuard:
    """
    ControllerAgent._dispatch_conversational_with_empty_guard(): the
    controller-owned floor + bounded forced-web_search retry for empty
    completions. See conversational_agent.py's own (narrower) legacy-path
    guard and its tests in test_conversational_agent_empty_guard.py — that
    guard deliberately does NOT cover the prebuilt-prompt path, since this
    is the layer that owns it.
    """

    @staticmethod
    def _plan(tools_to_call: list[str] | None = None) -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            priority       = 6,
            tools_to_call  = tools_to_call or [],
        )

    @staticmethod
    def _tool_result(text: str = "TSM reported Q2 2026 earnings on July 16, beating estimates."):
        return _ToolResult(
            tool_name  = "web_search",
            parameters = "query='TSM earnings'",
            result     = text,
            success    = True,
        )

    def test_empty_then_real_answer_on_retry_forces_web_search(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent_with_answers(["", "TSM reported strong Q2 2026 earnings."])
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._plan(tools_to_call=[])

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [self._tool_result()]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "Look up TSM's July 16 2026 earnings and report back a summary."}
            )

        assert result["status"] == "complete"
        assert result["answer"] == "TSM reported strong Q2 2026 earnings."
        assert len(conv._received) == 2, "expected exactly one retry (original + 1)"

        _, dispatch_kwargs = mock_dispatcher.dispatch.call_args
        assert "web_search" in dispatch_kwargs["tools_to_call"]

        retry_prompt = conv._received[1].context["_prebuilt_prompt"]
        assert "TSM reported Q2 2026 earnings" in retry_prompt

    def test_empty_then_empty_falls_back_to_canned_message(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent_with_answers(["", ""])
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._plan(tools_to_call=[])

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [self._tool_result()]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "Look up TSM's July 16 2026 earnings and report back a summary."}
            )

        assert result["status"] == "complete"
        assert result["answer"] == _EMPTY_RESPONSE_FALLBACK
        assert result["answer"].strip() != ""
        assert result["sources"] == []
        assert len(conv._received) == 2, "retry must be bounded to exactly one attempt"

    def test_retry_bounded_even_when_dispatcher_raises(self, mm):
        """A tool-dispatch exception during retry must fall back cleanly, never raise."""
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent_with_answers([""])
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._plan(tools_to_call=[])

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.side_effect = RuntimeError("MCP session failed")

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "Look up TSM's July 16 2026 earnings and report back a summary."}
            )

        assert result["status"] == "complete"
        assert result["answer"] == _EMPTY_RESPONSE_FALLBACK

    def test_non_empty_first_answer_never_triggers_retry(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent_with_answers(["Direct answer, no retry needed."])
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._plan(tools_to_call=[])

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [self._tool_result()]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task({"instruction": "What is 2+2?"})

        assert result["answer"] == "Direct answer, no retry needed."
        assert len(conv._received) == 1
        mock_dispatcher.dispatch.assert_not_called()

    def test_memory_receives_exactly_one_entry_not_two(self, mm, db_path):
        """
        Regression guard for the ordering bug this design specifically
        avoids: memory_manager.add_agent_result() serializes synchronously
        at call time, so dispatching the original (empty) attempt through
        the normal self._dispatch() path would have already persisted the
        empty answer into working memory before any retry could replace
        it. Exactly one entry must be written for this turn, carrying the
        final (retried) answer.
        """
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent_with_answers(["", "Real retried answer."])
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._plan(tools_to_call=[])

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [self._tool_result()]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            ctrl.handle_task({
                "instruction": "Look up TSM's July 16 2026 earnings and report back a summary.",
                "task_id":     "mem-write-once-test",
            })

        entries = mm.get_context_window(task_id="mem-write-once-test", limit=50)
        agent_entries = [e for e in entries if e["role"] == "agent"]
        assert len(agent_entries) == 1, (
            f"expected exactly one persisted agent result, got {len(agent_entries)}: "
            f"{agent_entries}"
        )
        assert "" not in [e["content"] for e in agent_entries]
        assert "Real retried answer." in agent_entries[0]["content"]


# ---------------------------------------------------------------------------
# Step 3b — corpus fallback tool_name matching
# (ollama-web-search-mcp-tool-scoping.md, 2026-07-31)
# ---------------------------------------------------------------------------

class TestWebSearchProviderStep3bCorpusFallback:
    """
    Regression coverage for the Step 3b tool_name-matching gap found (and
    fixed) while building the WEB_SEARCH_PROVIDER Ollama fallback: a plain
    `tool_name == "web_search"` equality check silently stopped catching
    failures once mcp_tool_dispatcher.py started retagging fallback/primary
    results "web_search:ollama_fallback" / "web_search:ollama_primary" —
    the identical gap news_search hit before and fixed the same way
    (startswith(), see controller_agent.py's Step 3b comment block). Fixed
    by widening the first clause to tool_name.startswith("web_search").

    These three tests are the fast, isolated unit-level lock-in for that
    match itself — one per tag variant that can appear in
    dispatched_tool_results — via the same wholesale-MCPToolDispatcher-mock
    pattern TestEmptyCompletionGuard above uses, so no real localist-mcp
    process is needed. The live, end-to-end proof that a real Brave/
    LangSearch failure still grounds the answer (and that it no longer
    makes a live Ollama call while doing so) lives in
    test_tool_dispatcher_phase6.py's
    test_web_search_missing_key_triggers_corpus_fallback — that test
    already pins WEB_SEARCH_PROVIDER=brave and blanks OLLAMA_API_KEY as
    part of the same fix, so it isn't duplicated here.
    """

    @staticmethod
    def _plan() -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            priority       = 6,
            tools_to_call  = ["web_search"],
        )

    @staticmethod
    def _failed_result(tool_name: str) -> _ToolResult:
        return _ToolResult(
            tool_name  = tool_name,
            parameters = "query='Zylophonic quarterly earnings'",
            result     = "ERROR: web_search failed",
            success    = False,
        )

    def _run(self, mm, db_path, tool_name: str) -> str:
        # Same content/instruction pair test_tool_dispatcher_phase6.py's
        # corpus-fallback test already uses — proven to clear the 0.55
        # Jaccard threshold Step 3b's corpus query filters on.
        mm.index_document(
            path     = db_path.parent / "zylophonic-notes.md",
            doc_type = "raw",
            content  = "Zylophonic quarterly earnings update web search",
        )
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Here is what I found.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._plan()

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [self._failed_result(tool_name)]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            ctrl.handle_task({
                "instruction": "do a web search for Zylophonic quarterly earnings update",
            })

        return conv._received[0].context["_prebuilt_prompt"]

    def test_plain_web_search_failure_still_triggers_corpus_fallback(self, mm, db_path):
        """Unretagged tool_name=="web_search" — the pre-existing case,
        confirming the startswith() widening didn't regress it."""
        prompt = self._run(mm, db_path, "web_search")
        assert "[CONTEXT]" in prompt

    def test_ollama_fallback_tagged_failure_triggers_corpus_fallback(self, mm, db_path):
        prompt = self._run(mm, db_path, "web_search:ollama_fallback")
        assert "[CONTEXT]" in prompt

    def test_ollama_primary_tagged_failure_triggers_corpus_fallback(self, mm, db_path):
        prompt = self._run(mm, db_path, "web_search:ollama_primary")
        assert "[CONTEXT]" in prompt


# ---------------------------------------------------------------------------
# Path 2 — RAG path (Priority 4)
# ---------------------------------------------------------------------------
# P4 now fires on explicit wiki/vault trigger keywords, not corpus scoring.

class TestRAGPath:

    def test_fetch_rag_true_when_wiki_keyword_present(self, mm, db_path):
        mm.index_document(
            path     = db_path.parent / "fake_wiki.md",
            doc_type = "wiki",
            content  = "check the wiki LORA research assistant agentic",
        )

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("RAG answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "check the wiki for LORA research assistant"})

        assert result["status"] == "complete"
        routing = conv._received[0].context["_routing"]
        assert routing["fetch_rag"] is True

    def test_rag_sources_appear_in_prompt(self, mm, db_path):
        mm.index_document(
            path     = db_path.parent / "fake_wiki.md",
            doc_type = "wiki",
            content  = "check the wiki LORA SQLite memory storage",
        )

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "check the wiki for LORA SQLite memory"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" in prompt
        assert "Source:" in prompt

    def test_no_rag_when_corpus_empty(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is the capital of France?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" not in prompt


# ---------------------------------------------------------------------------
# Path 3 — Ingest path (Priority 1)
# ---------------------------------------------------------------------------

class TestIngestPath:

    def test_ingest_routes_to_wiki_agent(self, mm):
        rt   = make_runtime()
        conv = make_conv_agent()
        wiki = make_wiki_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv, wiki],
                               memory_manager=mm)

        ctrl.handle_task({
            "instruction": "ingest this document",
            "context":     {"raw_path": "/data/notes.md"},
        })

        assert wiki.run.called
        assert conv.run.called is False

    def test_ingest_does_not_set_fetch_rag(self, mm):
        rt   = make_runtime()
        wiki = make_wiki_agent()

        received: list[SubTask] = []
        def capture(subtask):
            received.append(subtask)
            return wiki.run.return_value
        wiki.run.side_effect = capture

        ctrl = ControllerAgent(runtime=rt, agents=[wiki], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "ingest file",
            "context":     {"raw_path": "/x.md"},
        })

        assert received[0].context["_routing"]["fetch_rag"]      is False
        assert received[0].context["_routing"]["fetch_episodic"] is False

    def test_ingest_fallback_when_wiki_not_registered(self, mm):
        """P1 fires but wiki_agent not registered → fallback to conv agent."""
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Fallback answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({
            "instruction": "ingest file",
            "context":     {"raw_path": "/x.md"},
        })

        assert result["status"] == "complete"
        assert conv.run.called


# ---------------------------------------------------------------------------
# Wiki diff review path — ControllerAgent._build_wiki_diff_result()
# (scope-review-then-apply-diff-ui.md)
# ---------------------------------------------------------------------------

class TestWikiDiffResultPath:

    @staticmethod
    def _plan(priority: int = 1) -> RoutingPlan:
        return RoutingPlan(
            agent          = "wiki_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            priority       = priority,
            diff_target    = "some-page",
        )

    @staticmethod
    def _diff_result(
        applied: bool = False,
        diffs=None,
        status: TaskStatus = TaskStatus.COMPLETE,
    ) -> AgentResult:
        return AgentResult(
            subtask_id = "wiki-0",
            agent_name = "wiki_agent",
            status     = status,
            output     = {
                "new_pages": [],
                "diffs": diffs if diffs is not None else [
                    {"page_name": "some-page", "diff": "@@ -1,1 +1,1 @@\n-old\n+new\n"}
                ],
                "applied": applied,
            },
        )

    def test_pending_diff_populates_metadata(self, mm):
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        task = Task(task_id="t1", instruction="update page some-page")

        result = ctrl._build_wiki_diff_result(
            task, self._plan(), "wiki_agent", [self._diff_result(applied=False)]
        )

        assert result is not None
        assert result.metadata["pending_diffs"] == [
            {"page_name": "some-page", "diff": "@@ -1,1 +1,1 @@\n-old\n+new\n", "status": "pending"}
        ]
        assert result.metadata["priority"] == 1
        assert "Proposed" in result.answer
        assert "some-page" in result.answer

    def test_applied_diff_has_applied_status_and_answer(self, mm):
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        task = Task(task_id="t1", instruction="x")

        result = ctrl._build_wiki_diff_result(
            task, self._plan(), "wiki_agent", [self._diff_result(applied=True)]
        )

        assert result.metadata["pending_diffs"][0]["status"] == "applied"
        assert "Applied" in result.answer

    def test_multiple_diffs_in_one_turn_all_reach_metadata(self, mm):
        """
        docs/architecture/17-wiki-agent-diff-target.md's open item: the
        model has only ever proposed one diff per instruction to date, so
        this path (WikiAgent.output["diffs"] with len > 1) has never been
        live-exercised. _build_wiki_diff_result's own list comprehension
        already has no [0]/single-item assumption — this is the missing
        direct coverage proving that, not a code change (episode-browsing-
        ui-plan.md Phase 3).
        """
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        task = Task(task_id="t1", instruction="update page-a and page-b")

        result = ctrl._build_wiki_diff_result(
            task, self._plan(), "wiki_agent",
            [self._diff_result(diffs=[
                {"page_name": "page-a", "diff": "@@ -1,1 +1,1 @@\n-old a\n+new a\n"},
                {"page_name": "page-b", "diff": "@@ -1,1 +1,1 @@\n-old b\n+new b\n"},
            ])],
        )

        assert result is not None
        assert result.metadata["pending_diffs"] == [
            {"page_name": "page-a", "diff": "@@ -1,1 +1,1 @@\n-old a\n+new a\n", "status": "pending"},
            {"page_name": "page-b", "diff": "@@ -1,1 +1,1 @@\n-old b\n+new b\n", "status": "pending"},
        ]
        assert "page-a" in result.answer
        assert "page-b" in result.answer

    def test_empty_diffs_returns_none(self, mm):
        """Regression guard: a pure ingest (new pages only, no diffs) must
        still fall through to the generic synthesizer, unaffected."""
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        task = Task(task_id="t1", instruction="x")

        result = ctrl._build_wiki_diff_result(task, self._plan(), "wiki_agent", [self._diff_result(diffs=[])])
        assert result is None

    def test_non_wiki_agent_returns_none(self, mm):
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        task = Task(task_id="t1", instruction="x")

        result = ctrl._build_wiki_diff_result(
            task, self._plan(), "conversational_agent", [self._diff_result()]
        )
        assert result is None

    def test_multi_result_returns_none(self, mm):
        """Compound dispatch (>1 result) is out of scope for this pass."""
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        task = Task(task_id="t1", instruction="x")

        result = ctrl._build_wiki_diff_result(
            task, self._plan(), "wiki_agent", [self._diff_result(), self._diff_result()]
        )
        assert result is None

    def test_failed_result_returns_none(self, mm):
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        task = Task(task_id="t1", instruction="x")

        result = ctrl._build_wiki_diff_result(
            task, self._plan(), "wiki_agent", [self._diff_result(status=TaskStatus.FAILED)]
        )
        assert result is None

    def test_end_to_end_ingest_with_diffs_bypasses_synthesizer(self, mm):
        """Full pipeline: P1 routes to wiki_agent, whose output carries a
        diff — handle_task()'s result must carry pending_diffs, and the
        generic synthesizer's inference call must never fire (proves the
        branch actually short-circuits self._synthesizer.synthesize())."""
        rt = make_runtime()
        wiki = MagicMock()
        wiki.name = "wiki_agent"
        wiki.can_handle.return_value = True
        wiki.run.return_value = self._diff_result(applied=False)
        ctrl = ControllerAgent(runtime=rt, agents=[wiki], memory_manager=mm)

        result = ctrl.handle_task({
            "instruction": "ingest this document",
            "context":     {"raw_path": "/data/notes.md"},
        })

        assert result["status"] == "complete"
        assert result["metadata"]["pending_diffs"][0]["page_name"] == "some-page"
        rt.infer.assert_not_called()


# ---------------------------------------------------------------------------
# Priority 1c compound plan — tool_context wiring for wiki_agent diff-only
# path (Planner._priority1c_pinned_diff())
# ---------------------------------------------------------------------------

class TestPinnedDiffToolContextWiring:
    """
    A RoutingPlan carrying both diff_target and a non-empty tools_to_call
    (Priority 1c's compound case) must have the dispatched tool's result
    text wired into SubTask.context["tool_context"] alongside
    context["diff_target"] — WikiAgent's diff-only path builds its own
    prompt and never reads _prebuilt_prompt/_prebuilt_system, so this is
    the only way fetched content reaches it. Bypasses real Planner routing
    (patch.object on ctrl._planner) and patches MCPToolDispatcher wholesale,
    matching TestDeferredFileOpDispatch's convention.
    """

    @staticmethod
    def _compound_plan() -> RoutingPlan:
        return RoutingPlan(
            agent          = "wiki_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            priority       = 1,
            diff_target    = "some-page",
            diff_target_source = "pinned",
            tools_to_call  = ["url_fetch"],
            compound       = True,
            tool_signal_source = "keyword",
        )

    def test_successful_tool_result_wired_into_tool_context(self, mm):
        rt   = make_runtime()
        wiki = make_wiki_agent()
        received: list[SubTask] = []
        def capture(subtask):
            received.append(subtask)
            return wiki.run.return_value
        wiki.run.side_effect = capture

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name  = "url_fetch",
                parameters = "url='https://example.com/changelog'",
                result     = "Title: Changelog\nSource: https://example.com/changelog\n\nAdded X.",
                success    = True,
            )
        ]

        ctrl = ControllerAgent(runtime=rt, agents=[wiki], memory_manager=mm)
        with patch.object(ctrl._planner, "route", return_value=self._compound_plan()), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            ctrl.handle_task({
                "instruction": "fetch this url https://example.com/changelog and update page to reflect it",
            })

        assert received[0].context["diff_target"]  == "some-page"
        assert received[0].context["tool_context"] == (
            "Title: Changelog\nSource: https://example.com/changelog\n\nAdded X."
        )

    def test_no_tool_results_leaves_tool_context_absent(self, mm):
        rt   = make_runtime()
        wiki = make_wiki_agent()
        received: list[SubTask] = []
        def capture(subtask):
            received.append(subtask)
            return wiki.run.return_value
        wiki.run.side_effect = capture

        # Same compound plan, but the dispatcher fails entirely — no
        # successful results.
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name  = "url_fetch",
                parameters = "url='https://example.com/changelog'",
                result     = "ERROR: could not reach host",
                success    = False,
            )
        ]

        ctrl = ControllerAgent(runtime=rt, agents=[wiki], memory_manager=mm)
        with patch.object(ctrl._planner, "route", return_value=self._compound_plan()), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            ctrl.handle_task({
                "instruction": "fetch this url https://example.com/changelog and update page to reflect it",
            })

        assert received[0].context["diff_target"] == "some-page"
        assert "tool_context" not in received[0].context


# ---------------------------------------------------------------------------
# Episodic write path (Priority 2)
# ---------------------------------------------------------------------------

class TestEpisodicWritePath:

    def test_write_episode_true_on_memory_keyword(self, mm, db_path):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task(
            {"instruction": "remember that I prefer step-by-step instructions"}
        )

        routing = conv._received[0].context["_routing"]
        assert routing["write_episode"] is True

    def test_episode_written_to_db(self, db_path, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task(
            {"instruction": "remember that I prefer step-by-step instructions"}
        )

        reader  = EpisodicMemoryReader(db_path=db_path)
        records = reader.by_recency(project_context="general")
        # At least one episode was written this session
        assert len(records) >= 1


# ---------------------------------------------------------------------------
# Episodic retrieval path (fetch_episodic=True)
# ---------------------------------------------------------------------------

class TestEpisodicRetrievalPath:

    def test_episodic_bullets_appear_in_prompt(self, db_path, mm):
        # Seed an episode directly
        writer = EpisodicMemoryWriter(db_path=db_path)
        writer.insert(
            episode_type    = "preference",
            subject         = "output format",
            content         = "User prefers step-by-step instructions.",
            source          = "explicit",
            confidence      = 1.0,
            project_context = "general",
        )

        # Instruction contains episodic keyword ("preferences") → P5 keyword match
        rt = make_runtime(infer_return="yes")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What are my formatting preferences?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[EPISODIC MEMORY]" in prompt

    def test_episodic_flag_in_routing(self, db_path, mm):
        writer = EpisodicMemoryWriter(db_path=db_path)
        writer.insert(
            episode_type    = "correction",
            subject         = "vault resolver",
            content         = "raw_path passed explicitly.",
            source          = "explicit",
            project_context = "general",
        )

        rt = make_runtime(infer_return="yes")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is my workflow preference?"})

        routing = conv._received[0].context["_routing"]
        assert routing["fetch_episodic"] is True


# ---------------------------------------------------------------------------
# Recency cache (by_recency() cached per project_context, invalidated on write)
# ---------------------------------------------------------------------------

class TestRecencyCache:
    """
    ControllerAgent._recency_cache — by_recency() depends only on
    project_context (not the instruction), so its result is reused across
    consecutive fetch_episodic turns until an episodic write invalidates
    the cache. RoutingPlans are injected directly via patch.object on
    ctrl._planner.route (same pattern as TestGraphQueryFetch) so each
    turn's fetch_episodic/write_episode flags are precise and don't depend
    on real Planner keyword routing.
    """

    @staticmethod
    def _episodic_plan() -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = True,
            fetch_rag      = False,
            write_episode  = False,
            priority       = 5,
        )

    @staticmethod
    def _write_plan() -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            write_episode  = True,
            priority       = 2,
        )

    def test_by_recency_reused_across_turns_without_write(self, db_path, mm):
        rt   = make_runtime(infer_return="Test answer.")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._episodic_plan()

        # process_implicit_extraction is pinned to None here — the post-
        # response implicit hook runs on every turn where write_episode is
        # False (real code, not part of what this test targets), and
        # "my preferences" incidentally substring-matches the implicit
        # signal gate's "my preference" trigger. Left unpatched, the real
        # pipeline would write a genuine (if spurious) episode and clear
        # the cache as a side effect, defeating the point of this test.
        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(EpisodicMemoryReader, "by_recency", return_value=[]) as by_recency_mock, \
             patch("localist.controller_agent.process_implicit_extraction", return_value=None):
            ctrl.handle_task({"instruction": "What are my preferences?"})
            ctrl.handle_task({"instruction": "Remind me of my preferences again."})

        # Second turn is a cache hit — by_recency() must not run again.
        assert by_recency_mock.call_count == 1

    def test_by_recency_requeried_after_write_invalidates_cache(self, db_path, mm):
        rt   = make_runtime(infer_return="Test answer.")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        episodic_plan = self._episodic_plan()
        write_plan    = self._write_plan()
        extraction    = ExtractionResult(
            episode_type = "preference",
            subject      = "output format",
            content      = "User prefers step-by-step instructions.",
            source       = "explicit",
            confidence   = 1.0,
        )

        with patch.object(
            ctrl._planner, "route",
            side_effect=[episodic_plan, write_plan, episodic_plan],
        ), patch.object(
            EpisodicMemoryReader, "by_recency", return_value=[],
        ) as by_recency_mock, patch(
            "localist.controller_agent.process_explicit_signal", return_value=extraction,
        ), patch(
            "localist.controller_agent.process_implicit_extraction", return_value=None,
        ):
            ctrl.handle_task({"instruction": "What are my preferences?"})
            ctrl.handle_task({"instruction": "remember that I prefer step-by-step instructions"})
            ctrl.handle_task({"instruction": "What are my preferences now?"})

        # Turn 1: cache miss (queried, cached). Turn 2: the write invalidates
        # the cache *before* Step 5 runs, and Step 5 now runs unconditionally
        # for every conversational-agent turn (Episodic Injection rule,
        # §4.3a) regardless of fetch_episodic — so it re-queries into an
        # empty cache and repopulates it. Turn 3: cache hit against what
        # turn 2 just repopulated — no further query. Net count still 2,
        # same total as before this rule, just via a different turn-2 path.
        assert by_recency_mock.call_count == 2


# ---------------------------------------------------------------------------
# Episodic Injection rule (§4.3a of 04-planner-routing-model.md):
# Step 5 decoupled from plan.fetch_episodic + new Mode 4 graph-neighbor
# expansion
# ---------------------------------------------------------------------------

class TestEpisodicInjectionUnconditionalRecall:
    """
    Step 5 now runs on every conversational-agent, non-graph-query turn
    regardless of plan.fetch_episodic — closing the class of gap where a
    real matching episode exists but the instruction trips no P5
    keyword/semantic signal at all (a true P6 fallback). Quality control
    stays entirely on the existing confidence>=0.7 formatting floor
    (format_episodic_summary(), §2.7) — unchanged.
    """

    def test_recall_runs_on_true_fallback_turn_with_no_episodic_signal(self, db_path, mm):
        writer = EpisodicMemoryWriter(db_path=db_path)
        writer.insert(
            episode_type    = "project_fact",
            subject         = "Claude Impact Lab",
            content         = "User is participating in a Claude Impact Lab on August 6th.",
            source          = "explicit",
            confidence      = 1.0,
            project_context = "general",
        )

        rt   = make_runtime(infer_return="no")  # "no" → P5 semantic gate does not fire
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        # No episodic/RAG/tool keyword anywhere in this instruction — a true
        # P6 fallback under the old gate, which never attempted recall here.
        ctrl.handle_task({"instruction": "What should I pack for a trip to Denver?"})

        routing = conv._received[0].context["_routing"]
        assert routing["fetch_episodic"] is False, (
            "P5 keyword/semantic gate correctly did not fire — "
            "fetch_episodic stays informational, not a gate, per §4.3a"
        )

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[EPISODIC MEMORY]" in prompt
        assert "Claude Impact Lab" in prompt

    def test_no_episodes_no_slot_rendered(self, mm):
        """Empty store — Step 5 runs but produces nothing to inject."""
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What should I pack for a trip to Denver?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[EPISODIC MEMORY]" not in prompt

    def test_p3c_graph_query_turn_still_excluded(self, db_path, mm):
        """
        The unconditional-recall change must not regress §5b's P3c
        mutual-exclusivity guarantee — see also
        TestGraphQueryFetch.test_p3c_purity_no_rag_or_episodic_slots for the
        full real-Planner end-to-end version of this check.
        """
        writer = EpisodicMemoryWriter(db_path=db_path)
        writer.insert(
            episode_type    = "project_fact",
            subject         = "graph query purity",
            content         = "PURITY_LEAK: should not appear on a P3c turn.",
            source          = "explicit",
            confidence      = 1.0,
            project_context = "general",
        )

        rt   = make_runtime(infer_return="yes")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            priority       = 3,
            graph_query    = ("incoming", 1, "some-page"),
        )

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(mm, "get_backlinks", return_value=[]):
            ctrl.handle_task({"instruction": "what links to some-page"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[EPISODIC MEMORY]" not in prompt
        assert "PURITY_LEAK" not in prompt


class TestEpisodicGraphNeighborExpansion:
    """
    Mode 4 (controller_agent.py Step 5): when the top Mode 3 candidate has
    an episode graph node (§8.9 Phase B), its resolved outgoing edge(s) are
    followed to a wiki concept node, and that concept's backlinks —
    including sibling episode-sourced ones — are pulled in as additional
    Slot 3a candidates. Uses a MagicMock memory_manager with EpisodicMemoryReader's
    query methods patched directly, mirroring TestGraphQueryFetch's style,
    so the controller's Mode 4 orchestration is tested in isolation from
    real BM25/graph-write mechanics (covered separately in
    test_memory_phase1.py and 08-graph-retrieval-layer.md's own tests).
    """

    @staticmethod
    def _record(id_, subject="s", content="c", confidence=1.0):
        return EpisodeRecord(
            id=id_, episode_type="project_fact", subject=subject, content=content,
            confidence=confidence, source="model_extracted", task_id=None,
            conversation_id=None, project_context="general", status="active",
            created_at=time.time(), last_accessed=None,
        )

    def _wire_graph(self, mm_mock, sibling_doc_path: str = "episode://2"):
        mm_mock.get_graph_node_by_doc_path.side_effect = lambda p: (
            {"id": 10, "doc_path": "episode://1", "title": None} if p == "episode://1"
            else {"id": 20, "doc_path": "/wiki/omlx.md", "title": "oMLX"} if p == "/wiki/omlx.md"
            else None
        )
        mm_mock.get_outgoing_links.return_value = [
            GraphEdgeResult(
                link_text="oMLX", target_path="omlx", target_resolved=True,
                node_title="oMLX", node_doc_path="/wiki/omlx.md",
            )
        ]
        mm_mock.get_backlinks.return_value = [
            GraphEdgeResult(
                link_text=None, target_path=None, target_resolved=True,
                node_title=None, node_doc_path=sibling_doc_path,
            ),
        ]

    def test_sibling_episode_surfaced_via_shared_concept(self, db_path, mm):
        top_match = self._record(1, subject="oMLX version", content="oMLX 0.4.2 is current.")
        sibling   = self._record(2, subject="oMLX release notes", content="0.4.2 fixed the paging bug.")

        mm_mock = MagicMock()
        mm_mock._db_path = db_path
        self._wire_graph(mm_mock)

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=False, priority=6)

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(EpisodicMemoryReader, "by_recency", return_value=[]), \
             patch.object(EpisodicMemoryReader, "by_similarity", return_value=[top_match]), \
             patch.object(EpisodicMemoryReader, "get_by_ids", return_value=[sibling]):
            ctrl.handle_task({"instruction": "tell me about oMLX"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "oMLX 0.4.2 is current." in prompt
        assert "0.4.2 fixed the paging bug." in prompt

    def test_low_confidence_sibling_not_injected(self, db_path, mm):
        """Mode 4 only adds *candidates* — the existing 0.7 injection floor
        (format_episodic_summary(), §2.7) still governs what surfaces."""
        top_match = self._record(1, subject="oMLX version", content="oMLX 0.4.2 is current.")
        sibling   = self._record(
            2, subject="oMLX rumor", content="Unverified: oMLX may add X next.",
            confidence=0.5,
        )

        mm_mock = MagicMock()
        mm_mock._db_path = db_path
        self._wire_graph(mm_mock)

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=False, priority=6)

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(EpisodicMemoryReader, "by_recency", return_value=[]), \
             patch.object(EpisodicMemoryReader, "by_similarity", return_value=[top_match]), \
             patch.object(EpisodicMemoryReader, "get_by_ids", return_value=[sibling]):
            ctrl.handle_task({"instruction": "tell me about oMLX"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "oMLX 0.4.2 is current." in prompt
        assert "Unverified: oMLX may add X next." not in prompt

    def test_graph_lookup_failure_degrades_gracefully(self, db_path, mm):
        top_match = self._record(1, subject="oMLX version", content="oMLX 0.4.2 is current.")

        mm_mock = MagicMock()
        mm_mock._db_path = db_path
        mm_mock.get_graph_node_by_doc_path.side_effect = RuntimeError("SQLite locked")

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=False, priority=6)

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(EpisodicMemoryReader, "by_recency", return_value=[]), \
             patch.object(EpisodicMemoryReader, "by_similarity", return_value=[top_match]):
            result = ctrl.handle_task({"instruction": "tell me about oMLX"})

        assert result["status"] == "complete"
        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "oMLX 0.4.2 is current." in prompt

    def test_no_similarity_match_skips_graph_lookup_entirely(self, db_path, mm):
        mm_mock = MagicMock()
        mm_mock._db_path = db_path

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=False, priority=6)

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(EpisodicMemoryReader, "by_recency", return_value=[]), \
             patch.object(EpisodicMemoryReader, "by_similarity", return_value=[]):
            ctrl.handle_task({"instruction": "What is 2+2?"})

        mm_mock.get_graph_node_by_doc_path.assert_not_called()


# ---------------------------------------------------------------------------
# Working memory path
# ---------------------------------------------------------------------------

class TestWorkingMemoryPath:

    def test_prior_turns_appear_in_slot3(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Answer 2.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        task_id = "wm-test-task"

        # Add a prior turn directly to memory
        mm.add(
            role    = "user",
            content = "What is LORA?",
            task_id = task_id,
        )
        mm.add(
            role    = "agent",
            content = "LORA is a local research assistant.",
            task_id = task_id,
        )

        ctrl.handle_task({
            "task_id":     task_id,
            "instruction": "Tell me more about it.",
        })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[WORKING MEMORY]" in prompt
        assert "LORA is a local research assistant." in prompt


# ---------------------------------------------------------------------------
# Backend-tier-aware context window (context_profile.py)
# ---------------------------------------------------------------------------

class TestContextProfileTiering:
    """
    Neither tier caps by turn count any more (working_memory_limit=None on
    both LOCAL_PROFILE and CLOUD_PROFILE — context_profile.py). Local
    runtimes (is_local truthy — including the MagicMock default used by
    every other test in this file) are now budgeted from the runtime's
    max_model_len instead of a fixed row count; a MagicMock's unconfigured
    `max_model_len` attribute isn't a real int, so profile_for() falls back
    to its conservative default and floors at 300 tokens (see
    context_profile.py's _LOCAL_MAX_MODEL_LEN_FALLBACK /
    _LOCAL_WORKING_MEMORY_FLOOR_TOKENS) — small enough to still trim old
    turns by token budget, just no longer by row count.
    """

    def test_local_runtime_trims_by_token_budget_not_row_count(self, mm):
        rt = make_runtime(infer_return="no")
        rt.is_local = True   # explicit, though MagicMock's default is already truthy
        conv = make_conv_agent("Answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        task_id = "local-tier-token-trim-test"
        # Long enough per-turn that the 300-token floor (not a row count)
        # is what trims the oldest ones — a plain "turn-marker-N" alone is
        # short enough that even 60 of them fit inside 300 tokens.
        padding = "x" * 100
        for i in range(20):
            mm.add(role="user", content=f"turn-marker-{i}-{padding}", task_id=task_id)

        ctrl.handle_task({"task_id": task_id, "instruction": "Continue."})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        # No row-count cap: it's the 300-token ceiling, not "keep the last
        # 5 rows", that drops the oldest markers here.
        assert "turn-marker-0-" not in prompt
        assert "turn-marker-19-" in prompt

    def test_cloud_runtime_removes_turn_cap(self, mm):
        rt = make_runtime(infer_return="no")
        rt.is_local = False   # e.g. Ollama serving a "-cloud"-suffixed model
        conv = make_conv_agent("Answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        task_id = "cloud-tier-no-cap-test"
        for i in range(12):
            mm.add(role="user", content=f"turn-marker-{i}", task_id=task_id)

        ctrl.handle_task({"task_id": task_id, "instruction": "Continue."})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        # All twelve turns survive — no 5-row cap under the cloud profile,
        # and the (tiny, well under 60k tokens) content never nears the
        # raised token ceiling either.
        for i in range(12):
            assert f"turn-marker-{i}" in prompt


# ---------------------------------------------------------------------------
# Tool stub path (Priority 3)
# ---------------------------------------------------------------------------

class TestToolStubPath:

    def test_tool_signal_sets_tools_to_call(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What are the latest oMLX changes?"})

        routing = conv._received[0].context["_routing"]
        assert "web_search" in routing["tools_to_call"]

    def test_tool_stub_does_not_add_tool_results_slot(self, mm):
        """
        MCPToolDispatcher is fully wired (Phase 6) and does fire for
        web_search-shaped instructions like this one — so this test now
        forces localist-mcp to be unreachable via the _open_session test
        seam (see its docstring in mcp_tool_dispatcher.py) to exercise the
        graceful-failure path: no [TOOL RESULTS] slot, not a hallucinated
        result. Without this mock, dispatch() would attempt a real SSE
        connection to _MCP_SERVER_URL — succeeding (and adding real search
        content to the prompt) whenever a live localist-mcp happens to be
        reachable, e.g. via start_localist.sh.
        """
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        async def fake_open_session_unreachable(stack):
            raise ConnectionError("mock: localist-mcp unreachable")

        with patch(
            "localist.mcp_tool_dispatcher.MCPToolDispatcher._open_session",
            side_effect=fake_open_session_unreachable,
        ):
            ctrl.handle_task({"instruction": "What are the latest oMLX changes?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" not in prompt


# ---------------------------------------------------------------------------
# Chart artifact -> ControllerResult.metadata["chart"] (_execute_plan Step 3)
# ---------------------------------------------------------------------------

class TestChartArtifactMetadata:
    """
    ToolResult.artifact (mcp_tool_dispatcher.py's _run_chart, Phase 3) must
    reach ControllerResult.metadata["chart"] on a successful chart dispatch,
    and the key must be absent entirely (not null) on any other turn — see
    _build_conversational_result's chart_artifact parameter.

    Same wholesale-MCPToolDispatcher-mock pattern as
    TestDeferredFileOpDispatch above; dispatch()'s own MCP-session behavior
    is already covered by test_mcp_tool_dispatcher.py.
    """

    def _make_chart_plan(self) -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            priority       = 3,
            compound       = True,
            tools_to_call  = ["chart"],
        )

    def test_successful_chart_dispatch_populates_metadata_chart(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Here's your chart.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._make_chart_plan()

        chart_config = {
            "chart_type": "bar",
            "title":      "Fruit Inventory",
            "labels":     ["apples", "oranges"],
            "datasets":   [{"label": "Count", "data": [5, 3]}],
        }
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name  = "chart",
                parameters = "chart_type='bar'",
                result     = "Generated bar chart: Fruit Inventory",
                success    = True,
                artifact   = {"png_path": "charts/abc123.png", "chart_config": chart_config},
            )
        ]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "chart this: apples 5, oranges 3"}
            )

        assert result["metadata"]["chart"] == {
            "png_path":     "charts/abc123.png",
            "chart_config": chart_config,
        }

    def test_non_chart_turn_omits_chart_key_entirely(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Test answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "What is 2+2?"})

        assert "chart" not in result["metadata"]


# ---------------------------------------------------------------------------
# workflow_id/workflow_steps -> ControllerResult.metadata (_execute_plan
# Step 3, episode-browsing-ui-plan.md Phase 2)
# ---------------------------------------------------------------------------

class TestWorkflowStepsMetadata:
    """
    Every ToolResult a research-loop run produces shares one workflow_id
    (mcp_tool_dispatcher._run_research_loop) — _execute_plan pulls that id
    and the ordered step list out into ControllerResult.metadata, same
    "wholesale-MCPToolDispatcher-mock, pull the dict out of metadata"
    pattern as TestChartArtifactMetadata above.
    """

    def _make_research_plan(self) -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            priority       = 3,
            compound       = True,
            tools_to_call  = ["research"],
        )

    def test_research_loop_populates_workflow_metadata(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("The Basic plan is $10/month.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._make_research_plan()

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name   = "web_search",
                parameters  = "query='basic plan cost'",
                result      = "See https://vendor.example/pricing for plan details.",
                success     = True,
                workflow_id = "wf-1",
            ),
            _ToolResult(
                tool_name   = "url_fetch",
                parameters  = "url='https://vendor.example/pricing'",
                result      = "The Basic plan is $10 per month.",
                success     = True,
                workflow_id = "wf-1",
            ),
        ]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "what does the basic plan cost"}
            )

        assert result["metadata"]["workflow_id"] == "wf-1"
        assert result["metadata"]["workflow_steps"] == [
            {
                "tool_name":  "web_search",
                "parameters": "query='basic plan cost'",
                "success":    True,
                "result":     "See https://vendor.example/pricing for plan details.",
            },
            {
                "tool_name":  "url_fetch",
                "parameters": "url='https://vendor.example/pricing'",
                "success":    True,
                "result":     "The Basic plan is $10 per month.",
            },
        ]

    def test_non_research_turn_omits_workflow_keys_entirely(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Test answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "What is 2+2?"})

        assert "workflow_id" not in result["metadata"]
        assert "workflow_steps" not in result["metadata"]

    def test_plain_web_search_without_workflow_id_omits_workflow_keys(self, mm):
        """A non-research web_search tool call has workflow_id=None on its
        ToolResult — must not be mistaken for a research-loop workflow."""
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Zebras are striped.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            priority       = 3,
            compound       = True,
            tools_to_call  = ["web_search"],
        )

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name  = "web_search",
                parameters = "query='zebras'",
                result     = "Zebras are striped equines.",
                success    = True,
            ),
        ]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task({"instruction": "tell me about zebras"})

        assert "workflow_id" not in result["metadata"]
        assert "workflow_steps" not in result["metadata"]


# ---------------------------------------------------------------------------
# _strip_false_tool_attribution — narrow mitigation for Open Item 2's
# 2026-07-20 repro (docs/architecture/14-localist-mcp-tool-layer.md §14.7):
# a model claiming "(Source: Tool output)" on a turn where tools_to_call
# was empty. Not a fix for Open Item 2 itself — see the function's own
# docstring and the doc update.
# ---------------------------------------------------------------------------

class TestStripFalseToolAttribution:

    def test_strips_parenthetical_source_tool_output(self):
        answer = (
            "I have generated a pie chart showing the Browser Market Share: "
            "Chrome (65), Safari (19), Firefox (8), Edge (6), and Other (2).\n"
            "(Source: Tool output)"
        )
        result = _strip_false_tool_attribution(answer)
        assert "(Source: Tool output)" not in result
        assert "Browser Market Share" in result

    def test_strips_bare_source_tool_output_phrase(self):
        answer = "Here is the summary. Source: Tool output"
        result = _strip_false_tool_attribution(answer)
        assert "tool output" not in result.lower()

    def test_strips_via_tool_output_phrase(self):
        answer = "The result was computed via tool output."
        result = _strip_false_tool_attribution(answer)
        assert "tool output" not in result.lower()

    def test_case_insensitive_match(self):
        answer = "Done. (SOURCE: TOOL OUTPUT)"
        result = _strip_false_tool_attribution(answer)
        assert "tool output" not in result.lower()

    def test_no_match_returns_answer_unchanged(self):
        answer = "Just a normal answer with no tool citation."
        assert _strip_false_tool_attribution(answer) == answer

    def test_falls_back_to_caveat_when_stripping_would_empty_the_answer(self):
        answer = "(Source: Tool output)"
        result = _strip_false_tool_attribution(answer)
        assert "(Source: Tool output)" not in result
        assert "no tool actually ran" in result


class TestFalseToolAttributionIntegration:
    """
    Wiring into _build_conversational_result — only strips when
    plan.tools_to_call is empty for the turn.
    """

    def test_empty_tools_turn_strips_false_attribution(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent(
            "I have generated a pie chart. (Source: Tool output)"
        )
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "What is 2+2?"})

        assert "(Source: Tool output)" not in result["answer"]

    def test_real_tool_fired_citation_not_stripped(self, mm):
        """The more important test: a legitimate tool citation on a turn
        where a tool actually ran (tools_to_call non-empty) must survive
        untouched — a false positive here would strip real citations."""
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent(
            "I have generated a pie chart. (Source: Tool output)"
        )
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            priority       = 3,
            compound       = True,
            tools_to_call  = ["chart"],
        )

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name  = "chart",
                parameters = "chart_type='pie'",
                result     = "Generated pie chart: Browser Market Share",
                success    = True,
            )
        ]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "chart this: apples 5, oranges 3"}
            )

        assert "(Source: Tool output)" in result["answer"]


# ---------------------------------------------------------------------------
# _extract_file_op_content / _file_op_confirmation_line — unit tests
# ---------------------------------------------------------------------------

class TestExtractFileOpContent:

    def test_strips_leading_label_and_trailing_parenthetical(self):
        answer = (
            "Haiku about the sea:\n\n"
            "Blue expanse so wide,\n"
            "Waves that whisper ancient tales,\n"
            "Horizon holds sky.\n\n"
            "(Attempting to save content to haiku.md)"
        )
        assert _extract_file_op_content(answer) == (
            "Blue expanse so wide,\n"
            "Waves that whisper ancient tales,\n"
            "Horizon holds sky."
        )

    def test_no_label_or_parenthetical_passes_through_unchanged(self):
        answer = "Blue expanse so wide,\nWaves that whisper ancient tales,\nHorizon holds sky."
        assert _extract_file_op_content(answer) == answer

    def test_label_only_no_trailing_parenthetical(self):
        answer = "Summary:\n\nThe meeting covered three topics."
        assert _extract_file_op_content(answer) == "The meeting covered three topics."

    def test_trailing_parenthetical_only_no_label(self):
        answer = "The meeting covered three topics.\n\n(Saving this to summary.md)"
        assert _extract_file_op_content(answer) == "The meeting covered three topics."

    def test_content_with_internal_colon_not_mistaken_for_label(self):
        """A single-line answer with a colon must not be treated as a label
        line stripped down to nothing — the guard against zeroing out real
        content should keep it intact."""
        answer = "Remember: buy milk and walk the dog."
        assert _extract_file_op_content(answer) == answer

    def test_multiline_answer_with_colon_in_body_not_first_line(self):
        answer = "Notes:\n\nTODO: buy milk\nTODO: walk the dog"
        assert _extract_file_op_content(answer) == "TODO: buy milk\nTODO: walk the dog"

    def test_markdown_italicized_trailing_aside_is_stripped(self):
        """Observed live: the model wrapped its whole aside in markdown
        italics with a backticked filename, e.g.
        '*(This haiku has been generated and is ready to be saved as
        `haiku.md`.)*' — the plain (...) pattern alone doesn't match this
        because of the leading/trailing '*'."""
        answer = (
            "Blue expanse so wide,\n"
            "Waves crash on the sandy shore,\n"
            "Salt wind fills the air.\n\n"
            "*(This haiku has been generated and is ready to be saved as `haiku.md`.)*"
        )
        assert _extract_file_op_content(answer) == (
            "Blue expanse so wide,\n"
            "Waves crash on the sandy shore,\n"
            "Salt wind fills the air."
        )

    def test_parenthetical_that_is_the_whole_answer_is_not_stripped_to_empty(self):
        """Guard: if stripping the trailing parenthetical would leave nothing,
        keep the original text instead of writing an empty file."""
        answer = "(just kidding, no real content here)"
        assert _extract_file_op_content(answer) == answer


class TestFileOpConfirmationLine:

    def test_success_reports_actual_written_filename(self):
        result = _ToolResult(
            tool_name="file_op", parameters="", success=True,
            result="OK: wrote 34 characters to haiku.md",
        )
        assert _file_op_confirmation_line(result, "haiku.md") == "\n\n*(Saved to haiku.md)*"

    def test_success_reports_versioned_filename_when_original_existed(self):
        result = _ToolResult(
            tool_name="file_op", parameters="", success=True,
            result="OK: wrote 34 characters to haiku_2.md",
        )
        # fallback_path is the pre-versioning plan.file_op_path; the actual
        # written name (post version-cap fallback) must win when present.
        assert _file_op_confirmation_line(result, "haiku.md") == "\n\n*(Saved to haiku_2.md)*"

    def test_success_falls_back_to_plan_path_when_message_has_no_filename(self):
        result = _ToolResult(
            tool_name="file_op", parameters="", success=True,
            result="OK: skipped duplicate append for turn_id=abc (already applied)",
        )
        assert _file_op_confirmation_line(result, "log.md") == "\n\n*(Saved to log.md)*"

    def test_failure_strips_error_prefix(self):
        result = _ToolResult(
            tool_name="file_op", parameters="", success=False,
            result="ERROR: path traversal outside project_root is not permitted",
        )
        assert _file_op_confirmation_line(result, "x.md") == (
            "\n\n*(Could not save — path traversal outside project_root is not permitted)*"
        )


# ---------------------------------------------------------------------------
# Deferred file_op dispatch (_execute_plan Step 7b)
# ---------------------------------------------------------------------------

class TestDeferredFileOpDispatch:
    """
    plan.file_op_deferred means Planner detected a file_op-shaped
    instruction whose content had to be generated by the agent first (see
    planner.py's P3 content-present-vs-deferred split). These tests bypass
    real Planner routing (patch.object on ctrl._planner) and patch
    MCPToolDispatcher wholesale — dispatch()'s own MCP-session behavior is
    already covered by test_mcp_tool_dispatcher.py; these only verify
    _execute_plan's new Step 7b wiring (content extraction, dispatch args,
    and the appended confirmation/failure line).
    """

    def _make_deferred_plan(self, path: str = "haiku.md", action: str = "write") -> RoutingPlan:
        return RoutingPlan(
            agent            = "conversational_agent",
            fetch_episodic   = False,
            fetch_rag        = False,
            priority         = 3,
            compound         = True,
            file_op_deferred = True,
            file_op_path     = path,
            file_op_action   = action,
        )

    def test_success_strips_label_paren_for_content_and_appends_confirmation(self, mm):
        raw_answer = (
            "Haiku about the sea:\n\n"
            "Blue expanse so wide,\n"
            "Waves that whisper ancient tales,\n"
            "Horizon holds sky.\n\n"
            "(Attempting to save content to haiku.md)"
        )
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent(raw_answer)
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._make_deferred_plan()

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name  = "file_op",
                parameters = "action='write' path='haiku.md'",
                result     = "OK: wrote 74 characters to haiku.md",
                success    = True,
            )
        ]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "write a haiku about the sea and save it as haiku.md"}
            )

        # The label/parenthetical framing must not leak into the saved content.
        _, kwargs = mock_dispatcher.dispatch.call_args
        assert kwargs["context"]["file_op_content"] == (
            "Blue expanse so wide,\n"
            "Waves that whisper ancient tales,\n"
            "Horizon holds sky."
        )
        assert kwargs["context"]["file_op_path"]   == "haiku.md"
        assert kwargs["context"]["file_op_action"] == "write"
        assert kwargs["tools_to_call"] == ["file_op"]

        # The displayed/persisted answer keeps the model's own text verbatim
        # plus a deterministic (never model-narrated) confirmation line.
        assert result["answer"] == raw_answer + "\n\n*(Saved to haiku.md)*"

    def test_failure_appends_deterministic_failure_line(self, mm):
        raw_answer = "Blue expanse so wide, waves that whisper old secrets, horizon holds the sky."
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent(raw_answer)
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._make_deferred_plan(path="../../etc/passwd", action="write")

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name  = "file_op",
                parameters = "action='write' path='../../etc/passwd'",
                result     = "ERROR: path traversal outside project_root is not permitted",
                success    = False,
            )
        ]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "write a haiku and save it as ../../etc/passwd"}
            )

        assert result["answer"] == raw_answer + (
            "\n\n*(Could not save — path traversal outside project_root is not permitted)*"
        )

    def test_dispatch_exception_appends_failure_line_without_raising(self, mm):
        raw_answer = "Blue expanse so wide, waves that whisper old secrets, horizon holds the sky."
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent(raw_answer)
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._make_deferred_plan()

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.side_effect = RuntimeError("localist-mcp unreachable")

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("localist.controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "write a haiku about the sea and save it as haiku.md"}
            )

        assert result["status"] == "complete"
        assert result["answer"] == raw_answer + "\n\n*(Could not save — localist-mcp unreachable)*"


# ---------------------------------------------------------------------------
# Prebuilt prompt passthrough in ConversationalAgent
# ---------------------------------------------------------------------------

class TestPrebuiltPromptPassthrough:

    def test_prebuilt_path_skips_internal_rag(self):
        """When _prebuilt_prompt is present, ConversationalAgent skips
        its own corpus query and uses the prebuilt prompt verbatim."""
        from localist.conversational_agent import ConversationalAgent

        mm = MagicMock()   # mock MM — must NOT be called for query_corpus
        rt = MagicMock()
        rt.infer.return_value = "Prebuilt answer."

        agent = ConversationalAgent(runtime=rt, memory_manager=mm)

        subtask = MagicMock()
        subtask.subtask_id = "pb-test"
        subtask.instruction = "What is LORA?"
        subtask.context = {
            "_prebuilt_prompt": "[USER]\nWhat is LORA?",
            "_prebuilt_system": "You are LORA.",
            "_routing":         {"fetch_rag": True},
        }

        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == "Prebuilt answer."
        mm.query_corpus.assert_not_called()

    def test_without_prebuilt_normal_path_runs(self):
        from localist.conversational_agent import ConversationalAgent

        rt = MagicMock()
        rt.infer.return_value = "Normal answer."

        agent = ConversationalAgent(runtime=rt, memory_manager=None)

        subtask = MagicMock()
        subtask.subtask_id  = "normal-test"
        subtask.instruction = "What is 2+2?"
        subtask.context     = {}

        result = agent.run(subtask)
        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == "Normal answer."


# ---------------------------------------------------------------------------
# wiki_doc wiring — _load_persona() frontmatter handling
# ---------------------------------------------------------------------------

_PERSONA_CONTENT = (
    "You are {{ASSISTANT_NAME}}, a local‑first thinking partner.\n"
    "You speak clearly, directly, and in a natural conversational tone.\n"
    "You use tools when they are needed and follow tool instructions precisely.\n"
    "When you state facts, you cite where they came from."
)


def _mock_doc(path_str: str, content: str):
    doc = MagicMock()
    doc.path = Path(path_str)
    doc.content = content
    return doc


class TestLoadPersonaWikiDoc:

    def test_no_frontmatter_byte_identical(self):
        """Zero-behavior-change: plain content (name substituted) → cache equals the substituted content[:2000] exactly."""
        mm = MagicMock()
        mm.get_assistant_name.return_value = "Localist"
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/persona.md", _PERSONA_CONTENT)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        ctrl._load_persona()

        expected = _PERSONA_CONTENT.replace("{{ASSISTANT_NAME}}", "Localist")
        assert ctrl._persona_cache == expected[:2000]

    def test_frontmatter_stripped_body_only(self):
        """Forward-looking: frontmatter lines are excluded; body text is present."""
        content = (
            "---\n"
            "title: Persona\n"
            "type: system\n"
            "created: 2026-06-01\n"
            "---\n"
            "\n"
            "You are {{ASSISTANT_NAME}}, a local-first thinking partner.\n"
            "You are helpful, concise, and precise.\n"
        )
        mm = MagicMock()
        mm.get_assistant_name.return_value = "Localist"
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/persona.md", content)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        ctrl._load_persona()

        assert "title: Persona" not in ctrl._persona_cache
        assert "type: system" not in ctrl._persona_cache
        assert "---" not in ctrl._persona_cache
        assert "You are Localist" in ctrl._persona_cache
        assert "You are helpful" in ctrl._persona_cache

    def test_web_search_description_names_langsearch_by_default(self, monkeypatch):
        """SEARCH_PROVIDER unset → persona's "Web search" line keeps LangSearch."""
        monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
        content = "Web search fires automatically. It returns real results from LangSearch."
        mm = MagicMock()
        mm.get_assistant_name.return_value = "Localist"
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/persona.md", content)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        ctrl._load_persona()

        assert "It returns real results from LangSearch." in ctrl._persona_cache

    def test_web_search_description_names_brave_when_configured(self, monkeypatch):
        """SEARCH_PROVIDER=brave → persona's LangSearch mention is swapped."""
        monkeypatch.setenv("SEARCH_PROVIDER", "brave")
        content = "Web search fires automatically. It returns real results from LangSearch."
        mm = MagicMock()
        mm.get_assistant_name.return_value = "Localist"
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/persona.md", content)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        ctrl._load_persona()

        assert "It returns real results from Brave Search." in ctrl._persona_cache
        assert "LangSearch" not in ctrl._persona_cache

    def test_unknown_search_provider_raises_before_corpus_fetch(self, monkeypatch):
        """Bad SEARCH_PROVIDER fails loud rather than silently dropping persona."""
        monkeypatch.setenv("SEARCH_PROVIDER", "bing")
        mm = MagicMock()

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        with pytest.raises(ValueError, match="ERROR: unknown SEARCH_PROVIDER 'bing'"):
            ctrl._load_persona()
        mm.query_corpus.assert_not_called()

    def test_assistant_name_substituted_into_persona(self):
        """The persona doc's {{ASSISTANT_NAME}} placeholder is swapped for the configured name."""
        mm = MagicMock()
        mm.get_assistant_name.return_value = "Percy"
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/persona.md", _PERSONA_CONTENT)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        ctrl._load_persona()

        assert "You are Percy," in ctrl._persona_cache
        assert "{{ASSISTANT_NAME}}" not in ctrl._persona_cache

    def test_name_change_invalidates_cache_on_next_load(self):
        """A changed get_assistant_name() value re-fetches and re-substitutes."""
        mm = MagicMock()
        mm.get_assistant_name.return_value = "Percy"
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/persona.md", _PERSONA_CONTENT)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        first = ctrl._load_persona()
        assert "You are Percy," in first
        assert mm.query_corpus.call_count == 1

        mm.get_assistant_name.return_value = "Ada"
        second = ctrl._load_persona()

        assert "You are Ada," in second
        assert "Percy" not in second
        assert mm.query_corpus.call_count == 2, (
            "a name change must trigger a real re-fetch, not just re-serve "
            "the stale cached string"
        )

    def test_unchanged_name_serves_cache_without_requerying(self):
        """Same name across calls hits the cache — no redundant corpus query."""
        mm = MagicMock()
        mm.get_assistant_name.return_value = "Percy"
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/persona.md", _PERSONA_CONTENT)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        ctrl._load_persona()
        ctrl._load_persona()

        assert mm.query_corpus.call_count == 1

    def test_invalidate_persona_cache_forces_requery(self):
        """invalidate_persona_cache() clears the cache even when the name hasn't changed."""
        mm = MagicMock()
        mm.get_assistant_name.return_value = "Percy"
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/persona.md", _PERSONA_CONTENT)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        ctrl._load_persona()
        ctrl.invalidate_persona_cache()
        ctrl._load_persona()

        assert mm.query_corpus.call_count == 2


# ---------------------------------------------------------------------------
# wiki_doc wiring — _load_user_profile() frontmatter handling
# ---------------------------------------------------------------------------

_PROFILE_CONTENT_NO_FM = (
    "## About Michael\n"
    "\n"
    "- Name: Michael\n"
    "- Role: Solo developer\n"
    "\n"
    "## Preferences\n"
    "\n"
    "- I prefer concise answers.\n"
)


class TestLoadUserProfileWikiDoc:

    def _make_ctrl_with_embed(self):
        mm = MagicMock()
        mm._embed_fn = lambda _: [0.0] * 768
        return ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)

    def test_no_frontmatter_byte_identical(self, tmp_path: Path):
        """Zero-behavior-change: no frontmatter → profile_lines identical to old logic."""
        profile_file = tmp_path / "michael.md"
        profile_file.write_text(_PROFILE_CONTENT_NO_FM, encoding="utf-8")

        # Reproduce what the old logic (raw.splitlines()) would have produced
        expected = [
            line.lstrip("- ").strip()
            for line in _PROFILE_CONTENT_NO_FM.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        ctrl = self._make_ctrl_with_embed()
        with patch("pathlib.Path.exists", return_value=True), \
             patch("localist.controller_agent.load_wiki_doc",
                   side_effect=lambda _: load_wiki_doc(profile_file)):
            ctrl._load_user_profile()

        assert ctrl._profile_lines == expected

    def test_frontmatter_excluded_from_profile_lines(self, tmp_path: Path):
        """Forward-looking: frontmatter YAML lines are not ingested as fact lines."""
        content = (
            "---\n"
            "title: Michael Profile\n"
            "type: user-profile\n"
            "created: 2026-06-01\n"
            "---\n"
            "\n"
            "## About Michael\n"
            "\n"
            "- Name: Michael\n"
            "- Role: Solo developer\n"
        )
        profile_file = tmp_path / "michael.md"
        profile_file.write_text(content, encoding="utf-8")

        ctrl = self._make_ctrl_with_embed()
        with patch("pathlib.Path.exists", return_value=True), \
             patch("localist.controller_agent.load_wiki_doc",
                   side_effect=lambda _: load_wiki_doc(profile_file)):
            ctrl._load_user_profile()

        lines = ctrl._profile_lines
        assert not any("title:" in l for l in lines)
        assert not any("type:" in l for l in lines)
        assert not any("---" in l for l in lines)
        assert "Name: Michael" in lines
        assert "Role: Solo developer" in lines


# ---------------------------------------------------------------------------
# _memory_key() and conversation_id/session_id cross-turn working memory
# ---------------------------------------------------------------------------

class TestMemoryKey:

    def test_prefers_conversation_id_over_session_id(self):
        """conversation_id takes precedence over session_id and task_id."""
        task = Task(
            task_id="xyz",
            instruction="hi",
            context={"conversation_id": "conv-1", "session_id": "abc"},
        )
        assert _memory_key(task) == "conv-1"

    def test_falls_back_to_session_id_when_no_conversation_id(self):
        """session_id in context is used when conversation_id is absent."""
        task = Task(task_id="xyz", instruction="hi", context={"session_id": "abc"})
        assert _memory_key(task) == "abc"

    def test_falls_back_to_task_id_when_absent(self):
        """Callers without conversation_id/session_id (e.g. ingest path) keep today's behavior."""
        task = Task(task_id="xyz", instruction="hi", context={})
        assert _memory_key(task) == "xyz"

    def test_same_conversation_id_shares_working_memory(self, db_path: Path):
        """Two handle_task() calls with the same conversation_id accumulate in one log."""
        mm   = MemoryManager(db_path=db_path)
        rt   = make_runtime()
        conv = make_conv_agent("Answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({
            "task_id":     "task-1",
            "instruction": "First question",
            "context":     {"conversation_id": "conv-shared", "session_id": "tab-session"},
        })
        ctrl.handle_task({
            "task_id":     "task-2",
            "instruction": "Second question",
            "context":     {"conversation_id": "conv-shared", "session_id": "tab-session"},
        })

        entries = mm.get_context_window(task_id="conv-shared", limit=20)
        user_instructions = [e["content"] for e in entries if e["role"] == "user"]
        assert "First question" in user_instructions
        assert "Second question" in user_instructions

    def test_new_conversation_id_in_same_session_is_isolated(self, db_path: Path):
        """Clicking 'New chat' (new conversation_id, same page-load session_id)
        must not leak the prior conversation's turns into working memory."""
        mm   = MemoryManager(db_path=db_path)
        rt   = make_runtime()
        conv = make_conv_agent("Answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({
            "task_id":     "task-1",
            "instruction": "Old conversation question",
            "context":     {"conversation_id": "conv-old", "session_id": "tab-session"},
        })
        ctrl.handle_task({
            "task_id":     "task-2",
            "instruction": "New conversation question",
            "context":     {"conversation_id": "conv-new", "session_id": "tab-session"},
        })

        entries_new = mm.get_context_window(task_id="conv-new", limit=20)
        contents_new = [e["content"] for e in entries_new]

        assert "New conversation question" in contents_new
        assert "Old conversation question" not in contents_new

    def test_different_task_ids_without_session_are_isolated(self, db_path: Path):
        """Callers without conversation_id/session_id keep isolated per-request memory."""
        mm   = MemoryManager(db_path=db_path)
        rt   = make_runtime()
        conv = make_conv_agent("Answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({
            "task_id":     "task-a",
            "instruction": "Question A",
        })
        ctrl.handle_task({
            "task_id":     "task-b",
            "instruction": "Question B",
        })

        entries_a = mm.get_context_window(task_id="task-a", limit=20)
        entries_b = mm.get_context_window(task_id="task-b", limit=20)

        contents_a = [e["content"] for e in entries_a]
        contents_b = [e["content"] for e in entries_b]

        assert "Question A" in contents_a
        assert "Question B" not in contents_a
        assert "Question B" in contents_b
        assert "Question A" not in contents_b


# ---------------------------------------------------------------------------
# wiki_doc wiring — RAG source frontmatter stripping
# ---------------------------------------------------------------------------

# Shape of how-localist-works.md frontmatter — confirmed from real corpus.
_RAG_WITH_FRONTMATTER = (
    "---\n"
    "title: Localist Agent Framework Manifest and Schema\n"
    "type: research-note\n"
    "query: Analyze how-localist-works.md\n"
    "created: 2026-06-18\n"
    "updated: 2026-06-18\n"
    "---\n"
    "\n"
    "## Summary\n"
    "\n"
    "Localist is a local-first AI agent framework.\n"
    "It runs entirely on-device with no cloud dependencies.\n"
)

_RAG_WITHOUT_FRONTMATTER = (
    "## Overview\n"
    "\n"
    "This document has no frontmatter block at all.\n"
    "All content is body text.\n"
)


def _make_rag_ctrl(docs: list) -> tuple:
    """Return (ctrl, conv_agent) with query_corpus() returning *docs*."""
    mm = MagicMock()
    mm.query_corpus.return_value = docs
    conv = make_conv_agent()
    ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)
    return ctrl, conv


class TestRagSourceFrontmatterStripping:

    def test_frontmatter_stripped_from_rag_source(self):
        """Frontmatter lines must not appear in the RagSource content passed to PromptBuilder."""
        doc = _mock_doc("/wiki/how-localist-works.md", _RAG_WITH_FRONTMATTER)
        doc.relevance_score = 0.9  # above threshold

        ctrl, conv = _make_rag_ctrl([doc])
        ctrl.handle_task({"instruction": "check the wiki for Localist framework"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "title: Localist Agent Framework" not in prompt
        assert "type: research-note"              not in prompt
        assert "query: Analyze how-localist-works" not in prompt
        assert "created: 2026-06-18"              not in prompt
        assert "---"                              not in prompt
        # Body text must still be present
        assert "Localist is a local-first AI agent framework" in prompt

    def test_no_frontmatter_rag_source_unchanged(self):
        """Zero-behavior-change: content without frontmatter is passed through unmodified."""
        doc = _mock_doc("/wiki/plain-doc.md", _RAG_WITHOUT_FRONTMATTER)
        doc.relevance_score = 0.9

        ctrl, conv = _make_rag_ctrl([doc])
        ctrl.handle_task({"instruction": "check the wiki for plain doc overview"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        # The full body text must appear in the prompt, character-for-character
        assert "This document has no frontmatter block at all." in prompt
        assert "All content is body text." in prompt

    def test_rag_filter_and_exclusion_unaffected(self):
        """Score filter and persona.md exclusion must be unchanged after the fix."""
        low_score_doc   = _mock_doc("/wiki/low-score.md", "low relevance content")
        low_score_doc.relevance_score = 0.3   # below 0.55 threshold — must be excluded

        persona_doc     = _mock_doc("/wiki/persona.md", "persona content")
        persona_doc.relevance_score = 0.9     # above threshold but excluded by path rule

        good_doc        = _mock_doc("/wiki/good.md", "relevant body text about Localist")
        good_doc.relevance_score = 0.9

        ctrl, conv = _make_rag_ctrl([low_score_doc, persona_doc, good_doc])
        ctrl.handle_task({"instruction": "check the wiki for Localist"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        # Only the good_doc should appear
        assert "relevant body text about Localist" in prompt
        assert "low relevance content"             not in prompt
        assert "persona content"                   not in prompt

    def test_bm25_scored_doc_bypasses_0_55_floor(self):
        """
        A low BM25 score (scored_by_embedding=False) must still be included —
        raw BM25 is unbounded, so 0.55 was never a meaningful floor for it;
        controller_agent.py trusts query_corpus()'s ranking instead for
        these. A cosine-scored doc (scored_by_embedding=True, the default)
        at the same low score must still be excluded — the floor stays in
        force there.
        """
        bm25_low_doc = _mock_doc("/wiki/bm25-low.md", "bm25 scored content")
        bm25_low_doc.relevance_score     = 0.1
        bm25_low_doc.scored_by_embedding = False

        cosine_low_doc = _mock_doc("/wiki/cosine-low.md", "cosine scored content")
        cosine_low_doc.relevance_score     = 0.1
        cosine_low_doc.scored_by_embedding = True

        ctrl, conv = _make_rag_ctrl([bm25_low_doc, cosine_low_doc])
        ctrl.handle_task({"instruction": "check the wiki for Localist"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "bm25 scored content"    in prompt
        assert "cosine scored content" not in prompt


# ---------------------------------------------------------------------------
# Graph query fetch — Step 5c wiring
# ---------------------------------------------------------------------------

class TestGraphQueryFetch:
    """
    Tests for _execute_plan() Step 5c: graph query fetch and Slot 5b rendering.

    Tests 1-5 inject a specific RoutingPlan directly (bypassing real Planner
    routing) via patch.object so the graph_query field can be set precisely.
    Test 6 routes through the real Planner with a real MemoryManager to verify
    the "pure/minimal" guarantee end-to-end.
    """

    def _make_graph_plan(self, direction: str, node_id: int, stem: str) -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            compound       = False,
            priority       = 3,
            graph_query    = (direction, node_id, stem),
        )

    # 1. Incoming, 2 edges — [GRAPH RESULT] appears with both source page stems
    def test_incoming_populated(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []
        mm_mock.get_backlinks.return_value = [
            GraphEdgeResult(
                link_text       = "[[lora-persona]]",
                target_path     = "lora-persona",
                target_resolved = True,
                node_title      = "How Localist Works",
                node_doc_path   = "/wiki/how-localist-works.md",
            ),
            GraphEdgeResult(
                link_text       = "[[lora-persona]]",
                target_path     = "lora-persona",
                target_resolved = True,
                node_title      = "Localist Build Order",
                node_doc_path   = "/wiki/localist-build-order.md",
            ),
        ]

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Incoming answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = self._make_graph_plan("incoming", 7, "lora-persona")

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "what links to lora-persona"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" in prompt
        assert "Pages linking to lora-persona:" in prompt
        assert "how-localist-works" in prompt
        assert "localist-build-order" in prompt

    # 2. Outgoing, mixed resolved+unresolved — confirms link_text used for
    #    unresolved display (not target_path), catching the link_text-vs-
    #    target_path bug described in the prompt spec.
    def test_outgoing_mixed_link_text_vs_target_path(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []
        mm_mock.get_outgoing_links.return_value = [
            GraphEdgeResult(
                link_text       = "localist-master-project-outline",
                target_path     = "localist-master-project-outline",
                target_resolved = True,
                node_title      = "Localist Master Project Outline",
                node_doc_path   = "/wiki/localist-master-project-outline.md",
            ),
            GraphEdgeResult(
                link_text       = "Localist Software Stack Overview",  # original casing
                target_path     = "localist-software-stack-overview",  # normalized — must NOT appear
                target_resolved = False,
                node_title      = None,
                node_doc_path   = None,
            ),
        ]

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Outgoing answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = self._make_graph_plan("outgoing", 3, "localist-build-order")

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "what does localist-build-order link to"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" in prompt
        assert "localist-master-project-outline" in prompt
        # Unresolved entry must show original link_text, not the normalized target_path
        assert '"Localist Software Stack Overview"' in prompt
        assert "localist-software-stack-overview" not in prompt

    # 3. Zero edges — [GRAPH RESULT] still present (clean-omission exception)
    def test_zero_edges_slot_still_emitted(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []
        mm_mock.get_backlinks.return_value = []

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("No backlinks answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = self._make_graph_plan("incoming", 5, "lora-persona")

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "what links to lora-persona"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" in prompt
        assert "No pages link to lora-persona." in prompt

    # 4. Fetch failure — _execute_plan does not raise; [GRAPH RESULT] absent
    def test_fetch_failure_degrades_gracefully(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []
        mm_mock.get_backlinks.side_effect = RuntimeError("SQLite locked")

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Degraded answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = self._make_graph_plan("incoming", 5, "lora-persona")

        with patch.object(ctrl._planner, "route", return_value=plan):
            # Must not raise
            result = ctrl.handle_task({"instruction": "what links to lora-persona"})

        assert result["status"] == "complete"
        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" not in prompt

    # 5. Non-graph-query plan — get_backlinks/get_outgoing_links never called
    def test_no_graph_query_no_edge_fetch(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Direct answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)

        # "What is 2+2?" triggers no graph pattern → plan.graph_query is None
        ctrl.handle_task({"instruction": "What is 2+2?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" not in prompt
        mm_mock.get_backlinks.assert_not_called()
        mm_mock.get_outgoing_links.assert_not_called()

    # 6. Purity end-to-end: real Planner + real MemoryManager.
    #    Episodic records and RAG docs exist in the DB and WOULD appear if
    #    their fetch conditions fired. Verify they do not leak into the prompt.
    def test_p3c_purity_no_rag_or_episodic_slots(self, tmp_path):
        db_path = tmp_path / "purity.db"
        mm = MemoryManager(db_path=db_path)

        # Graph: "lora-persona" node + "how-localist-works" backlink source
        lp_id  = mm.upsert_graph_node(
            doc_path  = str(tmp_path / "lora-persona.md"),
            node_type = "wiki",
            title     = "LORA Persona",
        )
        src_id = mm.upsert_graph_node(
            doc_path  = str(tmp_path / "how-localist-works.md"),
            node_type = "wiki",
            title     = "How Localist Works",
        )
        mm.upsert_graph_edge(
            source_node_id  = src_id,
            source_doc_path = str(tmp_path / "how-localist-works.md"),
            target_path     = "lora-persona",
            target_node_id  = lp_id,
            target_resolved = True,
            link_text       = "lora-persona",
        )

        # Episodic record (would appear in [EPISODIC MEMORY] if fetch_episodic fired)
        writer = EpisodicMemoryWriter(db_path=db_path)
        writer.insert(
            episode_type    = "preference",
            subject         = "output format",
            content         = "PURITY_LEAK_EPISODIC: should not appear in prompt",
            source          = "explicit",
            confidence      = 1.0,
            project_context = "general",
        )

        # RAG document (would appear in [CONTEXT] if fetch_rag fired)
        mm.index_document(
            path     = tmp_path / "background-doc.md",
            doc_type = "wiki",
            content  = "PURITY_LEAK_RAG localist persona links wiki architecture",
        )

        # Real Planner routes "what links to lora-persona" → P3c
        rt   = make_runtime(infer_return="yes")  # "yes" would fire episodic if reached
        conv = make_conv_agent("Graph answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "what links to lora-persona"})

        prompt = conv._received[0].context["_prebuilt_prompt"]

        # Graph result must be present
        assert "[GRAPH RESULT]" in prompt
        assert "how-localist-works" in prompt

        # No other context slots must leak — purity guarantee
        assert "[CONTEXT]"       not in prompt
        assert "[EPISODIC MEMORY]" not in prompt
        assert "[USER PROFILE]"  not in prompt
        assert "PURITY_LEAK_EPISODIC" not in prompt
        assert "PURITY_LEAK_RAG"      not in prompt



# ---------------------------------------------------------------------------
# Step 5d — WorkingMemoryState (Slot 6A) wiring
# ---------------------------------------------------------------------------

class TestWorkingStateSlot6A:
    """
    Verifies Step 5d: WorkingMemoryState assembly and the P3c exclusivity guard.

    Tests inject RoutingPlans directly via patch.object so the graph_query
    field can be set precisely — matching the pattern used in TestGraphQueryFetch.
    """

    def _make_rag_plan(self, *, fetch_rag: bool = True, graph_query=None) -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = fetch_rag,
            priority       = 4,
            graph_query    = graph_query,
        )

    # 1. Non-P3c route with RAG sources present → working_state constructed,
    #    active_artifacts matches the RAG source paths exactly.
    def test_non_p3c_with_rag_sources_builds_working_state(self):
        doc = _mock_doc("/wiki/localist-arch.md", "Localist architecture content here.")
        doc.relevance_score = 0.9

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)
        plan = self._make_rag_plan(fetch_rag=True)

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "check the wiki for Localist architecture"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[WORKING STATE]" in prompt
        assert "active_artifacts:" in prompt
        # Path must be the exact path from the RAG source
        assert "/wiki/localist-arch.md" in prompt

    # 2. Non-P3c route with no RAG sources and no usable current_project →
    #    working_state stays None; no [WORKING STATE] block in prompt.
    def test_non_p3c_no_rag_no_project_working_state_absent(self):
        mm = MagicMock()
        mm.query_corpus.return_value = []
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)
        plan = self._make_rag_plan(fetch_rag=False)

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "What is 2+2?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[WORKING STATE]" not in prompt

    # 3. P3c exclusivity guard: graph_query is not None with fetch_rag=True and
    #    docs present in scope → working_state is NOT constructed regardless.
    #    This is the regression test for the Phase C purity guarantee.
    def test_p3c_graph_query_excludes_working_state(self):
        doc = _mock_doc("/wiki/lora-persona.md", "Some persona content.")
        doc.relevance_score = 0.9

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]
        mm.get_backlinks.return_value = []
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)

        # Plan with both graph_query set AND fetch_rag=True — hypothetical scenario
        # that exercises the guard directly, regardless of what the Planner produces.
        plan = self._make_rag_plan(
            fetch_rag    = True,
            graph_query  = ("incoming", 5, "lora-persona"),
        )

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "what links to lora-persona"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[WORKING STATE]" not in prompt, (
            "Slot 6A must not be constructed on P3c routes (graph_query is not None)"
        )

    # 4. Regression guard: existing RAG path behaviour is unchanged — answer,
    #    sources, and [CONTEXT] slot are unaffected by the Step 5d addition.
    def test_regression_rag_answer_and_sources_unchanged(self):
        doc = _mock_doc("/wiki/localist-arch.md", "check the wiki Localist architecture content.")
        doc.relevance_score = 0.9

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]
        conv = make_conv_agent("Architecture answer.")
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)
        plan = self._make_rag_plan(fetch_rag=True)

        with patch.object(ctrl._planner, "route", return_value=plan):
            result = ctrl.handle_task(
                {"instruction": "check the wiki for Localist architecture"}
            )

        # Core result unchanged
        assert result["status"] == "complete"
        assert result["answer"] == "Architecture answer."
        # [CONTEXT] slot still present — RAG wiring unaffected
        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" in prompt
        assert "Localist architecture content" in prompt


# ---------------------------------------------------------------------------
# Post-P4a removal: query_corpus always called without doc_type (Part 4B)
# ---------------------------------------------------------------------------

class TestQueryCorpusNeverReceivesDocType:
    """
    After removing force_rag, Step 4's query_corpus() call must never pass
    doc_type — the kwarg was dropped entirely (not set to None explicitly),
    so it should be absent from call_kwargs.

    Any plan with fetch_rag=True exercises this code path.
    """

    def test_step4_query_corpus_has_no_doc_type_kwarg(self):
        """Step 4 query_corpus() must not pass doc_type under any plan."""
        mm = MagicMock()
        mm.query_corpus.return_value = []
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)

        plan = RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = True,
            priority       = 4,
        )

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "check the wiki for Localist"})

        assert mm.query_corpus.call_count >= 1
        _, step4_kwargs = mm.query_corpus.call_args_list[0]
        assert "doc_type" not in step4_kwargs, (
            f"doc_type must not be passed to query_corpus after force_rag removal; "
            f"got doc_type={step4_kwargs.get('doc_type')!r}"
        )


# ---------------------------------------------------------------------------
# Post-P4a removal: relevance threshold is now unconditional (Part 4C)
# ---------------------------------------------------------------------------

class TestRelevanceThresholdUnconditional:
    """
    Under the old code, force_rag=True bypassed the 0.55 relevance_score
    threshold so documents below it were included in rag_sources.  After
    removing force_rag, the threshold is unconditional: a low-scoring document
    must always be excluded regardless of plan contents.
    """

    def test_low_score_doc_excluded_no_bypass(self):
        """A doc with relevance_score < 0.55 must not appear in the prompt."""
        low_doc = _mock_doc("/wiki/low-relevance.md", "Some low-relevance content.")
        low_doc.relevance_score = 0.40   # below 0.55 — would have been included by force_rag

        mm = MagicMock()
        mm.query_corpus.return_value = [low_doc]
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)

        plan = RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = True,
            priority       = 4,
        )

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "check the wiki for LORA"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "Some low-relevance content." not in prompt, (
            "Low-score document must be excluded: threshold is now unconditional"
        )
        assert "[CONTEXT]" not in prompt


# ---------------------------------------------------------------------------
# RoutingPlan no longer accepts force_rag keyword argument (Part 4D)
# ---------------------------------------------------------------------------

class TestRoutingPlanNoForceRagField:
    """
    Confirm force_rag was genuinely removed from the RoutingPlan dataclass,
    not merely left unused.  Passing force_rag=True must raise TypeError.
    """

    def test_force_rag_kwarg_raises_type_error(self):
        import pytest
        with pytest.raises(TypeError):
            RoutingPlan(
                agent          = "conversational_agent",
                fetch_episodic = False,
                fetch_rag      = False,
                force_rag      = True,   # no longer a field — must raise
            )
