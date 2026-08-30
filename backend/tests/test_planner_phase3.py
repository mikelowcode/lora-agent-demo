"""
Phase 3 integration tests — Localist rule-based Planner.

Covers:
  - Each priority level fires correctly (P1–P6)
  - Compound detection fires before P1
  - ControllerAgent._execute() uses RoutingPlan for agent selection
  - Fallback when requested agent is not registered
  - RoutingPlan metadata is passed into SubTask context
  - Priority 3c graph-query wiring (P3c)

All tests use mocks — no SQLite, no runtime calls for P1–P4 and P6.
P3c tests use a real MemoryManager with a temporary SQLite database.
"""

import logging
import math
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from localist import session_files
from localist.memory_manager import MemoryManager
from localist.planner import Planner, RoutingPlan, extract_graph_query, resolve_graph_target, _has_explicit_remember_signal
from localist.controller_agent import (
    ControllerAgent, Task, TaskStatus, SubTask, AgentResult
)


def make_mm_with_nodes(tmp_path: Path) -> tuple[MemoryManager, dict[str, int]]:
    """Create a MemoryManager with five known graph_nodes for P3c tests."""
    db = tmp_path / "planner_p3c_test.db"
    mm = MemoryManager(db_path=db)
    node_ids: dict[str, int] = {}
    for stem in [
        "how-localist-works",
        "localist-build-order",
        "localist-master-project-outline",
        "localist-software-stack",
        "lora-persona",
    ]:
        nid = mm.upsert_graph_node(
            doc_path  = str(tmp_path / f"{stem}.md"),
            node_type = "wiki",
            title     = stem.replace("-", " ").title(),
        )
        node_ids[stem] = nid
    return mm, node_ids


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_runtime(infer_return="no", embed_return=None):
    rt = MagicMock()
    rt.infer.return_value = infer_return
    rt.embed.return_value = embed_return or ([0.0] * 768)
    return rt


def make_agent(name, answer="Mock answer."):
    agent = MagicMock()
    agent.name = name
    agent.can_handle.return_value = True
    agent.run.return_value = AgentResult(
        subtask_id = "test-0",
        agent_name = name,
        status     = TaskStatus.COMPLETE,
        output     = {"answer": answer, "sources": [], "grounded": False},
    )
    return agent


# ---------------------------------------------------------------------------
# Planner unit tests — priority firing
# ---------------------------------------------------------------------------

class TestPlannerPriorities:

    def test_p1_raw_path_routes_to_wiki(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("do something", context={"raw_path": "/f.md"})
        assert plan.agent == "wiki_agent"
        assert plan.fetch_rag      is False
        assert plan.fetch_episodic is False

    def test_p1_keyword_routes_to_wiki(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("ingest this document", context={})
        assert plan.agent == "wiki_agent"

    def test_p1_beats_p2(self):
        """raw_path in context + memory keyword → Priority 1 wins."""
        p = Planner(runtime=make_runtime())
        plan = p.route("remember that you should ingest this",
                       context={"raw_path": "/x.md"})
        assert plan.agent         == "wiki_agent"
        assert plan.write_episode is False

    def test_p2_memory_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("remember that I prefer dark mode", context={})
        assert plan.write_episode is True
        assert plan.compound      is True
        assert plan.agent         == "conversational_agent"

    def test_p2_forget_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("forget that preference", context={})
        assert plan.write_episode is True

    def test_p2_keep_in_mind_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("keep in mind I'm allergic to peanuts", context={})
        assert plan.write_episode is True

    def test_p2_make_a_note_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("make a note that my flight is on the 10th", context={})
        assert plan.write_episode is True

    def test_p3_web_search_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("What are the latest oMLX changes?", context={})
        assert "web_search" in plan.tools_to_call
        assert plan.compound is True

    def test_p3_file_op_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("read the file notes.md", context={})
        assert "file_op" in plan.tools_to_call

    def test_p3_file_op_content_present_quoted_dispatches_immediately(self):
        """Destination phrase + quoted literal content → unchanged old
        behavior: tools_to_call=["file_op"], not deferred."""
        p = Planner(runtime=make_runtime())
        plan = p.route('save it as notes.md: "buy milk"', context={})
        assert "file_op" in plan.tools_to_call
        assert plan.file_op_deferred is False
        assert plan.compound is True

    def test_p3_file_op_content_present_fenced_dispatches_immediately(self):
        """Destination phrase + fenced literal content → unchanged old
        behavior: tools_to_call=["file_op"], not deferred."""
        p = Planner(runtime=make_runtime())
        plan = p.route("save it as notes.md: ```buy milk```", context={})
        assert "file_op" in plan.tools_to_call
        assert plan.file_op_deferred is False

    def test_p3_file_op_generation_required_is_deferred(self):
        """No literal content in the instruction — content must be composed
        by the agent first — so file_op is NOT dispatched yet."""
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "write a haiku about the sea and save it as haiku.md", context={}
        )
        assert "file_op" not in plan.tools_to_call
        assert plan.file_op_deferred is True
        assert plan.file_op_path   == "haiku.md"
        assert plan.file_op_action == "write"
        assert plan.compound is True

    def test_p4_explicit_wiki_keyword_fires(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("check the wiki for LORA memory system", context={})
        assert plan.fetch_rag      is True
        assert plan.fetch_episodic is True

    def test_p4_no_wiki_keyword_falls_through(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("Tell me about the LORA memory system", context={})
        assert plan.fetch_rag is False

    def test_p5_yes_returns_episodic(self):
        p = Planner(runtime=make_runtime(infer_return="yes"))
        plan = p.route("What are my formatting preferences?", context={})
        assert plan.fetch_episodic is True

    def test_p5_no_falls_to_p6(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("What is 2+2?", context={})
        assert plan.fetch_episodic is False
        assert plan.fetch_rag      is False
        assert plan.agent          == "conversational_agent"

    def test_p6_direct_all_false(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("What is 2+2?", context={})
        assert plan.tools_to_call == []
        assert plan.write_episode  is False
        assert plan.compound       is False


# ---------------------------------------------------------------------------
# Priority 2 — bare-"remember" write-command pattern (2026-07-23)
# ---------------------------------------------------------------------------
# Live bug: "I want you to remember I'm participating in a Claude Impact Lab
# on August 6th." matched no _MEMORY_KEYWORDS entry (only literal "remember
# that" was covered). Semantic gating was tried first (mirroring the same-day
# Priority 5 fix) and rejected on measured evidence — two rounds of template
# tuning against the real embedding model both found negative-max > positive-
# min (recall questions like "Do you remember my name?" scored higher than
# genuine write commands). See diagnostics/reports/
# explicit_memory_write_gate_2026-07-23.md for both rounds' data and the
# adopted deterministic rule tested here.

class TestExplicitRememberPattern:
    """Unit tests for Planner._has_explicit_remember_signal() directly, plus
    its wiring into _priority2_memory()/route()."""

    def test_live_bug_repro_fires(self):
        assert _has_explicit_remember_signal(
            "i want you to remember i'm participating in a claude impact "
            "lab on august 6th."
        ) is True

    def test_bare_remember_without_that_fires(self):
        assert _has_explicit_remember_signal("please remember i prefer dark mode.") is True

    def test_do_you_remember_question_does_not_fire(self):
        assert _has_explicit_remember_signal(
            "do you remember what i told you about the migration?"
        ) is False

    def test_what_do_you_remember_question_does_not_fire(self):
        assert _has_explicit_remember_signal(
            "what do you remember about my project?"
        ) is False

    def test_will_you_remember_question_does_not_fire(self):
        assert _has_explicit_remember_signal("will you remember to feed the cat?") is False

    def test_i_remember_reminiscing_does_not_fire(self):
        assert _has_explicit_remember_signal(
            "i remember when we talked about this before."
        ) is False

    def test_trailing_question_mark_does_not_fire(self):
        assert _has_explicit_remember_signal(
            "should i remember to bring my laptop tomorrow?"
        ) is False

    def test_no_remember_word_does_not_fire(self):
        assert _has_explicit_remember_signal("what is the capital of france?") is False

    def test_route_writes_episode_for_live_bug_instruction(self):
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "I want you to remember I'm participating in a Claude Impact "
            "Lab on August 6th.",
            context={},
        )
        assert plan.write_episode is True
        assert plan.priority == 2

    def test_route_does_not_write_episode_for_recall_question(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("Do you remember what I told you about the migration?", context={})
        assert plan.write_episode is False


class TestNegatedForgetPattern:
    """
    Closed 2026-07-23, same day as TestExplicitRememberPattern, follow-up
    request: "don't forget that X" collided with _RETRACTION_SIGNALS'
    "forget that" substring in episodic_extractor.detect_explicit_signal()
    (incorrectly triggering delete instead of remember). This class covers
    _has_explicit_remember_signal()'s negated-forget branch and
    _priority2_memory()'s routing; the retraction-vs-insert fix itself lives
    in episodic_extractor.py (see test_episodic_phase5.py).
    """

    def test_dont_forget_that_fires(self):
        assert _has_explicit_remember_signal(
            "don't forget that i have a dentist appointment."
        ) is True

    def test_dont_forget_without_that_fires(self):
        assert _has_explicit_remember_signal(
            "don't forget i have a dentist appointment next tuesday."
        ) is True

    def test_do_not_forget_variant_fires(self):
        assert _has_explicit_remember_signal("do not forget i have a meeting at 3pm.") is True

    def test_never_forget_variant_fires(self):
        assert _has_explicit_remember_signal("never forget that i prefer dark mode.") is True

    def test_bare_forget_without_negation_does_not_fire(self):
        # "forget that" alone (no "don't"/"do not"/"never") is NOT this
        # pattern's concern — it's the existing, correct retraction signal,
        # untouched by this fix.
        assert _has_explicit_remember_signal("forget that preference about formatting.") is False

    def test_route_writes_episode_for_dont_forget(self):
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "Don't forget I have a dentist appointment next Tuesday.", context={},
        )
        assert plan.write_episode is True


# ---------------------------------------------------------------------------
# Post-P4a-removal: former identity phrasings now route to P6
# ---------------------------------------------------------------------------
#
# Discovery run (2026-06-26): all 13 former _IDENTITY_KEYWORDS phrasings
# resolve to priority=6 (P6 direct-answer fallback) with no MemoryManager.
# P4 Path B is skipped because there is no corpus to score against; if a
# corpus were present with a sufficiently high-scoring document, some of
# these might reach P4. The assertions below lock in the P6 outcome for
# the no-corpus case, which is the clean-room unit-test baseline.

class TestFormerP4aIdentityPhrasingsRouteToPSix:

    def _check(self, phrase: str) -> None:
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route(phrase, context={})
        assert plan.priority == 6, (
            f"Expected priority=6 for {phrase!r}; got priority={plan.priority}"
        )
        assert plan.fetch_rag is False, (
            f"Expected fetch_rag=False for {phrase!r}; got {plan.fetch_rag}"
        )
        assert plan.fetch_episodic is False, (
            f"Expected fetch_episodic=False for {phrase!r}; got {plan.fetch_episodic}"
        )
        assert plan.agent == "conversational_agent"

    def test_who_are_you(self):
        self._check("who are you")

    def test_what_are_you(self):
        self._check("what are you")

    def test_tell_me_about_yourself(self):
        self._check("tell me about yourself")

    def test_what_can_you_do(self):
        self._check("what can you do")

    def test_are_you_an_ai(self):
        self._check("are you an ai")

    def test_are_you_a_bot(self):
        self._check("are you a bot")

    def test_what_is_lora(self):
        self._check("what is lora")

    def test_who_is_lora(self):
        self._check("who is lora")

    def test_what_is_localist(self):
        self._check("what is localist")

    def test_are_you_made_by_google(self):
        self._check("are you made by google")

    def test_are_you_chatgpt(self):
        self._check("are you chatgpt")

    def test_are_you_gemma(self):
        self._check("are you gemma")

    def test_introduce_yourself(self):
        self._check("introduce yourself")


# ---------------------------------------------------------------------------
# Compound detection
# ---------------------------------------------------------------------------

class TestCompoundDetection:

    def test_tool_ingest_compound(self):
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "Search for the latest oMLX release notes and add to wiki",
            context={},
        )
        assert plan.agent         == "wiki_agent"
        assert "web_search"       in plan.tools_to_call
        assert plan.compound      is True

    def test_compound_fires_before_p1(self):
        """Without compound detection, P1 would win and drop the tool signal."""
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "Get the latest release notes and ingest them",
            context={},
        )
        assert "web_search" in plan.tools_to_call

    def test_ingest_only_not_compound(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("ingest this document", context={})
        assert plan.tools_to_call == []
        assert plan.compound      is False

    def test_tool_only_not_wiki_compound(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("What are the latest changes?", context={})
        assert plan.agent    == "conversational_agent"
        assert plan.compound is True   # P3 sets compound=True for tool+response


# ---------------------------------------------------------------------------
# Priority 3 — chart generation trigger keywords
# ---------------------------------------------------------------------------

class TestPlannerP3Chart:

    @pytest.mark.parametrize("phrase", sorted(Planner._CHART_KEYWORDS))
    def test_chart_keyword_routes_to_chart(self, phrase):
        p = Planner(runtime=make_runtime())
        plan = p.route(f"{phrase}: apples 5, oranges 3, bananas 7", context={})
        assert plan.tools_to_call == ["chart"]
        assert plan.compound is True

    def test_explain_bar_chart_does_not_route_to_chart(self):
        """Negative control from the diagnostic corpus (negative_control
        category, diagnostics/diag_shadow_chart_toolcall.py) — no imperative
        chart trigger present, must stay negative."""
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("explain what a bar chart is", context={})
        assert "chart" not in plan.tools_to_call

    def test_chart_compounds_with_web_search(self):
        """Same combination-ordering behavior _WEB_SEARCH_KEYWORDS /
        _FILE_OP_KEYWORDS already compound with (TestCompoundDetection
        above) — a message matching both a chart keyword and a web_search
        keyword gets both tools, in the order _priority3_tool() checks them
        (web_search before chart)."""
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "make a bar chart of the latest oMLX release notes", context={}
        )
        assert plan.tools_to_call == ["web_search", "chart"]
        assert plan.compound is True

    def test_diagnostic_corpus_turn_into_pie_chart_routes_to_chart(self):
        """Regression test for the exact chart_keyword_clear corpus
        instruction (diagnostics/diag_shadow_chart_toolcall.py,
        expects_tool=True) that fell through to P6 (tools=[]) before
        "turn this into a {chart type}" was added to _CHART_KEYWORDS —
        caught live 2026-07-20."""
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "Turn this into a pie chart: Chrome 65, Safari 19, Firefox 8, "
            "Edge 6, Other 2",
            context={},
        )
        assert "chart" in plan.tools_to_call

    def test_diagnostic_corpus_turn_into_something_visual_routes_to_chart(self):
        """Same gap, the chart_semantic_implicit corpus instruction
        (expects_tool=True) — "turn this into something visual" phrasing."""
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "Turn this into something visual: Q1 100, Q2 150, Q3 130, Q4 180",
            context={},
        )
        assert "chart" in plan.tools_to_call


# ---------------------------------------------------------------------------
# Priority 0 — explicit slash-command tool bypass ("/chart", "/research")
# ---------------------------------------------------------------------------

class TestPlannerP0SlashCommand:

    def test_slash_chart_short_circuits_before_p3_evaluates(self, monkeypatch):
        """Not just "both paths happen to agree" — spy on _priority3_tool to
        confirm P0 returns before P3 is ever reached, even though "chart
        this" alone would also have matched _CHART_KEYWORDS."""
        p = Planner(runtime=make_runtime())
        with patch.object(
            Planner, "_priority3_tool", wraps=p._priority3_tool
        ) as spy:
            plan = p.route(
                "/chart chart this: apples 5, oranges 3, bananas 7", context={}
            )
        spy.assert_not_called()
        assert plan.tools_to_call == ["chart"]
        assert plan.priority == 0
        assert plan.tool_signal_source == "slash_command"
        assert plan.compound is True

    def test_slash_research_bypasses_the_research_loop_flag(self, monkeypatch):
        """The whole point of the bypass: "research" must enter
        tools_to_call even though LOCALIST_RESEARCH_LOOP_ENABLED — which
        normally gates the web_search -> research upgrade entirely (§18.2)
        — is off/unset."""
        monkeypatch.delenv("LOCALIST_RESEARCH_LOOP_ENABLED", raising=False)
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "/research what does the enterprise plan cost", context={}
        )
        assert plan.tools_to_call == ["research"]
        assert plan.priority == 0
        assert plan.tool_signal_source == "slash_command"

    def test_bare_slash_chart_reaches_the_tool_without_crashing(self):
        """No data/topic following the command — still reaches the tool;
        empty-content chart requests degrading gracefully is the
        dispatcher's problem, not Planner's."""
        p = Planner(runtime=make_runtime())
        plan = p.route("/chart", context={})
        assert plan.tools_to_call == ["chart"]
        assert plan.priority == 0

    @pytest.mark.parametrize("phrase", ["/Chart", "/CHART this data: a 1, b 2"])
    def test_slash_chart_is_case_insensitive(self, phrase):
        p = Planner(runtime=make_runtime())
        plan = p.route(phrase, context={})
        assert plan.tools_to_call == ["chart"]
        assert plan.priority == 0

    def test_mid_sentence_slash_chart_does_not_match(self):
        """Only a LEADING token counts — a mid-sentence occurrence must not
        trigger the P0 bypass."""
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route(
            "what's a good site to /chart my finances", context={}
        )
        assert plan.priority != 0
        assert plan.tool_signal_source != "slash_command"

    def test_existing_chart_keyword_routing_unaffected(self):
        """Additive only — an instruction with no leading slash command
        still routes via P3's keyword match unchanged."""
        p = Planner(runtime=make_runtime())
        plan = p.route("chart this: apples 5, oranges 3, bananas 7", context={})
        assert plan.tools_to_call == ["chart"]
        assert plan.priority == 3
        assert plan.tool_signal_source == "keyword"


# ---------------------------------------------------------------------------
# ControllerAgent integration
# ---------------------------------------------------------------------------

class TestControllerAgentRouting:

    def test_conversational_query_short_circuits_synthesizer(self):
        """ConversationalAgent result bypasses Synthesizer."""
        rt = make_runtime(infer_return="no")
        conv = make_agent("conversational_agent", answer="42 is the answer.")
        wiki = make_agent("wiki_agent")

        ctrl = ControllerAgent(runtime=rt, agents=[conv, wiki])
        result = ctrl.handle_task({"instruction": "What is 42?"})

        assert result["status"]  == "complete"
        assert result["answer"]  == "42 is the answer."
        assert conv.run.called
        assert wiki.run.called   is False

    def test_ingest_routes_to_wiki_agent(self):
        """raw_path in context → wiki_agent is dispatched."""
        rt = make_runtime()
        conv = make_agent("conversational_agent")
        wiki = make_agent("wiki_agent")
        # WikiAgent returns ingest-style output (no "answer" key → Synthesizer)
        wiki.run.return_value = AgentResult(
            subtask_id = "test-0",
            agent_name = "wiki_agent",
            status     = TaskStatus.COMPLETE,
            output     = {"new_pages": [], "applied": False},
        )

        ctrl = ControllerAgent(runtime=rt, agents=[conv, wiki])
        result = ctrl.handle_task({
            "instruction": "ingest this file",
            "context":     {"raw_path": "/data/notes.md"},
        })

        assert wiki.run.called
        assert conv.run.called is False

    def test_routing_metadata_in_subtask_context(self):
        """RoutingPlan fields are forwarded into SubTask.context._routing."""
        rt = make_runtime(infer_return="no")
        captured_subtasks = []

        conv = MagicMock()
        conv.name = "conversational_agent"
        conv.can_handle.return_value = True

        def capture_run(subtask):
            captured_subtasks.append(subtask)
            return AgentResult(
                subtask_id = subtask.subtask_id,
                agent_name = "conversational_agent",
                status     = TaskStatus.COMPLETE,
                output     = {"answer": "ok", "sources": [], "grounded": False},
            )
        conv.run.side_effect = capture_run

        ctrl = ControllerAgent(runtime=rt, agents=[conv])
        ctrl.handle_task({"instruction": "What is LORA?"})

        assert len(captured_subtasks) == 1
        routing = captured_subtasks[0].context.get("_routing")
        assert routing is not None, "_routing key must be present in SubTask.context"
        assert "fetch_rag"      in routing
        assert "fetch_episodic" in routing
        assert "tools_to_call"  in routing
        assert "write_episode"  in routing

    def test_fallback_when_requested_agent_not_registered(self):
        """If wiki_agent not registered but instruction triggers P1,
        fall back to conversational_agent."""
        rt = make_runtime()
        conv = make_agent("conversational_agent")

        ctrl = ControllerAgent(runtime=rt, agents=[conv])
        # P1 will fire (raw_path present) → requests wiki_agent → not found
        # → fallback to conversational_agent
        result = ctrl.handle_task({
            "instruction": "process this",
            "context":     {"raw_path": "/x.md"},
        })

        assert result["status"] == "complete"
        assert conv.run.called

    def test_no_agents_returns_failed(self):
        """No agents registered at all → failed result."""
        rt = make_runtime()
        ctrl = ControllerAgent(runtime=rt, agents=[])
        result = ctrl.handle_task({"instruction": "What is LORA?"})
        assert result["status"] == "failed"


# ---------------------------------------------------------------------------
# Graph-query extraction and name resolution
# ---------------------------------------------------------------------------

# 5-stem test set per LOCALIST-Architecture.md §8.7.
# Note: the real wiki/ directory also contains a "michael" page not in this
# set. Tests use this fixed list so they are isolated from filesystem state.
_TEST_STEMS = [
    "how-localist-works",
    "localist-build-order",
    "localist-master-project-outline",
    "localist-software-stack",
    "lora-persona",
]


class TestGraphQueryExtraction:

    # 1. Pattern A matches (outgoing, anchored regex)
    def test_pattern_a_outgoing(self):
        result = extract_graph_query("What does localist-build-order link to?")
        assert result == ("outgoing", "localist-build-order")

    # 2. Pattern A does NOT match trailing content after "link to"
    def test_pattern_a_no_trailing_content(self):
        result = extract_graph_query("what does X link to and why")
        assert result is None

    # 3. Pattern B longest-phrase-first: "show me backlinks for" wins.
    #    If phrases were checked shortest-first, a hypothetical shorter prefix
    #    could steal the match; this test confirms the longest phrase fires
    #    and the remainder is correctly extracted.
    def test_pattern_b_longest_phrase_first(self):
        result = extract_graph_query("show me backlinks for lora-persona")
        assert result == ("incoming", "lora-persona")

    # 4. Pattern B basic case (case + trailing ?)
    def test_pattern_b_what_links_to(self):
        result = extract_graph_query("What links to lora-persona?")
        assert result == ("incoming", "lora-persona")

    # 5. Pattern C matches (outgoing lead-phrase)
    def test_pattern_c_outgoing(self):
        result = extract_graph_query("links from localist-build-order")
        assert result == ("outgoing", "localist-build-order")

    # 6. Degenerate empty remainder — extraction still returns a match;
    #    resolution handles the failure, not extraction.
    def test_pattern_b_empty_remainder(self):
        result = extract_graph_query("what links to")
        assert result == ("incoming", "")

    # 7. No pattern matches at all
    def test_no_match(self):
        result = extract_graph_query("what is the weather today")
        assert result is None


class TestGraphNameResolution:

    # 8. Tier 1: remainder is a substring of exactly one stem
    def test_tier1_remainder_in_stem(self):
        result = resolve_graph_target("software stack", _TEST_STEMS)
        assert result == "localist-software-stack"

    # 9. Tier 1: stem is a substring of remainder (other direction)
    def test_tier1_stem_in_remainder(self):
        # "lora-persona" is a substring of the normalized remainder
        result = resolve_graph_target("info about lora-persona page", _TEST_STEMS)
        assert result == "lora-persona"

    # 10. Tier 1: ambiguous — "localist" substring-matches multiple stems
    def test_tier1_ambiguous(self):
        # "localist" appears in 4 of the 5 test stems → multi-match → None
        result = resolve_graph_target("localist", _TEST_STEMS)
        assert result is None

    # 11. Tier 2 fallback: Tier 1 finds zero, Tier 2 succeeds with ratio ≥ 0.5
    def test_tier2_fallback_single_match(self):
        # Normalized: "build-order-for-localist"
        # Tier 1: no stem is a substring of this and it is not in any stem.
        # Tier 2: query_tokens={"build","order","localist"} (3, after removing "for")
        #   localist-build-order: intersection={"build","order","localist"} → ratio 3/3 = 1.0 ✓
        #   all others: only "localist" overlaps → ratio 1/3 < 0.5 ✗
        result = resolve_graph_target("build order for localist", _TEST_STEMS)
        assert result == "localist-build-order"

    # 12. Tier 2 skipped: <2 meaningful tokens after stopword removal.
    #     "the overview" → normalized "the-overview"; Tier 1 finds no matches;
    #     Tier 2 tokens = {"the","overview"} → after stopwords: {"overview"} → 1 < 2.
    #     This is the "skipped, not scored" path (not a ratio failure).
    def test_tier2_skipped_too_few_tokens(self):
        result = resolve_graph_target("the overview", _TEST_STEMS)
        # Confirm the token count before the skip:
        from localist.planner import _normalize_graph_text, _GRAPH_STOPWORDS
        normalized = _normalize_graph_text("the overview")
        meaningful = set(normalized.split("-")) - _GRAPH_STOPWORDS
        assert len(meaningful) < 2, (
            f"Expected <2 meaningful tokens, got {meaningful} — "
            "this would exercise Tier 2 scoring, not the skip path"
        )
        assert result is None

    # 13. Tier 2 ambiguous: two stems both clear the 0.5 ratio threshold
    def test_tier2_ambiguous(self):
        # Normalized: "localist-project-build"
        # Tier 1: no substring match either direction.
        # Tier 2: query_tokens={"localist","project","build"} (3 tokens)
        #   localist-build-order: intersection={"localist","build"} → 2/3 ≈ 0.67 ✓
        #   localist-master-project-outline: intersection={"localist","project"} → 2/3 ≈ 0.67 ✓
        #   two matches → ambiguous → None
        result = resolve_graph_target("localist project build", _TEST_STEMS)
        assert result is None

    # 14. Empty remainder: Tier 1 treats "" as substring of every stem
    #     (all 5 match) → ambiguous → None. No Tier 2 is reached.
    def test_empty_remainder(self):
        result = resolve_graph_target("", _TEST_STEMS)
        assert result is None

    # 15. Completely unrelated remainder: no substring matches, no token overlap
    def test_unrelated_remainder(self):
        result = resolve_graph_target("the weather forecast", _TEST_STEMS)
        assert result is None


# ---------------------------------------------------------------------------
# Priority 3c — graph-query wiring (uses real SQLite via make_mm_with_nodes)
# ---------------------------------------------------------------------------

class TestPlannerP3c:

    # 1. Basic incoming graph-query matches and returns graph_query tuple
    def test_incoming_graph_query_resolves(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "incoming"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]
        assert plan.tools_to_call == []
        assert plan.fetch_rag      is False
        assert plan.fetch_episodic is False
        assert plan.compound       is False

    # 2. Outgoing graph-query ("what does X link to") resolves correctly
    def test_outgoing_graph_query_resolves(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("What does lora-persona link to?", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "outgoing"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]

    # 3. file_op guard: "create a file" triggers inline guard → P3c returns None,
    #    P3 fires on "create a file" and returns file_op plan
    def test_file_op_guard_defers_to_p3(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona, create a file with the results", context={})
        # P3c defers; P3 fires. "the results" isn't literal content present in
        # the instruction (it's the not-yet-computed graph-query output), so
        # this is a generation-required file_op — deferred, not dispatched.
        assert plan.priority == 3
        assert plan.file_op_deferred is True
        assert "file_op" not in plan.tools_to_call
        assert plan.graph_query is None

    # 4. Ordering regression: graph-query wins over a web_search-only P3 match.
    #    "recent" is a web_search keyword; under the old (wrong) ordering where
    #    P3c ran AFTER P3, P3 would win. P3c's inline guard does NOT block
    #    web_search, so with correct ordering P3c resolves the graph query first.
    def test_p3c_beats_web_search_p3(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona, recent activity", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "incoming"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]
        # Must NOT have triggered a web_search
        assert "web_search" not in plan.tools_to_call

    # 5. Unresolvable name → P3c returns None and falls through to P6
    def test_unresolvable_name_falls_through(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(infer_return="no"), memory_manager=mm)
        plan = p.route("what links to the localist pages", context={})
        # Name resolution fails for ambiguous "localist" → P3c returns None.
        # No other priority fires → P6 direct answer.
        assert plan.graph_query is None
        assert plan.agent == "conversational_agent"
        assert plan.tools_to_call == []

    # 6. P1 beats P3c: ingest keyword ensures wiki_agent routing before P3c runs
    def test_p1_beats_p3c(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("ingest this file, what links to lora-persona", context={})
        assert plan.agent == "wiki_agent"
        assert plan.graph_query is None

    # 7. No MemoryManager: P3c skips and route() falls through to P6
    def test_no_memory_manager_skips_p3c(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("what links to lora-persona", context={})
        assert plan.graph_query is None
        assert plan.agent == "conversational_agent"
        assert plan.tools_to_call == []


# ---------------------------------------------------------------------------
# Priority 3-news — news_search tool routing (news-query-routing plan)
# ---------------------------------------------------------------------------

class TestPlannerP3News:

    # 1. Basic news keyword match routes to news_search only.
    def test_news_keyword_routes_to_news_search(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("what's the latest news on APC?", context={})
        assert plan.tools_to_call == ["news_search"]
        assert plan.compound is True
        assert plan.priority == 3
        assert plan.tool_signal_source == "keyword"
        assert plan.fetch_rag is False
        assert plan.fetch_episodic is False

    # 2. "headlines"/"breaking"/"top stories" all match _NEWS_KEYWORDS too.
    @pytest.mark.parametrize("instruction", [
        "any headlines about APC?",
        "is there breaking news about APC?",
        "what are the top stories today?",
    ])
    def test_other_news_keywords_route_to_news_search(self, instruction):
        p = Planner(runtime=make_runtime())
        plan = p.route(instruction, context={})
        assert plan.tools_to_call == ["news_search"]

    # 3. news_search wins over a web_search-only P3 match on the same turn —
    #    "recent" is a _WEB_SEARCH_KEYWORDS entry; P3-news runs first in
    #    route() so it must win, not merge with web_search.
    def test_news_beats_web_search_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("any recent news on APC?", context={})
        assert plan.tools_to_call == ["news_search"]
        assert "web_search" not in plan.tools_to_call

    # 4. "news" no longer fires plain web_search on its own (moved out of
    #    _WEB_SEARCH_KEYWORDS into _NEWS_KEYWORDS — see planner.py's
    #    2026-07-22 comment above _WEB_SEARCH_KEYWORDS).
    def test_news_word_not_in_web_search_keywords(self):
        from localist.planner import _WEB_SEARCH_KEYWORDS, _NEWS_KEYWORDS
        assert "news" not in _WEB_SEARCH_KEYWORDS
        assert "news" in _NEWS_KEYWORDS

    # 5. file_op guard: a literal _FILE_OP_KEYWORDS hit ("read the file")
    #    alongside "news" defers P3-news to P3, which resolves it as file_op.
    def test_file_op_guard_defers_to_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("read the file for news updates", context={})
        assert plan.tools_to_call == ["file_op"]
        assert "news_search" not in plan.tools_to_call

    # 6. url_fetch guard: a raw URL alongside "news" defers P3-news to P3.
    def test_url_guard_defers_to_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("summarize the news at https://example.com/article", context={})
        assert "url_fetch" in plan.tools_to_call
        assert "news_search" not in plan.tools_to_call

    # 7. Ordering regression: P3c (graph-query) beats P3-news, same as it
    #    already beats plain P3 web_search — P3-news must not run before P3c.
    def test_p3c_beats_news_search(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona, any recent news?", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "incoming"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]
        assert "news_search" not in plan.tools_to_call

    # 8. P1 beats P3-news: ingest keyword routes to wiki_agent regardless of
    #    a news keyword also being present.
    def test_p1_beats_p3_news(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("ingest this file, it's about recent news", context={})
        assert plan.agent == "wiki_agent"
        assert "news_search" not in plan.tools_to_call

    # 9. No news keyword at all → P3-news defers, falls through normally
    #    (P6 direct answer, since nothing else in this instruction matches).
    def test_no_news_keyword_falls_through(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("how do I bake bread?", context={})
        assert plan.tools_to_call == []
        assert plan.agent == "conversational_agent"


# ---------------------------------------------------------------------------
# Priority 3-github — github_search tool routing
# ---------------------------------------------------------------------------

class TestPlannerP3Github:

    # 1. Basic github keyword match routes to github_search only.
    def test_github_keyword_routes_to_github_search(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("what's in the anthropics github repo?", context={})
        assert plan.tools_to_call == ["github_search"]
        assert plan.compound is True
        assert plan.priority == 3
        assert plan.tool_signal_source == "keyword"
        assert plan.fetch_rag is False
        assert plan.fetch_episodic is False

    # 2. "readme"/"pull request"/"repository" all match _GITHUB_KEYWORDS too.
    @pytest.mark.parametrize("instruction", [
        "can you read the readme for that project?",
        "is there an open pull request for this?",
        "find a python repository for parsing PDFs",
    ])
    def test_other_github_keywords_route_to_github_search(self, instruction):
        p = Planner(runtime=make_runtime())
        plan = p.route(instruction, context={})
        assert plan.tools_to_call == ["github_search"]

    # 3. github_search wins over a web_search-only P3 match on the same turn —
    #    "recent" is a _WEB_SEARCH_KEYWORDS entry; P3-github runs before P3.
    def test_github_beats_web_search_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("any recent activity on that github repo?", context={})
        assert plan.tools_to_call == ["github_search"]
        assert "web_search" not in plan.tools_to_call

    # 4. file_op guard: a literal _FILE_OP_KEYWORDS hit ("read the file")
    #    alongside "github" defers P3-github to P3, which resolves it as file_op.
    def test_file_op_guard_defers_to_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("read the file about that github project", context={})
        assert plan.tools_to_call == ["file_op"]
        assert "github_search" not in plan.tools_to_call

    # 5. url_fetch guard: a raw URL alongside "github" defers P3-github to P3.
    def test_url_guard_defers_to_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("summarize this github page: https://example.com/article", context={})
        assert "url_fetch" in plan.tools_to_call
        assert "github_search" not in plan.tools_to_call

    # 6. Ordering regression: P3c (graph-query) beats P3-github, same as it
    #    already beats plain P3 web_search and P3-news.
    def test_p3c_beats_github_search(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona, any related github repo?", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "incoming"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]
        assert "github_search" not in plan.tools_to_call

    # 7. P1 beats P3-github: ingest keyword routes to wiki_agent regardless of
    #    a github keyword also being present.
    def test_p1_beats_p3_github(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("ingest this file, it's about a github repo", context={})
        assert plan.agent == "wiki_agent"
        assert "github_search" not in plan.tools_to_call

    # 8. No github keyword at all → P3-github defers, falls through normally
    #    (P6 direct answer, since nothing else in this instruction matches).
    def test_no_github_keyword_falls_through(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("how do I bake bread?", context={})
        assert plan.tools_to_call == []
        assert plan.agent == "conversational_agent"


# ---------------------------------------------------------------------------
# Priority 3-github-release — github_release tool routing
# ---------------------------------------------------------------------------

class TestPlannerP3GithubRelease:

    # 1. The motivating case: no literal "github"/"repo" anywhere, just a
    #    release-shaped phrase + project name — must route to github_release,
    #    not fall through to plain web_search or nothing at all.
    def test_release_notes_with_no_github_keyword_routes_to_github_release(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("fetch the oMLX 0.5.3 release notes and summarize them", context={})
        assert plan.tools_to_call == ["github_release"]
        assert plan.compound is True
        assert plan.priority == 3
        assert plan.tool_signal_source == "keyword"
        assert plan.fetch_rag is False
        assert plan.fetch_episodic is False

    # 2. "changelog"/"latest release"/"new release" all match
    #    _GITHUB_RELEASE_KEYWORDS too.
    @pytest.mark.parametrize("instruction", [
        "check the changelog for oMLX",
        "what's the latest release of oMLX",
        "is there a new release for oMLX yet?",
    ])
    def test_other_release_keywords_route_to_github_release(self, instruction):
        p = Planner(runtime=make_runtime())
        plan = p.route(instruction, context={})
        assert plan.tools_to_call == ["github_release"]

    # 3. github_release wins over the plain github_search match when both
    #    signals are present in the same turn — P3-github-release runs
    #    before P3-github in route().
    def test_release_beats_plain_github_search(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("check the release notes for that github repo", context={})
        assert plan.tools_to_call == ["github_release"]
        assert "github_search" not in plan.tools_to_call

    # 4. Bare "release" alone does NOT trigger github_release — deliberately
    #    excluded from _GITHUB_RELEASE_KEYWORDS as too weak/ambiguous a
    #    signal (see its module-level comment).
    def test_bare_release_word_does_not_trigger(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("please release the quarterly report", context={})
        assert "github_release" not in plan.tools_to_call

    # 5. file_op guard: a literal _FILE_OP_KEYWORDS hit defers to P3.
    def test_file_op_guard_defers_to_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("read the file about the oMLX release notes", context={})
        assert plan.tools_to_call == ["file_op"]
        assert "github_release" not in plan.tools_to_call

    # 6. url_fetch guard: a raw URL defers to P3.
    def test_url_guard_defers_to_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("summarize these release notes: https://example.com/releases/v1", context={})
        assert "url_fetch" in plan.tools_to_call
        assert "github_release" not in plan.tools_to_call

    # 7. Ordering regression: P3c (graph-query) beats P3-github-release, same
    #    as it already beats P3-news/P3-github/plain P3.
    def test_p3c_beats_github_release(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona, any release notes?", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "incoming"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]
        assert "github_release" not in plan.tools_to_call

    # 8. P1 beats P3-github-release: ingest keyword routes to wiki_agent
    #    regardless of a release keyword also being present.
    def test_p1_beats_p3_github_release(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("ingest this file, it's about some release notes", context={})
        assert plan.agent == "wiki_agent"
        assert "github_release" not in plan.tools_to_call

    # 9. No release keyword at all → P3-github-release defers, falls through
    #    normally.
    def test_no_release_keyword_falls_through(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("how do I bake bread?", context={})
        assert plan.tools_to_call == []
        assert plan.agent == "conversational_agent"


# ---------------------------------------------------------------------------
# Priority 3-hacker-news — hacker_news_search tool routing
# ---------------------------------------------------------------------------

class TestPlannerP3HackerNews:

    # 1. Basic keyword match routes to hacker_news_search only.
    def test_hacker_news_keyword_routes_to_hacker_news_search(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("what's trending on hacker news today?", context={})
        assert plan.tools_to_call == ["hacker_news_search"]
        assert plan.compound is True
        assert plan.priority == 3
        assert plan.tool_signal_source == "keyword"
        assert plan.fetch_rag is False
        assert plan.fetch_episodic is False

    # 2. "y combinator"/"ycombinator"/"hackernews"/bare "hn" all match too.
    @pytest.mark.parametrize("instruction", [
        "any discussion of this on hackernews?",
        "what's y combinator saying about this",
        "search ycombinator for this topic",
        "what's on hn today",
        "check hn for this",
    ])
    def test_other_hacker_news_keywords_route_to_hacker_news_search(self, instruction):
        p = Planner(runtime=make_runtime())
        plan = p.route(instruction, context={})
        assert plan.tools_to_call == ["hacker_news_search"]

    # 3. hacker_news_search wins over a web_search-only P3 match on the same
    #    turn — "recent" is a _WEB_SEARCH_KEYWORDS entry; P3-hacker-news
    #    runs before P3.
    def test_hacker_news_beats_web_search_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("any recent discussion of this on hacker news?", context={})
        assert plan.tools_to_call == ["hacker_news_search"]
        assert "web_search" not in plan.tools_to_call

    # 4. file_op guard: a literal _FILE_OP_KEYWORDS hit ("read the file")
    #    alongside "hacker news" defers P3-hacker-news to P3, which resolves
    #    it as file_op.
    def test_file_op_guard_defers_to_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("read the file about that hacker news story", context={})
        assert plan.tools_to_call == ["file_op"]
        assert "hacker_news_search" not in plan.tools_to_call

    # 5. url_fetch guard: a raw URL alongside "hacker news" defers to P3.
    def test_url_guard_defers_to_p3(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("summarize this hacker news page: https://example.com/article", context={})
        assert "url_fetch" in plan.tools_to_call
        assert "hacker_news_search" not in plan.tools_to_call

    # 6. Ordering regression: P3c (graph-query) beats P3-hacker-news, same as
    #    it already beats plain P3 web_search/P3-news/P3-github.
    def test_p3c_beats_hacker_news_search(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona, any hacker news discussion?", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "incoming"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]
        assert "hacker_news_search" not in plan.tools_to_call

    # 7. P1 beats P3-hacker-news: ingest keyword routes to wiki_agent
    #    regardless of a hacker news keyword also being present.
    def test_p1_beats_p3_hacker_news(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("ingest this file, it's about a hacker news story", context={})
        assert plan.agent == "wiki_agent"
        assert "hacker_news_search" not in plan.tools_to_call

    # 8. No hacker news keyword at all → P3-hacker-news defers, falls
    #    through normally (P6 direct answer).
    def test_no_hacker_news_keyword_falls_through(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("how do I bake bread?", context={})
        assert plan.tools_to_call == []
        assert plan.agent == "conversational_agent"


# ---------------------------------------------------------------------------
# Priority 1b — diff-only wiki update (standalone diff instructions scope doc)
# ---------------------------------------------------------------------------

class TestPlannerP1bDiff:

    # 1. Keyword + resolvable target → wiki_agent + diff_target set.
    def test_resolvable_target_routes_to_wiki_agent_with_diff_target(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route(
            "update page localist-software-stack.md to reflect the new "
            "Ollama runtime backend and MCP tool layer",
            context={},
        )
        assert plan.agent          == "wiki_agent"
        assert plan.diff_target    == "localist-software-stack"
        assert plan.fetch_rag      is False
        assert plan.fetch_episodic is False

    # 2. Keyword + ambiguous target → clarification path, not P2+.
    def test_ambiguous_target_routes_to_conversational_agent_for_clarification(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(infer_return="no"), memory_manager=mm)
        plan = p.route("update page localist", context={})
        assert plan.diff_target is None
        assert plan.agent       == "conversational_agent"
        assert plan.tools_to_call == []
        assert plan.fetch_rag      is False
        assert plan.fetch_episodic is False

    # 3. No _DIFF_KEYWORDS lead phrase → P1b does not fire; P1-P6 ordering
    #    unaffected (regression guard).
    def test_no_diff_keyword_falls_through_to_other_priorities(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)

        # P3c (graph-query) still fires — P1b doesn't swallow it.
        plan = p.route("what links to lora-persona", context={})
        assert plan.diff_target is None
        assert plan.graph_query is not None

        # P2 (memory command) still fires — P1b doesn't swallow it.
        p2 = Planner(runtime=make_runtime(), memory_manager=mm)
        plan2 = p2.route("remember that I prefer dark mode", context={})
        assert plan2.diff_target   is None
        assert plan2.write_episode is True

    # 4. P1 (ingest) beats P1b when both would match.
    def test_p1_ingest_beats_p1b(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route(
            "ingest this file, update page localist-software-stack.md",
            context={},
        )
        assert plan.agent       == "wiki_agent"
        assert plan.diff_target is None

    # 5. P1b beats P2 — a diff instruction that also contains a memory
    #    keyword still routes via P1b, not P2.
    def test_p1b_beats_p2(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route(
            "update page localist-software-stack.md, remember that I like it",
            context={},
        )
        assert plan.agent         == "wiki_agent"
        assert plan.diff_target   == "localist-software-stack"
        assert plan.write_episode is False

    # 6. No MemoryManager: keyword match but nothing to resolve against →
    #    clarification path, same as an ambiguous/failed resolution.
    def test_no_memory_manager_routes_to_conversational_agent_for_clarification(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("update page localist-software-stack.md", context={})
        assert plan.diff_target is None
        assert plan.agent       == "conversational_agent"
        assert plan.tools_to_call == []


# ---------------------------------------------------------------------------
# Priority 1c — pinned-wiki-page diff short-circuit
# ---------------------------------------------------------------------------

class TestPlannerP1cPinnedDiff:

    @pytest.fixture(autouse=True)
    def _clear_session_files(self):
        session_files.clear()
        yield
        session_files.clear()

    # (a) exactly-one-pin + diff keyword anywhere (not a lead phrase) →
    # short-circuits, no resolve_graph_target() needed.
    def test_pinned_page_diff_keyword_anywhere_short_circuits(self, tmp_path):
        session_files.add_file(
            "localist-software-stack.md", "# Stack\n", source="wiki_pin",
        )
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)

        plan = p.route(
            "Read my live repo then propose diffs to update the wiki corpus",
            context={},
        )

        assert plan.agent              == "wiki_agent"
        assert plan.diff_target        == "localist-software-stack"
        assert plan.diff_target_source == "pinned"
        assert plan.tools_to_call      == []
        assert plan.compound           is False

    # (b) exactly-one-pin + a Priority 3 tool need (url_fetch) → compound
    # plan carrying both diff_target and tools_to_call.
    def test_pinned_page_with_tool_need_routes_compound(self, tmp_path):
        session_files.add_file(
            "localist-software-stack.md", "# Stack\n", source="wiki_pin",
        )
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)

        plan = p.route(
            "fetch this url https://example.com/changelog and "
            "update page to reflect it",
            context={},
        )

        assert plan.agent              == "wiki_agent"
        assert plan.diff_target        == "localist-software-stack"
        assert plan.diff_target_source == "pinned"
        assert plan.tools_to_call      == ["url_fetch"]
        assert plan.compound           is True
        assert plan.tool_signal_source == "keyword"

    # (c) two or more pins → ambiguous; falls through to normal priority
    # evaluation exactly as if nothing were pinned.
    def test_two_pins_falls_through_to_normal_evaluation(self, tmp_path):
        instruction = "Read my live repo then propose diffs to update the wiki corpus"
        mm, _ = make_mm_with_nodes(tmp_path)

        session_files.add_file("localist-software-stack.md", "# A\n", source="wiki_pin")
        session_files.add_file("how-localist-works.md",      "# B\n", source="wiki_pin")
        p_with_pins = Planner(runtime=make_runtime(), memory_manager=mm)
        plan_with_pins = p_with_pins.route(instruction, context={})

        session_files.clear()
        p_without_pins = Planner(runtime=make_runtime(), memory_manager=mm)
        plan_without_pins = p_without_pins.route(instruction, context={})

        assert plan_with_pins.diff_target_source is None
        assert plan_with_pins.agent       == plan_without_pins.agent
        assert plan_with_pins.diff_target == plan_without_pins.diff_target

    # (d) zero pins → Priority 1c's own guard is what causes pass-through
    # (not accidental non-invocation); the existing TestPlannerP1bDiff
    # suite above is the full byte-for-byte regression check.
    def test_zero_pins_priority1c_returns_none(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)

        result = p._priority1c_pinned_diff(
            "update page localist-software-stack.md to reflect the new backend",
            "update page localist-software-stack.md to reflect the new backend",
        )
        assert result is None

    # (e) Regression coverage for real-world phrasings that initially fell
    # through to conversational_agent despite a page being pinned — neither
    # matched the original five-phrase _DIFF_KEYWORDS set. "apply diffs",
    # "apply this diff", "propose diffs", "propose a diff" were added.
    @pytest.mark.parametrize("instruction", [
        "Apply diffs updating my new tool route logic.",
        "Propose diffs for the localist-runtime-tooling-update wiki file.",
    ])
    def test_real_world_diff_phrasings_short_circuit(self, tmp_path, instruction):
        session_files.add_file(
            "localist-runtime-tooling-update.md", "# Tooling\n", source="wiki_pin",
        )
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)

        plan = p.route(instruction, context={})

        assert plan.agent              == "wiki_agent"
        assert plan.diff_target        == "localist-runtime-tooling-update"
        assert plan.diff_target_source == "pinned"


# ---------------------------------------------------------------------------
# Diagnostic Slot 1 — _semantic_search_intent tests
# ---------------------------------------------------------------------------

def _unit_vector(dim: int = 4) -> list[float]:
    """Return a simple normalised vector [1/sqrt(dim), ...] of length `dim`."""
    v = 1.0 / math.sqrt(dim)
    return [v] * dim


class TestSemanticSearchIntent:
    """Unit tests for Planner._semantic_search_intent (Diagnostic Slot 1)."""

    def test_returns_none_when_embed_fn_is_none(self):
        """No embed_fn → always None, no templates cached."""
        p = Planner(runtime=make_runtime())
        assert p._embed_fn is None
        assert p._template_embeddings == []
        result = p._semantic_search_intent("why don't you search for something")
        assert result is None

    def test_negative_filter_no_conflict_returns_none_without_model_call(self):
        """
        2026-07-16 redesign: the negative filter is checked AFTER scoring,
        not before — so embed_fn IS now called even for a filter-matched
        phrase (unlike the pre-redesign short-circuit this test used to
        assert). But when the resulting scores don't clear any gated
        group's threshold, filter and embedding already agree — no
        conflict, so no model call (runtime.infer) is made either.
        """
        fixed_vec = _unit_vector(4)                    # [0.5, 0.5, 0.5, 0.5]
        orthogonal_vec = [0.5, 0.5, -0.5, -0.5]         # cosine(fixed, orthogonal) == 0

        embed_fn = MagicMock(return_value=fixed_vec)
        runtime = make_runtime()
        p = Planner(runtime=runtime, embed_fn=embed_fn)
        embed_fn.reset_mock()
        embed_fn.return_value = orthogonal_vec

        result = p._semantic_search_intent("did you search for that already?")
        assert result is None
        embed_fn.assert_called_once()                  # now called, unlike pre-redesign
        runtime.infer.assert_not_called()               # but no conflict -> no model call

    def test_negative_filter_conflict_confirmed_by_tiebreak_returns_none(self):
        """
        A filter phrase matches AND the score clears a gated threshold (a
        genuine conflict) — the tie-break model call fires. When it
        answers anything other than "lookup" (make_runtime()'s default
        infer_return="no"), the filter's suppression is confirmed and the
        method still returns None.
        """
        fixed_vec = _unit_vector(4)
        embed_fn = MagicMock(return_value=fixed_vec)    # same vector everywhere -> cosine 1.0
        runtime = make_runtime()                         # infer() returns "no" by default
        p = Planner(runtime=runtime, embed_fn=embed_fn)

        result = p._semantic_search_intent("did you search for that already?")
        assert result is None
        runtime.infer.assert_called_once()

    def test_negative_filter_conflict_overridden_by_tiebreak_returns_result(self):
        """
        Same genuine-conflict setup as above, but the tie-break answers
        "lookup" — the filter is overridden and the real (best_group,
        best_score, all_scores) result is returned instead of None.
        """
        fixed_vec = _unit_vector(4)
        embed_fn = MagicMock(return_value=fixed_vec)
        runtime = make_runtime(infer_return="lookup")
        p = Planner(runtime=runtime, embed_fn=embed_fn)

        result = p._semantic_search_intent("did you search for that already?")
        assert result is not None
        runtime.infer.assert_called_once()

    def test_returns_group_and_score_for_matching_vector(self):
        """
        When embed_fn always returns the same unit vector, every template gets
        that vector at __init__ time and the query also gets it — cosine
        similarity of 1.0 with every template, so the returned score ≈ 1.0.
        The returned group must be one of the known template groups.
        """
        from localist.planner import _SEARCH_INTENT_TEMPLATES
        fixed_vec = _unit_vector(8)
        embed_fn = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)

        embed_fn.reset_mock()
        result = p._semantic_search_intent("why don't you do a web search for APC")
        assert result is not None
        best_group, best_score, all_scores = result
        assert best_group in _SEARCH_INTENT_TEMPLATES
        assert abs(best_score - 1.0) < 1e-6  # cosine(v, v) == 1.0

    def test_returns_none_gracefully_when_embed_fn_raises(self):
        """embed_fn raising RuntimeError must not propagate — returns None instead."""
        embed_fn = MagicMock(side_effect=RuntimeError("model unavailable"))
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)
        # __init__ will fail to embed templates (error swallowed), so
        # _template_embeddings will be empty → returns None immediately.
        # OR embed_fn raises during the query call if templates somehow cached.
        # Either way, the method must return None, never raise.
        result = p._semantic_search_intent("look it up online")
        assert result is None

    def test_returns_none_gracefully_when_embed_fn_raises_only_on_query(self):
        """
        embed_fn succeeds for template embedding at __init__ but raises when
        called for the query string — _semantic_search_intent must return None.
        """
        call_count = {"n": 0}
        template_total = sum(
            len(v) for v in __import__("localist.planner", fromlist=["_SEARCH_INTENT_TEMPLATES"])._SEARCH_INTENT_TEMPLATES.values()
        )

        def embed_fn_that_fails_later(text: str) -> list[float]:
            call_count["n"] += 1
            if call_count["n"] > template_total:
                raise RuntimeError("model unavailable on query call")
            return _unit_vector(8)

        p = Planner(runtime=make_runtime(), embed_fn=embed_fn_that_fails_later)
        assert len(p._template_embeddings) == template_total  # templates loaded OK

        result = p._semantic_search_intent("look it up")
        assert result is None


class TestEpisodicSemanticRelevance:
    """
    Unit + integration tests for Planner._episodic_semantic_relevance() and
    its wiring into _priority5_episodic() (2026-07-23 addition — see
    diagnostics/reports/episodic_relevance_semantic_gate_2026-07-23.md).

    Deliberately parallels TestSemanticSearchIntent's fixed_vec/orthogonal_vec
    pattern rather than inventing a new one, since this gate was built to
    mirror _semantic_search_intent()'s mechanism.
    """

    def test_returns_none_when_embed_fn_is_none(self):
        p = Planner(runtime=make_runtime())
        assert p._embed_fn is None
        assert p._episodic_template_embeddings == []
        assert p._episodic_semantic_relevance("help me prepare for this") is None

    def test_returns_high_score_for_matching_vector(self):
        fixed_vec = _unit_vector(8)
        embed_fn = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)

        score = p._episodic_semantic_relevance("anything at all")
        assert score is not None
        assert abs(score - 1.0) < 1e-6  # cosine(v, v) == 1.0

    def test_returns_low_score_for_orthogonal_vector(self):
        fixed_vec = _unit_vector(4)                     # templates embedded with this
        orthogonal_vec = [0.5, 0.5, -0.5, -0.5]          # cosine(fixed, orthogonal) == 0

        embed_fn = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)
        embed_fn.return_value = orthogonal_vec

        score = p._episodic_semantic_relevance("totally unrelated text")
        assert score is not None
        assert abs(score) < 1e-6

    def test_returns_none_gracefully_when_embed_fn_raises(self):
        embed_fn = MagicMock(side_effect=RuntimeError("model unavailable"))
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)
        # __init__ swallows the error too, so _episodic_template_embeddings
        # stays empty — either way this must return None, never raise.
        assert p._episodic_semantic_relevance("help me prepare for this") is None

    def test_disabled_when_semantic_gating_disabled(self):
        # Same _semantic_gating_disabled flag _semantic_search_intent() uses
        # (mismatched embedding model) — must disable this gate too, since
        # the "cosine thresholds aren't portable across models" rationale
        # applies to this threshold exactly as much as P3's.
        fixed_vec = _unit_vector(8)
        embed_fn = MagicMock(return_value=fixed_vec)
        p = Planner(
            runtime=make_runtime(), embed_fn=embed_fn,
            embedding_model_name="nomic-embed-text",
        )
        assert p._semantic_gating_disabled is True
        assert p._episodic_semantic_relevance("help me prepare for this") is None

    def test_priority5_fires_via_semantic_gate_with_no_keyword_match(self):
        # "Help me prepare for the upcoming Claude Impact Lab on August 6th."
        # (the live failure this gate was added for) contains no
        # _EPISODIC_KEYWORDS match — only the semantic path can catch it.
        fixed_vec = _unit_vector(8)
        embed_fn = MagicMock(return_value=fixed_vec)  # cosine 1.0 with every template
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)

        instruction = "Help me prepare for the upcoming Claude Impact Lab on August 6th."
        assert not any(
            kw in instruction.lower()
            for kw in [
                "preference", "remember", "decision", "correction", "workflow",
                "last time", "previously", "before", "my project", "my setup",
                "my environment", "my name", "who am i",
            ]
        )

        plan = p._priority5_episodic(instruction)
        assert plan is not None
        assert plan.fetch_episodic is True
        assert plan.fetch_rag is False

    def test_priority5_returns_none_when_neither_keyword_nor_semantic_fires(self):
        fixed_vec = _unit_vector(4)
        orthogonal_vec = [0.5, 0.5, -0.5, -0.5]

        embed_fn = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)
        embed_fn.return_value = orthogonal_vec

        plan = p._priority5_episodic("What is 2+2?")
        assert plan is None

    def test_priority5_keyword_path_unaffected_by_embed_fn_presence(self):
        # Regression guard: adding the semantic gate must not change the
        # pre-existing keyword-match behavior for an instruction that hits
        # both signals.
        fixed_vec = _unit_vector(8)
        embed_fn = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)

        plan = p._priority5_episodic("What are my formatting preferences?")
        assert plan is not None
        assert plan.fetch_episodic is True


class TestTunedEmbeddingModelGuard:
    """
    Unit tests for the _TUNED_EMBEDDING_MODEL guard (docs/architecture/
    16-runtime-backend-layer.md §16.4): semantic search-intent gating is
    disabled whenever the active embedding model doesn't match the model
    every _SEMANTIC_GATE_THRESHOLDS / _RESEARCH_INTENT_THRESHOLD value was
    tuned against, confirmed live 2026-07-16 (lookup_request scored 0.7119
    under mlx-community/embeddinggemma-300m-4bit vs 0.578 under
    nomic-embed-text for the identical utterance).
    """

    def _matching_vec_planner(self, **kwargs) -> "Planner":
        """Planner whose embed_fn returns the identical vector everywhere —
        every template group scores ~1.0, clearing every gate threshold —
        so any None result can only be attributed to the guard, not to a
        genuinely low score."""
        fixed_vec = _unit_vector(8)
        embed_fn = MagicMock(return_value=fixed_vec)
        return Planner(runtime=make_runtime(), embed_fn=embed_fn, **kwargs)

    def test_mismatched_model_disables_semantic_gating(self, caplog):
        with caplog.at_level(logging.WARNING, logger="localist.planner"):
            p = self._matching_vec_planner(embedding_model_name="nomic-embed-text")

        assert p._semantic_gating_disabled is True
        assert "does not match the model" in caplog.text
        assert "nomic-embed-text" in caplog.text

        result = p._semantic_search_intent("why don't you do a web search for APC")
        assert result is None

    def test_matching_model_name_unaffected(self, caplog):
        from localist.planner import _TUNED_EMBEDDING_MODEL

        with caplog.at_level(logging.WARNING, logger="localist.planner"):
            p = self._matching_vec_planner(embedding_model_name=_TUNED_EMBEDDING_MODEL)

        assert p._semantic_gating_disabled is False
        assert caplog.text == ""

        result = p._semantic_search_intent("why don't you do a web search for APC")
        assert result is not None
        best_group, best_score, all_scores = result
        assert abs(best_score - 1.0) < 1e-6

    def test_none_model_name_unaffected_no_warning(self, caplog):
        """embedding_model_name=None (keyword-only / no embedding source
        configured) is not a mismatch — no warning, gating untouched."""
        with caplog.at_level(logging.WARNING, logger="localist.planner"):
            p = self._matching_vec_planner(embedding_model_name=None)

        assert p._semantic_gating_disabled is False
        assert caplog.text == ""

        result = p._semantic_search_intent("why don't you do a web search for APC")
        assert result is not None


class TestSemanticSearchIntentDiag2:
    """Diagnostic Slot 2 additions — per-group score dict shape and correctness."""

    def test_all_group_scores_contains_exactly_five_expected_keys(self):
        """all_group_scores must contain exactly the five template-group keys."""
        from localist.planner import _SEARCH_INTENT_TEMPLATES
        expected_keys = set(_SEARCH_INTENT_TEMPLATES.keys())
        assert expected_keys == {
            "explicit_search_action",
            "lookup_request",
            "knowledge_request_open",
            "freshness_request",
            "research_intent",
        }

        fixed_vec = _unit_vector(8)
        p = Planner(runtime=make_runtime(), embed_fn=MagicMock(return_value=fixed_vec))
        result = p._semantic_search_intent("find out about this topic online")
        assert result is not None
        _, _, all_scores = result
        assert set(all_scores.keys()) == expected_keys

    def test_per_group_max_scores_are_independent(self):
        """
        When embed_fn returns different vectors for different template strings,
        the per-group score in all_group_scores reflects each group's own best
        cosine similarity, not the global best.

        Strategy: engineer two orthogonal basis vectors v1 and v2.
        At __init__ time, embed_fn always returns v1, so all template
        embeddings are v1. Then for the query, we switch the spy to return v2.
        cosine(v2, v1) ≈ 0 (orthogonal) for all templates.

        Then we patch _template_embeddings directly to give two groups
        different representative vectors and confirm per-group reporting.
        """
        from localist.planner import _SEARCH_INTENT_TEMPLATES

        # Build a planner with any embed_fn to get past __init__ validation.
        fixed_vec = _unit_vector(8)
        p = Planner(runtime=make_runtime(), embed_fn=MagicMock(return_value=fixed_vec))

        # Manually install two distinct group vectors:
        # group A ("explicit_search_action") gets [1, 0, 0, 0]
        # group B ("lookup_request")         gets [0, 1, 0, 0]
        # all others get [0, 0, 0, 1] (low similarity to both query vectors)
        def normalise(v: list[float]) -> list[float]:
            import math
            n = math.sqrt(sum(x * x for x in v))
            return [x / n for x in v]

        vec_a = normalise([1.0, 0.0, 0.0, 0.0])
        vec_b = normalise([0.0, 1.0, 0.0, 0.0])
        vec_other = normalise([0.0, 0.0, 0.0, 1.0])

        group_to_vec = {
            "explicit_search_action": vec_a,
            "lookup_request":         vec_b,
            "knowledge_request_open": vec_other,
            "freshness_request":      vec_other,
            "research_intent":        vec_other,
        }
        p._template_embeddings = [
            (g, group_to_vec[g]) for g in _SEARCH_INTENT_TEMPLATES
        ]

        # Query vector halfway between A and B: [1, 1, 0, 0] normalised.
        query_vec = normalise([1.0, 1.0, 0.0, 0.0])
        p._embed_fn = MagicMock(return_value=query_vec)

        result = p._semantic_search_intent("some query")
        assert result is not None
        best_group, best_score, all_scores = result

        # cos([1,1,0,0]/√2, [1,0,0,0]) = 1/√2 ≈ 0.707 for both A and B
        # cos([1,1,0,0]/√2, [0,0,0,1]) = 0 for others
        import math
        expected_ab = 1.0 / math.sqrt(2)
        assert abs(all_scores["explicit_search_action"] - expected_ab) < 1e-5
        assert abs(all_scores["lookup_request"] - expected_ab) < 1e-5
        assert abs(all_scores["knowledge_request_open"]) < 1e-5
        assert abs(all_scores["freshness_request"]) < 1e-5
        assert abs(all_scores["research_intent"]) < 1e-5

        # best_group is either A or B (tied — max() picks the first alphabetically)
        assert best_group in ("explicit_search_action", "lookup_request")
        assert abs(best_score - expected_ab) < 1e-5


def _planner_with_mocked_semantic(all_scores: dict[str, float]) -> "Planner":
    """
    Return a Planner whose _semantic_search_intent is patched to return
    all_scores directly, bypassing embed_fn entirely.  Used to test the
    gate logic in _priority3_tool with precise, normalization-free control.
    """
    p = Planner(runtime=make_runtime())
    best_group = max(all_scores, key=lambda g: all_scores[g])
    best_score = all_scores[best_group]
    p._semantic_search_intent = MagicMock(
        return_value=(best_group, best_score, all_scores)
    )
    return p


class TestPriority3SemanticGating:
    """
    Slot [Fix 1]: verifies the semantic gate correctly controls tools_to_call.

    Replaces TestPriority3ToolUnaffectedBySemantic (from Diagnostics 1/2).
    That class contained three methods:
      - test_no_literal_keyword_returns_none_despite_semantic_match
        REMOVED: asserted semantic never adds web_search — false after this slot.
      - test_route_unchanged_by_semantic_for_p6_instruction
        REMOVED: asserted that a fixed-vector instruction falls to P6 — false
        after this slot because a fixed unit vector scores 1.0 on all groups
        including explicit_search_action ≥ 0.68, so it now routes via P3.
      - test_existing_web_search_keyword_still_fires_with_embed_fn
        PRESERVED below as test_literal_keyword_still_fires_with_embed_fn:
        the invariant (literal keyword → web_search) is unchanged by this slot.

    Seven new tests cover the gate boundary, the protected negative group, and
    the deduplication guard.
    """

    def test_literal_keyword_still_fires_with_embed_fn(self):
        """Literal _WEB_SEARCH_KEYWORDS match still produces web_search.

        2026-07-22: instruction changed from "...latest news on APC?" to
        "...latest price on APC?" — "news" moved out of _WEB_SEARCH_KEYWORDS
        into _NEWS_KEYWORDS (news_search now wins for it via P3-news, which
        runs before P3 — see test_priority3_news.py), so the old wording no
        longer isolates the literal-web_search-keyword invariant this test
        is about. "latest"/"current price" remain untouched P3 keywords.
        """
        fixed_vec = _unit_vector(8)
        p = Planner(runtime=make_runtime(), embed_fn=MagicMock(return_value=fixed_vec))
        plan = p.route("what is the latest price on APC?", context={})
        assert "web_search" in plan.tools_to_call
        assert plan.compound is True

    def test_explicit_search_action_at_threshold_fires_web_search(self):
        """explicit_search_action ≥ 0.68 alone → gate fires → web_search added."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.90,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you check the internet for that")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_lookup_request_at_threshold_fires_web_search(self):
        """lookup_request ≥ 0.65 alone → gate fires → web_search added."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.90,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("go ahead and look it up")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_knowledge_group_alone_does_not_fire_gate(self):
        """
        knowledge_request_open at 0.95 with both gating groups below threshold
        must NOT add web_search — this is the exact failure mode the
        group-specific gate design prevents.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.30,
            "lookup_request":         0.30,
            "knowledge_request_open": 0.95,
            "freshness_request":      0.20,
        })
        result = p._priority3_tool("explain this code to me")
        assert result is None

    def test_score_just_below_lookup_threshold_does_not_fire(self):
        """
        lookup_request at 0.59 (below the 0.60 threshold; analogous to
        NEG-03 from Diagnostic 2 which scored 0.601) must not trigger the gate.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.30,
            "lookup_request":         0.59,
            "knowledge_request_open": 0.20,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("find me a good name for this variable")
        assert result is None

    def test_score_just_below_explicit_threshold_does_not_fire(self):
        """explicit_search_action at 0.67 (below the current 0.72 threshold) must not trigger the gate."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.67,
            "lookup_request":         0.30,
            "knowledge_request_open": 0.20,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you just check something for me")
        assert result is None

    def test_no_duplicate_web_search_when_literal_and_semantic_both_match(self):
        """
        An instruction with a literal _WEB_SEARCH_KEYWORDS match AND a high
        semantic score must produce exactly one 'web_search' in tools_to_call.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.90,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        # "latest" is a literal keyword; semantic gate also fires; no duplicate.
        result = p._priority3_tool("what's the latest on this topic, look it up")
        assert result is not None
        assert result.tools_to_call.count("web_search") == 1

    def test_route_falls_to_p6_when_all_semantic_scores_below_threshold(self):
        """
        An instruction with all semantic scores below both gating thresholds
        and no literal P3 keywords must reach P6 (no tools, no retrieval).
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.30,
            "lookup_request":         0.30,
            "knowledge_request_open": 0.40,
            "freshness_request":      0.20,
        })
        plan = p.route("what is 2 + 2?", context={})
        assert plan.tools_to_call == []
        assert plan.fetch_rag      is False
        assert plan.fetch_episodic is False
        assert plan.agent          == "conversational_agent"

    # ------------------------------------------------------------------
    # 2026-06-25: template-coverage fix for question-form lookup_request
    # (§8.8 Open Item 11)
    # ------------------------------------------------------------------

    def test_set1_lookup_request_templates_present(self):
        """
        All four Candidate Set 1 (object-specificity fix) templates are in lookup_request.
        The four 2026-06-25 question-form templates were removed 2026-06-28 — this test
        was renamed and updated from test_new_lookup_request_templates_present (pass→fail
        change: old test asserted the removed templates; updated to assert the new Set 1
        templates that replaced them).
        """
        from localist.planner import _SEARCH_INTENT_TEMPLATES
        templates = _SEARCH_INTENT_TEMPLATES["lookup_request"]
        for expected in (
            "can you look up the release date for this",
            "could you look up what year this happened",
            "can you look up information about the latest Apple products",
            "could you find out the current stock price for me",
        ):
            assert expected in templates, f"Missing Set 1 template: {expected!r}"

    def test_old_2026_06_25_lookup_request_templates_removed(self):
        """The 4 collision-prone templates added 2026-06-25 were removed 2026-06-28."""
        from localist.planner import _SEARCH_INTENT_TEMPLATES
        templates = _SEARCH_INTENT_TEMPLATES["lookup_request"]
        for removed in (
            "can you look up",
            "can you look that up for me",
            "could you look up",
            "can you look into this for me",
        ):
            assert removed not in templates, f"Removed template still present: {removed!r}"

    def test_original_lookup_request_templates_unchanged(self):
        """Regression guard: original five lookup_request templates are present and unmodified."""
        from localist.planner import _SEARCH_INTENT_TEMPLATES
        templates = _SEARCH_INTENT_TEMPLATES["lookup_request"]
        for expected in (
            "look up this",
            "look that up",
            "go ahead and look it up",
            "find information on this",
            "find out about this",
        ):
            assert expected in templates, f"Original template missing or edited: {expected!r}"

    def test_semantic_gate_thresholds_current_values(self):
        """Regression lock: _SEMANTIC_GATE_THRESHOLDS must match current calibrated values.
        lookup_request lowered 0.65 → 0.60 on 2026-06-25 (§10.4 Open Item 3 revisit).
        explicit_search_action raised 0.68 → 0.72 on 2026-06-28 per
        explicit_search_action_margin_assessment_2026-06-28.md (pass→fail change:
        old assertion was 0.68; updated to 0.72 after the threshold raise)."""
        from localist.planner import _SEMANTIC_GATE_THRESHOLDS
        assert _SEMANTIC_GATE_THRESHOLDS == {
            "explicit_search_action": 0.72,
            "lookup_request": 0.60,
        }

    def test_lookup_request_score_at_new_threshold_fires_gate(self):
        """lookup_request at 0.605 (≥ 0.60) must fire the gate → web_search added."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.605,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you look up something for me")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_lookup_request_score_just_below_new_threshold_does_not_fire(self):
        """lookup_request at 0.595 (< 0.60) must NOT fire the gate."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.595,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you look up something for me")
        assert result is None

    def test_other_template_groups_unmodified(self):
        """Regression guard: explicit_search_action, knowledge_request_open, freshness_request are unchanged."""
        from localist.planner import _SEARCH_INTENT_TEMPLATES
        assert _SEARCH_INTENT_TEMPLATES["explicit_search_action"] == (
            "search the web for this",
            "do a web search for this",
            "search online for this",
            "google this",
            "go look it up",
        )
        assert _SEARCH_INTENT_TEMPLATES["knowledge_request_open"] == (
            "what is this",
            "what do you know about this",
            "tell me about this",
            "explain this to me",
        )
        assert _SEARCH_INTENT_TEMPLATES["freshness_request"] == (
            "what's the latest on this",
            "what's the current status of this",
            "is there anything new about this",
        )


class TestExplicitDateSignal:
    """
    Explicit-date web-search signal (independent of both _WEB_SEARCH_KEYWORDS
    and _SEMANTIC_GATE_THRESHOLDS) — closes the live gap where "Look up
    TSM's July 16 2026 earnings..." had no keyword hit and scored 0.573 on
    lookup_request (just under the 0.60 gate), so no tool fired and the
    model produced an empty completion with zero grounding.
    """

    def test_month_day_year_fires_web_search_with_low_semantic_scores(self):
        """
        The exact live-bug phrasing: no keyword match, semantic scores all
        below threshold — only the date-pattern signal should fire.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.573,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool(
            "look up tsm's july 16 2026 earnings and report back a summary."
        )
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_day_month_year_fires(self):
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("what happened on 16 july 2026")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_iso_date_fires(self):
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("earnings were reported on 2026-07-16")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_month_year_without_day_fires(self):
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("tsm earnings for july 2026")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_bare_year_alone_does_not_fire_date_signal(self):
        """
        A bare 4-digit year with no month name is deliberately NOT enough —
        avoids false positives on version/model numbers. With semantic
        scores also below threshold and no keyword, nothing should fire.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("what happened with windows 2000")
        assert result is None

    def test_model_number_false_positive_guard(self):
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("tell me about the sr-2026 model specs")
        assert result is None

    def test_no_duplicate_web_search_when_keyword_and_date_both_match(self):
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("latest tsm earnings, july 16 2026")
        assert result is not None
        assert result.tools_to_call.count("web_search") == 1

    def test_existing_keyword_gate_cases_unaffected(self):
        """Regression guard: no threshold was touched by this signal."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.30,
            "lookup_request":         0.59,
            "knowledge_request_open": 0.20,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("find me a good name for this variable")
        assert result is None


class TestWebSearchKeywordLiteralFix2:
    """
    Slot Fix 2: 'web search' and 'do a search' added to _WEB_SEARCH_KEYWORDS.

    All tests use embed_fn=None so only the literal keyword path is reachable —
    proving the fix does not depend on the embedding layer.
    """

    def test_web_search_phrase_triggers_literal_path(self):
        """'web search' in instruction fires web_search via literal keyword match."""
        p = Planner(runtime=make_runtime())  # embed_fn=None — literal path only
        plan = p.route(
            "Why don't you do a web search for APC and then tell me if you "
            "still stand by your previous answer.",
            context={},
        )
        assert "web_search" in plan.tools_to_call

    def test_do_a_search_phrase_triggers_literal_path(self):
        """'do a search' in instruction fires web_search via literal keyword match."""
        p = Planner(runtime=make_runtime())  # embed_fn=None — literal path only
        plan = p.route("Can you do a search for recent AI papers?", context={})
        assert "web_search" in plan.tools_to_call

    def test_unrelated_sentence_does_not_trigger(self):
        """Sentence containing neither new phrase does not pick up a false positive."""
        p = Planner(runtime=make_runtime())  # embed_fn=None — literal path only
        plan = p.route("I searched online for new shoes yesterday.", context={})
        assert "web_search" not in plan.tools_to_call


class TestEmbedFnWiringSmoke:
    """Smoke-level test: Planner receives a non-None embed_fn when supplied."""

    def test_planner_stores_embed_fn_from_constructor(self):
        """Constructor argument embed_fn is stored as _embed_fn."""
        fn = MagicMock(return_value=_unit_vector())
        p = Planner(runtime=make_runtime(), embed_fn=fn)
        assert p._embed_fn is fn

    def test_controller_agent_threads_embed_fn_to_planner(self):
        """
        ControllerAgent accepts embed_fn and passes it through to the Planner.
        Verified by inspecting _planner._embed_fn on the constructed controller.
        """
        fn = MagicMock(return_value=_unit_vector())
        rt = make_runtime()
        agent = make_agent("conversational_agent")
        ctrl = ControllerAgent(runtime=rt, agents=[agent], embed_fn=fn)
        assert ctrl._planner._embed_fn is fn

    def test_controller_agent_embed_fn_defaults_to_none(self):
        """ControllerAgent with no embed_fn leaves Planner._embed_fn as None."""
        rt = make_runtime()
        agent = make_agent("conversational_agent")
        ctrl = ControllerAgent(runtime=rt, agents=[agent])
        assert ctrl._planner._embed_fn is None


# ---------------------------------------------------------------------------
# 2026-06-26: identity/capability negative-filter entries
# (confirmed false positives via diagnostics/score_lookup_request_templates.py)
# ---------------------------------------------------------------------------

class TestIdentityCapabilityNegativeFilter:
    """
    Five identity/capability phrases added to _SEARCH_NEGATIVE_FILTER on
    2026-06-26 after live diagnostic confirmed they cross the lookup_request
    0.60 gate via syntactic similarity with the four 2026-06-25 question-form
    templates ("can you look up", etc.).

    Tests 1–5: _semantic_search_intent returns None for each exact phrase.
    Test 6:    _priority3_tool returns None for "Who are you?" end-to-end.
    Test 7:    Non-regression — original 2026-06-25 incident utterances still
               fire the gate (negative filter does not intercept them).
    """

    def _make_planner_with_embed(self) -> "Planner":
        """Return a Planner with a stub embed_fn so _semantic_search_intent is reachable."""
        fixed_vec = _unit_vector(8)
        spy = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=spy)
        spy.reset_mock()
        return p

    def test_who_are_you_filtered(self):
        """'who are you' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("who are you") is None

    def test_what_are_you_filtered(self):
        """'what are you' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what are you") is None

    def test_what_can_you_do_filtered(self):
        """'what can you do' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what can you do") is None

    def test_what_can_you_help_with_filtered(self):
        """'what can you help with' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what can you help with") is None

    def test_what_do_you_do_filtered(self):
        """'what do you do' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what do you do") is None

    def test_priority3_tool_returns_none_for_who_are_you(self):
        """
        End-to-end: _priority3_tool("who are you?") returns None (no tools scheduled).
        Verifies the observed false-positive behavior is now blocked at the
        _priority3_tool level, not just the helper.
        """
        p = self._make_planner_with_embed()
        result = p._priority3_tool("who are you?")
        assert result is None

    def test_original_lookup_incident_utterances_still_fire_gate(self):
        """
        Non-regression: the 2026-06-25 incident's utterance family
        ("Can you look up ...") is NOT intercepted by the new negative-filter
        entries and still fires the semantic gate when scores are above threshold.

        Uses _planner_with_mocked_semantic to inject a score of 0.62 on
        lookup_request (above the 0.60 gate), exactly as measured for those
        live utterances post-update-B. Confirms gate_fired=True is preserved.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.62,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool(
            "can you look up apple's price hike for the macbook neo and ipad?"
        )
        assert result is not None, "Gate should fire for a real lookup-request utterance"
        assert "web_search" in result.tools_to_call


# ---------------------------------------------------------------------------
# 2026-06-27: greeting false-positive filter — §10.4
# ---------------------------------------------------------------------------

class TestGreetingFalsePositiveFilter:
    """
    Tests for the 4-phrase greeting block added to _SEARCH_NEGATIVE_FILTER
    on 2026-06-27 after the live diagnostic in
    diagnostics/score_greeting_collisions.py confirmed these utterances cross
    the lookup_request 0.60 gate on the real embedding model.

    What these tests DO verify:
        Substring-match behavior: the filter's `phrase in lowered` check fires
        correctly for each new entry and the surrounding confirmed-live
        utterance forms (trailing punctuation, mixed case, etc.).

        2026-07-16 redesign note: the filter is now checked AFTER scoring
        (see _semantic_search_intent's docstring), not before — embed_fn IS
        called for these utterances. Most tests below use
        _make_planner_with_embed's identical-vector stub, which makes every
        group score 1.0 (a guaranteed conflict), so suppression here is
        actually confirmed via the tie-break model call
        (_resolve_negative_filter_conflict, stubbed via make_runtime()'s
        default infer_return="no") rather than a pre-scoring short-circuit.

    What these tests CANNOT verify with a stub embed_fn:
        Whether these utterances would actually clear the 0.60 lookup_request
        gate on the REAL mlx-community/embeddinggemma-300m-4bit model. That
        was confirmed by the live diagnostic (run separately from this suite),
        not here. A stub embed_fn returning a synthetic vector has no
        relationship to the real model's embedding geometry.
    """

    def _make_planner_with_embed(self) -> "Planner":
        """Planner with a stub embed_fn; spy reset after __init__ template pre-embedding."""
        fixed_vec = _unit_vector(8)
        spy = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=spy)
        spy.reset_mock()
        return p

    # ------------------------------------------------------------------
    # Group 1 — filter membership
    # ------------------------------------------------------------------

    def test_hey_lora_in_negative_filter(self):
        """'hey lora' is a member of _SEARCH_NEGATIVE_FILTER (2026-06-27 addition)."""
        from localist.planner import _SEARCH_NEGATIVE_FILTER
        assert "hey lora" in _SEARCH_NEGATIVE_FILTER

    def test_hi_there_in_negative_filter(self):
        """'hi there' is a member of _SEARCH_NEGATIVE_FILTER (2026-06-27 addition)."""
        from localist.planner import _SEARCH_NEGATIVE_FILTER
        assert "hi there" in _SEARCH_NEGATIVE_FILTER

    def test_hey_there_in_negative_filter(self):
        """'hey there' is a member of _SEARCH_NEGATIVE_FILTER (2026-06-27 addition)."""
        from localist.planner import _SEARCH_NEGATIVE_FILTER
        assert "hey there" in _SEARCH_NEGATIVE_FILTER

    def test_whats_up_in_negative_filter(self):
        """"what's up" is a member of _SEARCH_NEGATIVE_FILTER (2026-06-27 addition)."""
        from localist.planner import _SEARCH_NEGATIVE_FILTER
        assert "what's up" in _SEARCH_NEGATIVE_FILTER

    # ------------------------------------------------------------------
    # Group 2 — behavioral: filter fires → _semantic_search_intent returns None
    # ------------------------------------------------------------------

    def test_hey_lora_exclamation_filtered(self):
        """
        'Hey LORA!' — the confirmed live false positive (2026-06-27) — is
        still suppressed post-redesign, but now via the tie-break conflict
        path rather than a pre-scoring short-circuit: embed_fn IS called
        (scores are computed first), the 'hey lora' substring match creates
        a conflict against the identical-vector stub (score 1.0 clears
        every gated threshold), and make_runtime()'s default
        infer_return="no" confirms the filter's suppression.

        Inlines the spy (rather than using _make_planner_with_embed) so
        call counts on both embed_fn and runtime.infer can be asserted
        explicitly, not just that the method returns None.

        Note: this stub cannot verify the real model would have scored this
        ≥ 0.60 — that was confirmed by the live diagnostic only.
        """
        spy = MagicMock(return_value=_unit_vector(8))
        runtime = make_runtime()
        p = Planner(runtime=runtime, embed_fn=spy)
        spy.reset_mock()

        result = p._semantic_search_intent("hey lora!")
        assert result is None
        spy.assert_called_once()          # now called — scores compute before the filter check
        runtime.infer.assert_called_once()  # genuine conflict (score 1.0) -> tie-break fires

    def test_hi_there_filtered(self):
        """'hi there' is caught by the filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("hi there") is None

    def test_whats_up_with_question_mark_filtered(self):
        """"what's up?" (trailing ?) matches 'what's up' substring → returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what's up?") is None

    def test_hey_lora_trailing_question_mark_filtered(self):
        """
        'hey lora?' — trailing punctuation — is caught by 'hey lora' substring.
        Confirms the substring check is not accidentally anchored or
        punctuation-sensitive at the tail of the string.
        """
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("hey lora?") is None

    # ------------------------------------------------------------------
    # Group 3 — non-regression: genuine lookup utterance not swallowed
    # ------------------------------------------------------------------

    def test_genuine_lookup_not_caught_by_greeting_filter(self):
        """
        'can you look up this' — a genuine lookup-request utterance — is NOT
        caught by any of the 4 new greeting-filter entries and still fires
        the semantic gate → web_search is in tools_to_call.

        Uses _planner_with_mocked_semantic (0.62 on lookup_request, above the
        0.60 threshold) matching the existing pattern in
        TestIdentityCapabilityNegativeFilter.test_original_lookup_incident_utterances_still_fire_gate.

        Guards against a future edit accidentally widening one of the 4 new
        phrases so it swallows real search intent.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.62,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you look up this")
        assert result is not None, (
            "Genuine lookup utterance must not be swallowed by the greeting filter"
        )
        assert "web_search" in result.tools_to_call

    # ------------------------------------------------------------------
    # Group 4 — documented gap: bare "hi" / "hey" deliberately NOT filtered
    # ------------------------------------------------------------------

    def test_bare_hi_not_filtered_documented_gap(self):
        """
        'hi' is deliberately NOT in _SEARCH_NEGATIVE_FILTER and is not
        intercepted by the filter.

        This is a DOCUMENTED GAP, not an omission. Bare 'hi' collides with
        common substrings in legitimate queries under this filter's
        substring-match mechanism ('history', 'this', 'high', 'vehicle',
        etc.) — adding it would silently suppress the semantic gate on
        unrelated queries.

        See the 2026-06-27 comment block in planner.py (_SEARCH_NEGATIVE_FILTER)
        and LOCALIST-Architecture.md §10.4 open items for the pending fix path
        (word-boundary-matched filter or a different mechanism).

        If you are reading this because you want to add 'hi' to the filter:
        resolve the collision risk in the §10.4 open item first.
        """
        from localist.planner import _SEARCH_NEGATIVE_FILTER
        assert "hi" not in _SEARCH_NEGATIVE_FILTER

        # Not intercepted → reaches the embedding path.
        # With a stub embed_fn the result is non-None (synthetic scores returned).
        p = self._make_planner_with_embed()
        result = p._semantic_search_intent("hi")
        assert result is not None, (
            "'hi' must reach the embedding path; the filter must not intercept it"
        )

    def test_bare_hey_not_filtered_documented_gap(self):
        """
        'hey' is deliberately NOT in _SEARCH_NEGATIVE_FILTER and is not
        intercepted by the filter.

        This is a DOCUMENTED GAP, not an omission. Bare 'hey' collides with
        'they' (and potentially other substrings) under this filter's
        substring-match mechanism — adding it would suppress the gate for
        utterances like 'what did they say?', 'can they look it up?', etc.

        See the 2026-06-27 comment block in planner.py (_SEARCH_NEGATIVE_FILTER)
        and LOCALIST-Architecture.md §10.4 open items for the pending fix path.

        If you are reading this because you want to add 'hey' to the filter:
        resolve the collision risk in the §10.4 open item first.
        """
        from localist.planner import _SEARCH_NEGATIVE_FILTER
        assert "hey" not in _SEARCH_NEGATIVE_FILTER

        p = self._make_planner_with_embed()
        result = p._semantic_search_intent("hey")
        assert result is not None, (
            "'hey' must reach the embedding path; the filter must not intercept it"
        )


# ---------------------------------------------------------------------------
# 2026-06-28: Candidate Set 1 template fix + explicit_search_action 0.68 → 0.72
# ---------------------------------------------------------------------------

class TestSet1TemplateFix20260628:
    """
    Validates the two 2026-06-28 planner.py changes:
      1. lookup_request: 4 collision-prone templates replaced with Candidate Set 1
         (object-specificity fix). Source: lookup_request_template_rework_2026-06-28.md,
         full_pertable_lr_set1_esa_2026-06-28.md.
      2. explicit_search_action threshold: 0.68 → 0.72. Source:
         explicit_search_action_margin_assessment_2026-06-28.md.

    Negative-filter protection for the 5 identity and 4 greeting phrases is
    unchanged — the filter fires pre-gate independent of template content.
    Verified by existing TestIdentityCapabilityNegativeFilter and
    TestGreetingFalsePositiveFilter; no new filter tests added here.

    All LR and ESA scores injected via _planner_with_mocked_semantic are
    the actual measured values from full_pertable_lr_set1_esa_2026-06-28.md
    and explicit_search_action_margin_assessment_2026-06-28.md.
    """

    # ------------------------------------------------------------------
    # (a) Cat C — 2026-06-25 incident utterances still fire via LR at Set 1
    # ------------------------------------------------------------------

    def test_cat_c1_still_fires_via_lr_set1(self):
        """
        "Can you look up Apple's price hike for the MacBook Neo and iPad?"
        LR(Set1)=0.7653, ESA=0.5424. LR ≥ 0.60 → gate fires.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.5424,
            "lookup_request":         0.7653,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool(
            "can you look up apple's price hike for the macbook neo and ipad?"
        )
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_cat_c2_still_fires_via_lr_set1(self):
        """
        "Can you look up their next-generation in-house Microsoft AI models?"
        LR(Set1)=0.6522, ESA=0.5785. LR ≥ 0.60 → gate fires.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.5785,
            "lookup_request":         0.6522,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool(
            "can you look up their next-generation in-house microsoft ai models?"
        )
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_cat_c3_still_fires_via_lr_set1(self):
        """
        "Can you look up Microsoft's next-generation in-house AI models?"
        LR(Set1)=0.6409, ESA=0.5735. LR ≥ 0.60 → gate fires.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.5735,
            "lookup_request":         0.6409,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool(
            "can you look up microsoft's next-generation in-house ai models?"
        )
        assert result is not None
        assert "web_search" in result.tools_to_call

    # ------------------------------------------------------------------
    # (b) Cat D — 2 utterances Set 1 fixes (LR now < 0.60 AND ESA < 0.72)
    # ------------------------------------------------------------------

    def test_can_you_help_does_not_fire_with_set1(self):
        """
        "Can you help?" — LR(Set1)=0.5901, ESA=0.5810. Both below thresholds.
        Nearest miss in the fixed-8 group (0.0099 below LR threshold).
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.5810,
            "lookup_request":         0.5901,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you help?")
        assert result is None

    def test_trip_to_japan_does_not_fire_with_set1(self):
        """
        "Would you help me plan a trip to Japan?" — LR(Set1)=0.4869, ESA=0.4735.
        Both well below thresholds — generic-domain domain item.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.4735,
            "lookup_request":         0.4869,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("would you help me plan a trip to japan?")
        assert result is None

    # ------------------------------------------------------------------
    # (c) ESA positive path still fires at the new 0.72 threshold
    # ------------------------------------------------------------------

    def test_esa_positive_still_fires_at_new_threshold(self):
        """
        A genuine explicit_search_action utterance with ESA score well above 0.72
        still triggers the gate at the raised threshold.
        LR injected below 0.60 to isolate the ESA path.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.85,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you search the web for the latest apple news")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_esa_score_just_above_new_threshold_fires(self):
        """explicit_search_action at 0.73 (just above the new 0.72 threshold) must fire."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.73,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("go search for that")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_esa_score_at_old_threshold_no_longer_fires(self):
        """
        explicit_search_action at 0.69 (above the old 0.68 but below the new 0.72)
        must NOT fire the gate after the threshold raise.
        The two ESA-floor Cat D items ("Would you look at this?" ESA=0.6990,
        "Will you look into this?" ESA=0.6874) fall in this range and would
        have fired before the 2026-06-28 change.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.69,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("would you look at this")
        assert result is None
