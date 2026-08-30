"""
PromptBuilder unit tests — persona injection and slot ordering.

Covers:
  PB-A — No persona: system message is identity constant only
  PB-B — Persona present: appended to system message; absent from user message
  PB-C — Persona truncation at 500-token ceiling
  PB-D — Slot ordering with all slots populated
"""

from datetime import datetime, timezone

import pytest

from localist.prompt_builder import (
    PromptBuilder,
    Turn,
    EpisodeBullet,
    UserProfileFact,
    RagSource,
    SessionFile,
    ToolResult,
    GraphLinkEntry,
    GraphQueryResult,
    GraphNeighborhood,
    WorkingMemoryState,
)

# Fixed datetime used across tests that don't specifically exercise
# [CURRENT DATETIME] rendering — keeps assertions deterministic.
_FIXED_DT = datetime(2026, 7, 17, 10, 10, 0, tzinfo=timezone.utc)


def test_pb_a_no_persona_system_is_identity_only():
    """build() with no persona returns bare identity string as system_prompt."""
    pb = PromptBuilder()
    system_prompt, user_prompt = pb.build(instruction="hello", current_datetime=_FIXED_DT)

    assert system_prompt == pb._system_message()
    assert "[INSTRUCTION]" in user_prompt
    assert "hello" in user_prompt


def test_pb_b_persona_appended_to_system_not_in_user():
    """Persona is appended to system_prompt and does not appear in user_prompt."""
    pb = PromptBuilder()
    persona = "I am LORA, your assistant."
    system_prompt, user_prompt = pb.build(instruction="hello", persona=persona, current_datetime=_FIXED_DT)

    assert system_prompt == pb._system_message() + "\n\n" + persona
    assert "[INSTRUCTION]" in user_prompt
    assert "I am LORA" not in user_prompt


def test_pb_c_persona_truncated_at_ceiling():
    """Persona longer than 500 tokens (2000 chars) is hard-truncated."""
    pb = PromptBuilder()
    long_persona = "word " * 600   # 3000 chars >> 2000-char (500-token) ceiling
    system_prompt, _ = pb.build(instruction="x", persona=long_persona, current_datetime=_FIXED_DT)

    assert len(system_prompt) < len(pb._system_message()) + len(long_persona)
    assert "… [truncated]" in system_prompt


def test_pb_d_slot_ordering_all_slots():
    """User message slots appear in strict static-first order."""
    pb = PromptBuilder()
    _, user_prompt = pb.build(
        instruction      = "final question",
        current_datetime = _FIXED_DT,
        episodic_summary = [EpisodeBullet("pref x", "preference", 0.9)],
        rag_snippets     = [RagSource("doc.md", "some context content here")],
        tool_results     = [ToolResult("search", "q=test", "search result text")],
        working_memory   = [Turn("user", "prior turn content")],
    )

    ep_pos    = user_prompt.index("[EPISODIC MEMORY]")
    ctx_pos   = user_prompt.index("[CONTEXT]")
    tools_pos = user_prompt.index("[TOOL RESULTS]")
    wm_pos    = user_prompt.index("[WORKING MEMORY]")
    inst_pos  = user_prompt.index("[INSTRUCTION]")

    assert ep_pos < ctx_pos < tools_pos < wm_pos < inst_pos, (
        f"Slot order wrong: ep={ep_pos} ctx={ctx_pos} "
        f"tools={tools_pos} wm={wm_pos} inst={inst_pos}"
    )


def test_pb_e_build_enforces_dynamic_suffix_slot_order():
    """
    Locks the dynamic-suffix slot order per LOCALIST-Architecture.md §3.7a.

    All optional parameters populated simultaneously. Asserts:
    - [EPISODIC MEMORY] < [USER PROFILE] < [CONTEXT] < [TOOL RESULTS]
      < [WORKING MEMORY] < [INSTRUCTION] in user_prompt
    - stable-prefix boundary: persona in system_prompt, absent from user_prompt
    - identity constant present in system_prompt

    This test must fail if PromptBuilder.build()'s slots list is reordered.
    Any failure here is a deliberate contract break requiring doc + review.
    """
    pb = PromptBuilder()
    persona = "Test persona content — stable prefix fixture."
    system_prompt, user_prompt = pb.build(
        instruction      = "test instruction",
        current_datetime = _FIXED_DT,
        persona          = persona,
        episodic_summary = [EpisodeBullet("episodic fact A", "preference", 0.9)],
        profile_facts    = [UserProfileFact("profile fact B")],
        rag_snippets     = [RagSource("test-doc.md", "rag context content C")],
        tool_results     = [ToolResult("web_search", "q=test", "tool result D")],
        working_memory   = [Turn("user", "prior turn E")],
    )

    ep_pos      = user_prompt.index("[EPISODIC MEMORY]")
    profile_pos = user_prompt.index("[USER PROFILE]")
    ctx_pos     = user_prompt.index("[CONTEXT]")
    tools_pos   = user_prompt.index("[TOOL RESULTS]")
    wm_pos      = user_prompt.index("[WORKING MEMORY]")
    inst_pos    = user_prompt.index("[INSTRUCTION]")

    assert ep_pos < profile_pos < ctx_pos < tools_pos < wm_pos < inst_pos, (
        f"Dynamic-suffix slot order violated (§3.7a): "
        f"ep={ep_pos} profile={profile_pos} ctx={ctx_pos} "
        f"tools={tools_pos} wm={wm_pos} inst={inst_pos}"
    )

    # Stable-prefix / dynamic-suffix boundary: persona goes into system, not user.
    assert pb._system_message() in system_prompt
    assert persona in system_prompt
    assert persona not in user_prompt


def test_pb_f_slot3_profile_only_precedes_context():
    """
    Locks Slot 3 profile-only sub-ordering per LOCALIST-Architecture.md §3.7a.

    When episodic_summary is absent but profile_facts are present, [USER PROFILE]
    must appear before [CONTEXT] and [EPISODIC MEMORY] must be absent entirely.
    Confirms the dynamic-suffix contract holds for the profile-only routing path
    (P4/P4a turns that fire profile injection but not episodic retrieval).
    """
    pb = PromptBuilder()
    _, user_prompt = pb.build(
        instruction      = "test instruction",
        current_datetime = _FIXED_DT,
        profile_facts    = [UserProfileFact("profile fact only")],
        rag_snippets     = [RagSource("test-doc.md", "rag context content")],
    )

    assert "[EPISODIC MEMORY]" not in user_prompt, (
        "Slot 3a must be cleanly absent when episodic_summary is empty (§3.7a)"
    )

    profile_pos = user_prompt.index("[USER PROFILE]")
    ctx_pos     = user_prompt.index("[CONTEXT]")

    assert profile_pos < ctx_pos, (
        f"[USER PROFILE] must precede [CONTEXT] in dynamic suffix (§3.7a): "
        f"profile={profile_pos} ctx={ctx_pos}"
    )


# ---------------------------------------------------------------------------
# TestAssistantName — Slot 1a identity is parameterized, not a constant
# ---------------------------------------------------------------------------

class TestAssistantName:

    def test_default_name_when_omitted(self):
        """build() with no assistant_name falls back to _DEFAULT_ASSISTANT_NAME."""
        pb = PromptBuilder()
        system_prompt, _ = pb.build(instruction="hello", current_datetime=_FIXED_DT)

        assert f"You are {PromptBuilder._DEFAULT_ASSISTANT_NAME}," in system_prompt

    def test_configured_name_interpolated(self):
        """build() with assistant_name substitutes it into the identity string."""
        pb = PromptBuilder()
        system_prompt, _ = pb.build(
            instruction="hello", current_datetime=_FIXED_DT, assistant_name="Percy",
        )

        assert "You are Percy," in system_prompt
        assert PromptBuilder._DEFAULT_ASSISTANT_NAME not in system_prompt

    def test_empty_string_name_falls_back_to_default(self):
        """An empty/falsy assistant_name is treated the same as None."""
        pb = PromptBuilder()
        system_prompt, _ = pb.build(
            instruction="hello", current_datetime=_FIXED_DT, assistant_name="",
        )

        assert f"You are {PromptBuilder._DEFAULT_ASSISTANT_NAME}," in system_prompt

    def test_name_and_persona_combine_in_system_prompt(self):
        """assistant_name and persona are independent — both land in system_prompt."""
        pb = PromptBuilder()
        persona = "Speak in short sentences."
        system_prompt, _ = pb.build(
            instruction="hello", current_datetime=_FIXED_DT,
            assistant_name="Percy", persona=persona,
        )

        assert "You are Percy," in system_prompt
        assert persona in system_prompt


# ---------------------------------------------------------------------------
# TestSlotDatetime — [CURRENT DATETIME] (unnumbered, always first)
# ---------------------------------------------------------------------------

class TestSlotDatetime:
    """
    Covers the datetime-hallucination fix: a ground-truth "now" anchor so
    the model stops treating real, recent/future-dated tool results as
    training-cutoff-violating errors (see LOCALIST-Architecture.md §3.2).
    """

    def _pb(self) -> PromptBuilder:
        return PromptBuilder()

    def test_renders_iso8601_weekday_and_tz(self):
        pb = self._pb()
        dt = datetime(2026, 7, 16, 9, 30, 0, tzinfo=timezone.utc)
        result = pb._slot_datetime(dt)

        assert result.startswith("[CURRENT DATETIME]\n")
        assert "2026-07-16T09:30:00+00:00" in result
        assert "Thursday" in result   # 2026-07-16 is a Thursday
        assert "UTC" in result

    def test_naive_datetime_omits_tz_label_cleanly(self):
        """No tzinfo → tzname() is None → no dangling ", " before the paren close."""
        pb = self._pb()
        dt = datetime(2026, 7, 16, 9, 30, 0)   # naive, no tzinfo
        result = pb._slot_datetime(dt)

        assert "(Thursday)" in result
        assert ", )" not in result

    def test_carries_trust_directive(self):
        pb = self._pb()
        result = pb._slot_datetime(_FIXED_DT)
        assert "ground truth" in result
        assert "not errors" in result

    def test_always_present_and_first_slot_in_user_message(self):
        """Unlike every other slot, [CURRENT DATETIME] is unconditional —
        it renders even when every other optional slot is absent, and it
        is always the first slot in the assembled user message."""
        pb = self._pb()
        _, user_prompt = pb.build(instruction="hello", current_datetime=_FIXED_DT)

        assert user_prompt.startswith("[CURRENT DATETIME]")
        dt_pos   = user_prompt.index("[CURRENT DATETIME]")
        inst_pos = user_prompt.index("[INSTRUCTION]")
        assert dt_pos < inst_pos

    def test_precedes_session_files_and_all_dynamic_suffix_slots(self):
        pb = self._pb()
        _, user_prompt = pb.build(
            instruction      = "test",
            current_datetime = _FIXED_DT,
            session_files    = [SessionFile("notes.md", "file content")],
            episodic_summary = [EpisodeBullet("fact", "preference", 0.9)],
            rag_snippets     = [RagSource("doc.md", "context")],
            tool_results     = [ToolResult("search", "q=x", "result")],
            working_memory   = [Turn("user", "prior turn")],
        )

        dt_pos    = user_prompt.index("[CURRENT DATETIME]")
        sf_pos    = user_prompt.index("[SESSION FILES]")
        ep_pos    = user_prompt.index("[EPISODIC MEMORY]")
        ctx_pos   = user_prompt.index("[CONTEXT]")
        tools_pos = user_prompt.index("[TOOL RESULTS]")
        wm_pos    = user_prompt.index("[WORKING MEMORY]")
        inst_pos  = user_prompt.index("[INSTRUCTION]")

        assert dt_pos < sf_pos < ep_pos < ctx_pos < tools_pos < wm_pos < inst_pos, (
            f"dt={dt_pos} sf={sf_pos} ep={ep_pos} ctx={ctx_pos} "
            f"tools={tools_pos} wm={wm_pos} inst={inst_pos}"
        )

    def test_not_memoized_across_build_calls(self):
        """PromptBuilder is documented stateless (class docstring). Two
        build() calls with different current_datetime values on the same
        instance must not reuse the first call's rendered slot."""
        pb = self._pb()
        dt1 = datetime(2026, 7, 17, 8, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2026, 7, 17, 20, 0, 0, tzinfo=timezone.utc)

        _, prompt_1 = pb.build(instruction="hello", current_datetime=dt1)
        _, prompt_2 = pb.build(instruction="hello", current_datetime=dt2)

        slot_1 = prompt_1.split("\n\n")[0]
        slot_2 = prompt_2.split("\n\n")[0]
        assert slot_1 != slot_2
        assert "08:00:00" in slot_1
        assert "20:00:00" in slot_2

    def test_recent_tool_result_does_not_read_as_an_error(self):
        """
        Regression guard mirroring the live TSM-earnings case: a tool
        result dated on/after the current datetime must sit alongside a
        [CURRENT DATETIME] slot that explicitly tells the model such dates
        are not errors, rather than the model being left to guess "now"
        from its training cutoff alone.
        """
        pb = self._pb()
        current = datetime(2026, 7, 16, 9, 0, 0, tzinfo=timezone.utc)
        _, user_prompt = pb.build(
            instruction      = "What were TSM's latest earnings?",
            current_datetime = current,
            tool_results      = [
                ToolResult(
                    "web_search",
                    "q=TSM earnings",
                    "TSM reported Q2 earnings on 2026-07-16.",
                ),
            ],
        )

        dt_pos    = user_prompt.index("[CURRENT DATETIME]")
        tools_pos = user_prompt.index("[TOOL RESULTS]")
        assert dt_pos < tools_pos, (
            "The trust-hierarchy anchor must precede tool results so the "
            "model reads ground-truth 'now' before evaluating their dates."
        )
        assert "not errors" in user_prompt
        assert "2026-07-16" in user_prompt


# ---------------------------------------------------------------------------
# TestSlotGraph — Slot 5b [GRAPH RESULT]
# ---------------------------------------------------------------------------

class TestSlotGraph:

    def _pb(self) -> PromptBuilder:
        return PromptBuilder()

    # 1. graph_result=None → slot returns ""
    def test_none_returns_empty(self):
        pb = self._pb()
        assert pb._slot_graph(None) == ""

    # 2. Incoming, populated — exact format match
    def test_incoming_populated(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="incoming",
            page_name="localist-software-stack",
            links=[
                GraphLinkEntry("how-localist-works", True),
                GraphLinkEntry("localist-master-project-outline", True),
            ],
        )
        expected = (
            "[GRAPH RESULT]\n"
            "Pages linking to localist-software-stack:\n"
            "- how-localist-works\n"
            "- localist-master-project-outline"
        )
        assert pb._slot_graph(gr) == expected

    # 3. Incoming, zero results — exact format match
    def test_incoming_empty(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="incoming",
            page_name="lora-persona",
            links=[],
        )
        expected = "[GRAPH RESULT]\nNo pages link to lora-persona."
        assert pb._slot_graph(gr) == expected

    # 4. Outgoing, all resolved — no unresolved / "also references" section
    def test_outgoing_all_resolved(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="outgoing",
            page_name="localist-build-order",
            links=[GraphLinkEntry("localist-master-project-outline", True)],
        )
        result = pb._slot_graph(gr)
        assert result == (
            "[GRAPH RESULT]\n"
            "localist-build-order links to:\n"
            "- localist-master-project-outline"
        )
        assert "also references" not in result
        assert "does not exist" not in result

    # 5. Outgoing, all unresolved — "references" WITHOUT "also"
    def test_outgoing_all_unresolved(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="outgoing",
            page_name="localist-build-order",
            links=[GraphLinkEntry("Localist Software Stack Overview", False)],
        )
        result = pb._slot_graph(gr)
        assert result == (
            "[GRAPH RESULT]\n"
            "localist-build-order references a page that does not exist:\n"
            '- "Localist Software Stack Overview" (no matching page found)'
        )
        assert "also references" not in result

    # 6. Outgoing, mixed — exact format match including "also" and blank line
    def test_outgoing_mixed(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="outgoing",
            page_name="localist-build-order",
            links=[
                GraphLinkEntry("localist-master-project-outline", True),
                GraphLinkEntry("Localist Software Stack Overview", False),
            ],
        )
        expected = (
            "[GRAPH RESULT]\n"
            "localist-build-order links to:\n"
            "- localist-master-project-outline\n"
            "\n"
            "localist-build-order also references a page that does not exist:\n"
            '- "Localist Software Stack Overview" (no matching page found)'
        )
        assert pb._slot_graph(gr) == expected

    # 7. Outgoing, zero results — exact format match
    def test_outgoing_empty(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="outgoing",
            page_name="lora-persona",
            links=[],
        )
        expected = "[GRAPH RESULT]\nlora-persona does not link to any other pages."
        assert pb._slot_graph(gr) == expected

    # 8. build() end-to-end: [GRAPH RESULT] positioned after [TOOL RESULTS]
    #    and before [WORKING MEMORY]
    def test_build_slot_order_with_graph(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="incoming",
            page_name="localist-software-stack",
            links=[GraphLinkEntry("how-localist-works", True)],
        )
        _, user_prompt = pb.build(
            instruction      = "test",
            current_datetime = _FIXED_DT,
            tool_results     = [ToolResult("search", "q=wiki", "some result")],
            graph_result     = gr,
            working_memory   = [Turn("user", "prior message")],
        )
        tools_pos = user_prompt.index("[TOOL RESULTS]")
        graph_pos = user_prompt.index("[GRAPH RESULT]")
        wm_pos    = user_prompt.index("[WORKING MEMORY]")
        assert tools_pos < graph_pos < wm_pos, (
            f"Slot 5b order wrong: tools={tools_pos} graph={graph_pos} wm={wm_pos}"
        )

    # 9. build() with graph_result=None and all other optionals absent →
    #    no [GRAPH RESULT] in output, no stray separators
    def test_build_none_graph_no_stray_output(self):
        pb = self._pb()
        _, user_prompt = pb.build(instruction="hello", current_datetime=_FIXED_DT)
        assert "[GRAPH RESULT]" not in user_prompt
        expected_dt_slot = pb._slot_datetime(_FIXED_DT)
        assert user_prompt == expected_dt_slot + "\n\n[INSTRUCTION]\nhello"


# ---------------------------------------------------------------------------
# TestSlotGraphNeighborhood — Phase A [RELATED CONTEXT]
# (memory-graph-inference-plan) — distinct from GraphQueryResult/[GRAPH
# RESULT] above: opportunistic, clean-omission on zero edges, flat 500-char
# truncation instead of the token-ceiling convention.
# ---------------------------------------------------------------------------

class TestSlotGraphNeighborhood:

    def _pb(self) -> PromptBuilder:
        return PromptBuilder()

    def test_none_returns_empty(self):
        pb = self._pb()
        assert pb._slot_graph_neighborhood(None) == ""

    def test_zero_edges_clean_omission(self):
        # Unlike GraphQueryResult, this must NOT render "zero edges" text —
        # it should look identical to no graph data at all.
        pb = self._pb()
        gn = GraphNeighborhood(page_name="lonely-page", backlinks=[], outgoing=[])
        assert pb._slot_graph_neighborhood(gn) == ""

    def test_backlinks_only(self):
        pb = self._pb()
        gn = GraphNeighborhood(
            page_name="localist-software-stack",
            backlinks=[GraphLinkEntry("how-localist-works", True)],
            outgoing=[],
        )
        expected = (
            "[RELATED CONTEXT: localist-software-stack]\n"
            "Referenced by:\n"
            "- how-localist-works"
        )
        assert pb._slot_graph_neighborhood(gn) == expected

    def test_outgoing_only(self):
        pb = self._pb()
        gn = GraphNeighborhood(
            page_name="localist-build-order",
            backlinks=[],
            outgoing=[GraphLinkEntry("localist-master-project-outline", True)],
        )
        expected = (
            "[RELATED CONTEXT: localist-build-order]\n"
            "Links to:\n"
            "- localist-master-project-outline"
        )
        assert pb._slot_graph_neighborhood(gn) == expected

    def test_both_backlinks_and_outgoing(self):
        pb = self._pb()
        gn = GraphNeighborhood(
            page_name="how-localist-works",
            backlinks=[GraphLinkEntry("localist-software-stack", True)],
            outgoing=[GraphLinkEntry("localist-master-project-outline", True)],
        )
        expected = (
            "[RELATED CONTEXT: how-localist-works]\n"
            "Referenced by:\n"
            "- localist-software-stack\n"
            "\n"
            "Links to:\n"
            "- localist-master-project-outline"
        )
        assert pb._slot_graph_neighborhood(gn) == expected

    def test_truncated_at_flat_500_chars_not_tokens(self):
        # Deliberately a flat char slice (index_document()'s convention),
        # not _truncate_to_tokens()'s word-boundary + "…[truncated]" scheme
        # — no ellipsis suffix, cut can land mid-word.
        pb = self._pb()
        gn = GraphNeighborhood(
            page_name="big-page",
            backlinks=[GraphLinkEntry(f"backlink-{i}", True) for i in range(100)],
            outgoing=[],
        )
        result = pb._slot_graph_neighborhood(gn)
        assert len(result) == 500
        assert "… [truncated]" not in result

    def test_build_slot_order_after_context_before_tools(self):
        pb = self._pb()
        gn = GraphNeighborhood(
            page_name="localist-software-stack",
            backlinks=[GraphLinkEntry("how-localist-works", True)],
            outgoing=[],
        )
        _, user_prompt = pb.build(
            instruction        = "test",
            current_datetime   = _FIXED_DT,
            rag_snippets       = [RagSource(path="wiki/localist-software-stack.md", content="body text")],
            tool_results       = [ToolResult("search", "q=wiki", "some result")],
            graph_neighborhood = gn,
        )
        context_pos = user_prompt.index("[CONTEXT]")
        related_pos = user_prompt.index("[RELATED CONTEXT")
        tools_pos   = user_prompt.index("[TOOL RESULTS]")
        assert context_pos < related_pos < tools_pos

    def test_build_none_neighborhood_no_stray_output(self):
        pb = self._pb()
        _, user_prompt = pb.build(instruction="hello", current_datetime=_FIXED_DT)
        assert "[RELATED CONTEXT" not in user_prompt


# ---------------------------------------------------------------------------
# TestSlot6AWorkingState — Slot 6A [WORKING STATE]
# ---------------------------------------------------------------------------

class TestSlot6AWorkingState:

    def _pb(self) -> PromptBuilder:
        return PromptBuilder()

    # 1. None state → clean omission; build() output byte-identical to no-arg baseline
    def test_none_state_clean_omission_regression_guard(self):
        pb = self._pb()
        baseline_sys, baseline_user = pb.build(instruction="test", current_datetime=_FIXED_DT)
        sys_with_none, user_with_none = pb.build(instruction="test", working_state=None, current_datetime=_FIXED_DT)
        assert sys_with_none == baseline_sys
        assert user_with_none == baseline_user
        assert "[WORKING STATE]" not in user_with_none

    # 2. Empty WorkingMemoryState → clean omission (both fields falsy)
    def test_empty_state_clean_omission(self):
        pb = self._pb()
        _, user_prompt = pb.build(
            instruction      = "test",
            current_datetime = _FIXED_DT,
            working_state    = WorkingMemoryState(),
        )
        assert "[WORKING STATE]" not in user_prompt

    # 3. current_project set, active_artifacts empty → only current_project line
    def test_current_project_only(self):
        pb = self._pb()
        state = WorkingMemoryState(current_project="localist-v2")
        result = pb._slot6a_working_state(state)
        assert result == "[WORKING STATE]\ncurrent_project: localist-v2"
        assert "active_artifacts" not in result

    # 4. active_artifacts set, current_project None → only active_artifacts line
    def test_active_artifacts_only(self):
        pb = self._pb()
        state = WorkingMemoryState(active_artifacts=["wiki/lora.md", "wiki/planner.md"])
        result = pb._slot6a_working_state(state)
        assert result == (
            "[WORKING STATE]\n"
            "active_artifacts: wiki/lora.md, wiki/planner.md"
        )
        assert "current_project" not in result

    # 5. Both fields set → both lines render in documented order
    def test_both_fields_render_in_order(self):
        pb = self._pb()
        state = WorkingMemoryState(
            current_project  = "localist-v2",
            active_artifacts = ["wiki/lora.md", "wiki/planner.md"],
        )
        result = pb._slot6a_working_state(state)
        assert result == (
            "[WORKING STATE]\n"
            "current_project: localist-v2\n"
            "active_artifacts: wiki/lora.md, wiki/planner.md"
        )
        assert result.index("current_project") < result.index("active_artifacts")

    # 6. Ceiling enforcement: active_artifacts truncated from the end
    def test_ceiling_truncates_artifacts_from_end(self):
        pb = self._pb()
        # Each artifact is ~20 chars; 30 artifacts × ~20 chars >> 400 chars (100-token ceiling)
        artifacts = [f"wiki/article-{i:04d}.md" for i in range(30)]
        state = WorkingMemoryState(
            current_project  = "localist-v2",
            active_artifacts = artifacts,
        )
        result = pb._slot6a_working_state(state)
        assert "[WORKING STATE]" in result
        assert "current_project: localist-v2" in result
        # Must be within ceiling
        assert pb._estimate_tokens(result) <= pb._CEIL_WORKING_STATE
        # Last artifact should be absent (truncated from end)
        assert "wiki/article-0029.md" not in result
        # First artifact should still be present (kept from front)
        assert "wiki/article-0000.md" in result

    # 7. Ordering: [WORKING STATE] after [TOOL RESULTS] and before [WORKING MEMORY]
    def test_build_slot_order_with_working_state(self):
        pb = self._pb()
        _, user_prompt = pb.build(
            instruction      = "test",
            current_datetime = _FIXED_DT,
            tool_results     = [ToolResult("search", "q=test", "some result")],
            working_state = WorkingMemoryState(
                current_project  = "localist-v2",
                active_artifacts = ["wiki/lora.md"],
            ),
            working_memory = [Turn("user", "prior turn")],
        )
        tools_pos = user_prompt.index("[TOOL RESULTS]")
        ws_pos    = user_prompt.index("[WORKING STATE]")
        wm_pos    = user_prompt.index("[WORKING MEMORY]")
        assert tools_pos < ws_pos < wm_pos, (
            f"Slot 6A order wrong: tools={tools_pos} ws={ws_pos} wm={wm_pos}"
        )


class TestEmitStructuredWorkingMemory:
    """
    Opt-in `emit_structured_working_memory=True` (default False): Slot 6
    ([WORKING MEMORY]) is withheld from user_prompt and its trimmed turns
    are returned as a third tuple element instead — additive, not a
    replacement for the default flattened-text path (see the ~8 other
    tests in this file and test_integration_phase7.py/test_controller_
    phase4.py/test_tool_dispatcher_phase6.py/test_warmup_hook.py that call
    build() without this flag and must keep asserting on flattened text
    unmodified).
    """

    def _pb(self) -> PromptBuilder:
        return PromptBuilder()

    def test_default_false_still_returns_2_tuple_with_flattened_text(self):
        """Sanity check: omitting the flag reproduces the pre-existing shape."""
        pb = self._pb()
        result = pb.build(
            instruction      = "hello",
            current_datetime = _FIXED_DT,
            working_memory   = [Turn("user", "prior turn content")],
        )
        assert len(result) == 2
        system_prompt, user_prompt = result
        assert "[WORKING MEMORY]" in user_prompt
        assert "prior turn content" in user_prompt

    def test_true_omits_flattened_slot_and_returns_3_tuple(self):
        pb = self._pb()
        turns = [
            Turn("user", "what is LORA?"),
            Turn("assistant", "a local research assistant."),
        ]
        system_prompt, user_prompt, working_memory_turns = pb.build(
            instruction      = "tell me more",
            current_datetime = _FIXED_DT,
            working_memory   = turns,
            emit_structured_working_memory = True,
        )
        assert "[WORKING MEMORY]" not in user_prompt
        assert "what is LORA?" not in user_prompt
        assert "a local research assistant." not in user_prompt
        # Everything else (e.g. the instruction) is still present as normal.
        assert "[INSTRUCTION]" in user_prompt
        assert "tell me more" in user_prompt

        assert working_memory_turns == turns

    def test_true_preserves_chronological_order(self):
        pb = self._pb()
        turns = [Turn("user", f"turn-{i}") for i in range(5)]
        _, _, working_memory_turns = pb.build(
            instruction      = "x",
            current_datetime = _FIXED_DT,
            working_memory   = turns,
            emit_structured_working_memory = True,
        )
        assert [t.content for t in working_memory_turns] == [
            "turn-0", "turn-1", "turn-2", "turn-3", "turn-4",
        ]

    def test_true_trims_oldest_first_against_ceiling(self):
        """
        Same drop-oldest-first ceiling enforcement as the flattened path
        (_slot6_working_memory), just returned as Turn objects instead of
        rendered text. Mirrors test_integration_phase7.py's ceiling-trim
        scenario but asserts on the structured list, not substring text.
        """
        pb = self._pb()
        # Each turn line is comfortably over 100 chars once formatted
        # ("Turn -N [user]: " + padding), so a tight ceiling forces drops.
        padding = "x" * 100
        turns = [Turn("user", f"turn-{i}-{padding}") for i in range(10)]

        _, user_prompt, working_memory_turns = pb.build(
            instruction             = "x",
            current_datetime        = _FIXED_DT,
            working_memory          = turns,
            working_memory_ceiling  = 50,   # 200 chars — only the newest turns fit
            emit_structured_working_memory = True,
        )

        assert "[WORKING MEMORY]" not in user_prompt
        assert len(working_memory_turns) < len(turns)
        # Oldest dropped first: whatever survives must be a chronological
        # (contiguous, order-preserved) suffix of the original list ending
        # at the newest turn.
        assert working_memory_turns == turns[-len(working_memory_turns):]
        assert turns[-1] in working_memory_turns   # newest always survives
        assert turns[0] not in working_memory_turns   # oldest dropped first

    def test_true_with_no_turns_returns_empty_list(self):
        pb = self._pb()
        _, user_prompt, working_memory_turns = pb.build(
            instruction      = "x",
            current_datetime = _FIXED_DT,
            working_memory   = None,
            emit_structured_working_memory = True,
        )
        assert "[WORKING MEMORY]" not in user_prompt
        assert working_memory_turns == []

    def test_true_matches_flattened_path_trim_count(self):
        """
        The structured path and the flattened path must trim identically —
        same ceiling, same survivors — since both delegate to
        _trim_working_memory() internally.
        """
        pb = self._pb()
        padding = "x" * 100
        turns = [Turn("user", f"turn-{i}-{padding}") for i in range(10)]

        _, flattened_prompt = pb.build(
            instruction            = "x",
            current_datetime       = _FIXED_DT,
            working_memory         = turns,
            working_memory_ceiling = 50,
        )
        _, _, working_memory_turns = pb.build(
            instruction            = "x",
            current_datetime       = _FIXED_DT,
            working_memory         = turns,
            working_memory_ceiling = 50,
            emit_structured_working_memory = True,
        )

        surviving_count = sum(1 for i in range(len(turns)) if f"turn-{i}-" in flattened_prompt)
        assert len(working_memory_turns) == surviving_count


class TestSlotSessionFilesSource:
    """SessionFile.source distinguishes an uploaded file from a pinned wiki
    page — a wiki_pin's opening label gets "(from the vault)" appended so
    the model can cite it distinctly, per LORA's honor-code citation style."""

    def _pb(self) -> PromptBuilder:
        return PromptBuilder()

    def test_wiki_pin_label_says_from_the_vault(self):
        pb = self._pb()
        result = pb._slot_session_files(
            [SessionFile("localist-software-stack.md", "content", source="wiki_pin")]
        )
        assert "--- localist-software-stack.md (from the vault) ---" in result
        assert "--- end localist-software-stack.md ---" in result

    def test_default_upload_source_has_no_vault_label(self):
        pb = self._pb()
        result = pb._slot_session_files(
            [SessionFile("notes.md", "content")]
        )
        assert "--- notes.md ---" in result
        assert "from the vault" not in result

    def test_mixed_sources_only_label_the_pin(self):
        pb = self._pb()
        result = pb._slot_session_files([
            SessionFile("notes.md", "a"),
            SessionFile("lora-persona.md", "b", source="wiki_pin"),
        ])
        assert "--- notes.md ---" in result
        assert "--- lora-persona.md (from the vault) ---" in result
