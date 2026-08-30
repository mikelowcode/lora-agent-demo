"""
P6-fallthrough tool-need classifier — feature-flagged, shadow-first.

Covers:
  - Flag off (default): _classify_tool_fallback is never invoked, zero
    behavior/latency change.
  - Gate 1: prior-turn tools_fired non-empty skips the classifier call.
  - Gate 2: missing memory_manager or runtime skips the classifier call.
  - Parser: valid tool name -> exact value; garbage/prose/empty -> None.
  - Shadow mode: classifier runs (and logs) but RoutingPlan is unaffected.
  - Active mode: classifier result sets tools_to_call + tool_signal_source;
    a genuine P3 keyword match never even reaches the classifier method.

All tests construct their own Planner instance and manage the
LOCALIST_TOOL_FALLBACK_CLASSIFIER env var via monkeypatch, so no state
leaks between tests or into the rest of the suite.
"""

from unittest.mock import MagicMock, patch

import pytest

from localist.planner import Planner, RoutingPlan

_ENV_VAR = "LOCALIST_TOOL_FALLBACK_CLASSIFIER"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_runtime(infer_return: str = "none"):
    rt = MagicMock()
    rt.infer.return_value = infer_return
    rt.embed.return_value = [0.0] * 768
    return rt


def make_memory_manager():
    """
    A minimal stand-in for Gate 2 (`is None` check only). query_corpus must
    return an empty list, not a bare MagicMock, so P3b/P4's corpus-score
    paths cleanly miss and fall through instead of raising on comparison.
    """
    mm = MagicMock()
    mm.query_corpus.return_value = []
    return mm


FALLTHROUGH_INSTRUCTION = "What is 2+2?"


# ---------------------------------------------------------------------------
# Flag off (default) — method never called, zero behavior change
# ---------------------------------------------------------------------------

class TestFlagOff:

    def test_default_env_unset_is_off(self, monkeypatch):
        monkeypatch.delenv(_ENV_VAR, raising=False)
        p = Planner(runtime=make_runtime(), memory_manager=make_memory_manager())
        assert p._tool_fallback_mode() == "off"

    def test_off_never_calls_classifier(self, monkeypatch):
        monkeypatch.delenv(_ENV_VAR, raising=False)
        p = Planner(runtime=make_runtime(), memory_manager=make_memory_manager())
        with patch.object(
            Planner, "_classify_tool_fallback", wraps=p._classify_tool_fallback
        ) as spy:
            plan = p.route(FALLTHROUGH_INSTRUCTION, context={})
        spy.assert_not_called()
        assert plan.priority == 6
        assert plan.tools_to_call == []
        assert plan.tool_signal_source is None

    def test_unrecognized_env_value_treated_as_off(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "banana")
        p = Planner(runtime=make_runtime(), memory_manager=make_memory_manager())
        assert p._tool_fallback_mode() == "off"


# ---------------------------------------------------------------------------
# Gate 1 — prior-turn tools_fired
# ---------------------------------------------------------------------------

class TestGate1PriorTurnToolsFired:

    def test_gate1_skips_when_prior_turn_fired_a_tool(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime(infer_return="web_search")
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        # Mocked prior-turn metadata fixture: previous turn's plan fired web_search.
        p.record_tools_fired(["web_search"])

        result = p._classify_tool_fallback(FALLTHROUGH_INSTRUCTION, context={})

        assert result is None
        rt.infer.assert_not_called()

    def test_gate1_proceeds_when_prior_turn_fired_nothing(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime(infer_return="web_search")
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        p.record_tools_fired([])  # prior turn fired no tools

        result = p._classify_tool_fallback(FALLTHROUGH_INSTRUCTION, context={})

        assert result == "web_search"
        rt.infer.assert_called_once()

    def test_route_populates_gate1_state_for_next_turn(self, monkeypatch):
        """route() itself records tools_to_call, so a P3 turn followed by a
        fallthrough turn skips the classifier without any external wiring."""
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime(infer_return="web_search")
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        # Turn 1 — P3 keyword match fires web_search. (Not "...latest
        # news?" — "news" now routes to news_search via P3-news, which
        # runs before P3; see test_priority3_news.py.)
        p.route("What is the latest price?", context={})
        assert p._last_turn_tools_fired == ["web_search"]

        # Turn 2 — fallthrough instruction; the classifier method is invoked
        # (mode="active" is not "off"), but Gate 1 inside it must skip the
        # model call before runtime.infer is ever reached.
        plan = p.route(FALLTHROUGH_INSTRUCTION, context={})
        rt.infer.assert_not_called()
        assert plan.tools_to_call == []


# ---------------------------------------------------------------------------
# Gate 2 — missing memory_manager / runtime
# ---------------------------------------------------------------------------

class TestGate2MissingDependencies:

    def test_gate2_skips_when_memory_manager_missing(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime(infer_return="web_search")
        p = Planner(runtime=rt, memory_manager=None)

        result = p._classify_tool_fallback(FALLTHROUGH_INSTRUCTION, context={})

        assert result is None
        rt.infer.assert_not_called()

    def test_gate2_skips_when_runtime_missing(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "active")
        p = Planner(runtime=None, memory_manager=make_memory_manager())

        result = p._classify_tool_fallback(FALLTHROUGH_INSTRUCTION, context={})

        assert result is None


# ---------------------------------------------------------------------------
# Parser — strict exact-match against the known tool-name set
# ---------------------------------------------------------------------------

class TestParser:

    @pytest.mark.parametrize("raw,expected", [
        ("web_search", "web_search"),
        ("file_op", "file_op"),
        ("url_fetch", "url_fetch"),
        ("  Web_Search  ", "web_search"),
        ("FILE_OP", "file_op"),
        ("none", None),
        ("None", None),
        ("", None),
        ("I think you should search the web for this.", None),
        ("web_search please", None),
        ("garbage_tool_name", None),
    ])
    def test_parse_variants(self, monkeypatch, raw, expected):
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime(infer_return=raw)
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        result = p._classify_tool_fallback(FALLTHROUGH_INSTRUCTION, context={})

        assert result == expected

    def test_runtime_infer_exception_parses_to_none(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime()
        rt.infer.side_effect = RuntimeError("backend unavailable")
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        result = p._classify_tool_fallback(FALLTHROUGH_INSTRUCTION, context={})

        assert result is None


# ---------------------------------------------------------------------------
# Shadow mode — runs and logs, RoutingPlan unaffected
# ---------------------------------------------------------------------------

class TestShadowMode:

    def test_shadow_runs_classifier_but_discards_result(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "shadow")
        rt = make_runtime(infer_return="web_search")
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        plan = p.route(FALLTHROUGH_INSTRUCTION, context={})

        rt.infer.assert_called_once()  # classifier ran
        assert plan.tools_to_call == []          # identical to flag-off behavior
        assert plan.tool_signal_source is None
        assert plan.priority == 6

    def test_shadow_identical_to_off_for_same_instruction(self, monkeypatch):
        rt_off = make_runtime(infer_return="web_search")
        p_off = Planner(runtime=rt_off, memory_manager=make_memory_manager())
        monkeypatch.delenv(_ENV_VAR, raising=False)
        plan_off = p_off.route(FALLTHROUGH_INSTRUCTION, context={})

        rt_shadow = make_runtime(infer_return="web_search")
        p_shadow = Planner(runtime=rt_shadow, memory_manager=make_memory_manager())
        monkeypatch.setenv(_ENV_VAR, "shadow")
        plan_shadow = p_shadow.route(FALLTHROUGH_INSTRUCTION, context={})

        assert plan_off.tools_to_call == plan_shadow.tools_to_call == []
        assert plan_off.tool_signal_source == plan_shadow.tool_signal_source is None
        assert plan_off.priority == plan_shadow.priority == 6


# ---------------------------------------------------------------------------
# Active mode — classifier result sets tools_to_call + tool_signal_source
# ---------------------------------------------------------------------------

class TestActiveMode:

    def test_active_sets_tools_to_call_and_signal_source(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime(infer_return="url_fetch")
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        plan = p.route(FALLTHROUGH_INSTRUCTION, context={})

        assert plan.tools_to_call == ["url_fetch"]
        assert plan.tool_signal_source == "classifier_fallback"
        assert plan.compound is True
        assert plan.agent == "conversational_agent"

    def test_active_no_tool_falls_through_to_p6(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime(infer_return="none")
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        plan = p.route(FALLTHROUGH_INSTRUCTION, context={})

        assert plan.tools_to_call == []
        assert plan.tool_signal_source is None
        assert plan.priority == 6

    def test_p3_keyword_match_never_reaches_classifier(self, monkeypatch):
        """A genuine P3 keyword match must short-circuit route() before the
        fallthrough classifier is even reached — not merely produce the
        same output as if it had been consulted."""
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime(infer_return="file_op")  # would answer differently if called
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        with patch.object(
            Planner, "_classify_tool_fallback", wraps=p._classify_tool_fallback
        ) as spy:
            plan = p.route("What are the latest oMLX changes?", context={})

        spy.assert_not_called()
        assert plan.tools_to_call == ["web_search"]
        assert plan.tool_signal_source == "keyword"
        assert plan.priority == 3

    def test_p3_keyword_match_not_overridden_by_fallback_output(self, monkeypatch):
        """Even if the classifier would have answered differently, a P3
        match's tool selection and provenance must be untouched."""
        monkeypatch.setenv(_ENV_VAR, "active")
        rt = make_runtime(infer_return="url_fetch")
        p = Planner(runtime=rt, memory_manager=make_memory_manager())

        plan = p.route("read the file notes.md", context={})

        assert plan.tools_to_call == ["file_op"]
        assert plan.tool_signal_source == "keyword"
        rt.infer.assert_not_called()
