"""
Unit tests for threshold_calibration.py (PLAN_semantic_gating_calibration.md
§2) — live, in-app semantic-gating threshold calibration.

Two layers, matching the module's own structure:
  - _youden_calibrate() tested directly against hand-constructed score
    lists — no embeddings involved, so the selection math itself (Youden's
    J, the degenerate floor) is pinned down exactly.
  - calibrate_thresholds() tested end-to-end with a stub embed_fn (mirrors
    test_planner_phase3.py's _unit_vector() pattern), verifying the fixture
    battery is actually wired up correctly and that embed_fn failures/
    non-unit-length vectors are handled the way the module promises.
"""

from __future__ import annotations

import math

from localist.planner import _SEARCH_INTENT_TEMPLATES, _EPISODIC_RELEVANCE_TEMPLATES
from localist.threshold_calibration import (
    calibrate_thresholds,
    _youden_calibrate,
    _MIN_ACCEPTABLE_J,
    GateCalibration,
    CalibrationResult,
)
from localist.threshold_calibration_fixtures import (
    LR_CAT_C, RI_CAT_T, EP_POSITIVES, EP_POSITIVES_LIVE,
)


def _unit_vector(dim: int = 4) -> list[float]:
    """Same helper as test_planner_phase3.py's _unit_vector()."""
    v = 1.0 / math.sqrt(dim)
    return [v] * dim


def _orthogonal_vector(dim: int = 4) -> list[float]:
    """A vector orthogonal to _unit_vector(dim) — cosine similarity 0."""
    v = [0.0] * dim
    half = dim // 2
    for i in range(half):
        v[i] = 1.0 / math.sqrt(half)
    for i in range(half, dim):
        v[i] = -1.0 / math.sqrt(dim - half)
    return v


class TestYoudenCalibrate:
    """Direct tests of the selection math — no embeddings involved."""

    def test_clean_separation_picks_a_threshold_between_the_pools(self):
        positives = [0.90, 0.85, 0.80]
        negatives = [0.30, 0.20, 0.10]
        grid = [x / 100 for x in range(10, 96, 5)]

        gate = _youden_calibrate(positives, negatives, grid)

        assert gate.degenerate is False
        assert gate.threshold is not None
        assert 0.30 < gate.threshold <= 0.80
        assert gate.tpr_at_threshold == 1.0
        assert gate.fpr_at_threshold == 0.0
        assert gate.min_positive_score == 0.80
        assert gate.max_negative_score == 0.30
        assert gate.positive_count == 3
        assert gate.negative_count == 3

    def test_picks_the_exact_expected_threshold_on_a_known_distribution(self):
        # Constructed so exactly one grid point maximizes TPR-FPR: at 0.55,
        # TPR=1.0 (both positives >= 0.55) and FPR=0.0 (both negatives <
        # 0.55) -- J=1.0, strictly better than every other grid point.
        positives = [0.60, 0.55]
        negatives = [0.50, 0.40]
        grid = [0.40, 0.45, 0.50, 0.55, 0.60]

        gate = _youden_calibrate(positives, negatives, grid)

        assert gate.threshold == 0.55
        assert gate.degenerate is False

    def test_fully_overlapping_pools_are_degenerate(self):
        positives = [0.5, 0.5, 0.5]
        negatives = [0.5, 0.5, 0.5]
        grid = [0.3, 0.4, 0.5, 0.6, 0.7]

        gate = _youden_calibrate(positives, negatives, grid)

        assert gate.degenerate is True
        assert gate.threshold is None
        assert gate.reason is not None
        assert "below floor" in gate.reason

    def test_degenerate_floor_is_the_documented_constant(self):
        # J just under the floor -> degenerate; just at/over -> not.
        # 3 positives, 10 negatives: threshold that keeps all 3 positives
        # (TPR=1.0) and lets through enough negatives to land J below/above
        # _MIN_ACCEPTABLE_J (0.5).
        positives = [0.9, 0.9, 0.9]
        # 6/10 negatives clear 0.5 -> FPR=0.6 -> J = 1.0 - 0.6 = 0.4 < 0.5
        negatives_below = [0.9] * 6 + [0.1] * 4
        gate_below = _youden_calibrate(positives, negatives_below, [0.5])
        assert gate_below.degenerate is True

        # 4/10 negatives clear 0.5 -> FPR=0.4 -> J = 1.0 - 0.4 = 0.6 >= 0.5
        negatives_above = [0.9] * 4 + [0.1] * 6
        gate_above = _youden_calibrate(positives, negatives_above, [0.5])
        assert gate_above.degenerate is False
        assert gate_above.tpr_at_threshold - gate_above.fpr_at_threshold >= _MIN_ACCEPTABLE_J

    def test_empty_positive_pool_is_degenerate_with_explanatory_reason(self):
        gate = _youden_calibrate([], [0.5, 0.6], [0.5])
        assert gate.degenerate is True
        assert gate.threshold is None
        assert "empty positive or negative pool" in gate.reason
        assert gate.min_positive_score is None
        assert gate.max_negative_score == 0.6

    def test_empty_negative_pool_is_degenerate_with_explanatory_reason(self):
        gate = _youden_calibrate([0.5, 0.6], [], [0.5])
        assert gate.degenerate is True
        assert gate.threshold is None
        assert gate.max_negative_score is None


class TestCalibrateThresholdsIntegration:
    """
    End-to-end tests of calibrate_thresholds() against the real fixture
    battery (threshold_calibration_fixtures.py) and the real
    _SEARCH_INTENT_TEMPLATES / _EPISODIC_RELEVANCE_TEMPLATES from planner.py
    — only embed_fn itself is a stub.
    """

    def test_identical_vector_everywhere_is_fully_degenerate(self):
        # A stub with zero discriminative power (every phrase embeds
        # identically) must never produce a threshold anybody could act on.
        fixed_vec = _unit_vector(8)
        result = calibrate_thresholds(lambda text: fixed_vec)

        assert set(result.gates.keys()) == {
            "explicit_search_action", "lookup_request",
            "research_intent", "episodic_relevance",
        }
        assert result.all_degenerate is True
        assert result.thresholds == {}

    def test_all_gates_calibrate_cleanly_with_a_membership_oracle_stub(self):
        # A stub that can't rely on keyword heuristics (the battery's
        # adversarial negatives are deliberately keyword-colliding with
        # their positives -- see planner.py's D-verb-swap/D-modal-swap
        # comments) but instead "knows" which exact utterances are the
        # battery's real positives, and embeds every template phrase
        # identically to them -- everything else gets an orthogonal
        # vector. This isn't asserting semantic correctness; it's asserting
        # the plumbing (fixtures -> scores -> per-gate Youden's J) actually
        # wires all four gates together end to end when real signal exists.
        positive_phrases = (
            {utt for _cat, utt in LR_CAT_C}
            | set(RI_CAT_T)
            | set(EP_POSITIVES) | set(EP_POSITIVES_LIVE)
        )
        template_phrases = {p for phrases in _SEARCH_INTENT_TEMPLATES.values() for p in phrases}
        template_phrases |= set(_EPISODIC_RELEVANCE_TEMPLATES)
        near_set = positive_phrases | template_phrases

        near_vec = _unit_vector(8)
        far_vec = _orthogonal_vector(8)

        def embed_fn(text: str) -> list[float]:
            return near_vec if text in near_set else far_vec

        result = calibrate_thresholds(embed_fn)

        assert result.all_degenerate is False
        for name in ("explicit_search_action", "lookup_request", "research_intent", "episodic_relevance"):
            assert result.gates[name].degenerate is False, result.gates[name]
        assert set(result.thresholds) == {
            "explicit_search_action", "lookup_request", "research_intent", "episodic_relevance",
        }

    def test_embed_fn_failure_is_skipped_not_raised(self):
        calls = {"count": 0}

        def flaky_embed(text: str) -> list[float]:
            calls["count"] += 1
            if "Xbox" in text:
                raise TimeoutError("simulated backend timeout")
            return _unit_vector(8)

        # Must not raise -- a wider range of user-picked models means more
        # chances of a mid-battery failure, and the whole point of the
        # per-gate degenerate fallback is that this never crashes the
        # caller (auto-calibration on model switch, or /memory/reembed).
        result = calibrate_thresholds(flaky_embed)

        assert calls["count"] > 0
        assert isinstance(result, CalibrationResult)
        # Every gate degenerate here too (uniform vector -> zero signal),
        # but the point is it returned at all instead of propagating the
        # TimeoutError.
        assert result.all_degenerate is True

    def test_embed_fn_returning_empty_vector_is_skipped(self):
        def embed_fn(text: str) -> list[float]:
            return [] if "Xbox" in text else _unit_vector(8)

        result = calibrate_thresholds(embed_fn)
        assert isinstance(result, CalibrationResult)
        # No crash; the empty-vector phrase is just excluded from scoring.

    def test_non_unit_length_vectors_score_identically_to_unit_vectors(self):
        # calibrate_thresholds() reuses planner._cosine_similarity, which
        # normalizes internally -- a scaled (non-unit-length) embed_fn
        # output must produce the same scores/verdicts as its unit-length
        # equivalent. This is the "don't assume unit-normalized embeddings"
        # guarantee from PLAN_semantic_gating_calibration.md §2.
        near_vec = _unit_vector(8)
        far_vec = _orthogonal_vector(8)
        scaled_near = [x * 37.0 for x in near_vec]     # arbitrary non-unit scale
        scaled_far  = [x * 0.02 for x in far_vec]       # a different arbitrary scale
        lookup_signal_words = ("look up", "find", "search", "track down", "check")

        def unit_embed(text: str) -> list[float]:
            return near_vec if any(w in text.lower() for w in lookup_signal_words) else far_vec

        def scaled_embed(text: str) -> list[float]:
            return scaled_near if any(w in text.lower() for w in lookup_signal_words) else scaled_far

        result_unit = calibrate_thresholds(unit_embed)
        result_scaled = calibrate_thresholds(scaled_embed)

        for name in ("explicit_search_action", "lookup_request", "research_intent", "episodic_relevance"):
            assert result_unit.gates[name].degenerate == result_scaled.gates[name].degenerate
            assert result_unit.gates[name].threshold == result_scaled.gates[name].threshold

    def test_thresholds_property_excludes_degenerate_gates(self):
        fixed_vec = _unit_vector(8)
        result = calibrate_thresholds(lambda text: fixed_vec)
        assert all(name not in result.thresholds for name in result.gates)
