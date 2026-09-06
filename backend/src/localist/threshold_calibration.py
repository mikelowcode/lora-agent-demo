"""
Live, in-app semantic-gating threshold calibration.

Turns the manual "run a diagnostic script, eyeball a markdown report,
hand-transcribe a number into planner.py" cycle (see
diagnostics/nomic_embed_text_threshold_probe.py) into something the running
app can do for itself, for any embedding model a user picks -- not just the
one model a developer has manually measured so far.

Reuses the exact same fixture battery (threshold_calibration_fixtures.py)
and _cosine_similarity scoring as the offline diagnostic, so a live result
is directly comparable to a hand-reviewed one. The two paths must never
score differently for the same model/embed_fn -- if you change how a pool
is built or filtered here, make the equivalent change in the diagnostic
script, or vice versa.

Selection method: Youden's J statistic (TPR - FPR, maximized over a
threshold grid) -- see docs/architecture/16-runtime-backend-layer.md §16.17
and PLAN_semantic_gating_calibration.md for why this was chosen over a
zero-false-positive rule. Per-gate degenerate fallback: a gate whose best
achievable separation on this model doesn't clear _MIN_ACCEPTABLE_J comes
back marked degenerate rather than shipping an unreviewed, possibly-bad
threshold -- callers should treat a degenerate gate as if this whole
function had returned nothing for it (fall through to a lower trust tier),
not use the numeric threshold field, which is None in that case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from .planner import (
    _ALL_SEARCH_NEGATIVE_FILTERS,
    _RESEARCH_NEGATIVE_FILTER,
    _SEARCH_INTENT_TEMPLATES,
    _EPISODIC_RELEVANCE_TEMPLATES,
    _cosine_similarity,
)
from .threshold_calibration_fixtures import (
    LR_CAT_A,
    LR_CAT_C,
    LR_CAT_D,
    RI_CAT_T,
    RI_CAT_L,
    RI_CAT_L_PRICE_ADJACENT,
    RI_CAT_K,
    RI_CAT_F,
    RI_CAT_E,
    RI_CAT_G,
    EP_POSITIVES,
    EP_POSITIVES_LIVE,
    EP_NEGATIVES,
)

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], list[float]]

# Same grids as diagnostics/nomic_embed_text_threshold_probe.py, kept in
# sync deliberately -- a coarser/finer grid here would make live-calibrated
# thresholds not comparable to hand-reviewed ones measured on the same grid.
_LR_ESA_GRID: list[float] = [x / 100 for x in range(30, 91, 2)]
_RI_EP_GRID: list[float] = [x / 100 for x in range(30, 96, 2)]

# Minimum Youden's J (TPR - FPR) required at the selected threshold for a
# gate to count as cleanly calibrated. Below this, positive/negative scores
# overlap too much to trust a threshold nobody has reviewed -- the gate
# falls back to a lower trust tier instead. Deliberately model-agnostic
# (not tuned against embeddinggemma/nomic specifically) since the whole
# point of this function is to work for embedding models neither of those
# tunings ever saw.
_MIN_ACCEPTABLE_J: float = 0.5


@dataclass
class GateCalibration:
    """Calibration outcome for one of the four semantic gates."""

    threshold: float | None
    degenerate: bool
    reason: str | None
    min_positive_score: float | None
    max_negative_score: float | None
    tpr_at_threshold: float | None
    fpr_at_threshold: float | None
    positive_count: int
    negative_count: int


@dataclass
class CalibrationResult:
    """Per-gate calibration outcomes for one embed_fn/model."""

    gates: dict[str, GateCalibration] = field(default_factory=dict)

    @property
    def thresholds(self) -> dict[str, float]:
        """Only the gates that calibrated cleanly -- what a caller persists."""
        return {
            name: gate.threshold
            for name, gate in self.gates.items()
            if not gate.degenerate and gate.threshold is not None
        }

    @property
    def all_degenerate(self) -> bool:
        return all(gate.degenerate for gate in self.gates.values())


def _embed_template_group(embed_fn: EmbedFn, phrases: tuple[str, ...]) -> list[list[float]]:
    vecs: list[list[float]] = []
    for phrase in phrases:
        try:
            vec = embed_fn(phrase)
        except Exception:
            logger.warning(
                "threshold_calibration: embed_fn raised on template %r, skipping",
                phrase, exc_info=True,
            )
            continue
        if vec:
            vecs.append(vec)
    return vecs


def _score_pool(embed_fn: EmbedFn, phrases: list[str], template_vecs: list[list[float]]) -> list[float]:
    """Best cosine similarity of each phrase against template_vecs. Skips any
    phrase whose embed_fn call fails or returns an empty vector, rather than
    raising -- a wider range of user-picked models means more chances of a
    timeout/error/wrong-dimension response mid-battery."""
    if not template_vecs:
        return []
    scores: list[float] = []
    for phrase in phrases:
        try:
            vec = embed_fn(phrase)
        except Exception:
            logger.warning(
                "threshold_calibration: embed_fn raised on utterance %r, skipping",
                phrase, exc_info=True,
            )
            continue
        if not vec:
            continue
        scores.append(max(_cosine_similarity(vec, tv) for tv in template_vecs))
    return scores


def _youden_calibrate(
    positive_scores: list[float],
    negative_scores: list[float],
    grid: list[float],
) -> GateCalibration:
    if not positive_scores or not negative_scores:
        return GateCalibration(
            threshold=None,
            degenerate=True,
            reason=(
                "empty positive or negative pool (embed_fn failures thinned the "
                "battery, or this gate has no labeled positive pool to begin with)"
            ),
            min_positive_score=min(positive_scores) if positive_scores else None,
            max_negative_score=max(negative_scores) if negative_scores else None,
            tpr_at_threshold=None,
            fpr_at_threshold=None,
            positive_count=len(positive_scores),
            negative_count=len(negative_scores),
        )

    best_j = -1.0
    best_threshold: float | None = None
    best_tpr: float | None = None
    best_fpr: float | None = None
    for th in grid:
        tpr = sum(1 for s in positive_scores if s >= th) / len(positive_scores)
        fpr = sum(1 for s in negative_scores if s >= th) / len(negative_scores)
        j = tpr - fpr
        if j > best_j:
            best_j, best_threshold, best_tpr, best_fpr = j, th, tpr, fpr

    degenerate = best_j < _MIN_ACCEPTABLE_J
    return GateCalibration(
        # None when degenerate, not the raw best_threshold -- a caller must
        # be able to trust `threshold is not None` alone (matching the
        # module docstring and the empty-pool branch above), never having
        # to remember to also check `degenerate` before using the value.
        threshold=best_threshold if not degenerate else None,
        degenerate=degenerate,
        reason=None if not degenerate else f"best Youden's J ({best_j:.2f}) below floor ({_MIN_ACCEPTABLE_J})",
        min_positive_score=min(positive_scores),
        max_negative_score=max(negative_scores),
        tpr_at_threshold=best_tpr,
        fpr_at_threshold=best_fpr,
        positive_count=len(positive_scores),
        negative_count=len(negative_scores),
    )


def calibrate_thresholds(embed_fn: EmbedFn) -> CalibrationResult:
    """
    Measure Youden's-J-optimal semantic-gating thresholds for the currently
    active embedding model (whatever embed_fn wraps), using the same fixture
    battery and _cosine_similarity scoring as the hand-tuning diagnostics.

    Never raises on an embed_fn failure -- a failed call just thins the
    affected pool, which naturally tends toward a degenerate verdict for
    that gate (see _youden_calibrate) rather than crashing the caller
    (auto-calibration on model switch, or the reembed endpoint).
    """
    esa_template_vecs = _embed_template_group(embed_fn, _SEARCH_INTENT_TEMPLATES["explicit_search_action"])
    lr_template_vecs = _embed_template_group(embed_fn, _SEARCH_INTENT_TEMPLATES["lookup_request"])
    ri_template_vecs = _embed_template_group(embed_fn, _SEARCH_INTENT_TEMPLATES["research_intent"])
    ep_template_vecs = _embed_template_group(embed_fn, _EPISODIC_RELEVANCE_TEMPLATES)

    result = CalibrationResult()

    # ---- lookup_request ----
    lr_positive_phrases = [utt for _cat, utt in LR_CAT_C]
    lr_negative_phrases = [utt for _cat, utt in LR_CAT_A] + [
        utt for _cat, _domain, utt in LR_CAT_D
        if not any(neg in utt.lower() for neg in _ALL_SEARCH_NEGATIVE_FILTERS)
    ]
    lr_pos_scores = _score_pool(embed_fn, lr_positive_phrases, lr_template_vecs)
    lr_neg_scores = _score_pool(embed_fn, lr_negative_phrases, lr_template_vecs)
    result.gates["lookup_request"] = _youden_calibrate(lr_pos_scores, lr_neg_scores, _LR_ESA_GRID)

    # ---- explicit_search_action ----
    # This battery has no dedicated true-positive category for ESA -- the
    # original hand-tuning diagnostics only ever scored LR_CAT_C against ESA
    # as a regression/false-positive check (see planner.py's
    # _SEMANTIC_GATE_THRESHOLDS comment: "Cat C max ESA score = 0.58, well
    # under either threshold"), never as a required-to-fire positive set.
    # Reusing it here as ESA's positive proxy is the same framing, not a new
    # assumption: if a given model doesn't treat "look up X" phrasing as
    # ESA-positive, Youden's J will legitimately fail to clear
    # _MIN_ACCEPTABLE_J and this gate falls back to a lower trust tier --
    # the honest outcome, not a guessed threshold.
    esa_pos_scores = _score_pool(embed_fn, lr_positive_phrases, esa_template_vecs)
    esa_neg_scores = _score_pool(embed_fn, lr_negative_phrases, esa_template_vecs)
    result.gates["explicit_search_action"] = _youden_calibrate(esa_pos_scores, esa_neg_scores, _LR_ESA_GRID)

    # ---- research_intent ----
    ri_positive_phrases = [
        u for u in RI_CAT_T if not any(neg in u.lower() for neg in _RESEARCH_NEGATIVE_FILTER)
    ]
    ri_negative_phrases = [
        u for u in (RI_CAT_L + [RI_CAT_L_PRICE_ADJACENT] + RI_CAT_K + RI_CAT_F + RI_CAT_E + RI_CAT_G)
        if not any(neg in u.lower() for neg in _RESEARCH_NEGATIVE_FILTER)
    ]
    ri_pos_scores = _score_pool(embed_fn, ri_positive_phrases, ri_template_vecs)
    ri_neg_scores = _score_pool(embed_fn, ri_negative_phrases, ri_template_vecs)
    result.gates["research_intent"] = _youden_calibrate(ri_pos_scores, ri_neg_scores, _RI_EP_GRID)

    # ---- episodic_relevance ----
    ep_positive_phrases = list(EP_POSITIVES) + list(EP_POSITIVES_LIVE)
    ep_negative_phrases = list(EP_NEGATIVES)
    ep_pos_scores = _score_pool(embed_fn, ep_positive_phrases, ep_template_vecs)
    ep_neg_scores = _score_pool(embed_fn, ep_negative_phrases, ep_template_vecs)
    result.gates["episodic_relevance"] = _youden_calibrate(ep_pos_scores, ep_neg_scores, _RI_EP_GRID)

    for name, gate in result.gates.items():
        logger.info(
            "threshold_calibration: gate=%s degenerate=%s threshold=%s "
            "tpr=%s fpr=%s pos_n=%d neg_n=%d reason=%s",
            name, gate.degenerate, gate.threshold,
            gate.tpr_at_threshold, gate.fpr_at_threshold,
            gate.positive_count, gate.negative_count, gate.reason,
        )

    return result
