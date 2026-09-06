"""
Diagnostic: calibrate_lexical_fallback_thresholds.py
=====================================================
One-time, offline calibration of the model-independent lexical/BM25
fallback tier for semantic gating (PLAN_semantic_gating_calibration.md
§9). Unlike threshold_calibration.calibrate_thresholds() (live, per-
embedding-model, run automatically by the app), this tier never needs
remeasuring per model -- BM25 doesn't depend on which embedding model (if
any) is active, so it's calibrated ONCE here and the result is
hand-transcribed into planner.py's _LEXICAL_FALLBACK_THRESHOLDS constant,
the same convention _VALIDATED_MODEL_THRESHOLDS already uses.

Background: a user who never gets real semantic gating at all (a model
that calibrates fully degenerate, or a true zero-config keyword-only
install with no embedding source configured) was, before this tier, stuck
with search-intent classification (Priority 3) and episodic-relevance
detection (Priority 5) both permanently disabled for the affected gates.
This is realistically the DOMINANT experience for many end users, who are
unlikely to know they need two different Ollama models (one chat, one
embedding) loaded to get real semantic gating in the first place -- so
this tier gets the same Youden's-J rigor as the embedding-based tiers,
not a cheaper heuristic.

READ-ONLY. Does not modify planner.py -- the printed dict below is meant
to be reviewed and pasted in by hand, the same way _VALIDATED_MODEL_
THRESHOLDS' nomic-embed-text:latest entry was originally derived.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/calibrate_lexical_fallback_thresholds.py
"""

from __future__ import annotations

from localist.threshold_calibration import calibrate_lexical_thresholds


def main() -> None:
    result = calibrate_lexical_thresholds()

    print("Lexical/BM25 fallback calibration results")
    print("=" * 60)
    for name, gate in result.gates.items():
        print(f"\n{name}:")
        print(f"  threshold        = {gate.threshold}")
        print(f"  degenerate       = {gate.degenerate}")
        if gate.reason:
            print(f"  reason           = {gate.reason}")
        print(f"  tpr / fpr        = {gate.tpr_at_threshold} / {gate.fpr_at_threshold}")
        print(f"  min_positive     = {gate.min_positive_score}")
        print(f"  max_negative     = {gate.max_negative_score}")
        print(f"  positive_count   = {gate.positive_count}")
        print(f"  negative_count   = {gate.negative_count}")

    print("\n" + "=" * 60)
    print("Paste into planner.py's _LEXICAL_FALLBACK_THRESHOLDS:\n")
    print("_LEXICAL_FALLBACK_THRESHOLDS: dict[str, float] = {")
    for name, threshold in result.thresholds.items():
        print(f"    {name!r}: {threshold!r},")
    print("}")

    missing = set(result.gates) - set(result.thresholds)
    if missing:
        print(f"\nWARNING: these gates calibrated degenerate and are NOT in the dict above: {sorted(missing)}")


if __name__ == "__main__":
    main()
