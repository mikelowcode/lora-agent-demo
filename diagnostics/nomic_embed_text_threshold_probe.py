"""
Diagnostic: nomic_embed_text_threshold_probe.py
================================================
Re-runs the exact production template sets and test-utterance batteries
already used to tune planner.py's four semantic-gating thresholds
(explicit_search_action, lookup_request, research_intent,
episodic_relevance) — all originally measured against
mlx-community/embeddinggemma-300m-4bit (_TUNED_EMBEDDING_MODEL) — against
nomic-embed-text:latest via the real, running Ollama daemon instead.

Background: docs/architecture/16-runtime-backend-layer.md §16.15. A user
switched the desktop build's embedding source to nomic-embed-text:latest
(the only local embedding path available there — no MLX in the packaged
build) via the new POST /settings/embedding-model (§16.14), and found a
known-relevant episodic memory ("The user games on an Xbox Series X")
never made it into the prompt for an Xbox-related question. Root cause:
Planner._semantic_gating_disabled fires for any embedding_model_name !=
_TUNED_EMBEDDING_MODEL (cosine similarity is not portable across
embedding-model geometries — confirmed 2026-07-16), which disables BOTH
the search-intent semantic gate (P3) and the episodic-relevance semantic
gate (P5) outright, falling back to keyword-only matching for episodic
fetch. This script measures whether nomic-embed-text can be given its own
validated thresholds instead of just being disabled.

Test utterance batteries live in localist.threshold_calibration_fixtures —
the single source of truth shared with the live in-app calibration path
(threshold_calibration.py) — copied verbatim from the original tuning
diagnostics (score_lookup_request_templates.py,
score_research_intent_templates.py, score_episodic_relevance_templates.py)
so results are directly comparable and the two paths never drift apart.

READ-ONLY. Does not modify planner.py. Embeds every utterance against the
live Ollama daemon's real /api/embed endpoint using the actual installed
OllamaRuntimeClient — no stubs, no mocks.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/nomic_embed_text_threshold_probe.py
"""

from __future__ import annotations

import pathlib
import sys
from datetime import date

from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.planner import (
    _SEARCH_INTENT_TEMPLATES,
    _EPISODIC_RELEVANCE_TEMPLATES,
    _ALL_SEARCH_NEGATIVE_FILTERS,
    _RESEARCH_NEGATIVE_FILTER,
    _cosine_similarity,
    _TUNED_EMBEDDING_MODEL,
    _SEMANTIC_GATE_THRESHOLDS,
    _RESEARCH_INTENT_THRESHOLD,
    _EPISODIC_RELEVANCE_THRESHOLD,
)
from localist.threshold_calibration_fixtures import (
    LR_CAT_A as _LR_CAT_A,
    LR_CAT_B as _LR_CAT_B,
    LR_CAT_C as _LR_CAT_C,
    LR_CAT_D as _LR_CAT_D,
    RI_CAT_T as _RI_CAT_T,
    RI_CAT_L as _RI_CAT_L,
    RI_CAT_L_PRICE_ADJACENT as _RI_CAT_L_PRICE_ADJACENT,
    RI_CAT_K as _RI_CAT_K,
    RI_CAT_F as _RI_CAT_F,
    RI_CAT_E as _RI_CAT_E,
    RI_CAT_G as _RI_CAT_G,
    EP_POSITIVES as _EP_POSITIVES,
    EP_NEGATIVES as _EP_NEGATIVES,
    EP_POSITIVES_LIVE as _EP_POSITIVES_LIVE,
)

MODEL_UNDER_TEST = "nomic-embed-text:latest"
TODAY = date.today().isoformat()

_LR_THRESHOLD_GRID = [x / 100 for x in range(30, 91, 2)]
_ESA_THRESHOLD_GRID = [x / 100 for x in range(30, 91, 2)]
_RI_THRESHOLD_GRID = [x / 100 for x in range(30, 96, 2)]
_EP_THRESHOLD_GRID = [x / 100 for x in range(30, 96, 2)]


def _trunc(s: str, n: int = 55) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    print(f"Constructing OllamaRuntimeClient for embedding_model={MODEL_UNDER_TEST!r}…")
    client = OllamaRuntimeClient(
        chat_model="gemma4:31b-cloud",  # required by constructor; never used, embed() only
        embedding_model=MODEL_UNDER_TEST,
    )
    health = client.health_check()
    if not health.get("reachable") or not health.get("embed_model_found"):
        print(f"ERROR: {MODEL_UNDER_TEST} not reachable/found via Ollama: {health}", file=sys.stderr)
        sys.exit(1)
    print("Ollama reachable, model confirmed present.\n")

    embed_cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text not in embed_cache:
            embed_cache[text] = client.embed(text)
        return embed_cache[text]

    # ---- Pre-embed all production template groups ----
    print("Pre-embedding _SEARCH_INTENT_TEMPLATES groups…")
    search_template_vecs: dict[str, dict[str, list[float]]] = {
        group: {phrase: embed(phrase) for phrase in phrases}
        for group, phrases in _SEARCH_INTENT_TEMPLATES.items()
    }
    print("Pre-embedding _EPISODIC_RELEVANCE_TEMPLATES…")
    episodic_template_vecs: dict[str, list[float]] = {
        phrase: embed(phrase) for phrase in _EPISODIC_RELEVANCE_TEMPLATES
    }
    print("Done.\n")

    def group_scores(text: str) -> dict[str, float]:
        qv = embed(text)
        return {
            group: max(_cosine_similarity(qv, tv) for tv in tvecs.values())
            for group, tvecs in search_template_vecs.items()
        }

    def episodic_score(text: str) -> float:
        qv = embed(text)
        return max(_cosine_similarity(qv, tv) for tv in episodic_template_vecs.values())

    md: list[str] = []
    md += [
        f"# Semantic-Gating Threshold Assessment — `{MODEL_UNDER_TEST}`",
        "",
        f"**Date:** {TODAY}",
        "**Script:** `diagnostics/nomic_embed_text_threshold_probe.py`",
        f"**Model under test:** `{MODEL_UNDER_TEST}` — real Ollama `/api/embed`, no stubs",
        f"**Tuned-model baseline:** `{_TUNED_EMBEDDING_MODEL}` "
        f"(current production thresholds: ESA={_SEMANTIC_GATE_THRESHOLDS['explicit_search_action']}, "
        f"LR={_SEMANTIC_GATE_THRESHOLDS['lookup_request']}, "
        f"RI={_RESEARCH_INTENT_THRESHOLD}, "
        f"episodic={_EPISODIC_RELEVANCE_THRESHOLD})",
        "**Status:** READ-ONLY. `planner.py` unmodified. All test batteries copied verbatim "
        "from the original tuning diagnostics for direct comparability.",
        "",
        "**Trigger:** docs/architecture/16-runtime-backend-layer.md §16.15 — a live report that "
        "an Xbox-related episodic memory never reached the prompt after switching the desktop "
        "build's embedding source to nomic-embed-text, because `_semantic_gating_disabled` "
        "unconditionally disables the episodic-relevance and search-intent semantic gates for "
        "any non-tuned embedding model.",
        "",
    ]

    # =========================================================================
    # lookup_request / explicit_search_action
    # =========================================================================
    print("Scoring lookup_request / explicit_search_action batteries…")
    lr_items: list[tuple[str, str]] = (
        [(cat, utt) for cat, utt in _LR_CAT_A]
        + [(cat, utt) for cat, utt in _LR_CAT_B]
        + [(cat, utt) for cat, utt in _LR_CAT_C]
        + [(cat, utt) for cat, _domain, utt in _LR_CAT_D]
    )
    lr_rows = []
    for cat, utt in lr_items:
        print(f"  [{cat}] {utt!r}")
        gs = group_scores(utt)
        filtered = any(neg in utt.lower() for neg in _ALL_SEARCH_NEGATIVE_FILTERS)
        lr_rows.append({"category": cat, "utterance": utt, "lr": gs["lookup_request"],
                         "esa": gs["explicit_search_action"], "filtered": filtered})

    md += [
        "## `lookup_request` / `explicit_search_action`",
        "",
        "Categories: A = live false positives (must NOT fire), B = `_ALL_SEARCH_NEGATIVE_FILTERS` "
        "phrases (filtered pre-gate regardless of threshold, shown for reference only), "
        "C = confirmed true positives (MUST fire), D = adversarial negatives (must NOT fire).",
        "",
        "| Cat | Utterance | LR score | ESA score | Filtered |",
        "|-----|-----------|---------:|----------:|:--------:|",
    ]
    for r in lr_rows:
        md.append(
            f"| {r['category']} | {_trunc(r['utterance'])} | {r['lr']:.4f} | {r['esa']:.4f} "
            f"| {'Y' if r['filtered'] else ''} |"
        )
    md.append("")

    lr_c_pool = [r for r in lr_rows if r["category"] == "C"]
    lr_neg_pool = [r for r in lr_rows if r["category"] in ("A",) or r["category"].startswith("D") and not r["filtered"]]
    md += [
        "### `lookup_request` threshold trade-off (Cat C must fire, Cat A/D must not)",
        "",
        "| Threshold | C survivors | A/D false positives |",
        "|:---------:|:-----------:|:--------------------:|",
    ]
    for th in _LR_THRESHOLD_GRID:
        c_ok = sum(1 for r in lr_c_pool if r["lr"] >= th)
        fp = sum(1 for r in lr_neg_pool if r["lr"] >= th)
        md.append(f"| {th:.2f} | {c_ok}/{len(lr_c_pool)} | {fp}/{len(lr_neg_pool)} |")
    md.append("")

    min_c_lr = min((r["lr"] for r in lr_c_pool), default=float("nan"))
    max_neg_lr = max((r["lr"] for r in lr_neg_pool), default=float("nan"))
    lr_clean = min_c_lr > max_neg_lr
    md += [
        f"Min Cat-C LR score: **{min_c_lr:.4f}**. Max Cat-A/D LR score: **{max_neg_lr:.4f}**. "
        + (
            f"**Clean separation** — any threshold in ({max_neg_lr:.4f}, {min_c_lr:.4f}] "
            f"achieves full separation."
            if lr_clean else
            "**No clean separation** in this battery — see trade-off table for the actual cost "
            "at each candidate."
        ),
        "",
    ]

    esa_neg_pool = [r for r in lr_rows if r["category"] in ("A",) or (r["category"].startswith("D") and not r["filtered"])]
    md += [
        "### `explicit_search_action` — same battery, no dedicated Cat-C-equivalent true positives "
        "exist in this battery (ESA has no positive templates tested here beyond what LR's Cat C "
        "also scores on); reported for completeness/regression only.",
        "",
        "| Threshold | A/D false positives (of " + str(len(esa_neg_pool)) + ") |",
        "|:---------:|:--------------------------------------------------------:|",
    ]
    for th in _ESA_THRESHOLD_GRID:
        fp = sum(1 for r in esa_neg_pool if r["esa"] >= th)
        md.append(f"| {th:.2f} | {fp}/{len(esa_neg_pool)} |")
    md.append("")

    # =========================================================================
    # research_intent
    # =========================================================================
    print("Scoring research_intent battery…")
    ri_items: list[tuple[str, str]] = (
        [("T", u) for u in _RI_CAT_T]
        + [("L", u) for u in _RI_CAT_L]
        + [("L-price-adj", _RI_CAT_L_PRICE_ADJACENT)]
        + [("K", u) for u in _RI_CAT_K]
        + [("F", u) for u in _RI_CAT_F]
        + [("E", u) for u in _RI_CAT_E]
        + [("G", u) for u in _RI_CAT_G]
    )
    ri_rows = []
    for cat, utt in ri_items:
        print(f"  [{cat}] {utt!r}")
        gs = group_scores(utt)
        filtered = any(neg in utt.lower() for neg in _RESEARCH_NEGATIVE_FILTER)
        ri_rows.append({"category": cat, "utterance": utt, "ri": gs["research_intent"], "filtered": filtered})

    md += [
        "## `research_intent`",
        "",
        "| Cat | Utterance | RI score | Filtered |",
        "|-----|-----------|---------:|:--------:|",
    ]
    for r in ri_rows:
        md.append(f"| {r['category']} | {_trunc(r['utterance'])} | {r['ri']:.4f} | {'Y' if r['filtered'] else ''} |")
    md.append("")

    ri_t_pool = [r for r in ri_rows if r["category"] == "T" and not r["filtered"]]
    ri_fp_pool = [r for r in ri_rows if r["category"] in ("L", "K", "F", "E", "G") and not r["filtered"]]
    md += [
        "### `research_intent` threshold trade-off",
        "",
        "| Threshold | T survivors | FP pool fires |",
        "|:---------:|:-----------:|:--------------:|",
    ]
    for th in _RI_THRESHOLD_GRID:
        t_ok = sum(1 for r in ri_t_pool if r["ri"] >= th)
        fp = sum(1 for r in ri_fp_pool if r["ri"] >= th)
        md.append(f"| {th:.2f} | {t_ok}/{len(ri_t_pool)} | {fp}/{len(ri_fp_pool)} |")
    md.append("")

    min_t_ri = min((r["ri"] for r in ri_t_pool), default=float("nan"))
    max_fp_ri = max((r["ri"] for r in ri_fp_pool), default=float("nan"))
    ri_clean = min_t_ri > max_fp_ri
    md += [
        f"Min Cat-T RI score: **{min_t_ri:.4f}**. Max FP-pool RI score: **{max_fp_ri:.4f}**. "
        + (
            f"**Clean separation** — any threshold in ({max_fp_ri:.4f}, {min_t_ri:.4f}] "
            f"achieves full separation."
            if ri_clean else
            "**No clean separation** in this battery."
        ),
        "",
    ]

    # =========================================================================
    # episodic_relevance
    # =========================================================================
    print("Scoring episodic_relevance battery…")
    ep_rows = []
    for utt in _EP_POSITIVES + _EP_POSITIVES_LIVE:
        print(f"  [POS] {utt!r}")
        ep_rows.append({"category": "positive", "utterance": utt, "score": episodic_score(utt)})
    for utt in _EP_NEGATIVES:
        print(f"  [NEG] {utt!r}")
        ep_rows.append({"category": "negative", "utterance": utt, "score": episodic_score(utt)})

    md += [
        "## `episodic_relevance`",
        "",
        "`positive` includes the original 10-utterance battery plus 3 live-motivated additions "
        "(the actual Xbox/gaming phrasing this investigation started from).",
        "",
        "| Cat | Utterance | Score |",
        "|-----|-----------|------:|",
    ]
    for r in ep_rows:
        md.append(f"| {r['category']} | {_trunc(r['utterance'])} | {r['score']:.4f} |")
    md.append("")

    ep_pos_pool = [r for r in ep_rows if r["category"] == "positive"]
    ep_neg_pool = [r for r in ep_rows if r["category"] == "negative"]
    md += [
        "### `episodic_relevance` threshold trade-off",
        "",
        "| Threshold | Positive survivors | Negative false positives |",
        "|:---------:|:-------------------:|:-------------------------:|",
    ]
    for th in _EP_THRESHOLD_GRID:
        p_ok = sum(1 for r in ep_pos_pool if r["score"] >= th)
        fp = sum(1 for r in ep_neg_pool if r["score"] >= th)
        md.append(f"| {th:.2f} | {p_ok}/{len(ep_pos_pool)} | {fp}/{len(ep_neg_pool)} |")
    md.append("")

    min_pos_ep = min((r["score"] for r in ep_pos_pool), default=float("nan"))
    max_neg_ep = max((r["score"] for r in ep_neg_pool), default=float("nan"))
    ep_clean = min_pos_ep > max_neg_ep
    md += [
        f"Min positive score: **{min_pos_ep:.4f}**. Max negative score: **{max_neg_ep:.4f}**. "
        + (
            f"**Clean separation** — any threshold in ({max_neg_ep:.4f}, {min_pos_ep:.4f}] "
            f"achieves full separation."
            if ep_clean else
            "**No clean separation** in this battery — see trade-off table."
        ),
        "",
    ]

    md += [
        "## Summary",
        "",
        "No threshold is auto-selected here beyond what's stated per section above — same "
        "discipline as the original tuning diagnostics (data informs the choice; the choice "
        "itself is recorded in planner.py's per-model threshold table with its rationale).",
        "",
        "---",
        "",
        "*Generated by `diagnostics/nomic_embed_text_threshold_probe.py` against the real, "
        "running Ollama daemon.*",
    ]

    report_dir = pathlib.Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"nomic_embed_text_threshold_assessment_{TODAY}.md"
    report_path.write_text("\n".join(md) + "\n")

    print("\n" + "=" * 72)
    print(f"Report written to:\n  {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
