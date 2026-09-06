# Plan: Live Semantic-Gating Threshold Calibration for New Embedding Models

**Design decisions locked in (2026-09-06):** Youden's J for threshold selection (§2);
visible UI trust badge, not log-line-only (§6); calibration stays bundled into "Re-embed
Corpus Now," no separate action (§5). Also added: per-gate degenerate-calibration fallback
and explicit vector normalization (§2), and a new BM25/lexical parity tier so users who
never get semantic gating at all (weak/incompatible model, or true zero-config keyword-only
setups) still get close-to-embedded routing quality (§9) — see rationale below each.

## Context (read this first — you have no memory of the session that produced this plan)

This is the Localist Framework repo (`lora-app-demo`). Read `CLAUDE.md` and `LOCALIST-Architecture.md` first, per repo convention.

Recent history relevant to this plan (see `docs/architecture/16-runtime-backend-layer.md` §16.14/§16.15 for full detail):

- `POST /settings/embedding-model` (`backend/src/localist/main.py`) lets a user pick which model the active runtime backend's `embed()` uses (e.g., Ollama's `nomic-embed-text:latest`) — built for the packaged desktop build, which has no local MLX embedding option.
- `Planner` (`backend/src/localist/planner.py`) has four cosine-similarity thresholds gating search-intent classification and episodic-memory-relevance detection (`_SEMANTIC_GATE_THRESHOLDS["explicit_search_action"|"lookup_request"]`, `_RESEARCH_INTENT_THRESHOLD`, `_EPISODIC_RELEVANCE_THRESHOLD`). All four were hand-tuned against exactly one embedding model, `mlx-community/embeddinggemma-300m-4bit` (`_TUNED_EMBEDDING_MODEL`), via a battery of curated test utterances and real diagnostic runs (`diagnostics/score_*.py`, `diagnostics/reports/*.md`).
- Cosine similarity does not transfer across embedding models' geometries. `Planner.__init__` guards against this: if the active `embedding_model_name` doesn't match `_TUNED_EMBEDDING_MODEL`, it checks a hardcoded `_VALIDATED_MODEL_THRESHOLDS` dict for a model-specific threshold set (currently containing exactly one entry, `"nomic-embed-text:latest"`, measured via `diagnostics/nomic_embed_text_threshold_probe.py` against a real live Ollama daemon). If the model isn't in that dict, semantic gating is disabled outright (fail-closed) rather than guessing.
- **The gap this plan closes:** `_VALIDATED_MODEL_THRESHOLDS` is a hardcoded dict populated by a developer manually running an offline diagnostic script and hand-transcribing results into source code. There is no way, from inside the running app, for a user who picks a *different* embedding model (anything other than the one already-measured entry) to get real semantic gating — they're permanently stuck with it disabled unless a developer repeats that whole manual cycle and ships a new build. This defeats the purpose of letting users freely pick any Ollama embedding model.

## The ask

Turn threshold calibration into something the running app can do for itself, triggered by the user, with results that persist across restarts — not a hardcoded dict requiring a source change per model.

## Design

### 1. Extract the test battery into shared, importable data

`diagnostics/nomic_embed_text_threshold_probe.py` currently hardcodes four test batteries inline (lookup_request/ESA Categories A–D, research_intent Categories T/L/K/F/E/G, episodic-relevance POSITIVES/NEGATIVES). Move these into a new module, e.g. `backend/src/localist/threshold_calibration_fixtures.py`, as plain data (tuples/lists of strings, same category labels). Update the diagnostic script to import from there instead of duplicating. This is the single source of truth both the offline diagnostic and the new live calibration path read from — they must never drift apart.

### 2. New calibration function

New module `backend/src/localist/threshold_calibration.py`:

```python
def calibrate_thresholds(embed_fn: Callable[[str], list[float]]) -> CalibrationResult:
    ...
```

Reuses `_SEARCH_INTENT_TEMPLATES`/`_EPISODIC_RELEVANCE_TEMPLATES`/`_cosine_similarity` from `planner.py` and the fixtures from step 1. For each of the four gates, scores every test utterance, then picks a threshold via **Youden's J statistic** (the value on the tested grid that maximizes true-positive rate minus false-positive rate) — **decided (2026-09-06)**: standard, deterministic, defensible default for turning a score table into one number without a human eyeballing a report. If a different selection philosophy is wanted later, this is the one function to change.

**Explicit vector normalization.** Since the whole point of this feature is to work against embedding models we have never tested (only gemma and nomic have been verified so far), don't inherit any assumption from those two about vectors already being unit-normalized. `calibrate_thresholds()` must normalize every vector it receives from `embed_fn` before computing cosine similarity, rather than relying on the model to hand back normalized output.

**Per-gate degenerate-calibration fallback.** A model that's simply weak at one of these four discrimination tasks (or has a score dynamic range wildly unlike gemma/nomic's) can produce a Youden's-J-selected threshold that still leaves heavy overlap between positive and negative scores. Silently shipping that threshold is worse than today's fail-closed behavior — a bad live threshold causes misrouting, whereas "disabled" just falls through to keyword behavior. So: after selecting each gate's threshold, check separation quality (e.g. TPR−FPR at the selected point against a floor, or overlap between the positive-score and negative-score ranges) computed relative to that model's own observed score spread, not a fixed number tuned against gemma/nomic. If a gate fails the check, mark it `"degenerate"` in `CalibrationResult` and that gate individually falls back to disabled — per-gate, not all-or-nothing, since a model might separate `lookup_request` cleanly but flop on `episodic_relevance`.

**Failure modes beyond scoring.** `embed_fn` can error or time out partway through the ~60-call battery, or return a wrong-dimension vector, especially for a wider range of user-picked Ollama models than the two tested so far. Treat any such failure as degenerate for the affected gate(s) rather than raising and leaving the caller (auto-trigger on model switch, or the reembed endpoint) with no result at all.

Returns both the four threshold values and enough metadata to log/display for transparency (min positive score, max negative score, whether clean separation was achieved per gate, which gates (if any) fell back to degenerate/disabled, sample sizes) — mirror the markdown report's content, just structured instead of prose.

### 3. Persist per-model calibration results (schema v17)

New `MemoryManager` table `embedding_model_thresholds`:

```sql
CREATE TABLE embedding_model_thresholds (
    model                   TEXT PRIMARY KEY,
    explicit_search_action  REAL NOT NULL,
    lookup_request          REAL NOT NULL,
    research_intent         REAL NOT NULL,
    episodic_relevance      REAL NOT NULL,
    calibrated_at           REAL NOT NULL
);
```

Follow the exact migration pattern already established in this file for every prior schema change (`_SCHEMA_VERSION` 16→17; add the table in the fresh-install DDL block in `_init_db()`, in the unconditional self-heal section, and in a new `if from_version < 17:` block in `_migrate()` — see the `episode_eviction_preset` addition, schema v16, for the exact three-site pattern to copy). New methods `get_calibrated_thresholds(model_name) -> dict|None` and `set_calibrated_thresholds(model_name, thresholds: dict) -> None`, mirroring `embedding_provenance`'s existing get/set shape (`_PROVENANCE_STORES`/`_check_embedding_provenance()` is a good structural reference, though this is a simpler single-row-per-model table, not a provenance-mismatch detector).

### 4. Three-tier threshold resolution in `Planner.__init__`

Currently two-tier (tuned model's own constants → `_VALIDATED_MODEL_THRESHOLDS` hardcoded dict → disabled). Extend to a **five-tier** resolution order (the fifth tier is new — see §9): tuned model → hardcoded validated dict (human-reviewed, highest trust) → **persisted live-calibration table (new, lower trust — auto-measured, never human-reviewed, per-gate degenerate fallback per above)** → **BM25/lexical fallback (new, §9 — model-independent, ships pre-calibrated)** → disabled. Per-gate resolution, not a single global tier: e.g. a model could land on "auto-calibrated" for three gates and "degenerate → BM25 fallback" for the fourth. `Planner` doesn't currently touch the database directly (it receives `embedding_model_name`/`embed_fn` by constructor injection) — keep that pattern: add a new optional constructor parameter, e.g. `calibrated_thresholds: dict[str, float] | None = None`, threaded through from `main.py`'s `_build_controller()` → `ControllerAgent.__init__()` → `Planner.__init__()`, exactly the way `embedding_model_name` already is. `main.py` looks it up via a new `_derive_calibrated_thresholds(mm, embedding_model_name)` helper (mirrors `_derive_active_embedding_model_name()`) at every `_build_controller()` call site.

Log clearly which tier is active **per gate** (`"validated"` / `"auto-calibrated"` / `"lexical-fallback"` / `"disabled"`) so future debugging can immediately tell whether a given session's routing behavior is running on hand-reviewed, self-measured, or keyword-based thresholds — and so the true "disabled" tier becomes a rare last resort rather than the common outcome for any model outside the two already tested.

### 5. When calibration runs

Two triggers, not one — they serve different goals:

- **Automatic, on `POST /settings/embedding-model`:** when switching to a model that has neither a `_VALIDATED_MODEL_THRESHOLDS` entry nor an existing row in `embedding_model_thresholds`, run calibration automatically as part of the switch, before rebuilding the controller — so a brand-new model choice is never silently degraded to fully-disabled without at least trying to do better, with zero extra clicks required.
- **Manual, explicit re-trigger:** the user asked for this bundled into the existing "Re-embed Corpus Now" flow specifically, since that's the button they already associate with "fix things after switching embedding models." Extend `POST /memory/reembed`'s handler to run calibration as a second phase after the corpus re-embed completes (using `_state.memory_manager.embed_fn`, persisting via step 3, rebuilding the controller the same way `POST /settings/embedding-model` does), and return both results in one response. This makes calibration idempotent and cheaply re-runnable — useful if the battery/methodology improves later, or if a model is re-pulled under the same tag with different weights.

Both call the same `calibrate_thresholds()` — no duplicated logic.

### 6. API & frontend

- `POST /memory/reembed`'s response gains a `calibration` field (the four new thresholds + summary stats + per-gate tier, alongside the existing `reembedded`/`total`/`model` fields).
- **Decided (2026-09-06): visible UI trust badge, not log-line-only.** The active embedding model's trust tier must be visible in the Settings UI on every page load, not just transiently after a reembed click — so whatever endpoint the Settings page already calls to display the current embedding model needs to also return per-gate tier (`"validated"` / `"auto-calibrated"` / `"lexical-fallback"` / `"disabled"`), sourced the same way `Planner`/`_derive_calibrated_thresholds()` resolves it. Render as a small badge next to the model name, e.g. "hand-validated" vs "auto-calibrated" vs "keyword fallback" vs "gating disabled" — reuse whatever badge/tag component pattern already exists in the Settings page, if any.
- `reembedCorpus.ts` / the Settings page's `handleReembedCorpus()` (`localist-ui/src/routes/settings/+page.svelte`) surfaces both results after the button click: `"Re-embedded 41 of 41 documents. Recalibrated semantic gating for nomic-embed-text:latest (ESA=0.68, LR=0.62, RI=0.58, episodic=0.62)."`, and updates the badge in place.
- Settings UI's Embedding Model card hint text should mention that switching models triggers automatic first-time calibration, and that "Re-embed Corpus Now" re-runs it.

### 7. Tests

- `threshold_calibration.py`: unit tests using a stub `embed_fn` with known, constructed cosine-similarity geometry (mirror `test_planner_phase3.py`'s `_unit_vector()` helper pattern) — verify Youden's J selection picks the expected threshold for a hand-constructed score distribution.
- Schema migration: a `TestSchemaV17Migration` class mirroring `test_retention_sweep.py`'s `TestSchemaV16Migration` exactly (real on-disk v16 database, migrated by opening it with `MemoryManager`, assert the new table + version).
- `Planner` threshold resolution: extend `test_planner_phase3.py`'s `TestValidatedModelThresholds` class with a `TestCalibratedModelThresholds` sibling covering the new tiers (persisted calibration present → used; persisted calibration present but marked degenerate for one gate → that gate falls to lexical fallback while siblings stay auto-calibrated; both embedding tiers absent → lexical fallback used; lexical fallback itself unavailable → disabled; hardcoded dict takes priority over a persisted row for the same model name, if both somehow exist).
- Endpoint tests for the calibration-triggering behavior in both `POST /settings/embedding-model` and `POST /memory/reembed` (new model → auto-calibrates; already-known model → doesn't re-run unnecessarily on switch, but does on explicit reembed).
- §9 lexical fallback: unit tests for the BM25-scored versions of the four gates using the same fixtures, and a `Planner` test confirming Priority 3 / Priority 5 route correctly with `embed_fn=None` but the lexical tier present (previously these would have just returned `None`/kept the static keyword list only).

### 8. Docs

New `docs/architecture/16-runtime-backend-layer.md` §16.17 documenting this (five-tier resolution, per-gate degenerate fallback, lexical/BM25 tier), and update §16.14/§16.15's "Open items" (they already flag needing per-model thresholds beyond the one hardcoded entry). Update the `LOCALIST-Architecture.md` index row for §16.

### 9. Keyword-only parity: extending BM25 into the two gates it doesn't reach yet

**Motivation.** A user picks an embedding model once, based on availability/compatibility, and lives with it — so the pain of calibration is meant to be a one-time cost per install, not a recurring one. But some users will end up on a model that calibrates as degenerate for a gate (§2), or will run true zero-config keyword-only (no `EmbeddingEngine`, no runtime-backend embed source — CLAUDE.md's documented zero-config default). For those users, semantic gating being simply "disabled" today is a real, noticeable quality gap versus the embedded experience. The premise for this section: BM25 plus the existing episodic-memory bypass can close most of that gap without embeddings at all.

**What's already there (confirmed in code, not assumed):** `bm25.py` is a working Okapi BM25 scorer already used in two places — the RAG corpus's stage-1 keyword prefilter (`memory_manager.py:3470-3489`), and as `MemoryManager`'s automatic fallback for episodic-memory scoring whenever no `embed_fn`/embedding is available (`_score_all_active()`, `memory_manager.py:4894-4918`). Separately, `Planner` Priority 5 already has a fully non-semantic bypass: a static `_EPISODIC_KEYWORDS` set (`planner.py:2692-2704` — "preference", "remember", "my name", "who am i", etc.) that triggers `fetch_episodic=True` and pulls the 5 most-recent preference/correction/decision/workflow episodes by pure recency (`by_recency()`, `memory_manager.py:4779-4826`) — no similarity or BM25 involved, and it works today with embeddings fully off.

**The actual gap:** the two *semantic gates inside Planner* — Priority 3 search-intent classification (`_semantic_search_intent`, `planner.py:1858-1925`) and Priority 5's semantic-relevance check (`_episodic_semantic_relevance`, `planner.py:2624-2657`) — both just short-circuit to `None` when `embed_fn is None` or gating is disabled (`planner.py:1858-1863,2639-2644`). RAG corpus retrieval and the static episodic keyword list are unaffected by this, but Planner's own routing decisions lose all of the nuance those two gates provide once cosine similarity is off the table.

**Design: a fifth, model-independent tier.** Since BM25 doesn't depend on which embedding model is active — it operates on the request text and the same fixture/template strings already used for calibration — it can be calibrated **once, offline, checked into source** (much like `_VALIDATED_MODEL_THRESHOLDS` is today), rather than needing live per-model calibration:

- Reuse the exact same fixtures from step 1 and the same `_SEARCH_INTENT_TEMPLATES`/`_EPISODIC_RELEVANCE_TEMPLATES` strings, but score them with `bm25.py`'s `score_documents()` instead of cosine similarity.
- BM25 scores are unbounded (`bm25.py:32`), so the selection method can't be a literal ported threshold — pick a scale-free rule instead, e.g. relative margin to the runner-up template/category (mirrors the existing `scored_by_embedding` distinction at `memory_manager.py:291-298`, which already treats BM25-ranked results by relative order rather than an absolute floor). Apply the same Youden's-J-over-a-grid method as §2, just on a normalized/rank-based BM25 signal instead of raw score, so it's one shared selection philosophy across both tiers.
- Store the result as a small constant table in source (e.g. `_LEXICAL_FALLBACK_THRESHOLDS` next to `_VALIDATED_MODEL_THRESHOLDS`) — it never needs a live-calibration UI flow of its own, since it isn't per-model.
- Wire it into Priority 3 and Priority 5 as the new fourth tier in the five-tier order (§4): when a gate resolves to `"disabled"` under the current logic, check the lexical tier before actually disabling. Log it as `"lexical-fallback"` per-gate, matching the tier-logging convention in §4.
- Priority 5's existing static `_EPISODIC_KEYWORDS` bypass is untouched and stays as a first, cheaper check — the new BM25-scored relevance check only needs to run for turns that don't already hit a static keyword.

**Explicitly out of scope for this pass:** extending the RAG corpus's BM25 prefilter or the chat-turn FTS5 BM25 (`memory_manager.py:1900-1901,2036`) — both already work regardless of embedding availability and aren't part of the gap this section addresses.

## Open decisions to confirm before/during building

All resolved (2026-09-06):

- Threshold-selection heuristic → Youden's J (§2).
- Trust labeling → visible UI badge (§6).
- Recalibration cost → bundled into "Re-embed Corpus Now" (§5).
- **Lexical-fallback selection rule (§9) → Youden's J**, same as the embedding tiers, applied to a normalized/rank-based BM25 signal rather than raw unbounded scores. Rationale: given how many end users will never realize they need two different Ollama models (one chat, one embedding) loaded to get real semantic gating, this BM25 tier is realistically the *dominant* routing experience for most installs, not a rare edge case — it earns the same rigor as the embedding tiers, not a cheaper heuristic.
- **Lexical-fallback scope (§9) → both gates.** Covers Priority 3 (search-intent) and Priority 5 (episodic relevance), not just one. Priority 5's existing static `_EPISODIC_KEYWORDS` bypass only catches literal trigger phrases; layering BM25 scoring on top improves recall for phrasings that don't hit it. Leaving Priority 3 with no lexical treatment would mean the majority of users (keyword-only) get materially worse search-intent routing than the minority running embeddings — directly against the "almost unnoticeable" bar for this feature.
