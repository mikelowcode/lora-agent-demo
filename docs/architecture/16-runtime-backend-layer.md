## 16. Runtime Backend Layer (BaseRuntimeClient)

### 16.1 Available Backends

`base_runtime_client.py` defines the `BaseRuntimeClient` Protocol (`infer`, `embed`,
`infer_stream`) that every concrete runtime client satisfies structurally, without inheritance.
`runtime_factory.py`'s `create_runtime()` is the single entry point that constructs the active
backend at process startup, selected via `LOCALIST_RUNTIME_BACKEND`. Concurrency posture is
swap-only: exactly one backend is active per process; dual-runtime/concurrent-backend operation is
explicitly out of scope for all three backends below.

Three backends are registered in `_REGISTRY` today:

- **`FoundryRuntimeClient`** (`foundry_runtime_client.py`) — Azure AI Foundry, local execution,
  ephemeral-port resolution via `foundry service status`.
- **`OMLXRuntimeClient`** (`omlx_runtime_client.py`) — oMLX local inference, fixed port 8000,
  OpenAI-compatible SSE streaming, native MarkItDown file ingestion (`infer_with_file()`, an
  oMLX-only capability not part of the Protocol).
- **`OllamaRuntimeClient`** (`ollama_runtime_client.py`, added 2026-07-08) — Ollama local
  inference, fixed port 11434, native NDJSON streaming (not OpenAI-compatible SSE). See §16.2.

### 16.2 OllamaRuntimeClient — Added 2026-07-08

> **Superseded in part (2026-07-14) — see §16.4.** `embed()` is no longer a permanent stub (it now
> implements `POST /api/embed`, with a real "not yet configured" `embedding_model` parameter
> mirroring oMLX's convention) and the "chat model left as an unfilled constructor default" framing
> below was superseded first by a hardcoded `gemma4:e4b-mlx` default (2026-07-08, later in this
> section) and then by a fail-fast `ValueError` (2026-07-14). The scoping narrative below is
> retained as a historical record of what was decided and why at the time.

**Scoping decisions (made before any code was written).** Full interchangeable-backend tier, same
as Foundry/oMLX — not an offload-style secondary backend as sketched in earlier dual-runtime
planning (the Apfel spec). Concurrency: swap-only, no dual-runtime logic added. `embed()`: stubbed
to always raise `NotImplementedError`, not left as a "not configured yet" placeholder the way
oMLX's empty-`embedding_model` case is — this backend has no path to an embedding model at all
under this scope, so its constructor doesn't even accept an `embedding_model` parameter. Chat
model left as an unfilled constructor default pending live confirmation rather than guessed up
front.

**Live verification before writing the Claude Code prompt.** Base URL confirmed as Ollama's actual
default (`http://localhost:11434` — no port conflict with 8000/8001/8003/5173). `GET /api/tags`
response shape confirmed via live `curl`: `{"models": [{"model": "...", "name": "...", ...}]}` — a
`models`/`model` shape, not oMLX's `{"data": [{"id": "..."}]}`. `POST /api/chat` streaming
confirmed via live `curl` to be NDJSON (one raw JSON object per line, terminated on `"done":
true`), not an SSE `data: {...}` envelope — meaning `foundry_runtime_client.py`'s
`_iter_sse_chunks` could not be reused; `ollama_runtime_client.py` implements its own
`_iter_ndjson_chunks`.

**Model selection — a live correction, not an assumption.** Initial framing considered Gemma 3
variants (the assistant's most recent reliable training knowledge). The user corrected this:
Ollama's model library now lists native Gemma 4 MLX-format models, and `gemma4:e4b-mlx` was chosen
specifically to match the oMLX baseline's `gemma-4-e4b-it-4bit` architecture class (E4B
effective-parameter size, 4-bit-class quantization) — not picked arbitrarily. This surfaced that
both Gemma 4 and Ollama's native MLX backend support post-date the assistant's reliable training
data; the live-verification step above, rather than the assistant's own knowledge of what Ollama
supports, is why the final choice held up.

**Implementation.** New file `backend/ollama_runtime_client.py`, structurally mirroring
`omlx_runtime_client.py` (docstring conventions, method ordering, `__repr__`,
`_assert_protocol_conformance()`). One flagged deviation from the `_make_omlx` registration
pattern: `_make_ollama` (`runtime_factory.py`) does not pass an `embedding_model` kwarg at all,
since `OllamaRuntimeClient.__init__` has no such parameter — `embed()` here is a permanent stub,
not an unconfigured state, so there is nothing to configure.

**Two mount-staleness false alarms surfaced and resolved during this work, not implementation
defects:**

- The assistant initially flagged the `label: str = ""` parameter (present in
  `BaseRuntimeClient`, `FoundryRuntimeClient`, and `OMLXRuntimeClient`, and carried into
  `OllamaRuntimeClient`) as a possibly-hallucinated, unverified claim — based on a stale
  project-knowledge mount that didn't contain it. A live `grep` against the actual repo confirmed
  `label` is genuinely present in all three; the original implementation report was correct
  throughout, and the mount, not the code, was stale.
- Separately, the assistant's mount showed `env_prefix="LORA_"` in `main.py`; a live `grep`
  confirmed the actual prefix is `env_prefix="LOCALIST_"`. Caught and corrected before any `.env`
  edit was made.

**Full live-verification chain, in order:** `GET /api/tags` reachable → backend startup with
`LOCALIST_RUNTIME_BACKEND=ollama` (clean init log chain, no errors) → health-check equivalent
confirmed via startup log (`chat_found=True embed_found=None`) → `POST /task` non-streaming, real
content, correct answer ("7 times 6 is 42") → `POST /task/stream` SSE relay, correct
token-by-token streaming, clean `[DONE]` termination. One call returning empty content
(`warmup.py`'s cache-warm call, `max_tokens=16`) was investigated and confirmed expected —
`run_cache_warmup()` deliberately discards completion content by design (KV-prefill exercise only),
not a bug in the new client.

**Test suite:** 595 tests collected/passed before this change, 595 passed after (file-scoped count
via a direct `.venv` pytest run, not carried forward from an earlier baseline). No new dedicated
test file was in scope for this addition.

**Open items — both closed 2026-07-08. OllamaRuntimeClient has no known open items as of this
update.**

**1. `LOCALIST_CHAT_MODEL` coupling. CLOSED 2026-07-08.**

*Originally:* `main.py`'s `Settings.chat_model` defaulted to the Foundry-specific string
`"Phi-4-mini-instruct-generic-gpu:5"` and was always passed explicitly to `create_runtime()`, so
`runtime_factory.py`'s per-backend `kwargs.get("chat_model", default)` fallback never triggered —
the key was never absent.

*Fix:* `Settings.chat_model` changed to `str | None = None`; `runtime_factory.py`'s
`_make_foundry`/`_make_omlx`/`_make_ollama` changed from `kwargs.get("chat_model", default)` to
`kwargs.get("chat_model") or default`, so `None`/empty string now correctly falls through to each
backend's own default.

*Live-verified:* with `LOCALIST_CHAT_MODEL` unset and `LOCALIST_RUNTIME_BACKEND=ollama`, the
constructed `OllamaRuntimeClient` reports `chat_model="gemma4:e4b-mlx"` (not the Foundry string,
not `None`). Foundry/oMLX fallback and explicit-override passthrough also spot-checked with no
regression.

*Test suite:* 595 passed (before and after).

**2. Missing `LOCALIST_OLLAMA_URL`. CLOSED 2026-07-08.**

*Originally:* `Settings` had `foundry_url` and `omlx_url` fields but no `ollama_url` field;
`create_runtime()` had no `ollama_url` kwarg wired in for the `ollama` backend.

*Fix:* added `ollama_url: str = "http://localhost:11434"` to `Settings`, documented
`LOCALIST_OLLAMA_URL` in the module docstring's environment variable list, and passed
`ollama_url=settings.ollama_url` into the `create_runtime()` call.

*Live-verified:* with `runtime_backend="ollama"` and `ollama_url` overridden to
`http://localhost:19999`, `health_check()` returns `base_url: 'http://localhost:19999'` — the
override, not the hardcoded `11434` default — confirmed even with nothing listening on that port,
since `health_check()` never raises.

*Test suite:* 595 passed (before and after).

### 16.3 Proposed — Live-Switchable Runtime Backend from Settings (scoped, not built, 2026-07-13)

> **Built and live-verified 2026-07-15 — see §16.5.** The proposal below was implemented
> essentially as scoped, with both open decisions resolved (persist to `.env`, not session-scoped;
> independent switch/pin endpoints, not folded together) and point 6's concurrency claim now
> live-verified rather than assumed. The narrative below is retained as the historical record of
> what was scoped and why; §16.5 is the current-state record of what shipped, and §16.6 closes the
> two follow-on items code review found there (the frontend now wired, the lock now `asyncio.Lock`).

**Status: proposal only. No code has been changed for this item.** The Settings page's Runtime
Backend segmented control (§7.10) is a display preference — selecting oMLX/Ollama/Foundry there
updates only the appbar's status-chip label, not the actual active runtime. This subsection records
a scoping pass done at the user's explicit request ("don't perform any edits, just scope out how
many steps it would take"). Full narrative: `sessions-log.md` §34.

**Why this is a real rebuild-and-replace, not a one-line state mutation.** At startup,
`main.py`'s `lifespan()` constructs one `runtime` object via `create_runtime()` and passes it **by
value** into the constructors of `WikiAgent`, `ConversationalAgent`, and `ControllerAgent`
(`self._runtime = runtime`). `ControllerAgent.__init__` further constructs its own
`Synthesizer(runtime)` and `_RulePlanner(runtime=runtime, ...)` from that same captured reference.
None of these look the runtime back up dynamically. The sole exception is `GET /health`, which reads
`_state.runtime` fresh via `_require_runtime()` on every call. So mutating `_state.runtime` alone
post-startup would make `/health` report a new backend while every actual chat turn kept running
inference through the *original* client.

**Proposed implementation shape:**
1. Extract the agent-construction block currently inline in `lifespan()` into a reusable function
   (e.g. `_build_controller(settings, runtime, memory_manager, embed_fn)`).
2. New `POST /settings/runtime-backend`: validate the target backend name; call `create_runtime()`
   for it; call `.health_check()` on the *new* client before committing anything, so an unreachable
   target fails cleanly and leaves the current backend running.
3. Rebuild `WikiAgent`/`ConversationalAgent`/`ControllerAgent` via the function from step 1 (which
   transparently rebuilds `Synthesizer`/`_RulePlanner` too, since they're constructed inside
   `ControllerAgent.__init__`).
4. Re-run `_run_cache_warmup` against the new controller/runtime, since persona caching
   (`ControllerAgent._load_persona()`) is per-instance and would otherwise start cold.
5. Atomically swap `_state.runtime`/`_state.wiki_agent`/`_state.controller` only after steps 3–4
   succeed, under a lock (so two rapid switch requests can't interleave).
6. In-flight requests are expected to be safe by construction — each request resolves
   `_state.controller` once at request time, so an active stream should finish on the pre-swap
   controller rather than crashing. **Live-verified 2026-07-15 (§16.5):** a switch fired mid-stream
   let the in-flight request complete its entire lifecycle on the pre-switch client, unaffected by
   the concurrent rebuild — no hang, no cross-contamination.

**The `chat_model`-per-backend interaction with §16.2's existing fix.** §16.2 already fixed the
*unset* case (`Settings.chat_model = None` correctly falls through to each backend's own hardcoded
default). This proposal is about the *explicit-override* case: a `LOCALIST_CHAT_MODEL` value set
for one backend would still get handed unchanged to whichever backend is live-switched to next
(e.g. a Foundry model id passed into `OllamaRuntimeClient`). Proposed fix: per-backend chat-model
storage (three settings fields or a dict) plus a new small `GET
/settings/runtime-backend/{backend}/models` endpoint — builds a throwaway client for that backend,
calls `.health_check()`, returns its `models` list, discards the client without touching `_state` —
letting the Settings UI's "Chat model ID" free-text field become a dropdown of models actually
available on the target backend instead of a field a user can mistype.

**A separate finding from the same scoping conversation, already acted on.** While tracing whether
Ollama could serve as an alternative embedding backend (unrelated to the switch proposal itself, but
raised in the same conversation), it became clear the Settings page's "Embedding model ID" field is
fully inert today — see `sessions-log.md` §33 for the finding and the fix (a `requirements.txt`
dependency gap, unrelated to runtime-backend switching). Recorded here only because it materially
affects what "fold Model Configuration into backend-switch config" should mean: that field should
likely be retired rather than made per-backend.

**Open decisions blocking a build session:**
- Does a live switch persist across a process restart (would need writing back to `.env` or a small
  runtime-config store), or is it session-scoped only?
- Accept the per-backend hardcoded chat-model fallback always, or let users pin a model per backend
  via the proposed dropdown?

**If built, would also require:** dedicated unit tests for the new endpoint(s) (unknown backend
rejected; unreachable backend rejected without side effects; successful swap updates `_state`);
updating `CLAUDE.md`'s current "Swapping backends is a config change only" line; updating this
section; updating the README's Configuration section, which currently and correctly describes the
control as display-only.

### 16.4 Embedding Support Across Backends, Chat-Model Fail-Fast, and Platform-Gated MLX
Fallback — Added 2026-07-14

`OllamaRuntimeClient.embed()` is no longer a permanent stub. It POSTs to Ollama's native
`/api/embed` endpoint (distinct from the OpenAI-compatible `/v1/embeddings` used internally by
`FoundryRuntimeClient` and `OMLXRuntimeClient` — Ollama's own API returns a plural `embeddings`
list rather than the singular `data[0].embedding` shape). The client accepts an `embedding_model`
constructor argument, defaulting to `""` ("not yet configured" — same convention as
`OMLXRuntimeClient`, not "this backend can never do this").

**Chat model configuration is now mandatory, not defaulted.** `OllamaRuntimeClient.DEFAULT_CHAT_MODEL`
was previously `"gemma4:e4b-mlx"` — a silent 8.8GB local-model fallback whenever `LOCALIST_CHAT_MODEL`
was unset. This has been removed. The constructor now raises `ValueError` immediately if
`chat_model` is falsy, failing at construction time (and therefore at FastAPI startup, surfacing as
`Application startup failed. Exiting.` with a clear traceback) rather than producing a confusing
failure on the first chat request. `runtime_factory._make_ollama()` was updated to match. This is
deliberate: a chat application should never start without a chat model configured, and Ollama in
particular spans model sizes from tiny local quantizations to 700B-parameter cloud models — there
is no safe default to assume. Live-verified 2026-07-14: unsetting `LOCALIST_CHAT_MODEL` produces a
clean startup crash with the expected `ValueError` message; restoring it produces normal startup.

**Embedding source precedence (all backends):**

At startup, `lifespan()` in `main.py` selects an embedding source via
`_configure_embedding_source(settings, runtime, health)` (extracted 2026-07-14 into a standalone,
unit-testable function — see Testing note below) in this order:

1. **Runtime-backend embed** — if `Settings.embedding_model` (`LOCALIST_EMBEDDING_MODEL`) is set
   and the active runtime's `health_check()` reports the model present, `MemoryManager.embed_fn`
   is bound to `runtime.embed`. This is wired through for the Foundry and Ollama backends and is
   platform-agnostic for both. **Not yet wired for the oMLX backend** —
   `runtime_factory._make_omlx()` hardcodes `embedding_model=""` regardless of
   `Settings.embedding_model`, so this tier can never actually engage while `runtime_backend="omlx"`
   is active, even though `OMLXRuntimeClient.embed()` itself supports a configurable
   `embedding_model` when constructed directly. Closing this gap is a follow-up, not part of this
   change.
2. **`EmbeddingEngine` (MLX-LM, standalone)** — used only when tier 1 isn't active, AND only on
   Apple Silicon. Gated behind
   `is_apple_silicon = platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")`,
   added 2026-07-14, since `mlx_lm` cannot run on Intel Mac, Windows, or Linux. On non-Apple-Silicon
   hosts with this tier otherwise eligible, the load attempt is skipped entirely (no
   `EmbeddingEngine()` construction, no import attempt) and an INFO-level log line fires — distinct
   from the WARNING-level "failed to load" message used when an actual load attempt fails, so the
   two situations ("this platform can't run this by design" vs. "this platform can but something
   went wrong") are distinguishable in logs.
3. **Keyword-only** — universal fallback when neither of the above is available.

**Updated 2026-08-30 (OSS packaging pass):** tier 2's gate,
`Settings.embedding_engine_enabled` (`LOCALIST_EMBEDDING_ENGINE_ENABLED`),
now defaults to `False` — previously `True`, meaning a fresh install used to
attempt the ~400MB EmbeddingGemma download silently on first startup
whenever Apple Silicon was detected. Keyword-only (tier 3) is now the true
zero-config default on every platform; enabling tier 2 is an explicit
opt-in, offered interactively by `start_localist.sh` on first run (only
when `.env` doesn't already set the key, this is Apple Silicon, and the
`localist[mlx]` extra — see `backend/pyproject.toml` — looks installed).
The precedence order and `_configure_embedding_source()`'s branch logic
above are unchanged; only the default value feeding into branch 2's
condition moved.

`GET /health`'s `embed_model_found` field mirrors this same precedence (fixed 2026-07-14 —
previously hardcoded to check `EmbeddingEngine` only, which misreported `false` whenever tier 1 was
actually active).

**Testing note:** the project's test suite convention (established in
`test_main_memory_episodes.py`) is to never trigger the real FastAPI `lifespan()` in tests — tests
swap `AppState` fields directly instead. The four-branch embedding-source selection was therefore
extracted out of `lifespan()`'s inline body into `_configure_embedding_source()` specifically to
make it unit-testable without violating that convention. `tests/test_main_embedding_source_selection.py`
covers all four branches, including the Apple-Silicon skip branch under mocked
`platform.system()`/`platform.machine()` calls (Linux/x86_64, Windows/AMD64) — this is
mocked/forced-condition testing, not real cross-platform execution; no non-Darwin hardware was
available to verify the skip branch by actually running on it.

**Verified configuration (2026-07-14, fully live end-to-end):** cloud chat model
(`gemma4:31b-cloud`, proxied through ollama.com — never resident locally) + local embedding model
(`nomic-embed-text:latest`, served on-device via `/api/embed`), both routed through the same local
Ollama daemon at `localhost:11434`. A real turn produced ~30 real embed calls (768-dim vectors)
plus a completed chat response and working-state extraction, all logged and confirmed via
`GET /health`. This is the reference configuration for Localist PC (non-Apple-Silicon) users, since
MLX/`EmbeddingEngine` is unavailable on that hardware entirely — Ollama's `/api/embed` is the only
local embedding path on Windows/Linux, and the platform gate above ensures those hosts never
attempt (and fail) an MLX load. One tag-mismatch false start along the way (`.env` needed the
`:latest` suffix Ollama's `/api/tags` actually reports) — corrected, not a code defect. Full
narrative: `sessions-log.md` §24 (2026-07-14 entry).

**Test suite:** 728 passed / 0 failed after `embed()` + wiring + `/health` fix; 732 passed / 0
failed after the chat-model fail-fast change (+4 new tests); 739 passed / 0 failed after the
platform-gating change (+7 new tests).

**Open items:**
- ~~Corpus-retrieval similarity thresholds may need to become embedding-model-aware~~ — **confirmed,
  2026-07-16.** The low top-score corpus miss (`0.028` under `nomic-embed-text`) noted below was the
  first hint; live testing the same day made it unambiguous: the identical `lookup_request`
  utterance scored `0.7119` under `mlx-community/embeddinggemma-300m-4bit` (the model
  `planner.py`'s semantic-gating thresholds were tuned against) vs `0.578` under `nomic-embed-text`
  — a swing large enough to flip `gate_fired` from `True` to `False` for the same input. Two more
  diagnostics required assuming a fixed embedding model to be meaningful at all:
  `diagnostics/reports/research_intent_threshold_assessment_2026-07-16*.md` and
  `diagnostics/reports/negative_filter_tiebreak_assessment_2026-07-16.md`. Cosine similarity is not
  portable across embedding models' geometries; a threshold tuned on one model's score distribution
  has no guaranteed meaning on another's.

  **Fixed on the Planner side (2026-07-16).** `planner.py` now declares `_TUNED_EMBEDDING_MODEL =
  "mlx-community/embeddinggemma-300m-4bit"`. `Planner.__init__` takes an `embedding_model_name`
  parameter; when it's set and doesn't match `_TUNED_EMBEDDING_MODEL`, semantic search-intent gating
  (`_semantic_search_intent()`, backing `explicit_search_action` / `lookup_request` /
  `research_intent`) is disabled for that Planner instance — a startup-time guard rather than a
  silent runtime degradation, with a `logger.warning` naming both models. `main.py` derives this name
  via the new `_derive_active_embedding_model_name()` (mirroring `_configure_embedding_source()`'s
  own three-tier precedence), stores it on `_state.active_embedding_model_name`, and threads it
  through `_build_controller()` into `ControllerAgent` at every construction site — startup
  (`lifespan()`) and both live-switch endpoints (`/settings/runtime-backend`,
  `/settings/runtime-backend/{backend}/chat-model`), read fresh from `_state` at request time rather
  than captured once. `embedding_model_name=None` (keyword-only mode, no embedding source
  configured) is not treated as a mismatch — `embed_fn` is already `None` in that case, which already
  short-circuits semantic scoring.

  **Fixed on the MemoryManager side too (2026-07-16).** A new `embedding_provenance` table
  (`store TEXT PRIMARY KEY` — `'corpus'` | `'episodes'` — `model TEXT NOT NULL`, schema v8) records
  which embedding model actually produced the vectors currently stored in `document_index` and
  `episodes`.

  **Extended to `'chat_turns'` (2026-07-21), Episode Browsing UI Phase 1.** Schema v9 adds an
  `embedding BLOB` column to `chat_turns`; `add_chat_turn()` embeds `content` (truncated to 500
  chars, same convention as `index_document()`/`reembed_corpus()`) whenever `embed_fn` is
  configured. `'chat_turns'` follows the same never-automatic path as `'corpus'` below — chat
  history can grow arbitrarily large under the `"forever"` eviction preset — with its own
  `self._chat_turns_stale` flag and `MemoryManager.reembed_chat_turns()` (exposed as
  `POST /memory/reembed-chat-turns`). `MemoryManager.get_chat_turns(mode="semantic")` does a
  full-table cosine scan (no `token_set`-style pre-filter column exists on `chat_turns` the way it
  does on `document_index`) and checks `not self._chat_turns_stale` before taking the embedding
  path — same fail-safe-to-keyword/FTS posture as the other two stores.

  `MemoryManager.__init__` gains `embedding_model_name`, threaded from `main.py`'s
  `_state.active_embedding_model_name` (the same value `ControllerAgent`/`Planner` now receive), and
  runs `_check_embedding_provenance()` once at construction time. Per store, per the recorded row:

  - No row, no embedded data yet → nothing to compare against — deferred; `'corpus'` seeds its own
    row the first time `index_document()` actually writes an embedding
    (`_maybe_seed_corpus_provenance()`).
  - No row, but embedded data already present → the pre-existing-database migration case (provenance
    tracking shipped after data was already embedded under whatever model was active at the time).
    Seeded silently, no warning, no re-embed — treating this as a mismatch would have triggered a
    surprise re-embed for every existing user on their very next boot.
  - Row present and it matches → no-op.
  - Row present and it disagrees → a genuine mismatch, handled per the decided split:
    - **`'corpus'`** (wiki/raw documents): potentially large/expensive to re-embed, so never
      automatic. `logger.warning`, `self._corpus_stale = True`, and an immediate retrieval-cache
      flush (stale rankings from the old model are wrong the instant the mismatch is detected, even
      before any re-embed happens). `query_corpus()` checks `not self._corpus_stale` alongside its
      existing `self._embed_fn is not None` check before taking the embedding re-rank path — same
      fail-safe-to-keyword-only posture as no `embed_fn` at all. Cleared only by the new
      `MemoryManager.reembed_corpus()` (exposed as `POST /memory/reembed`), which re-embeds every
      `document_index` row, flushes the cache, updates the provenance row, and clears
      `_corpus_stale` — idempotent, callable regardless of whether the corpus is currently flagged
      stale.
    - **`'episodes'`**: small and bounded, so auto-corrected in place before `__init__` returns —
      every `episodes` row is re-embedded via the active `embed_fn` (`"{subject}. {content}"`, the
      same convention `EpisodicMemoryWriter.insert()` uses) and the provenance row is advanced to the
      new model. Provenance only advances if every row succeeded; a partial failure leaves the old
      provenance value in place so the mismatch is detected and retried on the next boot rather than
      claiming a clean re-embed that didn't fully happen. A row-count tripwire
      (`_EPISODES_REEMBED_WARN_ROW_COUNT`) logs a warning if this ever needs revisiting at scale, but
      re-embedding runs regardless.

  This is the same detect-and-fail-safe pattern as the Planner-side fix above — a tuned-model
  comparison at construction time, disabling (not silently degrading) on a mismatch — applied to
  stored vectors instead of threshold constants, split into an automatic path where the fix is cheap
  and a manual path where it isn't.

  **`chat_turns.embedding` schema-drift bug found live, and a broader self-heal added (2026-07-22).**
  A running instance hit `sqlite3.OperationalError: table chat_turns has no column named embedding`
  from `add_chat_turn()`'s `INSERT` — its `schema_version` row already reported `9` (current), but the
  `chat_turns` table was physically missing the `embedding` column migration v8→v9 is supposed to add.
  `_init_db()`'s `row["version"] < _SCHEMA_VERSION` gate (§16.4 above) never re-triggers `_migrate()`
  once a DB reports the current version, so the `ALTER TABLE` never ran — the exact class of drift the
  `_check_embedding_provenance()` self-heal above already exists to catch, just on a different
  table/column than the one it was written for (the `embedding_provenance` table's own
  `CREATE TABLE IF NOT EXISTS`). Root cause on the affected DB: an `embedding_provenance` row for
  `'chat_turns'` already existed (seeded from an earlier state before the column was lost), so the
  `if row is None:` branch containing the vulnerable `embedding IS NOT NULL` query was never reached
  either — the provenance check itself couldn't have surfaced this at construction time on this
  particular DB. **Impact was low but real:** `add_chat_turn()`'s `INSERT` names all columns in one
  statement, so the missing column failed the entire insert, not just the embedding — every chat turn
  (both roles) silently stopped persisting to `chat_turns` since the drift happened, caught only by
  `main.py`'s best-effort `_persist_chat_turn()` try/except (never affecting the live response, but
  invisible to conversation history / the Episode Browsing UI's `chat_turns` search, §20). **Fix:** an
  unconditional self-heal added directly in `_init_db()` (not gated by `embedding_model_name`, since
  `add_chat_turn()`'s `INSERT` always names this column regardless of whether an embed source is
  configured) — a `PRAGMA table_info(chat_turns)` check + `ALTER TABLE ... ADD COLUMN embedding BLOB`
  if missing, cheap no-op on a healthy DB, runs on every construction. Live-verified, not just
  unit-tested: uvicorn's `--reload` picked up the code change on the user's actual running backend,
  the self-heal fired for real (`logs/backend.log`: `"chat_turns missing 'embedding' column despite
  schema_version=9 — self-healing via ALTER TABLE"`), and a follow-up `PRAGMA table_info` confirmed the
  column present with no restart required.
- Apple-Silicon skip branch remains verified only via mocked `platform` calls; real non-Darwin
  hardware verification is still outstanding, pending access to such a machine.
- oMLX's `embedding_model` wiring gap noted above (tier 1 never engages for that backend) — a
  follow-up, not yet scheduled.

### 16.5 Live-Switchable Runtime Backend — Built and Live-Verified, 2026-07-15

§16.3's proposal shipped as scoped. Implemented entirely in `backend/main.py`;
`runtime_factory.py`'s existing `create_runtime()`/`available_backends()` required no changes.
Full narrative, including the step-by-step live-verification checklist: `sessions-log.md` §36.

**What was built.** `_build_controller(settings, runtime, memory_manager, embed_fn, project_root,
templates_dir)` — the WikiAgent/ConversationalAgent/ControllerAgent construction + cache-warmup
block, extracted out of `lifespan()` so both startup and a live switch share one code path.
`_resolve_chat_model(settings, backend)` — single precedence rule: `Settings.chat_model` (global
override) beats a new per-backend pin (`chat_model_omlx`/`chat_model_ollama`/`chat_model_foundry`,
`LOCALIST_CHAT_MODEL_OMLX`/`_OLLAMA`/`_FOUNDRY`) beats `None` (falls through to
`runtime_factory.py`'s existing hardcoded per-backend default). `_write_env_var(project_root, key,
value)` — atomic (`tempfile` + `os.replace()`) read-modify-write against `.env`, preserving
comments/blank lines/unrelated keys, skipping commented-out lines when matching the target key.
`_runtime_switch_lock` (`threading.Lock`) serializes any sequence that reads-then-swaps
`_state.runtime`/`.wiki_agent`/`.controller`.

Three new endpoints:
- `POST /settings/runtime-backend` `{backend, chat_model?}` — validates the backend name;
  health-checks the candidate client before mutating anything; on success, rebuilds via
  `_build_controller()`, atomically swaps `_state`, then persists `LOCALIST_RUNTIME_BACKEND` to
  `.env`. An optional `chat_model` on the request also pins that backend (not a one-shot override —
  see the open call noted in the original Claude Code prompt, resolved this way to keep one source
  of truth). If the in-memory swap succeeds but the `.env` write itself fails, the swap is **not**
  rolled back — the response returns `persisted: false` plus a warning instead.
- `GET /settings/runtime-backend/{backend}/models` — builds a throwaway client for `backend`,
  health-checks it, returns its reported models, discards the client. Never touches `_state`.
- `POST /settings/runtime-backend/{backend}/chat-model` `{chat_model}` — always persists the pin
  (Settings field + `.env`) regardless of which backend is active; additionally triggers a live
  rebuild via the same lock/health-check/`_build_controller()` sequence if — and only if —
  `backend` is the currently active one.

**Test suite:** 757 passed / 0 failed (739 baseline + 18 new,
`backend/tests/test_main_runtime_backend_switch.py`), covering unknown-backend rejection,
unreachable-target rejection without side effects, successful-switch state/`.env` assertions, both
chat-model-pin paths (active vs. inactive backend), and `_write_env_var()`/`_resolve_chat_model()`
directly.

**Live verification (2026-07-15, full 9-step checklist against real oMLX/Ollama/Foundry
services):** every mechanic in the proposal confirmed working end to end — health-check-before-swap,
zero-side-effect read-only model listing, `.env` persistence surviving an actual process restart,
clean rejection of an unreachable target with the prior backend provably untouched, and (closing
§16.3 point 6's open item) an in-flight streaming request completing unaffected by a concurrent
switch. Full step-by-step results, including three findings that turned out to be pre-existing
design decisions or environmental constraints rather than defects (the Ollama fail-fast/pin
interaction, oMLX's memory ceiling, and `chmod`'s inability to trigger the `persisted: false` path
via POSIX rename semantics) are in `sessions-log.md` §36.

**Open items:**
- **`embed_fn` staleness across a switch — confirmed already-correct behavior, now hardened.**
  A follow-up code review found that neither `switch_runtime_backend()` nor
  `set_runtime_backend_chat_model()` ever re-derives or reassigns the embedding function from the
  candidate/new runtime: both read it once from `MemoryManager` and thread it through unchanged
  into `_build_controller()`. So a chat-backend switch already never touched the embedding source —
  this was not the bug the original finding above described, just an implicit consequence of how
  the code happened to be written. Hardened rather than fixed: `MemoryManager` gained a read-only
  public `embed_fn` property (replacing a private-attribute reach-through via
  `getattr(memory_manager, "_embed_fn", None)` at both call sites), both call sites gained an inline
  comment stating the decoupling is deliberate, and
  `backend/tests/test_main_runtime_backend_switch.py` gained two identity-check tests
  (`test_successful_switch_leaves_embed_fn_untouched`,
  `test_pin_for_active_backend_leaves_embed_fn_untouched`) asserting `embed_fn` is the exact same
  object before and after a live switch/pin. This guarantee is now explicit and test-covered instead
  of an accidental side effect a future edit could casually "fix" into a coupling bug. See
  `sessions-log.md` §37.
- ~~**Lock held across `await`.**~~ **Closed in §16.6** — `_runtime_switch_lock` is now an
  `asyncio.Lock`, acquired via `async with`.
- The `persisted: false` `.env`-write-failure path needs a different provocation technique for any
  future repeatable verification — a read-only *directory* (not a read-only file) would trigger it
  per POSIX rename semantics; not yet tried.
- ~~The Settings UI (`localist-ui`, §7.10) is not wired to any of these three endpoints.~~
  **Closed in §16.6** — the Runtime Backend segmented control, Chat Model dropdown, and per-backend
  model preview are now wired to all three endpoints.

### 16.6 Settings UI Wired to the Live Switch, and the Lock-Across-`await` Item Closed (2026-07-15)

Two remaining §16.5 open items closed in one pass — full narrative in `sessions-log.md` §38.

**Lock fix.** `_runtime_switch_lock` (`backend/main.py`) changed from `threading.Lock()` to
`asyncio.Lock()`, and both call sites from `with _runtime_switch_lock:` to
`async with _runtime_switch_lock:`. The prior synchronous lock, held across
`await asyncio.to_thread(...)`, didn't just serialize two overlapping switch/pin requests — it
busy-blocked the *entire event loop* while the second waited to acquire, since a plain
`threading.Lock.acquire()` has no cooperative-yield path back to the loop. Confirmed by hand: with
the lock reverted to `threading.Lock`, a new regression test
(`TestConcurrentSwitchRequestsDoNotBlockEventLoop` in
`backend/tests/test_main_runtime_backend_switch.py`) deadlocks outright rather than merely running
slow — the loop can never process the `asyncio.to_thread` completion callback that would let the
lock-holding coroutine resume and release, so nothing ever unblocks. That test now runs the
switch/heartbeat coroutines inside a daemon thread with an external `join(timeout=...)`, since an
in-process `asyncio` timeout can't fire either once the loop itself is frozen. With the
`asyncio.Lock` fix, the same test passes: two concurrent `switch_runtime_backend()` calls both
complete, and a concurrent heartbeat coroutine keeps ticking throughout (proving the loop stayed
responsive) rather than stalling for the ~150ms one request holds the lock.

**Backend prerequisite for the frontend.** `HealthResponse` (`backend/main.py`) gained a `backend:
str` field, sourced from `settings.runtime_backend` in `get_health()` — the one authoritative
signal the frontend needs to know which of `omlx`/`ollama`/`foundry` is actually live; `base_url`
alone can't disambiguate if two backends happen to share a host/port shape.

**Frontend — `localist-ui`.**
- `server.ts`'s `HealthState` gained the matching `backend: string` field (default `''`); no other
  change needed since `checkHealth()` already spreads the full response into the store.
- `model.ts`: `readStoredBackend()`'s `localStorage` value is now only a paint-before-first-health-
  check cache, not the record of truth. A new `health.subscribe(...)` sets `modelConfig.backend =
  $health.backend` whenever `$health.reachable` is true and `$health.backend` is non-empty — so a
  process restart (which reverts to `.env`) or an out-of-band switch (e.g. via `curl`) is reflected
  once the next health poll lands, not just whatever the browser last remembered.
- New `runtimeBackendSwitch.ts` store (follows `chatHistorySettings.ts`'s pattern — loading/error
  writables, no optimistic update on failure): `switchRuntimeBackend(backend, chatModel?)` →
  `POST /api/settings/runtime-backend`; `fetchBackendModels(backend)` →
  `GET /api/settings/runtime-backend/{backend}/models` (returns `[]` on failure rather than
  throwing); `pinChatModel(backend, chatModel)` → `POST /api/settings/runtime-backend/{backend}/chat-model`.
- `routes/settings/+page.svelte`: the Runtime Backend segmented control now calls
  `switchRuntimeBackend()` for real, gated by a plain `confirm()` dialog naming the target backend
  and noting in-flight requests are unaffected but the switch itself takes a few seconds; a no-op
  guard skips the confirm/switch entirely when the clicked backend is already active. The control
  disables and shows a spinner while `runtimeBackendSwitchLoading` is true. A new `selectedUiBackend`
  local variable — deliberately independent of the real active backend — lets a segmented-control
  click preview a *different* backend's models (via `fetchBackendModels()`, re-run whenever
  `selectedUiBackend` changes) before or even without confirming an actual switch to it; a
  `previewing` CSS modifier (`app.css`) distinguishes "previewing" from "active" on the segmented
  buttons. The former read-only "Chat Model" card is now a `<select>` sourced from that backend's
  real model list, `onchange` calling `pinChatModel()`; an inline note states whether the pin applies
  immediately (selected backend is the active one) or only takes effect on a future switch. The
  free-text "Embedding model ID" field (and the rest of the now-empty "Model Configuration" card) was
  deleted — confirmed inert for every backend per §33/§34, and there is still no independent
  embedding-source switch endpoint (out of scope, tracked separately).

**Test suite:** 761 passed / 0 failed (759 baseline + 2 new — the asyncio.Lock type assertion and
the concurrent-requests-don't-block-the-loop regression test). `npm run check` and `npm run build`
both clean.

**Live verification (2026-07-15).** No browser-automation tool is available in this environment
(same limitation §7.10 already flags), so verification was: (1) structural — SSR HTML for
`/settings` diffed for the expected segmented-control/dropdown markup; (2) direct, against the real
running backend (Ollama active) and through the Vite dev server's `/api` proxy — `GET /health`
confirmed the new `backend` field; two genuinely concurrent `POST
.../ollama/chat-model` requests (idempotent re-pin of the already-active model, so no functional
change) both returned `200`/`applied_live: true` in ~3s total with no hang, exercising the fixed
lock on the live server rather than only the mocked test; a `POST` through the `:5173/api` proxy
confirmed the frontend's exact fetch path reaches the backend correctly. `.env` was restored
byte-for-byte after each test. No live click-through/visual QA pass was performed — flagged as a
limitation, not asserted as done, consistent with §7.10's own posture.

**Open items:**
- No live human browser/visual QA pass of the wired Settings UI has been performed as of this
  writing — structural/API-level verification only (see above).
- The `persisted: false` `.env`-write-failure path still needs a different provocation technique
  (carried forward from §16.5 — a read-only *directory*, not a read-only file, would trigger it).

### 16.7 `is_local` — Backend-Tier Classification for Context-Window Ceilings (2026-07-18)

Every concrete `BaseRuntimeClient` now exposes a public `is_local: bool` attribute, set once at
construction and never recomputed per request. It is the single flag `context_profile.py`'s
`ContextProfile` (local/cloud ceiling pairs — Working Memory turn/token caps) keys off;
see §3's Slot 6 for the consuming side. Added specifically to stop truncating conversation history
to a small fixed budget regardless of what the active model can actually hold — the prior ceilings
(5 turns / 300 tokens) were sized for the 16GB Apple-Silicon local-inference case and applied
identically even when Ollama Cloud (~128K real context) was active.

**Per-backend derivation — grounded in live code, not assumed:**

- **`OMLXRuntimeClient.is_local = True`, always.** oMLX only ever runs on-device; no branching.
- **`FoundryRuntimeClient.is_local = True`, always.** The module's own docstring states Foundry is
  "local execution... (local only, never cloud)" (§16.1) — confirmed by grep before writing any
  branching logic for a cloud case that doesn't exist in this deployment.
- **`OllamaRuntimeClient.is_local`** — **not derivable from `base_url`.** The obvious-looking
  signal (a `localhost` vs. a cloud hostname) is wrong for this backend: §16.4's live-verified
  reference configuration runs a cloud chat model (`gemma4:31b-cloud`, "proxied through
  ollama.com — never resident locally") through the *same* local daemon at
  `http://localhost:11434` as any local pull. `base_url` is identical in both cases. The actual
  signal is the model tag itself — Ollama Cloud models carry a `-cloud` suffix on their tag (the
  part after `:`, e.g. `"31b-cloud"` in `"gemma4:31b-cloud"`); local pulls don't. `is_local` is
  computed once from `chat_model` at construction (`_is_cloud_model()` in
  `ollama_runtime_client.py`) and never re-derived from `base_url`.

**Consumers, threaded from `is_local` via `context_profile.profile_for()`:**
- `ControllerAgent._execute_plan()` resolves the profile once per turn (defaulting to local —
  `getattr(self._runtime, "is_local", True)` — so any test double lacking the attribute keeps
  today's behavior rather than silently landing in the cloud tier) and passes
  `working_memory_limit`/`working_memory_tokens` into `MemoryManager.get_context_window()` and
  `working_memory_tokens` into `PromptBuilder.build(working_memory_ceiling=...)`.
- `OllamaRuntimeClient` *used to* also set `options.num_ctx` on every chat request from its own
  resolved profile — **superseded by §16.9 (2026-07-19): confirmed inert and removed.** See §16.9
  for the corrected picture; the rest of this section is left as the historical record of what was
  built at the time, not a live description of current `options.num_ctx` behavior.

**`MemoryManager.get_context_window()`'s `limit` parameter now accepts `None`** (unbounded — no
`LIMIT` clause at all), used only by the cloud profile. The pre-existing `limit=50` default and its
other caller (`format_for_prompt()`'s Synthesizer/multi-agent-research path, `controller_agent.py`
line ~580 — a separate, unrelated code path from the Working Memory slot) are untouched; only the
one Working Memory call site was changed to pass an explicit profile-derived value instead of the
old hardcoded `limit=5, max_tokens=300`.

**Two independent truncation passes, both must scale together.** `MemoryManager.get_context_window()`'s
`max_tokens` trims rows before assembly; `PromptBuilder._slot6_working_memory()`'s own ceiling
(previously the hardcoded class constant `_CEIL_WORKING=300`, now an optional `ceiling` parameter
defaulting to that same constant) truncates again at render time. Raising only one silently leaves
the other re-truncating back down — `PromptBuilder.build()` gained a `working_memory_ceiling`
parameter for exactly this reason, always passed explicitly by `ControllerAgent` alongside the
`get_context_window()` call so the two stay in lockstep.

**Values chosen (`context_profile.py`), not copied from the model's raw ceiling — as of this
session's original implementation (superseded by §16.9 for `total_context_tokens`, since removed):**

| | Local | Cloud |
|---|---|---|
| `working_memory_tokens` | 300 (unchanged) | 60,000 |
| `working_memory_limit` | 5 (unchanged) | None (unbounded) |
| ~~`total_context_tokens` (→ Ollama `num_ctx`)~~ | ~~8,000~~ | ~~100,000~~ |

`total_context_tokens` and its ~28K-headroom-under-128K budget math (described in the paragraph
below, left for historical context) were removed in §16.9 after `num_ctx` was confirmed to have no
effect on Ollama, local or cloud.

Cloud's working_memory_tokens=60,000 was originally budgeted against a 100,000 `total_context_tokens`
figure (100,000 chosen to leave ~28K headroom under Gemma-4-31B's ~128K stated ceiling for
framework/tokenizer overhead), reserving the rest for the other slots' worst case (session files up
to 20K, persona/RAG/tool/graph/working-state slots combined ~3K, output generation ~4K) plus a
safety margin — a deliberate allocation, not the literal model ceiling. The 60,000 figure itself is
unchanged by §16.9's removal (it was never derived *from* `total_context_tokens` at runtime — only
documented alongside it); only the now-dead `total_context_tokens` field and its Ollama `num_ctx`
consumer were removed. Confirmed with the user before the original implementation (three options
presented; the above was selected over both a more conservative 30K/60K pairing and a more
aggressive 100K/128K pairing).

**Deliberately out of scope.** No cost-based guardrail was added for the cloud tier. The RAM-bloat
protection the original local ceilings existed for doesn't apply under cloud — the host Mac's RAM
never held the KV cache to begin with. A cloud prompt of ~60K tokens has a real, non-zero request
cost/latency on Ollama Cloud; this is a known, flagged trade-off, not a guardrail gap to close in
this pass.

**Test suite:** 871 passed / 0 failed (851 baseline + 20 new — `tests/test_runtime_is_local.py`
covering `is_local` per backend and `ContextProfile` value sanity; new cases in
`tests/test_ollama_runtime_client.py::TestIsLocalAndNumCtx` covering cloud/local model-name
derivation and the resulting `options.num_ctx`; `tests/test_memory_phase1.py`'s
`test_limit_none_returns_every_row`; two new cases in `test_controller_phase4.py`'s
`TestContextProfileTiering` confirming the local tier's 5-row cap is unchanged and the cloud tier
removes it end-to-end through a real `ControllerAgent.handle_task()` call).

**Open items:**
- ~~**Live verification against real Ollama Cloud**: confirm `num_ctx` is actually honored
  server-side~~ — **closed by `diagnostics/diag_ollama_num_ctx_probe.py` / `diagnostics/reports/
  ollama_cloud_num_ctx_findings.md`: it is not honored, in either direction, on either tier. See
  §16.9.**
- Whether Ollama Cloud's serving stack does cross-request prefix-hash KV-cache reuse at all is
  unconfirmed — see §3's Slot 6 note. If it doesn't, the cloud profile still delivers the
  reasoning-quality half of the original goal but not a latency/cost win from cache reuse.
- The 60,000/100,000 cloud figures are a deliberate starting budget, not empirically tuned against
  real long-conversation traffic — may need revisiting once there's live usage data.

### §16.8 — LOCAL_PROFILE working-memory ceiling investigated, left unchanged (2026-07-19)

Investigated whether `LOCAL_PROFILE`'s 300-token/5-turn working-memory ceiling (unchanged since
the pre-`ContextProfile` hardcoded behavior — see §16.7 above) could be safely raised. Full
methodology and data: `diagnostics/reports/local_working_memory_ram_findings.md`; the resulting
rationale also lives directly in `context_profile.py`'s module docstring.

**Result: no safe increase.** Built `diagnostics/diag_local_working_memory_ram_probe.py`, which
sends real `conversation_log` text (not synthetic filler) at working-memory sizes from 300 to
3,000 tokens against both local backends — oMLX and local (non-`-cloud`) Ollama — 3 trials each,
measuring per-process memory footprint and `vm_stat` swap-in/out deltas during live generation.

A real methodology bug surfaced and was fixed during this investigation: on Apple Silicon,
`psutil`'s per-process RSS does not capture Metal/unified-memory GPU-resident allocations — it
reported ~200MB for the Ollama runner process at the same instant `top`'s memory-footprint column
reported ~8.3GB for that same pid, a ~40x undercount. The probe uses `top -l 1 -pid <pid> -stats
mem` as ground truth instead.

Marginal cost came out modest in isolation (~320MB/1,000 tokens for local Ollama, ~190MB/1,000 for
oMLX — confirming the two local serving stacks have meaningfully different memory profiles, as
expected), but every single trial at every token size tested — including at today's unchanged
300-token baseline — produced measurable swap activity under this machine's ordinary real
(non-idle) background load. Per this project's own standard for this class of measurement (watch
the actual failure precursor, not just headroom arithmetic), that outweighs the naive "there's
still N GB free" calculation. `LOCAL_PROFILE` values are unchanged: `working_memory_tokens=300`,
`working_memory_limit=5`.

Also confirmed against the live `conversation_log` table: the turn-count and token ceilings are
both real, coupled constraints today — 107/318 (34%) of real conversation windows already hit the
300-token trim at 5 rows, so raising the row-count limit alone (without the token ceiling) would be
a near-total no-op for a third of real usage.

Added `context_profile.check_local_ram_headroom()` — not gating a raised budget (there isn't one),
but a lightweight, warn-only startup check (wired into `main.py`'s `lifespan()`, gated on the
active runtime's `is_local` flag) that surfaces the same already-happening swap-under-load
condition on whatever machine the app next runs on, mirroring `lifespan()`'s existing
reachable/not-reachable warn-and-continue pattern for runtime health. Thresholds
(`available < 8.0GB` or `swap > 40%`) are set from this session's own measured numbers (this
machine's pre-test state — 6.60GB available, 61.3% swap — already produced swapping throughout),
with margin added so the check doesn't always- or never-fire.

**Open items:**
- One-machine, one-session snapshot, taken under real (not synthetic-idle) background load — not
  a permanent hardware constant. Re-run the probe script if RAM changes, the local model changes,
  or a specifically quieter/busier baseline needs characterizing.
- `top`'s memory-footprint reporting has ~0.1GB quantization noise at the gigabyte boundary; the
  marginal-cost figures above are directional, not exact to the MB.

### §16.9 — `total_context_tokens`/`num_ctx` removed; `LOCAL_PROFILE.working_memory_tokens` made RAM-tiered (2026-07-19)

Two changes, done together because the second built directly on the first's cleanup.

**Part 1 — `total_context_tokens` deleted (confirmed-dead parameter).**
`diagnostics/reports/ollama_cloud_num_ctx_findings.md` (a prior session's investigation) found that
`options.num_ctx` — the field §16.7 describes `OllamaRuntimeClient` setting from this profile field
— has **zero observed effect on Ollama, local or cloud**: identical `prompt_eval_count` across a
25x range of requested `num_ctx` values, including a request smaller than the actual prompt with no
truncation; the real, always-enforced ceiling is each model's hardcoded native context length,
completely independent of what the client sends. §16.7's description of this mechanism (and the
open item asking to verify it) is now known incorrect and has been corrected in place above, not
rewritten — this section is the actual current state.

Removed: the `total_context_tokens` field from `ContextProfile`, its values from `LOCAL_PROFILE`
(8,000) and `CLOUD_PROFILE` (100,000), `OllamaRuntimeClient._num_ctx`, and the `"num_ctx"` key from
its request payload — rather than leave a dead parameter (and the now-unjustifiable "~28K headroom"
budget narrative built around it) in place. `tests/test_runtime_is_local.py` and
`tests/test_ollama_runtime_client.py` (formerly `TestIsLocalAndNumCtx`) updated to stop asserting on
the removed field/behavior and instead assert `"num_ctx"` is **absent** from the Ollama request
payload, so a future change can't silently reintroduce it without a test noticing.

**Part 2 — `LOCAL_PROFILE.working_memory_tokens` made RAM-tiered, fail-closed.**
§16.8's finding (no safe increase on this 16GB Mac) is a fact about *this machine*, not a formula for
other RAM sizes — memory headroom doesn't scale linearly with total RAM (OS overhead, background
apps, and the model's own fixed footprint don't scale with it either), so a 32GB budget needs its
own measurement, not "300 × 2".

`context_profile.py` now resolves `LOCAL_PROFILE.working_memory_tokens` at import time from
detected total RAM (`psutil.virtual_memory().total` — a system-wide read, confirmed *not* subject to
the per-process Metal/unified-memory RSS-undercount bug found in §16.8, since that bug is specific
to per-process RSS, a different metric) against `_VALIDATED_LOCAL_TIERS`, a dict seeded with
**exactly one entry: `{16: 300}`**, the §16.8 measurement. `resolve_local_working_memory_tokens()`
fails closed: any RAM size without its own validated entry — above 16GB (e.g. 32GB) or below it —
gets the same conservative 300-token value rather than an extrapolated guess, and logs clearly
(`logger.warning`) that it did so, distinguishing "below the smallest validated tier" (more
concerning — even the fallback is unproven at a smaller size) from "above the largest validated
tier" (falling back to the smaller machine's known-safe number). A machine that actually matches the
validated tier (within a 4GB margin, to absorb macOS's decimal-GB-vs-marketed-GiB rounding — this
dev Mac reports `total=17.18GB` for hardware sold as "16GB") logs an informational match instead.

**No 32GB (or other) tier was added — this ships the mechanism, not a second number.** The
follow-up is explicit and small: re-run `diagnostics/diag_local_working_memory_ram_probe.py` (already
backend- and machine-agnostic by construction, no changes needed) on real hardware of a new size and
add the measured result to `_VALIDATED_LOCAL_TIERS`.

**A real logging-visibility bug caught and fixed while wiring this in:** `context_profile.py` is
imported (resolving `LOCAL_PROFILE` and logging the tier match/fallback) before `main.py`'s
`lifespan()` calls `logging.basicConfig()` — Python's default no-handler-configured behavior only
surfaces WARNING+ via a last-resort stderr handler, so the common-case INFO "matched validated
tier" line would never reach the app's real log output, only the rarer WARNING fallback lines
would (by accident, not design). Added `context_profile.log_local_working_memory_tier()`, called
from `lifespan()` (alongside `check_local_ram_headroom()`) purely to re-emit the same resolution's
log line after logging is actually configured, so it's genuinely visible at startup rather than
lost to import ordering.

**Test suite:** 871 passed / 0 failed, unchanged from §16.8's baseline (removed/updated assertions
replaced 1:1, new tier-resolution tests added — `resolve_local_working_memory_tokens(16.0) == 300`,
a sub-16GB input falls back to 300 with a warning log, and a 32GB input falls back to 300 with a
distinct "no validated tier" warning log; deliberately no test asserts a specific 32GB token value).

### §16.10 — `OMLXRuntimeClient` exposes `max_model_len` from `health_check()` (2026-07-19)

`GET /v1/models` on oMLX returns a vLLM-compatible extension per model entry, `max_model_len` (the
loaded model's real effective context window; can be `null`, e.g. for oMLX's markitdown
pseudo-model entry). `health_check()`'s existing model-listing call now also locates the active
chat model's entry and caches its `max_model_len` onto a new `self.max_model_len` instance
attribute — the same plain-attribute pattern already used for `self.is_local`, not a new getter
method, so callers read it the same way. `null` and a missing field are treated identically (both
just mean "not reported") and fall back to a new `_DEFAULT_MAX_MODEL_LEN = 8192` module constant —
a safe floor most local chat models meet or exceed — via an explicit `isinstance(x, int)` check
rather than a falsy/truthy check, so a `0` or unexpected type can't silently pass through as if it
were a real reported value. This value stays at the fallback until the first successful health
check; nothing in this codebase calls `health_check()` synchronously before the first inference, so
early requests use the fallback rather than blocking on one.

This is the enabling change for §16.11 below — `context_profile.py`'s `LOCAL_PROFILE` budget is
built directly from this value rather than from detected host RAM.

**Test suite:** 878 passed / 0 failed (877 baseline + 1 net new — three new `health_check()` tests
for the real-integer/null/missing-field cases, offset by no removals).

### §16.11 — `LOCAL_PROFILE` retires RAM-tiering for a real per-model budget; `profile_for()` signature change (2026-07-19)

§16.8/§16.9's RAM-tiered lookup (`_VALIDATED_LOCAL_TIERS`, `resolve_local_working_memory_tokens()`)
is **retired outright, not re-measured or extended to a second tier.** It was confirmed to be
measuring the wrong layer: oMLX runs as a separate process with its own paged KV cache and memory
enforcer that already manages RAM dynamically per-machine, independent of total system RAM, so a
host-RAM-keyed lookup was never actually bounding the thing that matters. §16.10's `max_model_len`
gives a real, per-model, server-reported number to budget against instead.

`LOCAL_PROFILE.working_memory_tokens` is now `max_model_len - 27,000`, floored at `300`. The
27,000-token reservation is the same per-slot breakdown `CLOUD_PROFILE`'s docstring already used —
session files up to 20K (`PromptBuilder._CEIL_SESSION_FILES_TOTAL`), persona/RAG/tool/graph/
working-state slots combined ~3K, output generation ~4K — reused here because those slot ceilings
are shared `PromptBuilder` constants, not backend-tier-dependent; unlike `CLOUD_PROFILE` there is no
further "inexact context-length" safety margin stacked on top, since `max_model_len` is a value the
server reports for the model it actually loaded, not an assumption. The 300-token floor exists so a
small or unreported `max_model_len` (e.g. §16.10's own conservative fallback, before a health check
or on an oMLX version that never reports the field) can't drive the reservation past the window and
leave working memory at zero or negative — 300 is deliberately the same figure this profile used
under the retired RAM-tiered approach, so the degraded case lands on a familiar, previously-live
number rather than an untested new one. `working_memory_limit` is now `None` on **both** profiles —
the local tier's fixed 5-turn cap is gone; the token budget above is the only real limiter, matching
`CLOUD_PROFILE`'s existing pattern (that 5-turn number was never derived from anything about the
model or the machine in the first place, just carried over unexamined from the original 300-token/
5-turn pair).

**`profile_for()`'s signature changed**, from `profile_for(is_local: bool)` to
`profile_for(runtime: object)` — necessary, not cosmetic: the local budget now depends on a second
live value, `runtime.max_model_len`, which can change after a runtime-backend swap
(`POST /settings/runtime-backend`, §16.5/§16.6) or once the first health check completes, so
`LOCAL_PROFILE` can no longer be a fixed module-level constant resolved once at import time — it's
rebuilt fresh on every call from whatever the passed-in runtime currently reports. Both `is_local`
and `max_model_len` are read via `getattr` with a documented fallback rather than direct attribute
access: `is_local` defaults to `True` when absent (fails toward the more conservative tier, same
posture as the pre-existing pattern this replaces); `max_model_len` defaults to
`_LOCAL_MAX_MODEL_LEN_FALLBACK` (also `8192`) when absent *or present but not an int* — the
`isinstance` guard matters concretely for test doubles: a bare `MagicMock`'s unconfigured attribute
access returns another `MagicMock`, not an `AttributeError`, so a plain `getattr(..., default)`
alone would silently pass a non-numeric value into the subtraction below and raise `TypeError` at
call time — confirmed by a real test failure while wiring this in (`controller_agent.py`'s
`_dispatch_conversational_with_empty_guard`, a separate method/scope from the primary
`_execute_plan` call site, hadn't recomputed the runtime-type check it needed independently).

`controller_agent.py`'s one call site (`profile_for(getattr(self._runtime, "is_local", True))` →
`profile_for(self._runtime)`) and `main.py`'s startup log line (previously
`log_local_working_memory_tier()`, now removed — logging the resolved profile no longer needs the
import-time-vs-`logging.basicConfig()` ordering workaround §16.9 added, since `profile_for()` is no
longer resolved at import time at all) were both updated accordingly.
`check_local_ram_headroom()` **stays, but is now purely a startup observability log line** — its
docstring is updated to say so explicitly; nothing that computes `working_memory_tokens` reads its
return value any more.

**Test suite:** 884 passed / 0 failed (878 baseline + 6 net new — `test_runtime_is_local.py`
rewritten around a `_FakeRuntime` double covering the real-`max_model_len`, floor-triggering,
missing-attribute, and non-int fallback cases plus an integration check against a real
pre-health-check `OMLXRuntimeClient()`; `test_controller_phase4.py`'s
`TestContextProfileTiering::test_local_runtime_keeps_5_turn_cap` — asserting the now-retired 5-turn
cap — rewritten to assert token-budget trimming with no row cap instead).

### §16.12 — oMLX real multi-turn `messages` array, and "Prompt too long" 400 recovery (2026-07-19)

oMLX's own web UI sends a genuine multi-turn `messages` array — one leading system message, then
one array entry per prior conversation turn, then the live query last — which is what lets its
prefix cache reuse KV blocks turn-over-turn. `OMLXRuntimeClient.infer_stream()` previously packed
the *entire* assembled prompt (including all flattened working-memory turns) into one system
message + one user message, foreclosing that reuse for the working-memory portion of the prompt
entirely; see §3.7d in `03-unified-prompt-contract.md` for the prompt-assembly side of this change
and how it relates to §3.7b's speculative "future engine" framing.

`infer()`/`infer_stream()` gained an oMLX-only `working_memory_turns: list[Turn] | None = None`
keyword. This is invisible to Ollama/Foundry: `@runtime_checkable` Protocol conformance
(`base_runtime_client.py`) only checks member *presence*, not signatures, so an extra optional
parameter on one concrete client doesn't affect `isinstance()` checks against `BaseRuntimeClient`,
and `conversational_agent.py` only ever passes this kwarg when
`context["_prebuilt_working_memory_turns"]` is present — which `controller_agent.py` only sets when
`isinstance(self._runtime, OMLXRuntimeClient)` is true (checked independently at both the primary
`_execute_plan` call site and the separate-scope empty-answer retry path). Ollama/Foundry files have
zero diff from this change.

When provided, each turn becomes its own `messages` entry — chronological, oldest first — with
`turn.role` mapped `"agent"` → `"assistant"` (`memory_manager.py`'s `add_agent_result()` stores
`role="agent"`, the correct value everywhere else in this codebase — DB rows, `PromptBuilder`'s
flattened `[WORKING MEMORY]` text — but not a role oMLX's OpenAI-compatible endpoint recognizes;
mapped at this message-building boundary only). `None` (every non-oMLX call site) reproduces the
prior single system+user message shape exactly.

**"Prompt too long" 400 handling.** oMLX's `server.py` `validate_context_window()` 400s with a body
shaped `{"detail": "Prompt too long: {N} tokens exceeds max context window of {M} tokens"}`. A new
`_is_prompt_too_long_response()` matches on the distinctive `"Prompt too long:"` marker (JSON
`detail` field first, falling back to raw response text) so this is distinguishable from every other
400. On a match: drops the single oldest working-memory message from the outgoing array — tracked
via a `working_memory_start_idx`, never the system message or the current query — and retries
exactly once through a new shared `_post_chat_completion()` helper (factored out so the initial
attempt and the retry use identical transport/error-normalization code). A second "Prompt too long"
400 raises a clear, distinguishable `RuntimeError` (does not loop, does not drop a second turn); if
there was no working-memory turn available to drop in the first place, the same clear error is
raised immediately without attempting a retry. Any other non-200 (including a differently-shaped
400) is untouched — same generic-error handling as before this change.

**Test suite:** 888 passed / 0 failed (884 baseline + 4 net new, in
`test_omlx_runtime_client_concurrency.py::TestPromptTooLongRetry` — normal 200 never triggers the
retry path; a "Prompt too long" 400 then 200 drops exactly the oldest turn and returns the retried
response; two "Prompt too long" 400s in a row raise a distinguishable error with no third attempt;
a differently-shaped 400 falls through to the pre-existing generic-error path untouched). No live
oMLX server was used for any of this — all HTTP is mocked, consistent with the rest of this suite
per `CLAUDE.md`. Real end-to-end KV-cache-reuse verification (e.g. via oMLX's `/admin/api/cache/probe`,
§3.7c) was not performed this session and remains an explicit open item.

### §16.13 — Scoped and rejected: oMLX Context Benchmark API integration (decision record, 2026-07-31)

**No code changes.** oMLX v0.5.4rc1's release notes introduced a "Context Benchmark" tool
("measures the largest context the current Mac can prefill, verifies it with a real request, and
applies the result to model settings," available via "Admin API, web dashboard, and macOS app") —
a natural-looking candidate to feed §16.10/§16.11's `max_model_len`-based budget from a fresher,
Localist-triggered measurement instead of just reading oMLX's own reported value. Investigated live
against a real oMLX 0.5.4rc1 instance (upgraded in-place via `brew upgrade omlx` from 0.4.4, no
re-tap needed) to determine whether the "Admin API" claim was a stable, scriptable contract or just
the dashboard hitting its own internal routes.

**Findings.** The routes are real, live under `/admin/api/bench/context/`
(`start`/`active`/`{bench_id}/results`/`{bench_id}/stream` SSE/`{bench_id}/cancel`) — confirmed via
`/openapi.json` (one combined schema for `/v1/*` and `/admin/*`; no separate `/admin/openapi.json`)
and by reading the installed package source directly, since every one of these routes has an
untyped (`{}`) response schema in OpenAPI. Four properties make this a poor integration target:

- **No dry-run.** Every successful run ends by writing `max_context_window` into the model's own
  settings as a side effect (`context_benchmark.py`'s "apply" phase) — there is no request field to
  read the measured number without oMLX applying it.
- **Disruptive side effect.** Starting a run force-unloads every other currently-loaded model first
  (needed for a clean memory measurement), then reloads — not safe to trigger silently or on a
  schedule against a live system something else might be using.
- **Fragile surface.** `POST start` with no/malformed body threw an unhandled 500, not a clean 4xx,
  during testing — rc-quality, not a hardened contract.
- **Environment-dependent auth.** `/admin/*` routes gate on a session cookie via `/admin/api/login`
  (`require_admin`), not the bearer-token scheme `/v1/*` uses — a heavier flow than this codebase's
  existing runtime clients implement, and the test machine's `skip_api_key_verification=true` (which
  bypasses it entirely) can't be assumed for other installs.

No CLI equivalent exists (`omlx --help` exposes only `{start, stop, restart, serve, launch,
diagnose}`) — this feature is HTTP-only. **Stability verdict: low-to-medium confidence** — the
routes live under `/admin/api/`, not the versioned `/v1/` surface §16.10 already reads from, with
untyped responses and rc-quality error handling; expect the contract to move without notice across
oMLX releases.

**Decision: Localist will not integrate with the Context Benchmark API.** Separation of concerns
stays exactly where §16.11 already put it — oMLX (the inference engine) owns measuring and setting
its own context ceiling; a user who wants to run the Context Benchmark does so directly in oMLX's
own admin panel, which can present its own model-unload warning and apply its own result. Localist
stays a consumer only, reading `max_model_len` via `OMLXRuntimeClient.health_check()` (§16.10) and
deriving `working_memory_tokens` from it (§16.11), unchanged. No `OMLXBenchmarkClient`, no
admin-session-cookie auth handling, no SSE/polling logic, no version-compatibility shim for an
unstable rc endpoint was added. This also means Localist never calls `start`, so it never triggers
the disruptive multi-model unload or the no-dry-run auto-apply behavior described above.

**Reopen only if:** oMLX moves this benchmark under the versioned `/v1/` surface with typed
responses (a real stability signal), or a concrete product reason emerges for Localist to trigger
benchmarking on the user's behalf rather than pointing them at oMLX's own panel — and re-verify the
endpoint shapes fresh against whatever oMLX version is current at that time rather than trusting
this section's specifics indefinitely; they were measured against one rc build on one machine.

### §16.14 — `POST /settings/embedding-model`: a live, Ollama-backed embedding-model switch (2026-09-05)

Closes the "no independent embedding-source switch endpoint (out of scope, tracked separately)" gap
§16.6 left open. Built specifically for the packaged Tauri desktop build (see
`docs/architecture/07-localist-ui.md`'s packaging notes): the base-only PyInstaller freeze ships
with no `mlx-embeddings` at all (kept out deliberately for license/bundle-size reasons —
`THIRD_PARTY_LICENSES.md`), so tier 2 (`EmbeddingEngine`) can never engage there regardless of
`LOCALIST_EMBEDDING_ENGINE_ENABLED`. Ollama's `/api/embed` (tier 1, §16.4) is the only local
embedding path the desktop build can actually offer, and until this endpoint there was no UI or API
surface to configure it after first launch — only the CLI/dev-only `start_localist.sh` first-run
prompt, which the Tauri shell never runs (it spawns the frozen backend binary directly, per
`localist-ui/src-tauri/src/lib.rs`, not through the shell script).

**What was built.** `POST /settings/embedding-model` (`main.py`), request `{model: string}`. A
non-empty `model` health-checks it against the *currently active* runtime backend before committing
anything (`_create_and_check_embedding_candidate()`, mirroring `_create_and_check_backend()` but
overriding `embedding_model` instead of `backend`); an empty `model` clears the tier-1 override.
oMLX is rejected up front with a 409 (not left to surface as a confusing "model not found" 422) —
`runtime_factory._make_omlx()` hardcodes `embedding_model=""` regardless of what's requested
(§16.4's already-documented, still-unscheduled gap), so a health-check-based error there would blame
the wrong thing. Rebuilds `_state.runtime`/`.wiki_agent`/`.controller` under the same
`_runtime_switch_lock` a runtime-backend switch uses — even though the chat backend/model are
unchanged — specifically so `GET /health`'s `embed_model_found`/`embedding_model` (the latter field
newly added to `HealthResponse`, sourced from `settings.embedding_model`) reflect the change
immediately: each concrete runtime client's `health_check()` compares against the `embedding_model`
baked in at its own construction, not something passed fresh per call, so leaving `_state.runtime`
pointed at the old client would have left the health badge silently stale.

**Clearing falls back to tier 2, not straight to tier 3 — a live-caught bug, not a design choice
made correctly the first time.** The first implementation set `embed_fn = None` unconditionally
when `model` was empty, which is wrong whenever `_state.embedding_engine` already holds a loaded,
available instance from startup (the common case on the full dev build, where tier 2 engages
because `LOCALIST_EMBEDDING_MODEL` is normally unset) — clearing a tier-1 override would have
silently dropped a working embedder straight to keyword-only instead of restoring the
already-loaded `EmbeddingEngine`. Caught by manually exercising the real endpoint against a live
Ollama daemon (set → `nomic-embed-text:latest`, confirmed via `/api/embed`; clear → checked
`GET /health` came back `embed_model_found: false` when it should have stayed `true` via
`EmbeddingEngine`), not by the first pass of unit tests, which had mocked `_state.embedding_engine`
to `None` throughout and so never exercised the fallback path at all. Fixed: clearing now re-derives
`embed_fn` from `_state.embedding_engine.embed` when that instance is non-`None` and `.available`,
exactly mirroring what `_configure_embedding_source()` would produce for an unset
`LOCALIST_EMBEDDING_MODEL` at a fresh boot. Does not attempt to *newly construct* an `EmbeddingEngine`
live — only reuses one already loaded at startup — so this fallback is a no-op (falls through to
keyword-only) on the desktop build, where no such instance ever exists.

**A second, related bug caught in the same pass: naming, not just selection.**
`_derive_active_embedding_model_name(settings, embed_fn, embedding_engine)` checks
`embedding_engine is not None` *before* checking `embed_fn is not None` — a contract that only holds
because `_configure_embedding_source()` guarantees the two are mutually exclusive at startup
(`embedding_engine` is never constructed at all when tier 1 engages). This endpoint's rebuild path
doesn't share that guarantee automatically: `_state.embedding_engine` stays whatever `lifespan()`
left it as regardless of which tier this endpoint activates, so passing it through unconditionally
when *setting* a tier-1 model would have made the derived name resolve to the stale
`EmbeddingEngine`'s `model_path` instead of the actual Ollama model — silently defeating the
Planner's `_TUNED_EMBEDDING_MODEL` mismatch guard (§16.4) by comparing against the wrong name and
never warning on a real mismatch. Fixed by passing `embedding_engine=None` to the naming call
whenever a tier-1 `model` is set (restoring the mutual-exclusivity the function assumes) and the
real `_state.embedding_engine` only when clearing. The derived name — not the raw `model` string —
is what's threaded into `MemoryManager.set_embedding_source()`, so provenance checking (below)
compares against the vectors' actual source model in both directions.

**`MemoryManager.set_embedding_source(embed_fn, embedding_model_name)`** — new public method,
`memory_manager.py`. `embed_fn` is a read-only *property* (§16.5: a runtime-backend switch must
never implicitly change it), but this is the dedicated, intentional path for actually changing it on
a deliberate embedding-model switch. Re-runs `_check_embedding_provenance()` whenever
`embedding_model_name` is not `None`, so a live switch gets the exact same mismatch handling a fresh
restart with the new model configured would: episodes re-embed in place immediately (small, bounded
cost); corpus and chat_turns are flagged stale for a manual `POST /memory/reembed` /
`POST /memory/reembed-chat-turns` (both already existed, unchanged) rather than silently re-ranking
with vectors from a different model's geometry.

**Frontend.** New `embeddingModelSwitch.ts` store (`setEmbeddingModel()`,
`embeddingModelSwitchLoading`/`Error` writables) following `runtimeBackendSwitch.ts`'s
no-optimistic-update-on-failure pattern. `HealthResponse` gained `embedding_model: string`
(mirrored into `localist-ui`'s `HealthState`) specifically so the Settings UI has something to
select against — there was previously no GET surface for the configured value at all, only the
boolean `embed_model_found`. `routes/settings/+page.svelte` gained an "Embedding Model — Ollama"
card, shown only when `$modelConfig.backend === 'ollama'` (the *real* active backend, not the Chat
Model card's preview-only `selectedUiBackend` — an embedding-model switch always targets whichever
backend is actually live, there is no "preview a different backend's embedding models" concept
the way chat-model preview has). Model choices are fetched via the existing
`GET /settings/runtime-backend/{backend}/models` (unfiltered by embedding-capability, same
established convention the Chat Model dropdown already uses — Ollama's `/api/tags` doesn't cleanly
separate chat- from embedding-capable models). Foundry was deliberately left out of the UI for this
pass even though the backend endpoint is backend-agnostic and already supports it (§16.4) — a
smaller audience, scoped out rather than built speculatively; oMLX is excluded per the 409 above.

**Live-verified end-to-end (2026-09-05), not just unit-tested** — against the real Ollama daemon,
real `.env`, and the real dev `localist_memory.db`: set → `nomic-embed-text:latest`
(`GET /health` reflected it immediately, `.env` persisted, corpus/chat_turns correctly flagged
stale — a genuine cross-model mismatch); clear → confirmed falls back to the already-loaded
`EmbeddingEngine` (`embed_model_found` stayed `true`, `embedding_model` reported `""`) rather than
keyword-only; browser-driven (Playwright, no project-specific run skill existed for this yet)
click-through of the new Settings card confirmed the dropdown reflects `$health.embedding_model`,
selecting a model shows the "Embedding model set to …" confirmation and the pre-existing "Corpus
embeddings out of date" badge, and re-selecting "None" clears back to it. Real corpus/chat_turns
re-embedded back to a consistent state (`POST /memory/reembed` / `-chat-turns`) after testing so the
user's actual dev database was left exactly as found.

**Test suite:** `backend/tests/test_main_embedding_model_switch.py` (new — oMLX rejection,
unreachable-backend rejection, model-not-found rejection, successful set, the
embedding-engine-fallback-on-clear case, the stale-embedding-engine-ignored-for-naming case, and the
`.env`-write-failure/`persisted: false` path) plus three new cases in
`tests/test_embedding_provenance.py::TestSetEmbeddingSource` covering `set_embedding_source()`
directly (re-embeds episodes on a new model, flags corpus stale on mismatch, clearing skips the
provenance check). 1551 passed / 0 failed full-suite after this change (1540 baseline).

**Open items:**
- Foundry is not exposed in the Settings UI for this endpoint, though the backend supports it
  identically to Ollama — pick up if there's demand.
- No first-run/onboarding surface points a new desktop user at this control the way
  `start_localist.sh`'s interactive prompt does on the CLI/dev build — a fresh packaged install
  still defaults to keyword-only with no in-app nudge to configure Ollama embeddings. Tracked
  alongside the broader "first-run config UX" item already open for the Tauri shell (top-level
  `README.md`'s Roadmap, `backend/packaging/README.md`).
- **Closed 2026-09-06 (§16.17):** switching to a non-tuned embedding model with no validated
  threshold set used to mean permanently-disabled semantic gating for it, short of a developer
  manually re-running the offline diagnostic and shipping a hardcoded fix — live, in-app
  calibration now runs automatically on switch (and re-runs on demand via "Re-embed Corpus Now"),
  with results surfaced as a Settings UI trust badge.

### §16.15 — `window.confirm()` is a silent no-op in the packaged Tauri webview; fixed via `tauri-plugin-dialog` (2026-09-05)

Found immediately after §16.14 shipped: a user selected `nomic-embed-text:latest` in the new
Embedding Model card (correctly flagging `corpus_stale`, per §16.4's designed behavior — not a
bug), then clicked "Re-embed Corpus Now" and reported the warning never cleared. Backend-side
`POST /memory/reembed` was proven fully functional in isolation (direct `curl` against the running
packaged app: 41/41 documents re-embedded in under 2 seconds, `corpus_stale` correctly flipped to
`false`) — ruling out the user's own hypothesis that Ollama needed the model "loaded" first (Ollama
loads a pulled model into memory on first request automatically; no explicit load step exists or is
needed, and the health-check gate on the switch endpoint already requires the model to be present in
`ollama list`/`/api/tags` before the switch is even accepted).

**Root cause, proven empirically, not assumed.** `EpisodesPanel`/Settings' `handleReembedCorpus()`
and `selectRuntimeBackend()` both gate on the browser-native `window.confirm(...)`. Verified live by
temporarily injecting a bare `confirm('...')` call into `app.html`'s `<head>` (a synchronous call
that should block all further script execution and page rendering until dismissed) and rebuilding:
the packaged app's window rendered a fully-interactive page straight through it — no dialog ever
appeared, proving the packaged app's WKWebView instance has no `WKUIDelegate` wired up for JS
dialogs at all, so `confirm()` returns immediately (falsy) with nothing shown. This silently hit
every `if (!proceed) return;` gate in the codebase — both the Corpus re-embed button and the Runtime
Backend segmented-control switch were affected identically, the confirm-dialog bug is unrelated to
this session's embedding-model work and would have existed since §16.6's original Tauri wiring; it
simply had no prior report because nobody had hit a `confirm()`-gated control in the packaged app
and noticed the silence before now. The `:5173` dev flow was never affected — real browsers
implement `window.confirm()` natively.

**Fix.** Added `tauri-plugin-dialog` (`src-tauri/Cargo.toml`, registered via
`.plugin(tauri_plugin_dialog::init())` in `lib.rs`; `dialog:allow-confirm`/`dialog:allow-ask`
permissions added to `capabilities/default.json`) plus the `@tauri-apps/plugin-dialog` /
`@tauri-apps/api` npm packages — the first genuine use of any Tauri-specific JS API in this
frontend; previously the Rust shell was entirely invisible to the Svelte code. New
`localist-ui/src/lib/confirmDialog.ts` exports `confirmDialog(message, title?): Promise<boolean>`,
branching on `@tauri-apps/api/core`'s `isTauri()`: inside the packaged app it calls the plugin's
`confirm()` (a real native `NSAlert`-backed dialog, confirmed via the same injection technique —
this time the dialog rendered, showed Cancel/OK, and blocked as expected); in the browser dev flow
it falls back to `window.confirm()` unchanged. Both call sites in
`routes/settings/+page.svelte` (`handleReembedCorpus`, `selectRuntimeBackend`) now `await
confirmDialog(...)` instead of calling `confirm()` synchronously.

**Live-verified end-to-end (2026-09-05)**, both before and after, via the same startup-injection
technique (temporary, reverted before landing) against real debug builds — not just unit/type
checks: pre-fix, an injected `confirm()` call was silently skipped with no dialog; post-fix, an
injected `confirmDialog()` call showed a real native dialog and correctly blocked. Real corpus and
chat_turns staleness on the user's actual desktop install were then cleared directly via
`POST /memory/reembed` / `-chat-turns` (203/204 chat turns — one row's embed call failed and was
skipped per `reembed_chat_turns()`'s existing per-row try/except, left for a future manual retry,
not investigated further this session) so their real data reflects `nomic-embed-text:latest`
end-to-end.

**A related latent bug, not yet hit but now visible from the same root cause:**
`reembed_corpus()`/`reembed_chat_turns()` (`memory_manager.py`) unconditionally clear
`_corpus_stale`/`_chat_turns_stale` and advance the `embedding_provenance` row after a run,
regardless of how many individual documents actually failed to embed (`reembedded < total`). A
corpus where *every* row's embed call failed (e.g. a genuinely unreachable Ollama daemon mid-run)
would still be marked fresh and provenance-advanced, leaving stale vectors under a false "not stale"
flag. Not triggered here — the live run above succeeded for all but one row — but flagged as a
pre-existing gap worth closing (e.g., only clearing the flag when `reembedded == total`, or
surfacing a partial-failure warning distinct from full success) rather than assumed fixed by this
session's changes.

**Test suite:** no backend changes in this entry — frontend/Rust only.
`localist-ui`'s `npm run check` clean (0 errors/warnings) before and after.

**Open items:**
- The `reembedded < total` silent-success gap above (`memory_manager.py`) — not closed this
  session.
- No automated test covers the Tauri-webview dialog path (inherently hard to unit-test — this was
  caught and fixed via live injection against a real build, not a test file). A future regression
  here would need the same manual technique to catch, or a Tauri-specific e2e harness this project
  doesn't have yet.

### §16.17 — Live, in-app semantic-gating threshold calibration (2026-09-06)

**The gap this closes.** §16.15's `_VALIDATED_MODEL_THRESHOLDS` dict (planner.py) gave
`nomic-embed-text:latest` its own validated threshold set instead of falling all the way through to
disabled semantic gating — but that dict is populated entirely by a developer manually running
`diagnostics/nomic_embed_text_threshold_probe.py` against a real embedding model, reading a
markdown report, and hand-transcribing four numbers into source. A user who picks any *other*
Ollama embedding model via `POST /settings/embedding-model` (§16.14) had no way, from inside the
running app, to get real semantic gating for it — permanently stuck disabled unless a developer
repeated that whole manual cycle and shipped a new build. Full design record:
`PLAN_semantic_gating_calibration.md`.

**`threshold_calibration.py` — `calibrate_thresholds(embed_fn) -> CalibrationResult`.** Reuses the
exact same fixture battery as the offline diagnostic (extracted into
`threshold_calibration_fixtures.py`, the single source of truth both paths read from, so a live
result stays directly comparable to a hand-reviewed one) and `planner.py`'s own
`_cosine_similarity`/template constants. For each of the four gates
(`explicit_search_action`/`lookup_request`/`research_intent`/`episodic_relevance`), scores every
fixture utterance and picks a threshold via **Youden's J statistic** (max TPR−FPR over a fixed
grid, matching the diagnostic script's own grid so live and offline results stay comparable) —
chosen over a stricter "lowest threshold with zero false positives" rule because this tier is
realistically the dominant experience for most end users, who are unlikely to know they need two
different Ollama models (one chat, one embedding) loaded to get real semantic gating at all, so it
earns the same statistical rigor as the hand-tuned thresholds, not a cheaper heuristic.

A gate whose best achievable separation doesn't clear `_MIN_ACCEPTABLE_J` (0.5) comes back marked
**degenerate** (`threshold=None`) rather than shipping an unreviewed, possibly-harmful value —
resolution is **per-gate**, not per-model: a model can calibrate cleanly for `lookup_request` while
`episodic_relevance` stays degenerate for it. `explicit_search_action` has no dedicated
true-positive category in the original battery (a gap inherited from the hand-tuning diagnostics,
which never built one either); it reuses `lookup_request`'s positive pool as a proxy, exactly how
the original hand-tuning comments already framed that relationship — a model that doesn't actually
treat "look up X" phrasing as ESA-positive will legitimately calibrate degenerate for it rather than
getting a guessed threshold. `_cosine_similarity` normalizes internally, so calibration doesn't
inherit any assumption that a new, untested model returns unit-length vectors. An `embed_fn`
failure or empty-vector response mid-battery is caught per-utterance and just thins the affected
gate's pool (tending it toward degenerate) rather than raising — a wider range of user-picked
models means more chances of a timeout/error/wrong-dimension response than the two models
(`embeddinggemma`, `nomic-embed-text`) actually tested against this feature so far.

**Persistence — schema v17, `embedding_model_thresholds`.** One row per model
(`model TEXT PRIMARY KEY`, one nullable `REAL` column per gate, `calibrated_at REAL NOT NULL`).
Nullable, not `NOT NULL` as originally sketched — a fixed requirement caught while implementing:
the per-gate degenerate design means a row must be able to hold a partial result, and `NULL` means
"degenerate as of `calibrated_at`", not "unset". `MemoryManager.get_calibrated_thresholds(model)`
returns `None` only when the model has never been calibrated at all, distinct from an empty dict
(calibrated, every gate came back degenerate) — this distinction is what lets the auto-calibration
trigger below avoid repeatedly re-attempting a model that's already been tried and failed.
`set_calibrated_thresholds(model, thresholds)` always replaces the full row, so re-running
calibration correctly clears a gate that used to calibrate cleanly but no longer does (e.g. the
battery/methodology improved, or a model was re-pulled under the same tag with different weights).

**`Planner`'s tiered resolution, extended.** Previously two-tier (tuned model's own constants →
`_VALIDATED_MODEL_THRESHOLDS` → disabled, all-or-nothing per model). Now resolved **per gate**,
independently, across four tiers: tuned → validated (highest trust, hand-reviewed) →
**auto-calibrated (new — persisted, `Planner`'s new `calibrated_thresholds` constructor param,
threaded through `ControllerAgent` from `main.py`'s `_build_controller()` via a new
`_derive_calibrated_thresholds()` helper, mirroring `_derive_active_embedding_model_name()`)** →
disabled. A fifth, model-independent lexical/BM25 tier is designed
(`PLAN_semantic_gating_calibration.md` §9 — Youden's J applied to BM25 scores instead of cosine,
calibrated once offline and checked into source since BM25 doesn't shift per embedding model,
covering both the search-intent and episodic-relevance gates) but **not yet built** — see Open
items. The tier-resolution logic itself was extracted into a standalone, pure
`planner.resolve_gate_tiers(embedding_model_name, calibrated_thresholds) -> dict[str, str]`
function so `Planner.__init__` and `GET /health` (below) can never disagree about which tier is
active for a given gate.

**Two triggers, one shared code path (`main.py`'s `_calibrate_and_persist()`).** Automatic,
zero-extra-clicks first-time calibration on `POST /settings/embedding-model` — runs only when the
new model has neither a validated entry nor an existing persisted row (`_needs_first_time_
calibration()`), and runs *before* `_build_controller()` so the freshly-persisted thresholds are
what the new `Planner` instance actually resolves. Manual, always-re-runs recalibration bundled
into `POST /memory/reembed`'s existing "fix things after switching embedding models" flow — the
button users already associate with that task — as a second phase after the corpus re-embed
completes, followed by a controller rebuild so the refreshed thresholds take effect immediately
rather than waiting for a restart. Both endpoints' response bodies gained a `calibration` field
(the four thresholds plus per-gate `degenerate`/`reason`/sample-size metadata).

**Settings UI trust badge.** `GET /health` gained `active_embedding_model_name` (the *resolved*
name `Planner` actually compares against `_TUNED_EMBEDDING_MODEL`, not the raw `embedding_model`
setting — those can differ, e.g. under the `EmbeddingEngine`-fallback-on-clear case) and
`gate_tiers` (per-gate tier, via `resolve_gate_tiers()`), refreshed on every existing 15s health
poll — not just transiently after a switch/reembed response, so the badge is correct on page load
too. The Settings page's "Embedding Model — Ollama" card shows a badge for the *worst* tier across
the four gates (reusing the existing `badge badge-{variant}` pill pattern), with a hover tooltip
breaking down each gate individually, and both the switch/reembed confirmation messages now append
a `"Recalibrated semantic gating for {model} (ESA=…, LR=…, RI=…, episodic=…)."`-style summary when
calibration ran.

**Test suite:** `test_threshold_calibration.py` (new, 12 tests — Youden's J selection math,
the degenerate floor, empty-pool handling, and end-to-end `calibrate_thresholds()` behavior
including embed_fn-failure resilience and non-unit-length-vector consistency); `TestSchemaV17Migration`
added to `test_retention_sweep.py` (3 tests, mirroring `TestSchemaV16Migration`'s real-on-disk-DB
convention); `TestCalibratedModelThresholds` added to `test_planner_phase3.py` (6 tests covering the
new tier's priority, partial-calibration handling, and both-tiers-absent fallback);
`TestAutomaticFirstTimeCalibration`/`TestReembedRecalibration` added to
`test_main_embedding_model_switch.py` (6 tests covering both triggers' endpoint wiring);
`TestHealthGateTiers` (new file, 5 tests) covering `GET /health`'s new fields. Full suite 1576 → 1602
passed. A real bug was caught by this test-writing pass and fixed before landing:
`_youden_calibrate()`'s "poor separation" branch was returning a real numeric `threshold` even when
`degenerate=True` (only the separate "empty pool" branch honored the module's own documented
promise that `threshold is None` whenever `degenerate` is `True`) — now consistent across both
branches, so a caller checking `threshold is not None` alone is always safe.

**Open items:**
- The lexical/BM25 fifth tier (`PLAN_semantic_gating_calibration.md` §9) is designed but not built —
  today, a model that calibrates fully degenerate (or a true zero-config keyword-only install with
  no embedding source at all) still lands on fully-disabled semantic gating for the affected gates,
  same as before this feature. Priority 5's existing static `_EPISODIC_KEYWORDS` bypass (unrelated to
  and unaffected by this work) partially covers the keyword-only case for episodic relevance only.
- No UI surface yet for *triggering* recalibration independent of a corpus re-embed (e.g. if only
  the calibration methodology changes, not the corpus) — bundled into "Re-embed Corpus Now" per an
  explicit decision that a separate action wasn't worth the added UI surface, but revisit if that
  assumption turns out wrong in practice.
- `_VALIDATED_MODEL_THRESHOLDS`'s own addition (§16.15's cross-reference, `nomic-embed-text:latest`)
  was never actually documented in this file under its own section — a pre-existing doc gap noticed
  while writing this entry, not fixed here (out of scope for this pass).
