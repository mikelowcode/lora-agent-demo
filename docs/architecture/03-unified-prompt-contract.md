## 3. Unified Prompt Contract

### 3.1 Design Principles

The prompt contract is a **cognitive architecture**, not a formatting
convention. The order of slots, the presence of labels, and the token
ceilings all affect how Gemma 4B weights information under attention
constraints. A poorly ordered prompt does not just look wrong — it produces
worse answers.

The ordering principle is: **static before dynamic, stable before volatile.**

All content that is invariant across turns is placed first, forming a stable
prefix that inference backends can cache and reuse. All content that changes
per-turn is placed last. This is a KV-cache architectural constraint, not a
style preference: every backend that supports prefix caching (oMLX, MLX-LM,
Foundry Local, vLLM, llama.cpp, TGI, TensorRT-LLM, ONNX Runtime) requires
exact byte-identity from the start of the token sequence. A single character
change anywhere in the prefix causes a complete cache miss for everything
after it.

Content stability ranking, most stable to least:

| Rank | Content | Changes when |
|---|---|---|
| 1 | Identity | User changes the assistant-name setting (`PUT /settings/assistant-name`) — added 2026-08-03, see Slot 1a below. Previously listed as "Never"; the identity *template* is still a constant, the name it's parameterized with is not. |
| 2 | Persona | Wiki page updated, or the assistant-name setting changes (the persona's "You are {name}" line is re-substituted — see Slot 1b) |
| 3 | Episodic memory | New episode written |
| 4 | RAG snippets | Query topic changes |
| 5 | Tool results | New tool call issued |
| 6 | Working memory | Every turn |
| 7 | Current instruction | Every turn (always last) |

#### Cache eligibility under the current runtime contract

The table above describes conceptual content stability, not cache eligibility under the current runtime contract.

Per the oMLX single-turn finding (detailed fully in §3.7, now a resolved finding), only Slot 1a + Slot 1b (the system message) are byte-identical across separate HTTP requests today. Slots 3a–7 (the user message) are structurally single-shot per `OMLXRuntimeClient.infer()` and cannot participate in cross-request prefix matching regardless of internal ordering or byte-stability.

The canonical vocabulary for this distinction: **stable prefix** (system message: identity + persona) and **dynamic suffix** (user message: all of Slots 3a–7). These terms are used consistently in §3.7a and forward.

### 3.2 Slot Definitions

The prompt is assembled as two runtime arguments: a **system message**
(passed as `system=` to the runtime client) and a **user message** (passed
as the user turn). Slots 1a and 1b form the system message. Slots 3–7 form
the user message in strict stability order.

---

#### Slot 1a — Identity

**Purpose:** Establishes who the assistant is and how it reasons. The
anchor of every prompt.

**Updated 2026-08-03:** the assistant's name is now a user-configurable
setting (default `"Localist"`), not a hardcoded constant. The surrounding
template — behavioral constraint + epistemic stance — is still fixed and
never changes; only the `{name}` it's parameterized with does, and only
when the user explicitly changes it via the Settings page. This is a
deliberate exception carved into the stability ranking above (Rank 1),
not a design regression — see the KV-cache note below.

**Token ceiling:** ~50 tokens (the canonical value is 43 tokens at the
default name's length; a longer configured name eats into this budget,
same as before — the ceiling remains advisory, not hard-enforced)

**Content:** Identity name, core behavioral constraint, epistemic stance.

**Canonical template** (`PromptBuilder._SYSTEM_TEMPLATE`):
```
You are {name}, a local research assistant. You reason carefully, cite your
sources, and acknowledge when you don't know something. You do not simulate
certainty.
```

With the default name substituted (`PromptBuilder._DEFAULT_ASSISTANT_NAME
= "Localist"`):
```
You are Localist, a local research assistant. You reason carefully, cite
your sources, and acknowledge when you don't know something. You do not
simulate certainty.
```

**Rules:**
- The template is a constant defined in `PromptBuilder._SYSTEM_TEMPLATE`.
  It is never modified at runtime. The name is resolved by each caller
  (`ControllerAgent`, `ConversationalAgent`, `warmup.py`) from
  `MemoryManager.get_assistant_name()` and passed into
  `PromptBuilder.build(assistant_name=...)` / `_slot1_system(persona,
  assistant_name)` on every call — `PromptBuilder` itself stays free of
  any DB dependency and has no memory of the name between calls.
  `assistant_name=None` (no `MemoryManager`, e.g. `ConversationalAgent`'s
  ephemeral-memory mode) falls back to `_DEFAULT_ASSISTANT_NAME`.
- Persisted in a one-row `assistant_settings` SQLite table
  (`MemoryManager.get_assistant_name()`/`set_assistant_name()`), following
  the same precedent as `retention_settings` — plain GET/PUT, no
  health-check, no runtime-object rebuild, since this is a string
  preference, not an external-service switch (contrast §16.5's
  `/settings/runtime-backend`).
- Changing the name is a real, immediate KV-cache cost: it invalidates
  the stable system-message prefix (Slot 1a + 1b together — see §3.7a),
  so the very next request after a change is a guaranteed full cache miss
  on the system message. This is accepted as the same cost class as an
  ordinary persona edit already had before this change — not a new
  regression, just a second, user-triggerable way to pay it.
  `ControllerAgent.invalidate_persona_cache()` is called by
  `PUT /settings/assistant-name` so the change takes effect on the very
  next request rather than waiting for the persona cache to notice on
  its own (see Slot 1b).
- Keep the template minimal. Every token here is cached unconditionally
  by all backends — there is no cost to including it, but expanding it
  narrows the headroom for dynamic slots.

---

#### Slot 1b — Persona

**Purpose:** The assistant's voice, style, tool awareness, and honor code.
Loaded once per session from `wiki/persona.md` and appended to the
system message after the identity constant.

**Token ceiling:** 500 tokens (hard limit). Truncated by `PromptBuilder`
when the wiki page exceeds this budget.

**Format:** Appended to Slot 1a with a double newline separator. No label
is added; the persona content is inserted raw:

```
You are Localist, a local research assistant. You reason carefully, cite
your sources, and acknowledge when you don't know something. You do not
simulate certainty.

{persona content from wiki/persona.md}
```

**Persona structure (current `wiki/persona.md`):**
As of the 2026-06-20 rewrite, `wiki/persona.md` is five plain prose sentences with no internal section headers (~476 chars / ~119 tokens, roughly 24% of the 500-token hard ceiling). Persona content is intentionally undifferentiated prose rather than a fixed section template — this description should not be treated as a contract that future personas must follow.

**Rules:**
- Loaded by `ControllerAgent._load_persona()`, which caches the result in
  `self._persona_cache` after the first successful corpus query.
- Passed to `PromptBuilder.build()` as the `persona=` keyword argument.
- When persona is `None` or empty, the system message is Slot 1a only —
  no separator, no placeholder.
- WikiAgent's XML-only system prompt is a protected contract. WikiAgent
  does not pass `persona=` and never receives Slot 1b. See §3.5.
- The persona must remain byte-stable within a session. Re-querying the
  corpus on every turn would break prefix caching.
- **Updated 2026-08-03 — cache invalidation is now real, not aspirational.**
  This bullet previously claimed the cache "is invalidated only when
  WikiAgent writes a new persona page" — no code implementing that
  existed anywhere in the codebase at the time; it was aspirational, not
  a description of actual behavior. Real invalidation now exists, keyed
  on the assistant name: `_load_persona()` tracks which name the cached
  persona was last substituted with (`self._persona_cache_name`) and
  compares it against `MemoryManager.get_assistant_name()`'s current value
  on every call — a mismatch clears `self._persona_cache` and re-fetches.
  `ControllerAgent.invalidate_persona_cache()` additionally lets
  `PUT /settings/assistant-name` force this immediately rather than
  waiting for the next `_load_persona()` call to notice on its own. A
  future WikiAgent-write-triggered invalidation (the originally aspired
  behavior) is not yet built.
- `wiki/persona.md`'s "You are {{ASSISTANT_NAME}}" line is a **substitution
  target**, not hand-edited prose: `_load_persona()` does
  `body.replace("{{ASSISTANT_NAME}}", assistant_name)` at cache-fill time —
  the exact same mechanism already used for the `LangSearch` →
  provider-label swap (see `_web_search_provider_label()` immediately
  below). The literal placeholder `{{ASSISTANT_NAME}}` staying in the
  source file is correct; it is the pattern being matched, not a value to
  update by hand. **Updated 2026-08-30 (OSS packaging pass):** this was
  previously a blunt `body.replace("LORA", assistant_name)` keyed on the
  literal old product name — replaced with an explicit placeholder token
  precisely so this substitution no longer depends on a brand string that
  has no reason to survive an OSS rename; the file itself was renamed from
  `lora-persona.md` to `persona.md` in the same pass.
- `persona.md` is filtered from RAG results — it is already in the
  system message and must not appear twice in Slot 4.
- `_load_persona()` fetches top-3 corpus results and filters by
  `str(d.path).endswith("persona.md")` before accepting any document into
  Slot 1b. If `persona.md` is not in the top-3 results, persona
  is absent for that session and a warning is logged.
- Persona content is no longer required to stay minimal. It may grow, as plain undifferentiated prose with no internal section headers, to absorb durable, non-instruction-dependent behavioral content (the "static rules" referenced in §3.7a), up to the existing 500-token hard ceiling. No soft checkpoint or review threshold applies below that ceiling.

---

#### Slot DT — Current Datetime (unnumbered)

**Purpose:** Give the model a live, ground-truth anchor for "now." Without
this slot, the model has no signal for the actual current date/time and
falls back to its training cutoff — leading it to either refuse tasks
involving real recent/future-sounding dates as impossible, or to
second-guess correct tool results (e.g. a real earnings date) as likely
errors simply because they postdate training. Added 2026-07-17 to close
this gap; see the persona carve-out in §3.6-adjacent `wiki/persona.md`
for the matching trust-hierarchy rule.

**Token ceiling:** None formally enforced — content is fixed-shape
(timestamp + weekday + optional tz label + one fixed directive sentence),
not user- or corpus-supplied, so there is nothing to truncate. Typical
render is well under 50 tokens.

**Position:** Always the *first* slot in the user message — ahead of even
`[SESSION FILES]`. This is a deliberate exception to the "static before
dynamic, stable before volatile" ordering principle in §3.1: this slot's
content changes on every single call and can never be cache-stable
regardless of placement, so there is no caching argument for placing it
anywhere in particular. It is placed first purely for model salience —
small models under attention constraints weight what they read first more
heavily, and the goal here is maximum trust, not cache reuse. Practically,
this costs nothing beyond what already existed: the trailing partial block
after the system message is already unconditionally uncacheable at current
prompt lengths regardless of its content (§3.7c), so this slot does not
regress any cache behavior that exists today. It does forgo a *future*
option (§3.7c Lever 2 — making a second block cacheable by keeping the
front of the dynamic suffix stable); that tradeoff is accepted knowingly,
not overlooked.

**Format:**
```
[CURRENT DATETIME]
2026-07-17T10:10:00-04:00 (Thursday, EDT)
This is ground truth for "now." Tool results dated at or after your
training cutoff are not errors — trust this timestamp and tool output
over your training prior.
```

**Rules:**
- **Unconditional.** Unlike every other slot in the user message, this one
  is never omitted — there is no "empty" current time. `current_datetime`
  is a required parameter of `PromptBuilder.build()`.
- Computed by the caller — `datetime.now().astimezone()` (system local
  timezone; no new `.env` config) — immediately before each `build()` call,
  never by `PromptBuilder` itself. This keeps the builder free of a
  system-clock dependency and trivially testable with a fixed value, and
  guarantees the timestamp is fresh on every turn rather than captured
  once and reused — the same staleness trap `ControllerAgent` already
  guards against for the active runtime backend (see this doc's parent
  `CLAUDE.md` note on `_state.runtime`).
- Implemented in `_slot_datetime()` in `prompt_builder.py`. Weekday via
  `strftime("%A")`; tz label via `tzname()` when the datetime is
  timezone-aware, omitted cleanly when naive.
- All four `PromptBuilder.build()` call sites pass a freshly computed
  value: `ControllerAgent._execute_plan()`, `ConversationalAgent.run()`,
  `WikiAgent`'s raw-ingestion path, and the cache warm-up hook
  (`warmup.py`). WikiAgent's system prompt is still fully replaced per
  §3.5 — this slot renders in WikiAgent's discarded-system/kept-user-prompt
  the same as any other user-message slot, harmless there.

---

#### Slot SF — Session Files (unnumbered)

**Purpose:** Content of text/code files uploaded by the user during the current
chat session. Injected into every prompt for the lifetime of the session without
routing through the Planner ladder. Bypasses corpus ingestion, wiki processing,
and embedding — files are read directly into the prompt as literal text.

**Token ceiling:** 4,000 tokens per file; 20,000 tokens total across all attached
files. Both ceilings enforced at render-time in `PromptBuilder._slot_session_files()`
— never at upload time. Truncation appends `… [truncated]` consistent with all other
slot builders.

**Position:** First slot in the user message, before Slot 3 (`[EPISODIC MEMORY]`).
Placing uploaded file content as early as possible in the user message maximises the
stable prefix length on turns where session files are present.

**Format:**
```
[SESSION FILES]
--- filename.ext ---
{full file content, truncated only if per-file ceiling exceeded}
--- end filename.ext ---

--- second-file.md ---
{content}
--- end second-file.md ---
```

**Rules:**
- This slot is **conditional**. Cleanly omitted (no label, no whitespace) when no
  files are attached. Existing prompts with no session files are byte-identical to
  pre-feature prompts — no regression.
- Files are stored in `session_files.py`'s ephemeral module-level `OrderedDict`
  cache. Cache is process-lifetime: cleared on backend restart, not persisted to
  SQLite. Single local user — no session keying required.
- Populated by `_session_files.get_files()`, called unconditionally in
  `ControllerAgent._execute_plan()` Step 6 before `PromptBuilder.build()`. The call
  also exists in `ConversationalAgent.run()` as defense-in-depth for the rare
  non-prebuilt path.
- Allowlisted extensions only: `.md`, `.txt`, `.py`, `.ts`, `.js`, `.svelte`,
  `.json`, `.yaml`, `.yml`, `.toml`, `.sh`, `.env`, `.csv`, `.xml`, `.html`,
  `.css`, `.rs`, `.go`, `.rb`, `.java`, `.c`, `.cpp`, `.h`, `.hpp`, `.sql`.
  Extension gate is enforced server-side in `session_files.add_file()`; client-side
  check in `ChatPanel.svelte` is defense-in-depth only.
- Reject-with-error (HTTP 400, user-readable `detail`) when per-file ceiling or
  total budget would be exceeded. No silent LRU eviction.
- Files are never wiki-ingested, never embedded, never written to `document_index`
  or any SQLite table. They are prompt context only.
- PDF and image support RESOLVED 2026-08-01 — see §22. Not via oMLX's native
  multimodal support (that would tie the feature to one runtime backend); a
  local OCR tool extracts text once at upload time, so by the time content
  reaches this slot it's plain text like any other session file.

**PromptBuilder integration:**
- `SessionFile` dataclass: `filename: str`, `content: str`.
- `_CEIL_SESSION_FILES_EACH = 4000`, `_CEIL_SESSION_FILES_TOTAL = 20000`.
- `_slot_session_files(files)` method: iterates files in insertion order, applies
  per-file ceiling via `_truncate_to_tokens()`, stops adding files when total
  ceiling is reached, returns `""` when list is empty or `None`.
- `build()` gains `session_files: list[SessionFile] | None = None` as first optional
  keyword argument. Backward-compatible — all existing call sites with no
  `session_files` argument produce identical output.

---

#### Slot 3 — Episodic Memory + User Profile

**Purpose:** Durable facts about the user, project, and preferences (Slot 3a —
episodic bullets) and relevant lines from the user profile document (Slot 3b —
user profile facts).

**Token ceiling:** 250 tokens total. Two independent sub-budgets enforced by
`PromptBuilder._slot3_combined()`: 150 tokens for episodic bullets (Slot 3a),
100 tokens for user profile facts (Slot 3b).

**Format:**
```
[EPISODIC MEMORY]
- {content} ({episode_type}, {confidence:.1f})

[USER PROFILE]
- {fact line}
```

**Rules:**
- This slot is **conditional**. It is omitted entirely when both sub-blocks
  are empty. No empty label, no placeholder.
- Slot 3a (episodic): relevance determined by Priority 5. See §4.
- Slot 3b (profile): injected on P4, P5 routes and any turn where
  episodic bullets fire. See §3.6.
- When episodic bullets injected: 3–5 bullets, confidence ≥ 0.7, type-ordered per §2.7.
- Both `[EPISODIC MEMORY]` and `[USER PROFILE]` labels are mandatory when
  their respective sub-block is present.
- Placed first in the user message because episodic content changes rarely
  (only when a new episode is written), maximising the stable prefix shared
  across consecutive turns.

---

#### Slot 4 — RAG Snippets

**Purpose:** Relevant content from the wiki corpus and document index,
retrieved only when the user explicitly requests it.

**Token ceiling:** 800 tokens (800-token hard limit)

**Format:**
```
[CONTEXT]
Source: wiki/XML Parsing.md
{2–3 sentences of relevant content, not truncated mid-sentence}

Source: wiki/WikiAgent Architecture.md
{2–3 sentences of relevant content, not truncated mid-sentence}
```

**Rules:**
- This slot is **conditional**. Omitted entirely when Priority 4 does not fire.
- Priority 4 fires **only on explicit wiki/vault trigger keywords** — never
  on corpus scoring alone. See §4.2.
- Maximum 3 sources. Wiki sources are preferred over raw doc sources.
- When a source is truncated to fit the budget, the truncated content is suffixed
  with `… [truncated]` at the cut point (sentence boundary if possible, otherwise
  mid-content). This signals to the model that the source was incomplete so it can
  reflect that in its response rather than presenting a partial summary as complete.
- Content is truncated at a sentence boundary when possible.
- The `[CONTEXT]` label is mandatory. Source paths are mandatory.
- `persona.md` is excluded from RAG results (already in system message).

---

#### Slot 5 — Tool Results

**Purpose:** Fresh, request-specific evidence from tool calls made during
this request's routing phase.

**Token ceiling:** 1,500 tokens (hard limit)

**Format:**
```
[TOOL RESULTS]
{tool_name}({call_parameters}):
  {truncated result content}
```

**Rules:**
- This slot is **conditional**. Omitted entirely when no tools were called.
- Tool results are injected in the order tools were called.
- The `[TOOL RESULTS]` label is mandatory. Tool name and call parameters
  are mandatory for auditability.
- `url_fetch` results include title, source URL, word count, and full
  extracted text. PromptBuilder enforces the 1,500-token ceiling.
- `web_search`/`news_search` results arrive pre-formatted as deterministic
  `Title — Source — Summary` bullets (built by `mcp_server/search_format.py`,
  shared across both tools) — see §14 for the per-tool summary-length caps
  that feed into this slot's shared budget.

---

#### Slot 5b — Graph Result

**Purpose:** Structured answer to a P3c graph-query turn — which pages link
to a target page (incoming), or which pages a target page links to (outgoing).
Carries the complete graph-query answer to the model so the response can be
grounded in real graph state rather than generated from weights alone.

**Token ceiling:** 300 tokens

**Format (incoming direction):**
```
[GRAPH RESULT]
Pages linking to {page}:
- {page_name}
```

**Format (outgoing direction):**
```
[GRAPH RESULT]
{page} links to:
- {page_name}
{page} also references a page that does not exist:
- "{link_text}" (no matching page found)
```

When zero edges exist, the content is a single declarative sentence:
`No pages link to {page}.` or `{page} does not link to any other pages.`

**Rules:**
- **Exception to the clean-omission contract.** This slot renders whenever a
  graph query resolved a target page this turn — even when the edge list is
  empty. Zero edges is a real, correct answer that must be visible to the model.
  The only omission case is `graph_result is None`, meaning no graph query
  resolved this turn. Implemented in `_slot_graph()` in `prompt_builder.py`.
- **Mutual-exclusivity with RAG, episodic, profile, and tools.** P3c
  graph-query turns never co-render with Slot 3 (episodic/profile), Slot 4
  (RAG), or Slot 5 (tool results). This guarantee is not enforced by
  `PromptBuilder` — it falls out of the `RoutingPlan` produced by
  `planner._priority3c_graph_query()`, which sets `fetch_rag=False`,
  `fetch_episodic=False`, and `tools_to_call=[]`.
  `PromptBuilder` itself has no exclusivity guard.
- Ceiling enforced as a single post-render truncation via `_truncate_to_tokens()`.
- Conditional: omitted entirely (no label, no whitespace) when `graph_result is None`.

---

#### Slot 6A — Structured Working State

**Purpose:** Deterministic, per-turn working context — what RAG sources were
active this turn and (when Tier 2 rendering is wired in) what the model-assisted
session state is. Positioned after Slot 5b and before Slot 6 to place
state-carrying content after retrieval evidence and before conversational
scaffolding.

**Token ceiling:** 100 tokens

**Format:**
```
[WORKING STATE]
current_project: {current_project}
active_artifacts: {artifact1}, {artifact2}, ...
```

Each line is emitted only when its field is non-empty/non-None.
`active_artifacts` is truncated by dropping entries from the end until the
100-token ceiling is met.

**Tier 1 fields (deterministic — implemented and rendering):**
- `current_project` — derived from `task.context["project_context"]` in
  `controller_agent.py` Step 5d. Currently always `None` at all real call
  sites: `project_context` is a DB-scoping key (e.g., `"general"` or a project
  slug), not a human-readable project name. The Step 5d code explicitly sets
  `current_project = None` when the value is `None` or `"general"`. The field is
  wired and ready; it will activate without a code change once a real
  project-name source is available at this context key.
- `active_artifacts` — `[s.path for s in rag_sources]`: the filesystem paths of
  RAG documents retrieved during the same turn. Non-empty only on turns where
  RAG retrieval ran (P4, P5 routes).

**Tier 2 fields (model-assisted — updating, not yet rendering):**
Three fields — `current_focus`, `open_loops`, `recent_decisions` — are updated
post-response via `process_working_state_update()` in `episodic_extractor.py`
and persisted in the `working_state` SQLite table via `WorkingStateStore`. These
fields do **not** yet appear in the rendered `[WORKING STATE]` output. Wiring
them into this slot is a separate, future step.

The Tier 2 update runs after every completed turn, **regardless of route** —
including P3c (graph-query) turns that are excluded from rendering. A graph-query
turn still produces real conversational state worth persisting for the next
turn's render. The update-vs-render distinction is deliberate: suppressing the
update on P3c turns would discard valid session state to maintain a boundary that
only applies to rendering.

**Rules:**
- **Clean omission.** Returns `""` when `state` is `None` or when both
  `current_project` is falsy and `active_artifacts` is empty.
- **P3c render-gating.** This slot does **not** render on P3c (graph-query)
  routes. The gate is in `controller_agent.py` Step 5d:
  `if plan.graph_query is None:`. Rationale: P3c is a deliberate
  no-interpretation zone — structural, non-personalized, deterministic relative
  to graph state. Slot 6A is interpretive and state-carrying by definition.
  Co-rendering on the same P3c turn would blur that boundary by accident rather
  than by deliberate future design.
- Implemented in `_slot6a_working_state()` in `prompt_builder.py`.
  Input dataclass: `WorkingMemoryState(current_project: str | None,
  active_artifacts: list[str])`.

---

#### Slot 6 — Working Memory

**Purpose:** The immediate conversational context. What just happened.

**Token ceiling: backend-tier-aware, added 2026-07-18 (`context_profile.py`),
budget mechanism corrected 2026-07-19 (§16.11).**
The ceiling below was previously one hardcoded global constant; it now
comes from a `ContextProfile` resolved fresh on every call from the active
runtime client itself — `ControllerAgent._execute_plan()` calls
`profile_for(self._runtime)` (not just its `is_local` flag; see §16.11 for
why the signature had to change) and threads the result into both
`MemoryManager.get_context_window()` and
`PromptBuilder.build(working_memory_ceiling=...)`:

| Profile | `working_memory_limit` (rows fetched) | `working_memory_tokens` (ceiling) |
|---|---|---|
| **Local** (oMLX, Foundry, Ollama serving a locally-resident model) | None (unbounded, as of §16.11) | `max_model_len − 27,000`, floored at 300 (§16.11) |
| **Cloud** (Ollama Cloud — model tag ends `-cloud`) | None (unbounded) | 60,000 |

**Corrected 2026-07-19 (§16.11) — the RAM-tiered mechanism this paragraph
described through §16.9 is retired, not extended.** It measured the wrong
layer: oMLX manages its own paged KV cache and memory dynamically
per-machine, independent of total system RAM, so a host-RAM-keyed lookup
was never actually bounding the real constraint. The local tier is now
budgeted directly from `OMLXRuntimeClient.max_model_len` (§16.10 — read
from `GET /v1/models`, the model's real server-reported context window)
minus the same 27,000-token reservation `CLOUD_PROFILE` already used for
the other slots' worst case (session files up to 20K, persona/RAG/tool/
graph/working-state slots ~3K, generation ~4K), floored at 300 so a small
or unreported `max_model_len` can't drive the budget negative — 300 is the
same figure this profile used under the retired approach, kept as the
degraded-case floor rather than a live number. The local tier's
`working_memory_limit` is now `None` too — the fixed 5-turn cap is gone;
that number was never derived from the model or the machine, just carried
over from the original hardcoded 300-token/5-turn pair. The cloud row is
unchanged: a cloud-hosted model (Gemma-4-31B via Ollama Cloud, ~128K real
context) doesn't have oMLX's local-process RAM constraint at all — the
ceiling there is a deliberately-budgeted 60,000-token figure reserving the
same per-slot headroom plus an extra safety margin against an inexact
context-length assumption, spending less than the model's literal ceiling
(`CLOUD_PROFILE.total_context_tokens` and `OllamaRuntimeClient`'s
`options.num_ctx` — both referenced in early drafts of this section —
were removed in §16.9 after being confirmed to have zero effect on
Ollama's actual behavior). Oldest turns
are still dropped first when either ceiling is exceeded — enforced twice,
independently, over the same fetched rows: once by
`MemoryManager.get_context_window()`'s `max_tokens` argument (row-level
eviction before assembly) and again by `PromptBuilder`'s shared
`_trim_working_memory()` (char-level trim at render time, factored out
2026-07-19 so the flattened-text path below and the structured-turns path,
§3.7d, trim identically). Both must be
raised together — raising one alone leaves the other silently
re-truncating back down to its old value.

**Format:**
```
[WORKING MEMORY]
Turn -2 [user]: {prior user message}
Turn -2 [assistant]: {prior assistant response}
Turn -1 [user]: {most recent prior user message}
Turn -1 [assistant]: {most recent prior assistant response}
```

**Rules:**
- Local tier: last 5 turns, 300-token ceiling (unchanged pre-2026-07-18
  behavior). Cloud tier: no turn-count cap, 60,000-token ceiling — in
  practice this holds the full accumulated conversation for all but
  unusually long sessions.
- Turns are listed in chronological order (oldest first, newest last).
- Tool results from prior turns are included as
  `[tool: {tool_name}] {truncated result}` entries, capped at 2–3 lines.
- This slot is conditional. It is omitted when no prior turns exist.
- KV-cache implication: `OllamaRuntimeClient.infer_stream()` builds the
  same single-shot shape §3.7 established for oMLX — one system message +
  one user message per HTTP call, no accumulated `messages` array, no
  session/cache-control field. Growing this slot to "full history" doesn't
  change that shape; it's still one large user-message string sent fresh
  each call. Whether Ollama Cloud's serving stack independently does
  cross-request prefix-hash KV-cache reuse on the *content* of that string
  (as distinct from oMLX's per-session block-hash cache, §3.7b) is
  unconfirmed — not verified this session, flagged as an open item. If it
  doesn't, the cloud profile still delivers the reasoning-quality half of
  the goal (the model can see full history) but not a latency/cost win
  from cache reuse.
- **oMLX exception, added 2026-07-19 — see §3.7d.** oMLX no longer
  receives this slot as flattened text. `PromptBuilder.build()`'s opt-in
  `emit_structured_working_memory=True` (used only on the oMLX call path)
  omits `[WORKING MEMORY]` from `user_prompt` entirely and returns these
  same turns as a structured, chronologically-ordered `list[Turn]` instead,
  which `OMLXRuntimeClient` sends as discrete `messages` array entries —
  mirroring oMLX's own web UI, and the multi-turn shape §3.7b originally
  speculated a future engine might use. Ollama/Foundry are unaffected and
  still receive this slot exactly as described above.
  `working_memory_tokens`'s ceiling math (§16.11: `max_model_len - 27,000`,
  floored at 300) is unchanged by this — only the *representation* changed,
  not the token budget or the drop-oldest-first trim rule, which
  `_trim_working_memory()` now shares between the flattened and structured
  paths so both trim identically.

---

#### Slot 7 — Instruction

**Purpose:** The raw instruction from the current turn.

**Token ceiling:** Uncapped.

**Format:**
```
[INSTRUCTION]
{instruction}
```

**Rules:**
- No transformation of the instruction.
- Always the last slot in the user message. This is a KV-cache invariant:
  the most volatile content must be at the end so that all stable content
  above it can be prefix-cached.

---

### 3.3 Aggregate Token Budget

| Slot | Label | Ceiling | Presence | Message |
|---|---|---|---|---|
| 1a — Identity | *(none)* | ~50 | Always | System |
| 1b — Persona | *(none)* | 500 | Conditional | System |
| DT — Current datetime | `[CURRENT DATETIME]` | none (fixed-shape) | Always | User |
| SF — Session files | `[SESSION FILES]` | 4,000/file, 20,000 total | Conditional | User |
| 3a — Episodic memory | `[EPISODIC MEMORY]` | 150 | Conditional | User |
| 3b — User profile | `[USER PROFILE]` | 100 | Conditional | User |
| 4 — RAG snippets | `[CONTEXT]` | 800 | Conditional | User |
| 5 — Tool results | `[TOOL RESULTS]` | 1,500 | Conditional | User |
| 5b — Graph result | `[GRAPH RESULT]` | 300 | Conditional | User |
| 6A — Working state | `[WORKING STATE]` | 100 | Conditional | User |
| 6 — Working memory | `[WORKING MEMORY]` | 300 | Conditional | User |
| 7 — Instruction | `[INSTRUCTION]` | Uncapped | Always | User |
| **Worst-case total** | | **~3,850** | | |

Slot numbers 2 and the old Slot 2 label `[USER]` are retired. The gap
between 1b and 3 is intentional: slot numbering reflects cognitive role
and stability rank, not sequential position in the output string.

Gemma 4B quantized has an effective context window of approximately 8,000
tokens. The prompt contract consumes under 4,000 tokens in the worst case,
leaving substantial headroom for model output.

`_CEIL_TOOL` was raised from 500 to 1,500 tokens (2026-07-25) — search-shaped
tool results (`web_search`, `news_search`) were landing shallow and
mid-sentence-truncated at the old ceiling on top of each provider's own
300-char-per-summary cut. See §14's search-result formatting for the
corresponding per-result summary caps.

### 3.4 PromptBuilder Interface

`PromptBuilder` is the single point of prompt assembly. Every agent calls
it. No agent assembles its own prompt string.

```python
@dataclass
class Turn:
    role:    str   # "user" | "assistant" | "tool"
    content: str
    label:   str | None = None  # tool name, if role == "tool"

@dataclass
class EpisodeBullet:
    content:      str
    episode_type: str
    confidence:   float

@dataclass
class RagSource:
    path:    str
    content: str

@dataclass
class ToolResult:
    tool_name:  str
    parameters: str
    result:     str


class PromptBuilder:
    def build(
        self,
        instruction:      str,
        current_datetime: datetime,
        persona:          str | None            = None,
        assistant_name:   str | None            = None,  # added 2026-08-03, see Slot 1a
        episodic_summary: list[EpisodeBullet]   | None = None,
        rag_snippets:     list[RagSource]        | None = None,
        tool_results:     list[ToolResult]       | None = None,
        working_memory:   list[Turn]             | None = None,
        working_memory_ceiling: int              | None = None,
        emit_structured_working_memory: bool     = False,   # added 2026-07-19, see §3.7d
    ) -> tuple[str, str] | tuple[str, str, list[Turn]]:
        """
        Assembles the canonical 7-slot prompt (static-first ordering).

        current_datetime is required — callers must pass a freshly computed
        `datetime.now().astimezone()` on every call; PromptBuilder never
        reads the system clock itself (see Slot DT).

        assistant_name: resolved by the caller from
        `MemoryManager.get_assistant_name()` and interpolated into Slot 1a's
        template. None falls back to `_DEFAULT_ASSISTANT_NAME` ("Localist").
        PromptBuilder itself has no DB dependency and holds no memory of
        this value between calls — see Slot 1a.

        emit_structured_working_memory: opt-in, default False (reproduces
        the 2-tuple return below exactly). When True, Slot 6 is omitted
        from user_prompt and returned instead as a third tuple element —
        the same trimmed, chronological list[Turn] — for a caller (oMLX
        only, today) that sends each turn as its own discrete message
        rather than flattened text. See §3.7d.

        Returns
        -------
        (system_prompt, user_prompt)
            system_prompt : Slots 1a + 1b. Byte-stable when persona and
                            assistant_name are both unchanged — maximises
                            KV-cache prefix reuse.
            user_prompt   : Slot DT, then Slots 3–7 in stability order.
                            Slot DT is always present; empty optional
                            slots are omitted cleanly — no label, no
                            whitespace.
        """
        ...
```

**Enforcement rules:**
- `PromptBuilder` enforces all token ceilings internally. Callers do not
  truncate; they pass full content and let the builder enforce budgets.
- Empty optional slots produce no output — not an empty label, not
  whitespace, nothing.
- The builder is stateless. It is safe to call concurrently.

### 3.5 WikiAgent Prompt Exception

WikiAgent's system prompt is a protected contract: a compact XML-only
instruction block that must not be replaced by `PromptBuilder`'s identity
slot (`_SYSTEM_TEMPLATE`, renamed 2026-08-03 from `_SYSTEM` — see Slot 1a)
or extended with a persona. WikiAgent calls `PromptBuilder.build()` with
`instruction=` only (no `assistant_name=`, either). The returned
`system_prompt` is discarded; WikiAgent passes its own `SYSTEM_PROMPT`
constant to `runtime.infer()` directly.

This exception is intentional and permanent.

---

### 3.6 User Profile

`ControllerAgent` maintains a per-session user profile sourced from
`backend/wiki/users/michael.md`. This file is hand-authored (not produced
by WikiAgent) and contains structured fact lines in five sections: Identity,
Active Projects, Preferences, Working Patterns, and Decisions.

Each section has a maximum of 5 lines to enforce minimalism and prevent
prompt bloat. The ideal injection returns only the top relevant facts
for the current instruction, not the entire profile.

#### Embedding and scoring

On first request after startup, `ControllerAgent._load_user_profile()`
reads the file, strips section headers and blank lines, and embeds each
remaining fact line using `ControllerAgent._embed()` — which delegates
to `MemoryManager._embed_fn` (the EmbeddingEngine callable). Embeddings
are stored in parallel lists (`_profile_lines`, `_profile_embeddings`)
and cached for the session.

Per-turn scoring: `_score_profile_facts(instruction_embedding, top_n=5,
threshold=0.45)` computes cosine similarity between the instruction
embedding and each cached fact embedding. Lines scoring ≥ 0.45 are
returned as `UserProfileFact` objects, sorted by score descending,
capped at 5.

The 0.45 threshold is lower than the 0.55 RAG corpus threshold because
profile fact lines are short and produce lower raw cosine similarity
against full instruction embeddings even when semantically relevant.

#### Injection trigger

Profile facts are injected into Slot 3b on any turn where:
- `plan.fetch_rag` is True (P4 route)
- `plan.fetch_episodic` is True (P5 routes)
- Episodic bullets fired in the same turn

P6 direct-answer turns do not inject profile facts.

#### Update path

Manual for now: edit `wiki/users/michael.md` directly and restart the
backend to reload the cache. Automatic promotion from episodic memory
is planned as part of the graph retrieval layer (future session).

---

### 3.7 Resolved: Prefix Stability Is a Structural Property of the Stable Prefix, Not a User-Message Defect

The §3.1 design principle ("static before dynamic, stable before volatile") describes the intended ordering. The investigation in the 2026-06-20 session established that the zero-cache-hit result on the user turn (see Observed evidence below) is an **expected, structural property** of the current `infer()` contract — not a defect in `PromptBuilder`'s slot ordering.

#### Mechanism — routing-dependent slot presence

Slots 3–6 are all conditional. The first slot that actually appears in the user message
varies by routing path:

| Routing path | First user slot |
|---|---|
| P6 (direct answer) | `[WORKING MEMORY]` (if non-empty), else `[INSTRUCTION]` |
| P4 (RAG) | `[USER PROFILE]` or `[EPISODIC MEMORY]` (if either fires), else `[CONTEXT]` |
| P5 (episodic) | `[EPISODIC MEMORY]` |

A P6 turn followed by a P4 turn produces a different first byte in the user message.
Because KV-cache prefix matching must be exact from byte 0, this is a complete cache
miss for the user turn — not a partial miss from the point of divergence. This variability is expected and accepted under the dynamic-suffix model — it is no longer being treated as a defect to fix via reordering.

#### Compounding factor — similarity-scored profile selection

Even when Slot 3 is present on two consecutive turns (both are P4 routes, both fire
`[USER PROFILE]`), the Slot 3 content is not stable. `_score_profile_facts()` scores
each fact line by cosine similarity against the **current instruction embedding**.
Different instructions → different top-5 facts → different Slot 3 bytes → cache miss
at byte 1 of the user message, even within the same routing path. This per-turn variation is expected and accepted under the dynamic-suffix model — profile facts are deliberately instruction-dependent by design, and freezing them into a stable prefix would defeat their purpose.

#### Observed evidence (2026-06-19 live session)

Three-turn live session. Stable system message: `system_chars = 403` (≈ 101 tokens). *(Historical — this figure predates the persona rewrite completed later in the 2026-06-20 session. At current persona size, the system message would be approximately ~40 tokens identity + ~119 tokens persona ≈ ~159 tokens / ~636 chars under the same `len // 4` estimation convention used elsewhere in the codebase.)*
User turn longest common prefix (LCP):

| Turn pair | User LCP | Cause |
|---|---|---|
| T1 → T2 | 1 char | Slot 3/4 absent T1, present T2 — first byte differs |
| T2 → T3 | 1 char | Slot 3 content changed (different profile facts scored) |

oMLX dashboard: **zero KV-cache hits** across all three turns. The 403-char system
message may cache at the system-turn level (oMLX caching behavior not confirmed from
the codebase), but the user turn achieves no prefix reuse. This zero-cache-hit result on the user turn is now understood to be **expected**, not anomalous — see Root cause below.

#### What this is not

This is not a regression introduced by any change in the 2026-06-19 session. The
session_id working-memory fix and the RAG frontmatter-stripping fix are independently
verified and correct. The prefix-stability gap pre-dates this session and results from
the conditional-slot architecture as originally designed.

#### Root cause — oMLX single-turn request shape

`OMLXRuntimeClient.infer_stream()` sends exactly one system message + one user message per HTTP call, with no message-history accumulation and no session identifier in the payload. This was verified by reading `omlx_runtime_client.py` directly — there is no cache-control parameter, session field, or accumulated `messages` array anywhere in the request construction. The backend therefore has no mechanism to recognize a request as a continuation of a prior call. The system message is the only content that can ever be compared byte-for-byte against a previous request, and it is already stable (cached per-session via `ControllerAgent._load_persona()`). The user message can never achieve cross-request prefix reuse under this contract, independent of slot ordering.

#### Status

Resolved as a documentation/framing correction. No code change was required — `PromptBuilder.build()`'s existing slot order already matches the dynamic-suffix ordering this finding converged on. See §3.7a for the stable-prefix / dynamic-suffix contract and §3.7b for the future APC direction.

*Flagged 2026-07-19: the "Root cause" paragraph above is now stale specifically for oMLX. §3.7d implemented the multi-turn `messages` array §3.7b speculated about — oMLX's working-memory turns are no longer packed into one flattened user message; `OMLXRuntimeClient.infer_stream()` now sends one array entry per turn. This paragraph is left in place as the historical record of the finding that originally justified the dynamic-suffix ordering; see §3.7d for the current oMLX request shape.*

#### Design direction (historical record — both options superseded)

Two approaches were identified before the root-cause finding; tradeoffs were not evaluated:

1. **Emit all conditional slots with empty-state markers.** When Slot 3 is absent,
   emit `[EPISODIC MEMORY]\n(none)` rather than omitting the slot entirely. This
   guarantees the user message starts with the same bytes regardless of routing path —
   at the cost of a small fixed token overhead on every P6 turn.

2. **Reorder volatile content after a longer stable prefix.** Place a
   routing-invariant block first in the user message so that conditional slots always
   follow a guaranteed-stable prefix. Preserves clean omission but requires defining
   and maintaining a new static anchor that is byte-identical across all routing paths.

Both options are **superseded by the single-turn-request finding**, not rejected on their own merits. Both were designed to fix user-message prefix instability for a backend capable of comparing the user message across requests; since `OMLXRuntimeClient.infer()` structurally cannot make such a comparison today, neither option would produce any measurable effect. Retained here as a historical record of the options considered.

---

### 3.7a Stable Prefix / Dynamic Suffix Contract

This contract is the canonical boundary between cached and per-turn content. It is enforced by an automated test (`tests/test_prompt_builder.py`, `test_pb_e_build_enforces_dynamic_suffix_slot_order`) so that any future change to slot order is a deliberate, reviewed decision rather than a silent regression.

**Stable prefix** (system message, passed as `system=`): Slot 1a (identity) + Slot 1b (persona). Byte-identical for the lifetime of a session once persona is cached by `ControllerAgent._load_persona()` — **updated 2026-08-03:** and unchanged since the last assistant-name change, if any. Slot 1a's name is a user setting (`MemoryManager.get_assistant_name()`, default `"Localist"`, see Slot 1a above), not a true constant; changing it invalidates the persona cache (`ControllerAgent.invalidate_persona_cache()`) and forces a fresh substitution, which is a real full-prefix cache miss on the next request — the same cost class the wiki-page-update case already had, just now user-triggerable directly. No new slot is introduced. "Static rules" is not a separate artifact — it denotes invariant scaffolding that may be written directly into `persona.md` as plain prose, blended with voice and style content, with no internal section headers and no structural separation from the rest of the persona text. Persona may grow to absorb this kind of durable, non-instruction-dependent content, with no soft checkpoint or review threshold below the cap. The only constraint is the existing hard ceiling, `_CEIL_PERSONA = 500` tokens / 2000 chars in `prompt_builder.py`, which is unchanged and still governs KV-cache prefix stability. Current actual persona size: ~476 chars / ~119 tokens (roughly 24% of the cap) — substantial headroom (~381 tokens) exists.

**Dynamic suffix** (user message): Slot 3a (episodic) → Slot 3b (profile) → Slot 4 (RAG) → Slot 5 (tool results) → Slot 6 (working memory) → Slot 7 (instruction). This order is unchanged from the existing implementation and is preserved deliberately — it reflects conceptual layering (contextualizers → evidence providers → conversation scaffolding → instruction), not cache eligibility. Episodic and profile facts are *not* eligible to move into the stable prefix: profile is re-scored per turn via live cosine similarity; episodic presence is gated by routing path and session state. Freezing either into the prefix would defeat their purpose.

---

### 3.7b Future Direction: APC Layer for MLX-Based Engines

*(Future / unscheduled — direction statement only, no build sequence, no priority.)*

Localist is architected toward a future automatic/algorithmic prefix caching (APC) layer intended to work across MLX-based inference engines (oMLX, MLX-LM, and potentially other MLX-backed runtimes), either as a capability those engines add natively or as a plugin Localist builds itself.

The stable-prefix/dynamic-suffix contract locked in §3.7a is the groundwork for that future layer: if a future engine or plugin sends a growing multi-turn `messages` array (rather than today's single system+user shot) or otherwise gains the ability to compare prefixes across requests, the dynamic-suffix ordering already locked here becomes the correct stable-history/dynamic-tail shape for it to exploit, with no further prompt-assembly redesign needed.

This is explicitly a forward-looking architectural bet, not a fix for a currently-observed metric. No current backend in this codebase can produce a user-message cache hit; this is expected and accepted (§3.7) and should not be treated as a regression in future live-testing sessions.

*Flagged 2026-06-23: live `/admin/api/cache/probe` evidence in §3.7c below directly contradicts the "no current backend... can produce a user-message cache hit" claim above. §3.7 and this paragraph are NOT yet edited to reflect this — that edit is deliberately deferred to a future session per standing discipline (lock the finding, decide on action items, then update the doc). Treat the claim above as superseded-but-not-yet-corrected in the prose until that edit happens.*

---

### 3.7c Open Item — Live Cache-Mechanism Findings and Candidate Follow-Up Actions (2026-06-23)

**Status: forward-looking open item. Findings below are confirmed via live source-reading and a live probe call. The four candidate actions are NOT yet decided, scoped, or scheduled — this section exists to preserve the option space before the next session picks a subset to act on.**

#### Confirmed mechanism (supersedes §3.7's session-continuity framing)

Source-read directly from the installed oMLX package
(`/opt/homebrew/Cellar/omlx/0.4.2/libexec/lib/python3.11/site-packages/omlx/`)
and confirmed live via `POST /admin/api/cache/probe`:

- oMLX's prefix cache is **role-blind and session-blind**. It hashes fixed-size
  token-ID blocks (`compute_block_hash`, `paged_cache.py:78–119`) chained from a
  fixed root seed (`b"omlx-root"`) — it has no concept of "session," "history,"
  or "system vs. user message." §3.7's premise that caching requires the
  *client* to assert continuity (a session ID, a growing `messages` array) is
  incorrect. A client sending one isolated system+user pair per HTTP call —
  exactly what `OMLXRuntimeClient.infer()` does — can and does produce real
  cache hits on identical leading token blocks across separate, unrelated
  calls, with no client-side change required.
- Block size for the currently-loaded model is **512 tokens**, not the
  256-token default documented in `scheduler.py:904` —
  `_align_block_size_with_rotating_window()` overrides this at load time for
  Gemma 4's rotating-window attention. Confirmed live via probe response
  (`block_size: 512`), not assumed from source alone.
- Live probe data (two real Localist-shaped prompts, ~780–800 tokens each,
  system prompt + `[TOOL RESULTS]` + `[WORKING MEMORY]` held byte-identical,
  only `[INSTRUCTION]` varying): both prompts produced exactly 2 blocks. Block
  0 (tokens 0–511) was cached (`ssd_disk`) for both, confirmed already-warmed
  from a prior 20-call diagnostic run. Block 1 (the remaining ~270–280 tokens)
  was cold for both. Partial trailing blocks are never stored by
  `store_cache` — block 1 is **unconditionally** cold regardless of content,
  not cold because the instruction text differed. At current prompt lengths,
  instruction-content divergence has **zero effect** on cache behavior, since
  the only token range it could affect never gets cached in the first place.
- `blocks_ssd_hot: 0` on both probes — the cached block was sitting in the SSD
  cold tier, not the in-memory hot tier, at probe time (server/model had been
  reloaded since the diagnostic run). A real inference call right now would
  pay one SSD read for block 0 rather than a RAM hit.
- The endpoint is `POST /admin/api/cache/probe`, not `/admin/probe_cache` as
  an earlier investigation guessed from release notes alone — corrected here
  for any future session that wants to re-run this check. Response is
  aggregate counts only (`total_tokens`, `block_size`, `total_blocks`,
  `blocks_ssd_hot`, `blocks_ssd_disk`, `blocks_cold`, `ssd_hit_tokens`,
  `cold_tokens`) — no per-block detail.
- **Three structurally distinct system messages compete for block 0, with
  zero cross-sharing, confirmed by direct source read (2026-06-23).** The
  warm-up fixture's system message (Lever 3, below), the main
  conversational call's per-turn system message (persona + `PromptBuilder`
  slots), and `_WORKING_STATE_UPDATE_SYSTEM` (the fixed constant used by
  `extract_working_state_update()`'s post-response Tier 2 call) are three
  different byte sequences from token 0. Since the cache is content-hashed
  with no session or role awareness, none of these three call types can
  ever produce a block-0 cache hit against either of the other two — they
  are three separate, mutually non-overlapping cache lineages, not one
  cache being destabilized by another. This was identified live: a
  production session showed `blocks_cold` accumulating tokens turn over
  turn with `blocks_ssd_hot` frozen at its single post-warm-up value (512
  tokens cached out of 6,019 total prefill tokens after several turns —
  8.5% efficiency, down from 25% after the first conversational call). An
  embedding-call interference theory was proposed and **disproven by direct
  source read**: `EmbeddingEngine` (in `embedding_engine.py`) runs entirely
  in-process via `mlx_embeddings`, makes no HTTP call to oMLX at all, and
  the module's own docstring states "OMLXRuntimeClient.embed() is NOT
  called anywhere for corpus embeddings" — there is no code path by which
  an embedding call touches oMLX's cache. The three-system-message finding
  is the better-supported explanation, confirmed by reading
  `_WORKING_STATE_UPDATE_SYSTEM`'s literal content against the other two
  system messages, not yet confirmed by a live diagnostic isolating each
  lineage's contribution independently. **Not yet actioned** — see Lever 3
  update below and the live working-state-update outcome diagnostic
  currently in progress (§10.4-adjacent; not yet a numbered open item as
  of this update).

#### Candidate follow-up actions (Lever 3 now implemented and confirmed; 1, 2, 4 remain option space)

**Lever 1 — Grow the persona to widen the structurally-cacheable portion of
block 0.** The system message currently occupies only ~159 of block 0's 512
tokens; the remainder is dynamic-suffix content that happens to fit in block
0 only by coincidence of current prompt length. Growing persona content
toward the existing 500-token/2000-char ceiling (already sanctioned by
§3.7a, ~381 tokens of headroom) would increase the *guaranteed*-stable
portion of block 0, making cache reliability less sensitive to small changes
in dynamic-suffix length. Cheap; no routing or slot-order change required.

*Added context (2026-06-23, post-Lever-3 verification): the oMLX dashboard's
own Serving Stats reports 58.4% cache efficiency (512 of 877 total prefill
tokens cached) for the warm-up fixture's prompt shape in isolation — this is
the real per-call baseline Lever 1 would be improving on. The ceiling on
efficiency for any single prompt of this approximate length is structural:
block 1 (the trailing partial block) is unconditionally uncacheable
regardless of what Lever 1 does to block 0's composition. Separately, in a
real multi-turn session, overall efficiency falls well below even this
58.4% figure (observed: 8.5% after several turns) — see the new
three-system-message finding above. Lever 1 addresses block 0's *internal*
reliability for a single call shape; it does not address the
multi-lineage problem.*

**Lever 2 — Reconsider whether dynamic-suffix ordering has real cache payoff
at longer prompt lengths.** §3.7a currently states the dynamic-suffix slot
order (episodic → profile → RAG → tool results → working memory →
instruction) "reflects conceptual layering... not cache eligibility." That
statement was correct under the old (incorrect) session-continuity model. If
real multi-turn sessions push prompts past ~1024 tokens (a third block),
placing the most-stable dynamic slots (episodic, profile) immediately after
the system message could make a second block cacheable on turns where that
content doesn't change — but this is conditional on real session data
showing episodic/profile content is actually stable enough turn-to-turn, not
just stable within one frozen diagnostic fixture. Needs a live diagnostic
before any slot-order change; not assumed to pay off.

**Lever 3 — Pre-warm block 0 at startup. IMPLEMENTED AND CONFIRMED
(2026-06-23).** Implemented as `run_cache_warmup()` in `backend/warmup.py`,
hooked into `main.py`'s `lifespan()` immediately after
`_state.controller = controller` and before the "ControllerAgent ready" log
line. A single best-effort `runtime.infer()` call, built via the real
`PromptBuilder.build()` against a dedicated fixture
(`templates/warmup_fixture.md`, parsed by `parse_warmup_fixture()`), runs
once per backend process boot. Fails open: any failure (oMLX unreachable,
fixture missing/malformed, prompt assembly error) is logged as a warning
and startup proceeds normally with no warm cache, never raising and never
delaying server readiness beyond the warm-up call's own duration.

Design decision, locked during implementation: success is defined as block 0
reaching *any* cache-resident tier (disk or hot), not specifically the hot
tier. Disk-resident-but-not-yet-hot is an acceptable end state, since
disk→hot promotion was separately confirmed (see below) to be cheap and
synchronous whenever it happens. The hook issues exactly one call — no
polling, no retry, no second call to force hot-tier promotion.

Verification chain (all live, against the real running system, in order):
1. Initial probe-tool diagnostic (this section's "Confirmed mechanism"
   findings above) established the baseline mechanism and predicted Lever
   3's effect.
2. A zero-delay follow-up diagnostic confirmed disk→hot promotion is
   **synchronous, complete before `infer()` returns** (visible at a
   measured 2.3 µs post-call probe latency) — resolving an open question
   from the first Lever-3-shaped diagnostic about whether promotion was
   async. (The reverse direction — cold→disk write timing — was not
   resolved and remains genuinely untested; see "Still open" below.)
3. Implementation (`warmup.py` + `warmup_fixture.md` + `main.py` hook),
   352/352 tests passing, 0 regressions from the pre-implementation
   342-test baseline.
4. Fixture content reviewed directly for realism (concrete wiki-search and
   note-fetch tool outputs, a substantive six-turn research conversation) —
   judged qualitatively realistic, not a synthetic placeholder.
5. Fixture's real token/block shape confirmed against the actual tokenizer
   and probe endpoint: 877 tokens, 2 blocks, block 0 = tokens 0–511 —
   structurally identical in composition to the original diagnostic
   prompts despite being meaningfully longer in content. No 3-block
   overflow; no shape mismatch with production traffic's expected block 0.
6. Live verification against a real `start_localist.sh` boot (oMLX
   pre-confirmed reachable via health check; model not yet loaded —
   `loaded_count: 0` — so this run also exercised cold model load, not
   cache promotion alone): startup logs showed
   `Cache warm-up complete — block 0 promoted to cache-resident tier
   (8874 ms)`, followed immediately by `ControllerAgent ready` and
   `Application startup complete`, confirming hook ordering is correct —
   it runs and completes before the server accepts any request. A
   post-boot probe (preceded only by passive health-check polls, no real
   chat requests) showed: `total_tokens: 877`, `total_blocks: 2`,
   `blocks_ssd_hot: 1`, `blocks_cold: 1` (the unconditionally-cold tail,
   as expected), `ssd_hit_tokens: 512`. Cross-checked independently via
   the oMLX dashboard's Serving Stats panel (a separate UI/instrumentation
   path from the `/admin/api/cache/probe` endpoint): Total Prefill Tokens
   877, Cached Tokens 512 — exact agreement with the probe figures via a
   wholly independent measurement surface.

**Verdict: confirmed, with a real limitation surfaced after live
deployment.** A single warm-up call at backend boot reliably promotes
block 0 to a cache-resident tier before any real user request, with
negligible added risk (fail-open, single call, no production dependency on
the diagnostic-only probe endpoint) — this part of the original prediction
holds exactly as tested. **What was not anticipated:** in real multi-turn
production use, the warm-up's cache-resident block 0 is never hit again,
because neither the main conversational call nor the Tier 2 working-state
call shares a byte-identical system message with the warm-up fixture (see
the three-system-message finding above). The hook does exactly what it was
built and tested to do; it does not, by itself, improve steady-state
multi-turn cache efficiency, which depends on a separate question (whether
any of the three call types can be made to share a leading block) not
addressed by this implementation.

**One nuance for future reference:** the 8874 ms warm-up duration observed
in live verification includes cold *model load* time (the model was not
resident in oMLX at all before this test — `loaded_count: 0`), not warm-up
prefill time alone. On a boot where oMLX already has the model loaded (e.g.
if the backend restarts more often than oMLX does), the warm-up call's
duration would be substantially lower — this number should not be read as
the steady-state cost of the hook.

**Still open, not resolved by this work:**
- Cold→disk write timing (the original ambiguity from the first
  Lever-3-shaped diagnostic) was never directly tested — every subsequent
  run either found disk already populated or jumped straight from cold to
  hot in a way that didn't isolate the write step. Low practical urgency
  now, since disk→hot promotion is confirmed cheap regardless of how long
  the write itself takes, but it remains a genuine gap in the mechanism
  picture, not a closed question.
- The oMLX dashboard displays an idle-unload countdown for the loaded model
  (observed: "idle 4m 54s / ~10m 6s left" before eviction in one session,
  and separately "idle 13m 59s / ~1m 1s left" in another). If this is a
  real eviction policy and not just a display artifact, it has direct
  bearing on Lever 3: a long-idle Localist session could see oMLX unload
  the model entirely, after which the *next* real request — not the
  warm-up hook, which only runs once at backend boot — would pay the full
  cold-load cost again. Unscoped and unconfirmed; flagging only so it
  isn't lost.
- The dashboard's separate "Runtime Cache Observability" panel reports a
  distinct "Memory: _ MB / 2.0 GB · N entries" figure (observed values
  varying across sessions: 28 MB/1 entry shortly after one boot, 392
  MB/14 entries after a multi-turn session reached 8.5% efficiency). This
  appears to be a different cache-accounting surface from the
  `blocks_ssd_hot`/`blocks_ssd_disk` block-tier model documented above —
  its relationship to block-level tier state, and whether "entries" tracks
  per-cache-lineage state (which would make 14 entries plausibly
  correspond to the three-system-message finding's competing lineages,
  accumulating over turns), is not understood and was not investigated.
  Noted as an unexplained observation, not folded into the confirmed
  mechanism above.
- **New, highest-priority follow-up:** whether any of Levers 1/2/4, or a
  new fifth option (making the Tier 2 working-state call and/or the main
  conversational call share a byte-identical leading system message with
  each other or with the warm-up fixture), would address the real
  multi-turn efficiency problem the three-system-message finding
  describes. Not scoped. A live diagnostic logging pass on
  `extract_working_state_update()`'s outcomes (separate from cache
  mechanics — see working-state-update Tier 2 pre-gate work) is in
  progress as of this update and may independently reduce one of the
  three competing lineages if its outcome leads to gating that call on
  some turns.

**Highest-priority follow-up — IMPLEMENTED 2026-06-23 (same day as
diagnosis): all three calls now share a byte-identical leading prefix.**

Implemented via a new `_build_wsu_system(persona)` in `episodic_extractor.py`,
which constructs the Tier 2 system message as
`PromptBuilder._slot1_system(persona) + "\n\n" + _WSU_TASK_INSTRUCTIONS`
(the renamed, content-unchanged former `_WORKING_STATE_UPDATE_SYSTEM`).
`extract_working_state_update()` and `process_working_state_update()` each
gained a `persona: str | None = None` parameter, threaded from
`controller_agent.py`'s existing `self._load_persona()` call at the Tier 2
call site (~line 1249) — reusing the already-cached persona load from the
same turn's main-call construction, not a second corpus query. Lever 1
(persona growth) was folded into this same fix, since a shared prefix only
has practical cache value if it's long enough to cover a meaningful portion
of block 0: `persona.md` was grown from ~476 chars (~119 tokens) to
1,951 chars (~487 tokens) — real, previously-authored content (an earlier
draft of the persona, trimmed back down to fit the existing 500-token
`_CEIL_PERSONA` ceiling) rather than invented padding. `_SYSTEM` (160 chars
— since renamed to `_SYSTEM_TEMPLATE` and parameterized by assistant name,
2026-08-03; this measurement used the then-hardcoded "LORA", see Slot 1a)
+ this persona lands at ~528 tokens combined — 16 tokens past the 512-token
block-0 boundary, meaning block 0 is now covered entirely by content
genuinely shared across all three call sites, with a small uncontested
margin into block 1.

*What's confirmed, and at what strength of evidence:*
- **Confirmed by test, with the real on-disk persona file (not a
  placeholder string):** `_build_wsu_system(actual_persona)` produces output
  whose leading bytes are identical, string-for-string, to
  `PromptBuilder()._slot1_system(actual_persona)` — proven via a dedicated
  test that reads `wiki/persona.md` from disk, parses it through the
  same `parse_wiki_doc().body[:2000]` path `_load_persona()` itself uses,
  and asserts both the byte-identical prefix and the exact suffix shape
  (`"\n\n" + _WSU_TASK_INSTRUCTIONS`, nothing duplicated or mangled at the
  seam). This is real, code-level proof that the *construction* is correct.
- **Not yet confirmed live:** whether this construction-level fix actually
  changes `blocks_ssd_hot` / cache-efficiency behavior in a real multi-turn
  session — i.e., whether the original 8.5%-efficiency finding improves.
  The byte-identical-prefix test proves the prompts *would* hash to the
  same lineage; it does not, by itself, re-run the live diagnostic that
  measured the original problem. That re-verification (re-running the same
  kind of multi-turn session and re-probing `/admin/api/cache/probe` or the
  oMLX dashboard's Serving Stats) has not yet been done as of this writing
  and is the natural next step before this item can be marked fully closed
  rather than "implemented and unit-verified."
- **A separate, narrower gap worth naming:** `persona.md`'s on-disk
  edit required re-indexing into `document_index` for `_load_persona()`
  (which reads via `query_corpus()` from the DB, never from disk directly)
  to actually serve the new content — this was done via direct SQL update
  against `localist_memory.db` (content, token_set, and content_hash
  refreshed; existing embedding preserved rather than recomputed, since the
  identity/persona document is not typically retrieved via semantic
  similarity). The stray, unreferenced `lora_memory.db` was also updated in
  the same operation — outside the stated scope of the request that
  prompted it. Harmless given that file's confirmed unreferenced status,
  but logged explicitly per the project's standing discipline of not
  letting out-of-scope side-effects pass without comment, however benign.

**Lever 4 — Treat the trailing partial block's unconditional coldness as a
budget signal, not a defect to fix.** Since the trailing partial block is
*always* cold regardless of content or ordering, there may be limited ROI in
optimizing slot order within it. The more relevant move may be ensuring
expensive-to-recompute content (long RAG snippets, long tool results) lands
in the cacheable leading block(s) wherever possible, while accepting that
cheap, naturally-volatile content (the instruction itself) is what gets
recomputed every time regardless of ordering.

**Explicitly NOT recommended, regardless of which levers are pursued:**
artificially padding prompts to force a block boundary purely to improve a
cache-efficiency number — this trades a real, certain prefill cost for an
uncertain caching benefit with no evidence of net positive ROI.

**Current state (updated 2026-06-23):** Lever 3 is implemented, tested, and
live-verified as doing exactly what it was built to do — but live
multi-turn deployment surfaced that the original four-lever framing did not
anticipate the real driver of steady-state cache efficiency: three
structurally distinct system messages in rotation, none sharing a leading
block with any other. This is now the most important open question in this
section, ahead of Levers 1, 2, and 4 as originally scoped. Levers 1, 2, and
4 remain undecided option space, unchanged from the original framing, and
none has been scoped into a Claude Code prompt.

---

### 3.7c Update — Live Re-Verification of the Shared-Prefix Fix: Negative (2026-06-24)

The "not yet confirmed live" item flagged in the implementation writeup above has now
been tested directly, in a real multi-turn session, with a real `/admin/api/cache/probe`
cross-check. The result is negative — documented here in full so it is not mistaken for
"still pending" in any future session.

1. **Persona content and combined system-message length independently re-confirmed,
   byte-exact, from three separate sources this session:** `cat`'d directly from
   `wiki/persona.md` (1,951 chars, matching the implementation record above exactly);
   reconstructing `_SYSTEM + "\n\n" + persona` (`_SYSTEM` since renamed to
   `_SYSTEM_TEMPLATE`, 2026-08-03 — see Slot 1a) from this disk content produces exactly
   **2,113 characters**, matching both a real live backend log's own `system_chars=2113`
   debug field (from a `13:19:04` conversational turn this session) and an independently
   built `/admin/api/cache/probe` payload. All three agree exactly — there is no remaining
   doubt about what the real per-turn system message currently contains, nor any doubt that
   the construction described above is what's actually running in production right now.

2. **Two real conversational turns ran with this exact 2,113-character / 525-token system
   message** (confirmed via live backend log `_execute_plan: assembled system_prompt:`
   dumps, identical text on both turns), the model already warm for the second of the two
   (no cold-load confound).

3. **`POST /admin/api/cache/probe`, run immediately after, against this exact payload,
   twice in direct succession:** identical result both times —

   ```
   total_tokens: 525, block_size: 512, total_blocks: 2,
   blocks_ssd_hot: 0, blocks_ssd_disk: 0, blocks_cold: 2,
   ssd_hit_tokens: 0, cold_tokens: 525
   ```

   Fully cold, both blocks, both calls. Not the "block 0 hit, block 1 cold" pattern the fix
   was designed to produce for this lineage.

4. **Probe self-write ruled out as a confound:** sending identical content to the probe
   twice in a row with no real inference call in between also showed fully cold both times
   — consistent with the endpoint's own documented behavior (a hash-and-check walk against
   existing cache state, not a prefill) and confirming the probe itself never writes to
   cache. The two real conversational turns were the only events in this test capable of
   writing cache state for this content; neither produced a hit on the immediately-following
   probe.

5. **Aggregate dashboard efficiency climbed from 23.8% → 30.3%** over the same live session
   window (session-scoped, cleared beforehand; cached tokens 1,536 → 3,072, both exact
   multiples of the 512-token block size) — a real, positive trend, but uninformative about
   *which* of the three competing system-message lineages produced it, since the figure
   aggregates all prefill traffic without attribution. Fully consistent with this
   persona-bearing lineage getting zero hits while some other lineage (most plausibly the
   warm-up fixture re-hitting itself) accounts for the entire observed gain.

**Verdict: the fix remains correctly implemented at the construction level (re-confirmed
independently this session) but is NOT producing the cache benefit it was designed to
produce, based on direct live evidence rather than absence of evidence.** This should be
read as a genuine negative result going forward, not as "still awaiting verification."

**Root cause is not yet diagnosed; per standing project discipline, no fix should be
proposed before it is.** Candidate explanations, none yet investigated:
- The two turns observed may not have been the first time this content was ever sent —
  if the most recent model reload (confirmed to have happened earlier in this session)
  invalidated prior cache state, these two turns would both be "after invalidation, before
  re-caching," not proof the content has never cached. Not ruled out.
- Whether `OMLXRuntimeClient.infer_stream()`'s real streaming call path triggers the same
  `store_cache` behavior as the warm-up hook's non-streaming `infer()` call has not been
  directly checked — Lever 3's cache-writing behavior is confirmed only for the non-streaming
  path. If streaming and non-streaming calls are handled differently by oMLX's own caching
  logic, this would be a previously-unconsidered mechanism gap.
- Whether oMLX's block-hash chaining has any dependency beyond the system message's own
  token range (e.g. generation parameters such as `temperature`/`max_tokens`) is speculative
  and not source-confirmed, but not yet ruled out either.

**Next step (not yet started):** a direct source read of oMLX's `store_cache` call path
(same installed-package directory used for the original 2026-06-23 mechanism read), followed
by a targeted live test isolating streaming vs. non-streaming calls against identical content.

---

### 3.7c Update — Dashboard Observation, Sustained High Efficiency Across Restarts (2026-06-25)

A new aggregate data point, recorded as supporting evidence for this open investigation — **not**
a probe-confirmed finding, and explicitly not given the same evidentiary weight as the byte-level
verification above. Per this project's "verify the mechanism, not just the correlation" discipline,
this is logged as a dashboard reading pending live-probe confirmation, not as a resolved result.

**Correction, added immediately after this entry was first written:** this session was run on
**oMLX v0.4.4**, confirmed completed before this session began and consistent as a single binary for
the session's entire duration — not a mid-session change, and not a variable that differs between
the dashboard observation below and the rest of this session's work. Every prior entry in this §3.7c
thread — the original 2026-06-23 mechanism finding, the 2026-06-24 negative-result probe, and the
source-code read citing `/opt/homebrew/Cellar/omlx/0.4.2/...` — was conducted against v0.4.2, on
separate earlier sessions. The confound is at the boundary between this session and that prior work,
not within this session. The upgrade was not staged as a planned, controlled variable for this
investigation; it is logged here as a confound identified after this entry was first drafted, not as
a deliberate variable the analysis below was designed to isolate. **The "why is this notable"
analysis below was originally written assuming same-binary continuity with the 2026-06-24 probe —
that assumption is false.** The analysis is left in place rather than deleted, because the question
it raises (does sustained high efficiency survive restart/standby) is still a real question — but
every conclusion below must now be read as "possibly a v0.4.2 mechanism finding, possibly simply how
v0.4.4 behaves," with no way to distinguish those from a dashboard reading alone.

**Observation:** during this session's live testing (the Open Item 11 reproduction and gate-
calibration/backstop verification work, recorded above), the oMLX Serving Stats dashboard reported,
for `gemma-4-e4b-it-4bit`, all-time/session figures of: 41,236 total prefill tokens, 26,112 cached
tokens, **63.3% cache efficiency**. Reported as sustained — i.e., not reset to near-zero — despite
the Localist Runtime being restarted and the model going idle/into standby multiple times over the
course of the session.

**Why this is notable relative to the existing negative result — now qualified by the version
confound above:** the 2026-06-24 update found fully cold probe results (`blocks_ssd_hot: 0,
cold_tokens: 525`) for two real, identically-worded conversational turns on **v0.4.2**, against a
dashboard efficiency climbing only modestly (23.8% → 30.3%) over that session — and explicitly
concluded that modest aggregate gain was uninformative and likely attributable to a different
lineage entirely (the warm-up fixture self-hitting). 63.3%, persisting across multiple
restart/standby cycles, is a substantially larger and more durable figure than anything previously
recorded for this investigation — but it was recorded on **v0.4.4**, a different binary, so it
cannot be directly compared to the 23.8%/30.3%/cold-probe figures as if they were the same system
at two points in time. If this dashboard number reflects real, mechanism-confirmed cache reuse,
restart/standby persistence would be new information not covered by either candidate explanation
already on file (model reload invalidating cache state; streaming vs. non-streaming `store_cache`
differences) — but it would be new information about **v0.4.4 specifically**, not necessarily a
correction to what was found on v0.4.2.

**What this does NOT establish, stated plainly — now a longer list given the version confound:**
this is one dashboard reading, not a live `/admin/api/cache/probe` call against known, byte-confirmed
content, and not attributed to any specific lineage (system message vs. user message vs. some other
prefill source). It carries the same aggregation problem the 2026-06-24 entry already named — the
figure sums all prefill traffic without distinguishing which calls, paths, or content produced the
hits. It does not, on its own, overturn the 2026-06-24 negative result for the specific
persona-bearing system-message lineage that result was about — both because of the aggregation
problem and now because of the version difference, either of which independently breaks any direct
comparison. It also does not establish that v0.4.2's documented cold-probe behavior would reproduce
or fail to reproduce on v0.4.4; that is now an open, distinct question this entry cannot answer.

**Status: logged as supporting evidence, investigation remains open — now with an added
prerequisite.** This raises the priority of the next step already named above (direct source read
of `store_cache`, followed by a targeted live test isolating streaming vs. non-streaming calls) —
the restart/standby persistence detail is a new and specific enough observation that it may be worth
probing directly rather than waiting for it to recur. **However, given the v0.4.2 → v0.4.4 version
confound identified above, that source read must now target the v0.4.4 installed package path, not
the v0.4.2 path cited in the original 2026-06-23 finding (`/opt/homebrew/Cellar/omlx/0.4.2/...`) —
the two versions' `store_cache` implementations cannot be assumed identical without checking.** Not
yet investigated as of this entry; no mechanism claim is made for either version.

---

*(Reminder, unchanged from before: §3.7 and §3.7b's "no current backend...
can produce a user-message cache hit" line is still flagged inline as
superseded but not yet corrected — see line 824 of the current doc. That
edit remains a deliberately separate, unbundled task and was not addressed
in this update.)*

---

### 3.7d oMLX Multi-Turn `messages` Array Implemented for Slot 6 (2026-07-19)

§3.7b speculatively described a future direction: "if a future engine or plugin sends a growing
multi-turn `messages` array... the dynamic-suffix ordering already locked here becomes the correct
stable-history/dynamic-tail shape for it to exploit, with no further prompt-assembly redesign
needed." This entry implements that shape for oMLX specifically — a working-memory-scoped, opt-in
change, not a rewrite of the slot contract itself.

**Investigation first (no code change).** Before changing anything, `prompt_builder.py` was read in
full to confirm exactly how `build()` assembles its return value and where the risk of a breaking
change actually was. Findings: `build()` returns `tuple[str, str]`; `user_prompt` is 10
independently-built slot strings joined with `"\n\n".join(s for s in slots if s)` — no slot depends
on another slot's rendered text, the join is pure string concatenation over already-finished
strings. The *only* place working-memory turns get flattened into text is one line in
`_slot6_working_memory()` (`f"Turn {offset} [{role_str}]: {turn.content}"`) — Turn objects arrive
already structured from `controller_agent.py` (built from DB rows before `build()` is ever called).
Blast radius of a naive return-shape change: `base_runtime_client.py` and all three concrete runtime
clients declare `infer(prompt: str, system: str, ...)`; `conversational_agent.py`, `wiki_agent.py`,
`warmup.py`, and `controller_agent.py`'s retry path all unpack `build()`'s return as a 2-tuple; and
~8 existing tests assert on literal `"Turn -N"` / `"[WORKING MEMORY]"` text inside `user_prompt`.
**Verdict: an additive opt-in parameter is a contained change; changing the default return shape
would not have been.**

**What shipped.** `PromptBuilder.build(emit_structured_working_memory: bool = False)` — default
`False` reproduces the existing 2-tuple exactly (confirmed: the ~8 pre-existing flattened-text tests
were run unmodified before and after this change and their pass/fail status never moved). When
`True`: Slot 6 is omitted from `user_prompt` entirely, and `build()` returns a 3-tuple instead, with
the trimmed, chronologically-ordered (oldest first) `list[Turn]` as the third element. The
drop-oldest-first ceiling logic itself didn't change — it was factored out of
`_slot6_working_memory()` into a new `_trim_working_memory()`, shared by both the flattened-text
path and this new structured path, so both trim identically against the same
`working_memory_ceiling` (unaffected by §16.11's ceiling-math change — only which slot supplies the
ceiling value changed there, not how this method trims against it).

**Wiring (oMLX-only).** `controller_agent.py` passes `emit_structured_working_memory=True` only when
`isinstance(self._runtime, OMLXRuntimeClient)` — checked independently at both the primary
`_execute_plan` build() call and the separate-scope empty-answer retry path
(`_dispatch_conversational_with_empty_guard`, which needed its own recomputed check; a real test
failure during development caught the missing one). The resulting turns are threaded onto
`subtask_context["_prebuilt_working_memory_turns"]`, following the same passthrough convention as
`_prebuilt_prompt`/`_prebuilt_system`. **A plumbing gap was found and closed while wiring this in:**
`conversational_agent.py`'s prebuilt-prompt passthrough never read that context key at all, so the
controller-side wiring alone would have been inert — it now forwards
`working_memory_turns=context.get("_prebuilt_working_memory_turns")` to the runtime's
`infer()`/`infer_stream()`, but only when that key is present, so Ollama/Foundry — which never
populate it — call the runtime exactly as before with no new keyword at all.

**Consumption (`OMLXRuntimeClient`) and "Prompt too long" recovery.** See §16.12 in
`16-runtime-backend-layer.md` for the full detail: the outgoing `messages` array now mirrors oMLX's
own web UI (system message, one entry per working-memory turn, current query last) instead of one
flattened user message, and a specific 400 ("Prompt too long", oMLX's `validate_context_window()`)
triggers a single oldest-turn-drop retry rather than the generic non-200 error path.

**Status: implemented and unit-tested; the actual KV-cache-reuse benefit this shape targets is
unverified.** All tests use mocked HTTP — no live oMLX server, consistent with `CLAUDE.md`'s testing
posture. Whether this request shape actually produces more cache-hit-eligible prefill on a real
oMLX instance (the thing §3.7b's whole speculative direction was for) was not measured this session.
`/admin/api/cache/probe` (§3.7c) is the existing tool that would answer that — a natural next step,
not yet done.

**Test suite:** 884 passed / 0 failed after this change (878 baseline from §16.10/§16.11 + 6 new —
`TestEmitStructuredWorkingMemory` in `test_prompt_builder.py`, covering the default-False 2-tuple
shape, the True-path 3-tuple shape and text omission, chronological ordering, ceiling-based
oldest-first trimming, the empty-turns case, and flattened/structured trim-count parity). §16.12's
`OMLXRuntimeClient` changes add a further 4 tests (888 passed total) — see that section.

