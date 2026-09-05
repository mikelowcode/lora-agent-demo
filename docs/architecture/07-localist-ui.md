## 7. Localist UI

### 7.1 Overview

Localist UI is the SvelteKit frontend sub-product. It communicates with
the Localist backend exclusively via the REST/SSE API on port 8001. All
rendering is client-side; the backend has no knowledge of the UI.

**Tech stack:** SvelteKit, TypeScript, IBM Plex Sans / IBM Plex Mono,
CSS custom properties (no Tailwind, no component library).

**Directory:** `localist-ui/` at the project root.

**Dev server:** `npm run dev` from `localist-ui/` (port 5173).

### 7.2 Routes

| Route | Component | Purpose |
|---|---|---|
| `/conversation` | `ChatPanel.svelte` | Primary chat interface. Streams SSE responses. |
| `/memory` | `EpisodesPanel.svelte` | Episodic memory browser. |
| `/files` | `FileBrowser.svelte` | Full-width file preview pane. Wiki/Raw/Generated listing, upload, and ingest moved into the sidebar's Files sub-nav (§7.11, 2026-07-13) — no longer part of this route's own component. |
| `/settings` | Settings | Runtime backend status/health, live runtime-backend switch + per-backend chat-model dropdown (§7.10, §16.6), chat-history eviction preset, theme. |

### 7.3 Provenance Bar

Every completed assistant turn renders a **provenance bar** between the
response body and the source chips. It is driven by the `metadata` field
in the SSE `"done"` event.

**Chips rendered:**

| Chip | Condition | Colour |
|---|---|---|
| `P1 · Direct` | `priority === 1` | Muted |
| `P2 · Memory write` | `priority === 2` | Green |
| `P3 · Web search` / `P3 · File operation` / `P3 · Page fetch` / `P3 · Tool` | `priority === 3`, labeled by whichever tool fired (see below) | Blue |
| `P4 · Vault` | `priority === 4` | Purple |
| `P5 · Episodic` | `priority === 5` | Amber |
| `P6 · Inference` | `priority === 6` | Muted |
| `⚙ {tool_name}` | each entry in `tools_fired`; also rendered for a deferred `file_op` (see below) | Orange |
| `◎ episodic` | `fetch_episodic === true` | Amber |
| `◈ grounded` | `grounded === true` | Green |

Source chips (wiki/raw type + human-readable name) are rendered below the
provenance bar from the `sources` array.

**Correction 2026-07-07 — P3 chip was hardcoded to "Web search" regardless
of which tool actually fired.** Priority 3 (§4.2) is a generic tool-signal
priority — `web_search`, `file_op`, or `url_fetch` can each independently
match it — but `ChatPanel.svelte`'s provenance bar rendered a literal
`"P3 · Web search"` string for every Priority-3 turn, including ones where
only `file_op` or `url_fetch` had fired. Fix: a new `p3Provenance()`
helper reads `tools_fired` and picks the label from whichever tool actually
matched (`"P3 · Web search"` / `"P3 · File operation"` / `"P3 · Page
fetch"`), falling back to a generic `"P3 · Tool"` when none or more than
one match (compound turns). A deferred `file_op` (§4.4b) counts as a
`file_op` match for this purpose even though `tools_fired` stays empty for
it — the write happens after generation completes, so `file_op` never
enters `tools_to_call` — by checking `metadata.file_op_deferred` alongside
`tools_fired`. The same deferred case previously showed *no* tool chip at
all (the `⚙ {tool_name}` loop iterates `tools_fired`, which is empty here);
a matching `⚙ file_op` chip is now rendered explicitly whenever
`file_op_deferred` is true and `file_op` isn't already in `tools_fired`.
Verified by code-trace of `ResponseMetadata` (§4.4a) plus inspecting the
live task metadata for the `moon.md`/`ocean.md` deferred-dispatch repros
from §4.4b (`file_op_deferred: true`, `tools_fired: []`, chip renders `⚙
file_op` with no duplicate). A headless-browser screenshot check
(Playwright) was attempted for full pixel-level confirmation but the
install stalled in this environment; this is noted as a limitation on the
verification depth, not skipped as a shortcut — the API-level metadata
trace plus source-read of `p3Provenance()`'s branch logic against every
`tools_fired`/`file_op_deferred` combination is the verification actually
performed.

### 7.4 Episodic Memory Panel

The `/memory` route renders `EpisodesPanel.svelte`, which calls
`GET /api/memory/episodes` on mount and on manual refresh.

**State management:** `localist-ui/src/lib/stores/episodes.ts`
- `episodesStore` — writable store with episodes list, loading/error state,
  pagination, and active type filter
- `loadEpisodes(opts)` — fetches from `/api/memory/episodes` with optional
  `episode_type`, `offset`, `limit` query params
- `EPISODE_TYPES`, `TYPE_LABELS`, `TYPE_COLORS` — constants for the 7
  canonical episode types

**Episode card fields displayed:** type chip (colour-coded), subject,
date, content, confidence percentage, project context, source.

**Type filter bar:** All | Preference | Correction | Decision | Workflow |
Fact | Relationship | Context

### 7.5 API Proxy

Localist UI proxies all `/api/*` requests to `http://localhost:8001` via
the Vite dev server config. Production deployments should configure an
equivalent reverse proxy. The `/api` prefix is stripped before forwarding.

### 7.6 Status Bar & Live Turn Rendering

#### Header status chips

**Superseded 2026-07-13 — see §7.10.** The three-chip layout described in this subsection
(agents/model/connectivity) was replaced by a single consolidated chip (green dot + active
inference-engine name, Chat screen only) plus a separate pending-count chip (Memory screen only)
as part of the 1a desktop-UI port. The agents chip/popover described below no longer exists in the
UI at all — `agents.ts`'s store and polling are still running but unconsumed by any component (see
§7.10's open items). The rest of this subsection is retained as the historical record of the
pre-2026-07-13 SSE/status-event mechanics, which are unchanged by the redesign.

`StatusBar.svelte` (`localist-ui/src/lib/components/StatusBar.svelte`) renders three chip types in the right section of the application header: agents, model, and connectivity. A fourth chip — a "streaming" indicator driven by `$tasksStore.streaming` — existed in earlier versions and was removed 2026-06-28. It duplicated the in-bubble status line already present in `ChatPanel.svelte`; the in-bubble status line is now the canonical live-status indicator for in-flight tasks.

**Agents chip.** Reads `$agents.agents` (a `string[]` of agent names) and `$agents.loaded` (boolean) from `$lib/stores/server`. Hidden until `$agents.loaded === true && $agents.agents.length > 0`. The store exposes agent names only — not per-agent activity state, task assignment, or health. The chip label is the agent count (e.g. `2 agents`). Clicking it toggles a popover anchored below the chip (`position: absolute; top: calc(100% + 6px); right: 0`) listing each name as a `role="menuitem"` row. The button carries `aria-expanded` and `aria-haspopup="true"`.

**Popover close behavior.** Three paths close the popover: re-clicking the chip (`on:click={() => (agentsOpen = !agentsOpen)}`), clicking outside the `.agents-wrap` container, and pressing Escape. Click-away and Escape are handled via `window.addEventListener` calls registered in `onMount`. Cleanup runs in `onDestroy`, guarded by `if (browser)` — where `browser` is imported from `$app/environment`:

```svelte
onDestroy(() => {
  if (browser) {
    window.removeEventListener('click', handleWindowClick);
    window.removeEventListener('keydown', handleWindowKeydown);
  }
});
```

The guard is necessary because `onDestroy` runs during SSR teardown (SvelteKit pre-renders routes on the server), not only during client-side unmount. `window` is undefined in the server environment. An earlier unguarded version of this code caused a live `ReferenceError: window is not defined` crash on `/conversation` page load (2026-06-28); the `browser` guard was the direct fix.

#### SSE status event sequence

The streaming endpoint (`POST /api/task/stream`, `_stream_task()` in `backend/main.py`) yields the following sequence for a normal chat request:

| Order | Event type | `message` / payload | Emitted after |
|---|---|---|---|
| 1 | `status` | `"Planning task…"` | Immediately, before any blocking work |
| 2 | `status` | `"Routed to {agent}"` | `controller.route_task()` returns |
| 3–N | `token` | one chunk per event | Real-time from `on_token` drain loop; ConversationalAgent routes only (see §7.7) |
| N+1 | `status` | `"Updating working memory…"` | Before `process_working_state_update()`; conditional on post-dispatch gate (see §7.7) |
| N+2 | `sources` | sources array | After `handle_task_with_plan()` returns |
| N+3 | `done` | task_id, status, metadata, answer | — |
| N+4 | `task_complete` | task_id | Only after the full task — including `process_implicit_extraction()` and `process_working_state_update()` — has resolved. Emitted on every terminal path, success or error alike (see §7.7 Update 2026-07-05). |
| N+5 | `[DONE]` | (raw sentinel, not JSON) | — |

Event 1 is emitted unconditionally at the top of `_stream_task()` (`main.py:885`). Events 2 onward follow only after the corresponding blocking work completes. `task_complete` always immediately precedes `[DONE]` and is distinct from `done`: `done` can fire as soon as the visible answer is ready (see `on_answer_ready`, below), while `task_complete` fires only once the entire backend pipeline — memory hooks included — has actually finished. Clients must not treat `done` as the signal that it is safe to submit another task; see §7.7 Update 2026-07-05.

**Routing split.** The "Routed to {agent}" event is emitted after `controller.route_task()` (`controller_agent.py:836`) returns. `route_task()` is a thin wrapper around `self._planner.route()`, dispatched via `asyncio.to_thread` because some priority branches in `Planner.route()` call `embed_fn` or `runtime.infer()`, both synchronous. Once `route_task()` returns a `RoutingPlan`, `_stream_task()` yields event 2, then dispatches `controller.handle_task_with_plan()` (`controller_agent.py:847`) in a second `asyncio.to_thread`. `handle_task_with_plan()` calls `_execute_plan()` directly with the precomputed `RoutingPlan`, bypassing `_execute()` and therefore not calling `_planner.route()` again. Routing runs exactly once per streaming request.

**Unchanged surface.** `controller.handle_task()` (`controller_agent.py:786`) retains its original signature. `POST /task` (non-streaming) still calls `handle_task()` via `asyncio.to_thread`, unchanged.

#### Known limitation — word-replay resolved for ConversationalAgent; WikiAgent buffer path retained by design

**RESOLVED for ConversationalAgent (2026-06-28).** `_stream_task()` now uses a real producer/consumer bridge: an `asyncio.Queue[dict[str, str]]` populated from the worker thread via `loop.call_soon_threadsafe`, drained by `await asyncio.wait_for(queue.get(), timeout=0.05)` while the producer task runs. `ConversationalAgent.run()` calls `on_token(chunk)` for each chunk emitted by `infer_stream()`, and those chunks reach the SSE layer in real time — not as a post-hoc buffer replay. The word-split loop and its `asyncio.sleep(0)` separator are removed entirely. Full architecture in §7.7.

**WikiAgent — buffer path retained permanently.** WikiAgent's output contract is structured XML consumed by `parse_model_xml()`; streaming partial XML before the full response is available would surface malformed output to the user. For WikiAgent-routed plans, `on_token` is never called and the drain loop terminates promptly with an empty queue once `producer_task.done()`.

*Historical record of the pre-2026-06-28 word-replay approach (superseded):* `_stream_task()`'s former token loop split the completed answer on whitespace and yielded one `"token"` event per word, separated only by `await asyncio.sleep(0)`. Because both agents called `self._runtime.infer()` synchronously, the full answer was already in memory when the loop began; it completed faster than a single browser paint cycle, making the "Streaming answer…" status frame and the full answer content effectively simultaneous in the UI. The routing-status frames were the only phases where visible progressive rendering occurred. Documented as an accepted cosmetic gap in the original §7.6 entry.

**Correction to in-session pacing/streaming diagnosis (2026-06-28).** During this session — after the `infer_stream()` wiring described in the RESOLVED block above was already applied — specific test cases appeared to show "no visible word-by-word streaming" or "all arrived at once." An earlier draft of this subsection tentatively attributed this to residual blocking-`infer()` usage (the same causal mechanism as the pre-fix era, based on a stale in-session read of `conversational_agent.py`). **This diagnosis was wrong.** Confirmed against current on-disk `conversational_agent.py`: both the prebuilt-prompt branch (lines 219–229) and the main RAG branch (lines 364–375) call `self._runtime.infer_stream()` when `on_token` is not `None`. The wiring was complete at the time the incorrect diagnosis was made — the apparent "missing streaming" was not caused by `infer()` usage or missing pacing/`sleep(0)`. The pacing-and-sleep explanation was removed rather than softened; it was simply not the actual mechanism. The actual cause was a separate bug: the fabrication-correction propagation gap, documented in the next subsection. Garbled fabricated-tool-call text was being streamed live to the client with no correction ever arriving, which made genuine streaming look like "nothing is happening" or "all arrived at once" in the specific test cases triggered during this session.

#### Fabrication-correction propagation gap (fixed 2026-06-28)

A companion bug to the open-item fabrication detection (§8.8 Open Item 11): even when `_is_fabricated_toolcall()` correctly detected fabricated tool-call syntax and substituted `_SEARCH_UNAVAILABLE_FALLBACK`, the corrected answer never reached the client's chat bubble.

**Detection sequence (unchanged, pre-existing).** In `ConversationalAgent.run()`, both the prebuilt-prompt branch and the main RAG branch run the `on_token` streaming loop first, yielding each chunk to the SSE queue as inference progresses. `_is_fabricated_toolcall()` is called *after* that loop completes, on the fully-assembled `answer` string. On a positive match, the method returns an `AgentResult` with `output["answer"] = _SEARCH_UNAVAILABLE_FALLBACK`, `output["sources"] = []`, `output["grounded"] = False`. This correctly threads through `controller_agent.py`'s `_execute_plan()` fast-path into `ControllerResult.answer` → `result["answer"]` in the dict returned by `handle_task_with_plan()`.

**The bug.** `_stream_task()` in `main.py` emitted the `"done"` SSE event with only `task_id`, `status`, and `metadata`. The corrected `result["answer"]` existed in scope at that point but was never included in the event payload. `tasks.ts`'s `case 'done'` handler received no `answer` field, so it never overwrote the task's accumulated streamed text. The chat bubble permanently displayed whatever garbled fabricated-tool-call chunks had already been streamed, with no correction arriving — ever.

**Fix.**
- `main.py` (`_stream_task()`): The `"done"` SSE event now includes `"answer": result.get("answer", "")`. (See `main.py` lines 974–982, which also updated the SSE event table above from `task_id, status, metadata` to `task_id, status, metadata, answer`.)
- `tasks.ts` (`case 'done'`): Conditionally overwrites `task.answer` and clears `task.tokens` only when `event.answer` is present and differs from the already-accumulated streamed answer. For the normal (no fabrication) case, the condition `correctedAnswer !== t.answer` evaluates false and the patch is a no-op — no disruption to correctly streamed turns.
- `ChatPanel.svelte`: No changes required. Its existing reactive block already syncs bubble content from `tasksStore` on any store change.

**Live verification (same day).** The same incident shape was re-triggered after the fix: LangSearch returned a 500, the model fabricated tool-call syntax as its entire output, `_is_fabricated_toolcall()` detected and substituted the fallback. The chat bubble now shows `_SEARCH_UNAVAILABLE_FALLBACK` ("I don't have live search results for that — here's what I know from training, which may be stale or incomplete.") instead of the permanently garbled text seen before the fix.

#### Turn/task_id reconciliation — historical fix (2026-06-28)

**Prior bug.** `ChatPanel.svelte`'s `handleSubmit()` previously created the assistant turn with a temporary placeholder id (`const tempId = \`pending-${Date.now()}\``). The real `task_id` was only available once `submitTask()` resolved, which on the SSE path does not happen until the `[DONE]` sentinel arrives. `ChatPanel.svelte`'s reactive block (lines 30–46) matches live store updates to turns by `t.task_id === activeTask.task_id`; because the turn held `tempId` and `tasksStore` was keyed by the real UUID, no match ever occurred during an in-flight request. Every SSE status event updated the store correctly but nothing in `turns` reflected any of it until the stream ended. Only the final committed state was ever rendered; live status transitions and token-by-token content were never visible.

**Fix.** `submitTask()` in `tasks.ts` now accepts an optional third parameter (`task_id?: string`, line 88). Internally it uses `const id = task_id ?? crypto.randomUUID()` (line 90) for all store operations and the request body. `handleSubmit()` in `ChatPanel.svelte` generates `const task_id = crypto.randomUUID()` before creating either turn and passes it as the third argument to `submitTask(text, {}, task_id)`. Both the user turn and assistant turn receive the real `task_id` at creation time, so the reactive block finds a match from the first SSE event onward. The `tempId` variable and the post-`await` `turns.map(...)` reconciliation patch were removed entirely — no remaining references to `tempId` exist in `ChatPanel.svelte`.

This bug predated 2026-06-28's other changes. It was only surfaced when the addition of the "Routed to {agent}" status event created a visible gap: for the first time there was a status transition (routing → execution) that should have been visible mid-stream but was not, revealing that no live update ever reached the turn.

**Update 2026-06-29 — submitTask() resolves on 'done', not [DONE].** After the on_answer_ready fix caused `tasksStore.streaming` to flip false at the `'done'` SSE event, a secondary gap emerged: `submitTask()` in `tasks.ts` still resolved its `Promise<string>` only on the `[DONE]` sentinel, which `main.py` emits only after `producer_task` fully resolves (hooks included). This meant `submitting` in `ChatPanel.svelte`'s `handleSubmit()` stayed `true` for the full hooks duration — the textarea re-enabled visually but a fast follow-up Send was silently swallowed by the guard clause (`if (!text || submitting || $tasksStore.streaming) return`).

**Fix:** `submitTask()` restructured from `async function` to `function returning new Promise<string>((resolve) => { (async () => { ... })(); })`. `resolve(id)` is called at the `'done'` event (after `handleSSEEvent`'s store patch completes; the SSE read loop continues running un-awaited by the caller) and at `[DONE]` as a no-op safety-net (Promise spec: subsequent settle calls are silently ignored). Three additional safety-net resolves cover abrupt stream close, fetch error, and non-200 response. External signature (`(instruction, context?, task_id?) => Promise<string>`) is unchanged; the single call site in `ChatPanel.svelte` (`const task_id = await submitTask(text, {}, task_id)`) required no change.

*Minor open item:* a network drop after `'done'` resolves but before `[DONE]` causes `catch` to run post-resolve; `patchTask({status: 'failed'})` still executes, meaning a task the user experienced as complete could transiently show `status: 'failed'` in the store. Pre-existing race shape — the window is slightly wider now. Revisit on live evidence only.

---

### 7.7 Real-Time Token Streaming and In-Flight Status Visibility (2026-06-28)

#### Real-time token streaming — ConversationalAgent only

**Callback threading.** `handle_task_with_plan()` in `controller_agent.py` gained a fourth optional parameter `on_token: Callable[[str], None] | None = None`, and `_execute_plan()` gained the same parameter. When `on_token` is not None, `_execute_plan()` injects it into `subtask_context` under the key `"_on_token"`, alongside the existing `"_prebuilt_prompt"`, `"_prebuilt_system"`, and `"_routing"` keys. The `AgentInterface.run()` Protocol signature and `_dispatch()` are **unchanged** — the callback travels via `SubTask.context`, not the dispatch layer.

**`ConversationalAgent.run()`.** `on_token = context.get("_on_token")` is read once at the top of `run()`. At both existing `infer()` call sites — the prebuilt-prompt branch and the main RAG branch:
- If `on_token` is None: `self._runtime.infer(...)` is called exactly as before; behavior is unchanged.
- If `on_token` is not None: `self._runtime.infer_stream(...)` is called instead. Each yielded chunk is passed to `on_token(chunk)` and appended to a local list; the list is joined into the same `answer` variable that all downstream lines — `AgentResult` construction, `output["answer"]`, sources, grounded — already read. Both branches are wrapped in the same `try/except Exception` as the original blocking path.

**WikiAgent exclusion — permanent.** WikiAgent is not touched. Its output is raw XML consumed by `parse_model_xml()`; streaming partial XML before the full response is parseable would surface malformed output. This exclusion is structural: it falls out of WikiAgent never receiving `"_on_token"` in its `SubTask.context`.

**Queue-based SSE bridge in `main.py`.** `_stream_task()` uses an `asyncio.Queue[dict[str, str]]` (named `event_queue`) populated from the worker thread via `loop.call_soon_threadsafe(event_queue.put_nowait, item)`. Items are tagged dicts:
- `{"_kind": "token", "chunk": chunk}` — pushed by `on_token`
- `{"_kind": "status", "message": message}` — pushed by `on_status` (see next section)

`call_soon_threadsafe` was chosen over per-get `asyncio.to_thread` to avoid thread-crossing overhead for every token. The drain loop uses `await asyncio.wait_for(event_queue.get(), timeout=0.05)` while `producer_task.done()` is False, then a final synchronous `get_nowait()` drain after the task completes. A `_drain_item(item)` helper dispatches on `_kind`: `"status"` items yield `_sse({"type": "status", "message": ..., "task_id": ...})`; `"token"` items yield `_sse({"type": "token", "token": item["chunk"]})`. Both the live loop and the post-completion drain call `_drain_item()`.

For WikiAgent-routed plans, `on_token` is never called, the queue stays empty, the drain loop exhausts its 50 ms timeout on each iteration until `producer_task.done()`, and terminates immediately — no stall.

`handle_task_with_plan` is called with keyword arguments for both optional callbacks — `on_token=on_token, on_status=on_status` — making the binding explicit and position-safe against any future signature change.

#### on_status visibility event

A fifth optional parameter `on_status: Callable[[str], None] | None = None` follows `on_token` in both `handle_task_with_plan()` and `_execute_plan()`, threaded identically to `on_token`. Unlike `on_token`, it is **not** injected into `SubTask.context` — its sole call site is inside `_execute_plan()` itself, after the implicit extraction phase completes.

`on_status` is called exactly once per request at most: immediately after the `"TIMING implicit_extraction_end"` log line and before the `"TIMING working_state_start"` log line — i.e., right before `process_working_state_update()` runs. The call:
```python
if on_status is not None:
    on_status("Updating working memory…")
```
fires only inside the existing post-dispatch gate:
```python
if (db_path is not None and not plan.write_episode
        and results and results[0].status == TaskStatus.COMPLETE):
```
Turns where `plan.write_episode` is True, WikiAgent turns, failed results, or missing MemoryManager never enter this block — `on_status` is simply never called and no SSE event is emitted. No "done" counterpart is emitted for this status; the existing `"done"` SSE event already covers completion.

**Update 2026-06-29 — silently dropped after on_answer_ready.** Once `answer_ready_emitted` is set to `True` in `main.py`'s drain loop, subsequent queue events — including this `on_status("Updating working memory…")` — are dropped before reaching the SSE layer. The `'done'` event has already been sent by the time `on_status` fires (hooks run after `on_answer_ready` returns), so this status event no longer reaches the frontend on any qualifying conversational turn. The call site in `_execute_plan()` is unchanged; the suppression is entirely in the drain loop.

**Frontend compatibility — zero changes required.** `tasks.ts`'s `handleSSEEvent()` has a `case 'status'` handler at line 164 that patches `status_message` for any message string. `"Updating working memory…"` is rendered by the same in-bubble status line as `"Planning task…"` and `"Routed to {agent}"` — no frontend change was needed. This was confirmed by reading `tasks.ts` directly before writing code.

#### TIMING instrumentation

Seven `logger.info("TIMING %s t=%.4f", label, time.monotonic())` lines were added to `_execute_plan()` in `controller_agent.py` as permanent diagnostic instrumentation (not stripped). `import time` was added at module level. Labels and positions:

| Label | Position in `_execute_plan()` |
|---|---|
| `dispatch_start` | Immediately before `results = self._dispatch(...)` |
| `dispatch_end` | Immediately after `_dispatch()` returns |
| `implicit_extraction_start` | Before the `process_implicit_extraction` try block (inside the post-dispatch gate) |
| `implicit_extraction_end` | After that try/except closes |
| `working_state_start` | Before the `process_working_state_update` try block |
| `working_state_end` | After that try/except closes |
| `execute_plan_end` | Before the `if effective_agent_name == "conversational_agent"` final branching block |

`dispatch_end` marks the moment the full answer is already known — `ConversationalAgent.run()` has returned its complete `AgentResult` and `_dispatch()` has unblocked. For ConversationalAgent routes with streaming enabled, the last token was sent to the SSE queue before `_dispatch()` returned. The wall-clock gap `dispatch_end → execute_plan_end` is the total post-dispatch hook cost visible to the user as tail latency.

Grep pattern to isolate: `grep "TIMING" <server-log>`.

#### Tail-latency finding — process_working_state_update() dominates post-dispatch cost

Live timing (2026-06-28) confirmed `process_working_state_update()` accounts for the observed pause between the last streamed token and the "done" SSE event. Two live data points (normal conversational turns, `plan.write_episode=False`, `results[0].status=COMPLETE`):

| Turn | `working_state_start → working_state_end` | Outcome |
|---|---|---|
| 1 | 23.134 s | CHANGED |
| 2 | 18.835 s | CHANGED |

Both produced real state changes. Cross-reference: §9.5 Open Item 1 (pre-gate decision) and §9.5 Open Item 4 (reasoning-token exhaustion mechanism, previously confirmed at the inference layer). The `on_status("Updating working memory…")` event above was added specifically because of this finding — a 20+ second silent pause was the user-visible symptom.

*Update 2026-06-29 — user-visible impact closed.* The `on_answer_ready` early-completion callback (see next section) causes the `'done'` SSE event to fire immediately after dispatch, before either memory hook runs. The 18–23s pause still occurs server-side but `tasksStore.streaming` flips to `false` and the input re-enables before the hooks begin. Cross-reference: §9.5 Open Item 1 (pre-gate decision) remains open — this fix removes the user-visible consequence of the latency, not the latency itself.

#### Early-completion callback — on_answer_ready (2026-06-29)

The tail-latency finding above (18–23s post-dispatch pause) was confirmed to also cause full input lockout — the chat textarea and send button remained disabled for the entire duration of both memory hooks on every turn, not just during active streaming. Root cause: `on_status("Updating working memory…")` fired before `process_working_state_update()` ran, but the `'done'` SSE event (which flips `tasksStore.streaming` to `false` in `tasks.ts`, which drives `ChatPanel.svelte`'s `disabled` bindings) was only emitted after `_execute_plan()` returned — which required both hooks to complete first.

**Fix:** new `on_answer_ready: Callable[[dict[str, Any]], None] | None = None` parameter added to both `handle_task_with_plan()` and `_execute_plan()`, threaded identically to `on_token`/`on_status`. A new `_build_conversational_result()` helper (factored from the conversational_agent fast-path synthesizer block) constructs the `answer`/`sources`/`status`/`metadata` payload used by both the early callback and the final return path. `on_answer_ready` is called immediately after `results = self._dispatch(...)` returns, before either memory hook runs. Only fires on complete single-agent conversational dispatch — WikiAgent, failed, and synthesizer paths are unchanged.

In `main.py`, `on_answer_ready` bridges to `event_queue` via `call_soon_threadsafe` (same pattern as `on_token`/`on_status`). The drain loop handles `_kind == "answer_ready"` by immediately yielding `sources` + `done` SSE events and setting `answer_ready_emitted = True`. Subsequent queue events after this point (including the `on_status("Updating working memory…")` from the hooks) are silently dropped to avoid flickering the task status back to `'planning'` after `'done'`. After `await producer_task` completes (hooks finished), only the `[DONE]` sentinel is emitted. If `producer_task` raises after `'done'` was already sent, the error is logged but no error SSE event is emitted over the already-completed stream. All failure paths (routing exception, producer exception before `'done'`, `result.status == 'failed'`) are unaffected — they apply only when `on_answer_ready` was never called.

#### Update 2026-07-05 — runtime-level serialization + `finalizing` gate close the concurrency gap `on_answer_ready` reopened

`on_answer_ready` (above) removed the user-visible *wait*, but unlocking `tasksStore.streaming` at the same moment as `'done'` reopened a concurrency hazard: a user could submit turn N+1 while turn N's `process_implicit_extraction()` / `process_working_state_update()` were still running. Both calls invoke `runtime.infer()` against `omlx_runtime_client.py`, which talks to a single oMLX instance (port 8000, one Gemma 4B model) — so turn N+1's `main_dispatch` call and turn N's still-running background hook call ended up contending for the same instance. Confirmed by direct timestamp cross-reference (turn N+1's submission falling inside turn N's still-open `working_state_start`→`working_state_end` window) across two independent live sessions; full investigation trail, including the ruled-out call-frequency and thermal-throttling hypotheses that preceded this finding, is in `sessions-log.md` §16.

**Fix, defense in depth — either layer alone is sufficient:**
- **Backend — `omlx_runtime_client.py`.** A module-level `threading.Lock` (not `asyncio.Lock`: the full call chain from `conversational_agent.run()` down to the runtime client is synchronous, dispatched via `asyncio.to_thread()` from `main.py`, so serialization has to hold across worker threads, not coroutines) brackets the HTTP call + SSE consumption inside `infer_stream()`. `infer()` delegates to `infer_stream()`, so it is covered without separate locking. A `label` parameter (`"main_dispatch"`, `"implicit_extraction"`, `"working_state"`) threads through from each call site for diagnostic correlation with the existing `_log_infer_throughput()` lines. A `RUNTIME_OVERLAP` warning logs if any call ever finds the lock already held — structurally, this should never fire once the lock is in place (a contending call blocks at `.acquire()` rather than reaching the check); it ships as a live tripwire, not an expected event.
- **Backend — `main.py`.** New `task_complete` SSE event (see updated table, §7.6), emitted on every terminal path — success and error alike — only after `await producer_task` resolves, i.e. only after both memory hooks have actually finished.
- **Frontend — `tasks.ts` / `ChatPanel.svelte` / `ResearchView.svelte`.** New `finalizing` store field, separate from `streaming` (`streaming` is unchanged and still drives token-visible UI state). `finalizing` starts `true` at submission and clears only on `task_complete`, with fail-safes on `error`, `[DONE]`, and dropped-connection paths so a missed event cannot permanently disable submission. **`finalizing` gates only the submit action** — `handleSubmit()`'s guard clause and the send/query button's `disabled` binding in both components. The compose textarea (`ChatPanel.svelte`) and query input (`ResearchView.svelte`) are never disabled by it and remain freely editable throughout, including during the post-`'done'` finalizing window; the attach button in `ChatPanel.svelte` is likewise ungated by `finalizing` (only by the pre-existing local `submitting` flag), since attaching a file is a compose-time action, not a submission. A first pass disabled the whole textarea for the finalizing window and swapped its placeholder to a "saving" message; this was corrected same-day after review — for a sub-30s window, a disabled send button (with a native `title` tooltip explaining why) was judged sufficient, and a disabled compose box was not. `streaming` retains its prior meaning and call sites unchanged.

**Net effect:** `process_implicit_extraction()` and `process_working_state_update()` are now guaranteed to run without contending against the next turn's `main_dispatch` call, enforced at two independent layers (backend lock, frontend gate) so that either one holding is sufficient defense in depth. Live-verified via a deliberately fast-paced session (turn N+1 submitted during turn N's `working_state` window): zero `RUNTIME_OVERLAP` warnings, and a visible serialization gap in the timestamps (turn N+1's `main_dispatch` call blocked ~27s at the lock before its own HTTP POST completed) — the expected shape for a working blocking mutex.

*Scope note.* This closes the confirmed concurrency bug and its UX side effect (premature input unlock) only. The investigation that led here started from a laptop-heat observation; thermal throttling was directly tested and explicitly *not* confirmed as the mechanism (throughput varied non-monotonically across live samples, falsifying simple throttling before any fix was built around it). No heat/thermal claim is made here beyond what was verified: no concurrent runtime calls, no premature UI unlock. See `sessions-log.md` §16 for the full ruled-out-hypothesis trail.

#### Process note — mount staleness (recurring pattern, 2026-06-28)

Context staleness from mount-time reads recurred across multiple files in this session: `main.py`, `ChatPanel.svelte`, `controller_agent.py`, and `episodic_extractor.py` each exhibited stale-context issues traceable to reading file or variable state at initialization rather than at use time. Each instance was a new occurrence of the pattern documented in §3.7 (persona-cache staleness), §8.8 Open Item 6 (database-path disambiguation), and §8.8 Open Item 9 (cache-read disambiguation) — not a new principle. The existing discipline — verify the mechanism from current on-disk source, not from earlier in-context descriptions — applied uniformly across all four files. No new architectural rule is warranted.

### 7.8 Chat History Persistence (2026-06-29)

Conversation history (`turns: Turn[]`) was previously local component state in `ChatPanel.svelte`. Because Conversations and Files are separate SvelteKit routes (`+page.svelte` files rendered into `+layout.svelte`'s `<slot />`), navigating between tabs unmounted and remounted `ChatPanel`, resetting `turns` to `[]` on every navigation.

**Fix:** new `$lib/stores/chatHistory.ts` exports `chatHistoryStore: writable<Turn[]>([])` and the `Turn` interface (moved from `ChatPanel.svelte`). `ChatPanel.svelte` reads and writes through `$chatHistoryStore` / `chatHistoryStore.update()` exclusively — no local `turns` variable remains. The store lives at module level and survives any number of route navigations, resetting only on full page reload (by design — `SESSION_ID` in `tasks.ts` has the same lifecycle).

**Open item:** `chatHistoryStore` has no programmatic clear/reset path. Only a full page reload empties it. Not yet addressed; flagged for live observation.

*Forward reference: durable, cross-session, searchable persistence of chat turns — a separate concern from this session-only store — shipped later as the Chat History Tab; see §12.*

### 7.9 File Browser — `type` Discriminator Fix & Generated Files Listing (2026-07-07)

*Numbering note:* the File Browser (`/files`, §7.2) previously had no
dedicated subsection in this document — only a one-line row in the §7.2
routes table. Given this, the `type`-field fix below is filed as its own
new numbered entry rather than shoehorned into an unrelated existing
section; the provenance-badge fix above (§7.3) had a natural existing home
and was handled as a sub-entry there instead — different treatment for two
genuinely different situations, not an inconsistency.

**Bug: undefined preview badge and dead ingest-footer gating.**
`FileBrowser.svelte`'s content pane has always keyed off `selectedFile.type`
— a badge (`{selectedFile.type === 'wiki' ? 'badge-success' :
'badge-warning'}`) and a footer that only renders raw-file-only ingest
controls when `selectedFile.type === 'raw'`. But `FileEntry` (both the
Pydantic model in `main.py` and the TypeScript interface in
`stores/files.ts`) never actually had a `type` field — `/files/raw` and
`/files/wiki` returned metadata with no type discriminator at all, so
`selectedFile.type` was always `undefined`: the badge rendered the literal
text "undefined" with the `badge-warning` fallback style, and the
raw-only ingest footer never rendered for any file, `.type === 'raw'`
being false for every entry. Discovered live this session by opening
`generated_files/water.md` (created the prior session, 2026-07-06 18:39,
still 0 bytes — see `sessions-log.md`) in the browser and observing the
broken badge directly.

**Fix.** `FileEntry.type: Literal["raw", "wiki", "generated"]` added to
both the backend model (`main.py`) and the frontend interface
(`stores/files.ts`). `_file_entry()` (`main.py`) now takes an explicit
`type` parameter, threaded through from each call site (`get_files_raw`
→ `"raw"`, `get_files_wiki` → `"wiki"`, `post_file_upload` → `"raw"`, and
the new `get_files_generated` below → `"generated"`) rather than being
inferred — no ambiguity about which directory a listing came from.

**Shipped alongside: Generated Files listing.** Files written by `file_op`
(§4.6, §4.4b) land in `mcp_server/file_ops.py`'s sandboxed
`generated_files/` directory (§14.7) but had no UI surface at all before
this session — `water.md`'s 0-byte state was only discoverable by direct
filesystem inspection. New `GET /files/generated` endpoint (`main.py`,
mirrors `/files/raw`/`/files/wiki`'s shape exactly) backed by a new
`_state.generated_dir` (defaults to `project_root/generated_files`,
overridable via `LOCALIST_GENERATED_DIR`), and a matching entry added to
`/files/content`'s `allowed_roots` sandbox check. Frontend: new
`generatedFiles`/`generatedLoading`/`generatedError` stores and
`loadGeneratedFiles()` in `stores/files.ts`, and a new "Generated Files"
section in `FileBrowser.svelte` (loaded on mount alongside raw/wiki),
giving the file browser three panes total.

**Live-verified:** `GET /files/generated` confirmed returning `water.md`
(0 bytes, `type: "generated"`) both before and after the backend restart
that shipped this fix (`logs/backend.log`, `GET /files/generated` calls
either side of the `10:20:37` restart); `GET /files/content` for
`water.md` confirmed serving correctly through the sandbox check.

**Test suite:** no dedicated backend test added for the `type` field or
the new endpoint in this pass — covered only by the existing
`FilesResponse`/`FileEntry` Pydantic validation (a missing `type` on
construction is a hard `ValidationError`, not a silent gap) and the live
`GET /files/generated` round trip above. Flagged here as a real gap, not
elided silently: add an explicit `test_get_files_generated` case
alongside the existing `/files/raw`/`/files/wiki` tests if this endpoint
grows more logic than a directory listing.


### 7.10 Desktop UI Direction "1a — Inline Provenance" Ported to Web (2026-07-13)

A desktop-app UI direction, designed and approved separately as an HTML/React click-through
reference (`design_handoff_desktop_ui/reference.dc.html` + accompanying `README.md`), was ported
into this SvelteKit app as a visual/structural pass — existing stores' data-fetching logic, backend
contracts, and SSE handling were explicitly out of scope. Full session narrative and rationale for
individual decisions: `sessions-log.md` §30. This subsection is the current-state reference.

**Design tokens (`app.css`).** New dark palette (`--bg: #121214`, `--bg-panel: #1a1a1d`, etc.), a
new `--accent-2` (logo gradient only), `color-mix()`-based `--accent-dim/-mid/-glow`, and a
`[data-theme="light"]` override block (the `theme` store/`data-theme` attribute mechanism already
existed; only the light-theme values were added). `--topbar-h` 44px, `--radius`/`--radius-lg`
8px/12px, `--sidebar-w` default 236px.

**Sidebar (`Sidebar.svelte`).** Two-tone CSS gradient logo mark (`.brand-mark`, no image asset);
20×20 mono-letter nav icons (C/M/F/S). Chat and Files nav rows expand their sub-lists in place on
click (local `chatHistoryExpanded`/`filesNavExpanded` state) rather than always rendering them.
Files' sub-list contains the Wiki/Raw/Generated listing, upload, and per-file ingest that used to
live in `FileBrowser.svelte` — see §7.11. New `$lib/stores/sidebar.ts` (`sidebarWidth`,
`sidebarCollapsed`, both `localStorage`-persisted) backs a drag-to-resize divider (180–320px,
collapses fully below a 120px threshold) and the sidebar footer's theme-toggle switch.
`+layout.svelte` applies `sidebarWidth`/`sidebarCollapsed` to `#app-shell`'s `grid-template-columns`
directly via `document.getElementById('app-shell')` (necessary because `#app-shell` is defined in
`app.html`, outside the component tree `+layout.svelte` renders into) — animating between two fixed-
length grid tracks this way needs no `@property` registration, unlike animating the `--sidebar-w`
custom property value itself would have.

**Appbar (`StatusBar.svelte`).** Single consolidated chip (green dot + active inference-engine
name — `Ollama`/`oMLX`/`Foundry`, not the model id), shown only on the Chat screen; a separate
"N pending" chip shown only on Memory. New sidebar show/hide toggle button at the start of the bar.
Screen title now derives from `$page.url.pathname` rather than the component's previously-unused
`<slot />`. See §7.6 for what this replaced.

**Chat (`ChatPanel.svelte`).** The always-visible provenance bar (§7.3) is now collapsed by default:
a single `prov-toggle` pill (priority label + chevron) per completed assistant turn, expanding on
click to reveal the same tool/episodic/grounded/source chips §7.3 already documents — same
`metadata`/`sources` data and `p3Provenance()` labeling, purely a disclosure restructuring. Per-turn
expand state is a local `expandedProv` map keyed by `task_id` (falling back to `timestamp`).

**Memory (`EpisodesPanel.svelte`, `episodes.ts`).** `TYPE_COLORS` for `preference`/`decision`/
`workflow`/`correction` now reference `var(--accent-dim/mid)` / success / warning / error tokens
instead of bespoke hex triples. `fact`/`relationship`/`context` keep their pre-existing bespoke hex
colors — not covered by the design handoff, a known gap. Active filter-pill state is now solid
`--accent` background + white text; the pre-existing amber-tinted `Pending` chip's distinct active
state was intentionally preserved rather than unified into the same treatment.

**Settings.** Restyled as stacked cards. Runtime Backend segmented control
(`RUNTIME_BACKENDS`/`RUNTIME_BACKEND_LABELS`/`runtimeBackendLabel` in `model.ts`) drives the appbar
chip's label. **Live-wired as of §16.6 (2026-07-15)** — clicking a backend other than the active one
now calls `switchRuntimeBackend()` (a new `runtimeBackendSwitch.ts` store) against the real
`POST /settings/runtime-backend`, gated by a `confirm()` dialog; a no-op guard skips the confirm/
switch when the clicked backend is already active. The real active backend is synced from
`GET /health`'s new `backend` field (`model.ts`'s `health.subscribe(...)`), not just read from
`localStorage` — the browser's cached value is now only a paint-before-first-poll placeholder. The
former read-only "Chat Model" card is now a `<select>` populated from `fetchBackendModels()` for
whichever backend is currently selected in the segmented control (independent of which one is
actually active, so a different backend's models can be previewed before switching to it);
`onchange` calls `pinChatModel()` against `POST /settings/runtime-backend/{backend}/chat-model`. The
free-text "Embedding model ID" field (and the rest of the by-then-empty "Model Configuration" card)
was deleted — confirmed inert for every backend per §33/§34. New Streaming-responses /
Episodic-write-approval toggles are UI-only placeholders with no backing endpoint — flagged in the
code as such.

**Corpus Embeddings (2026-07-17).** New card directly below Runtime Backend, wired to the real
`POST /memory/reembed` (§16.4) via a new `reembedCorpus.ts` store — not a placeholder. A
"Re-embed Corpus Now" button, gated by the same `confirm()`-dialog pattern as the Runtime Backend
switch, calls `reembed_corpus()` and blocks (single `asyncio.to_thread` call, no progress
callback) until every wiki/raw document has been re-embedded; an indeterminate spinner covers the
wait, matching the Runtime Backend card's loading treatment. A `corpus_stale` badge — sourced from
the new `corpus_stale` field on `GET /memory/stats`, fetched once on mount and refreshed after a
re-embed completes — reads "Corpus embeddings out of date — re-ranking is running keyword-only"
when `MemoryManager._corpus_stale` is set. On success the card shows "Re-embedded X of Y
documents."; on failure it shows `reembedError` in `var(--error)`, same as the Runtime Backend
card's error paragraph.

**Embedding Model (2026-09-05, §16.14).** New card directly below Chat Model, shown only when
`$modelConfig.backend === 'ollama'` (the real active backend, not the Chat Model card's preview-only
`selectedUiBackend` — an embedding-model switch always targets whichever backend is actually live).
A `<select>` sourced from the same `fetchBackendModels()` used by Chat Model, `onchange` calling a
new `embeddingModelSwitch.ts` store's `setEmbeddingModel()` against `POST /settings/embedding-model`.
Closes the gap the "Embedding model ID" field's 2026-07-15 deletion above left open — that field was
removed for being inert everywhere; this replacement is wired to a real, working backend path (tier
1 of §16.4's precedence) and was specifically motivated by the packaged Tauri desktop build, whose
base-only PyInstaller freeze has no MLX `EmbeddingEngine` available at all, leaving Ollama's
`/api/embed` as the only local-embedding option that build can offer. Selected value reflects a new
`embedding_model` field on `GET /health` (there was previously no GET surface for the configured
value, only the boolean `embed_model_found`). oMLX and Foundry are both left out of the UI for this
pass — oMLX per its documented backend-side gap (§16.4), Foundry as a smaller audience scoped out
rather than built speculatively even though the backend endpoint supports it identically to Ollama.

**Episode Browsing — Superseded filter + total count (2026-09-05).** `EpisodesPanel.svelte` gained a
"Superseded" filter chip alongside the existing All/type/Pending/Retracted chips, and a total-episode
count (`allEpisodesTotal` in `episodes.ts`, sourced from `GET /memory/episodes?status=all&limit=1`)
shown next to the panel title regardless of the active filter. Motivated by a support case where a
user migrating their SQLite DB into the desktop build saw only 3 (of 68) episodes and suspected data
loss on transfer — the data had transferred completely; the UI simply had no way to see anything
outside the default `status=active` filter, and no chip at all for the `superseded` status
(`_supersede_existing()`'s normal fact-replacement lifecycle, not an error state). No backend change
needed — `status=superseded`/`status=all` already existed on `GET /memory/episodes`.

**Files.** See §7.11.

**Verification posture.** `npm run check` and a production `npm run build` both clean throughout.
No browser-automation tool is available in this environment; verification was structural (SSR HTML
diffed for expected markup against a live backend) rather than a rendered visual/pixel check —
explicitly flagged as a limitation, not asserted as a full visual pass.

**Open items:**
- `fact`/`relationship`/`context` episode-type colors don't participate in the token system.
- `agents.ts`'s `loadAgents()` polling is now unconsumed by any component (the agents chip/popover
  it fed no longer exists in the UI) — not removed this session.
- No live human browser/visual QA pass has been performed as of this writing.

### 7.11 File Browser Restructure: Sidebar-Driven Listing, Download, and Two-Step Delete (2026-07-13)

Continuation of §7.10's Files change, with two new real capabilities added in the same session.
Full narrative: `sessions-log.md` §31.

**Structure.** `FileBrowser.svelte` is now a full-width preview-only pane. Wiki/Raw/Generated
listing, upload, and per-file ingest live in `Sidebar.svelte`'s expandable Files sub-nav instead —
each group independently collapsible, all expanded by default. Selection state moved to a new
`$lib/stores/fileSelection.ts` (`selectedFile`, `fileContent`, `fileContentLoading`,
`fileContentError`, `selectFile()`, `closeFile()`), shared between the sidebar (selection UI) and
`FileBrowser.svelte` (preview rendering) — previously this was local component state inside
`FileBrowser.svelte` alone.

**New endpoint — `GET /files/download`.** Returns a `FileResponse` with an explicit
`Content-Disposition: attachment` header and a `mimetypes.guess_type()`-derived media type, so the
browser performs a real download (Safari's Downloads queue, specifically) rather than navigating to
raw content the way `GET /files/content`'s JSON response would. Gated by the same raw/wiki/generated
allowed-roots check as `/files/content`, but using `Path.is_relative_to()` rather than those
endpoints' string-prefix check (`str(target).startswith(str(root))`) — the prefix check is
vulnerable to a sibling-directory bypass (an allowed root of `/data/wiki` also matches
`/data/wiki_evil`); `/files/content` and `/files/upload` still use the older, narrower check
(flagged as an open item below, not fixed in this pass). Frontend: a plain `<a
href=".../files/download?path=..." download="filename">` in `FileBrowser.svelte`'s footer for
`type === 'generated'` files — no blob/JS handling; the anchor's `download` attribute plus the
response header alone trigger the native download.

**New endpoint — `DELETE /files`.** Same `is_relative_to()`-gated path check. Unlinks the file and
calls `MemoryManager.remove_document()` to purge any `document_index` row for that path — a no-op
for generated files (never indexed), necessary for raw/wiki so a deleted file doesn't linger in RAG
retrieval. Frontend: every sidebar file row gets a trash-icon button. First click swaps that row in
place for an inline `Delete "<filename>"?` / Confirm / Cancel prompt (`confirmDeletePath` local
state in `Sidebar.svelte`) — a two-step confirmation by design requirement, using the app's existing
inline-review pattern (cf. the wiki diff apply/discard flow, §17) rather than a browser-native
`confirm()` dialog. Confirming refreshes the affected list and closes the preview pane if the
deleted file was open.

**Verification.** Both endpoints exercised directly against the running backend: correct headers
and byte-identical content for download; real throwaway files created and deleted for the delete
path, confirmed removed from disk; a path-traversal attempt (`/etc/passwd` / `/etc/hosts`) 403s on
both. Each check repeated through the Vite dev server's `/api` proxy — the actual path the browser
UI uses — not just direct-to-backend, to rule out a proxy-layer discrepancy.

**Test suite.** No backend unit tests added for either endpoint — covered only by the live-request
verification above, following the same precedent set by §7.9's `GET /files/generated` (which also
shipped without dedicated tests). Flagged as a real gap.

**Open items:**
- No backend test coverage for `/files/download` or `DELETE /files`.
- `/files/content` and `/files/upload`'s path-containment checks remain on the older, narrower
  string-prefix pattern — not fixed in this pass, worth a consistency sweep later.

### 7.12 Math Rendering (KaTeX) in `MarkdownRenderer.svelte` (2026-07-18)

Live use surfaced literal LaTeX source (e.g. `$\rightarrow$`, quote-escape commands) appearing
verbatim in assistant replies instead of the symbols they encode. Investigation (full trail:
`sessions-log.md` §41) traced the entire backend prompt-assembly path — user-profile injection,
episodic memory, `PromptBuilder`'s slot rendering — and found no formatting bug there; the model
itself emits real LaTeX/MathJax source (`\rightarrow` is the standard command for →, wrapped in
`$...$` inline-math delimiters) because its training data biases arrow/symbol notation toward math
mode even in plain prose. This only surfaces as broken text because `MarkdownRenderer.svelte`
renders CommonMark, not KaTeX/MathJax — the delimiters previously passed through uninterpreted.
Since model output can't be controlled from this side, the fix is scoped entirely to the renderer.

**Fix.** `katex` (0.18.0) added as `localist-ui`'s first runtime dependency — a deliberate,
documented exception to this file's previous "no third-party deps" design comment, not a silent
violation of it. `katex/dist/katex.min.css` is imported once at the top of the component; Vite
bundles KaTeX's font files as hashed static assets, so there is no CDN dependency (consistent with
the project's local-first constraint).

`inlineFormat()` extracts math spans to placeholder tokens *before* the existing bold/italic/
code/link regexes run, then swaps in the real KaTeX HTML at the end — otherwise KaTeX's own markup
(full of literal `< > " '`) would get mangled by the later substitutions. Tokens use a
Private-Use-Area sentinel (`U+E000`) that survives `escape()` untouched and can't collide with
anything the other regexes match. A new `renderMath()` helper wraps `katex.renderToString()` with
`throwOnError: false, trust: false`; on any unexpected exception it falls back to the literal
source, so malformed LaTeX degrades to an inline KaTeX error span rather than breaking the render.

**Currency disambiguation.** `$$…$$` is always treated as display math — plain prose never doubles
dollar signs like that. Single `$…$` is only treated as inline math when its content contains a
backslash command, which covers every LaTeX symbol a model emits this way (`\rightarrow`,
quote/accent commands, `\alpha`, …) while leaving plain currency mentions (`$5`, `$10 total`)
completely untouched.

**Known gap.** No change to the pre-existing per-line paragraph architecture — a `$$...$$` display
block spanning multiple lines within one paragraph won't be recognized, since paragraph lines are
still formatted independently (§ design predates this change). Accepted rather than fixed: the
reported symptom is always single-line inline math embedded in prose, not multi-line display blocks.

**Verification.** `npm run check` (0 errors) and `npm run build` (succeeds, KaTeX assets bundle
correctly) both clean. A standalone Node smoke test against the real `katex` package confirmed
`$\rightarrow$` renders to a proper `→` glyph, `$5 and $10 total` passes through unchanged, and a
malformed math span degrades to an inline error span without crashing. No browser-automation tool
was available to screenshot the live UI directly (same limitation noted at §7.3, §7.10) — but the
user live-confirmed it afterward with a real chat transcript showing LORA's own
`Request → Tool/Vault Search → Grounding against Local Truth → Cited Response → Memory Update`
summary rendering with actual `→` glyphs instead of literal `$\rightarrow$` text.

### 7.13 Chart Rendering (`ChartRenderer.svelte`) (2026-07-20)

New `ChartRenderer.svelte` renders `turn.metadata.chart.chart_config` (bar/line/pie) via Chart.js —
`chart.js` added as this UI's second runtime dependency after KaTeX (§7.12), same "deliberate,
documented exception" posture. Full design (schema, color-token choice, wiring position, and the
end-to-end live verification against the real running stack) lives at §14.8, the `generate_chart`
MCP tool's own section, not duplicated here — this UI-side piece is the last of six steps in that
feature, not a standalone one. In short: `Task.metadata`/`Turn.metadata` gained an optional `chart`
field (§4.4a's `ChartArtifact`), inherited by the existing provenance-sync plumbing with no
pass-through code changes; the component reads this app's own CSS custom properties for series
colors rather than a separate palette, so charts track the live theme like every other surface; and
it's wired into `ChatPanel.svelte` right after the provenance-bar block (§7.3).

### 7.14 Daily News Brief UI: Header Button Retired in Favor of a Right-Side "Previews" Tab (2026-07-22)

Full feature design (data model, preferences endpoints, brief-generation/caching endpoints, rate-limit
budget, and the working-memory/`conversation_log` seeding fix) lives at `docs/daily-news-brief-plan.md`
(§6, §11) — not duplicated here. This section covers only the UI surface, which went through two
live-bug-driven revisions the same day it first shipped.

**Settings — two real bugs found via live use, not code review:**
- The "Local area" text input (`settings/+page.svelte`) had no `autocomplete` attribute, so Chrome/
  Safari's address-autofill heuristics (triggered by the nearby "Home country" field + a "city"-shaped
  label/placeholder) offered the user's real saved addresses in a native dropdown. Fixed with
  `autocomplete="off"`.
- A second, more serious bug survived that first fix: typing in the field was effectively disabled
  entirely, not just the autofill-selection case. Root-caused by compiling a minimal repro of the
  page's actual pattern with the real Svelte 4 compiler and reading the generated output directly
  (not guessed from behavior alone): a `$: { homeCountryInput = $newsPreferences.home_country; ... }`
  reactive block mixed a store-subscription read with variables that were also `bind:value` targets.
  Svelte's compiled `input` event handler for a two-way-bound variable invalidates *both* that
  variable and any store subscription read in the same reactive block — so every keystroke
  re-triggered the block, which immediately snapped the field back to the stale store value on the
  next tick. This is a general Svelte pitfall (mixing store-derived reactive assignment with
  `bind:value` in one block), not specific to this field. Fixed by deleting the reactive block
  entirely and replacing it with a one-time imperative sync inside `onMount`'s
  `loadNewsPreferences().then(...)` — local form state is meant to be freely editable until an
  explicit Save, so a continuous reactive mirror was never the right tool here.

**UI redesign — hover popover replaced by a persistent right-side tab.** The original build put a
📰 header button in `StatusBar.svelte` next to the runtime-backend chip, with a hover popover showing
a cosmetic progressive-reveal preview. Live use (a screenshot from the user) showed the popover was too
small and truncated real headlines. Rather than resize a popover, the preview content was relocated
into a new persistent, collapsible right-side panel:
- `$lib/stores/previewsPanel.ts` — a `previewsPanelCollapsed` writable, localStorage-persisted
  (`lora-previews-panel-collapsed`), defaulting to collapsed. Deliberately simpler than `sidebar.ts`
  (§7.10): two fixed widths (`--previews-w-collapsed: 40px` / `--previews-w: 320px` in `app.css`), no
  drag-resize, since nothing asked for one.
- `#app-shell`'s CSS grid gained a third column (`app.css`, `+layout.svelte`) sized the same way the
  sidebar's column already is — the existing `grid-template-columns` transition on `#app-shell` applies
  to this column for free, so expand/collapse animates consistently with the sidebar.
- `PreviewsPanel.svelte` (new) — collapsed state renders a slim always-visible vertical tab
  (`writing-mode: vertical-rl`) that expands on click; expanded state renders a header plus a
  scrollable stack of "blob blocks." The Daily News Brief block shows up to 3 full article
  title+source links per section (World/National/Local/topics), not the popover's one-truncated-line
  summary — confirmed live by the user clicking an article link and having it open correctly in a new
  Safari tab. Two further blocks, **GitHub** and **Hacker News**, are reserved layout only (`Coming
  soon` badge, no live data) per explicit instruction to scope the multi-block layout now and wire
  real daily-update APIs into it later.
- The 📰 header button was removed from `StatusBar.svelte` entirely (its hover-popover CSS and
  progressive-reveal timer went with it) — the only remaining trigger is a single underlined text
  link inside the panel's News block, reading "Daily News Brief Refresh," calling the same
  `openNewsBrief()`/navigate-to-conversation flow the old button used.

Verified via `svelte-check`/`vite build` only (0 errors/warnings both times across all three
revisions) — same no-frontend-test-framework posture as §20.9. No automated browser click-through was
possible in this environment (no `chromium-cli`/Playwright/Puppeteer installed); the user performed the
real interactive verification themselves (Safari, real article links).

### 7.15 Live Feed Refresh Not Populating the Panel Live (2026-07-23)

Live use surfaced a gap §7.14's redesign didn't catch: clicking "Daily News Brief Refresh" correctly
populated a new chat conversation, but the Live Feed panel itself stayed stale until a full browser
reload.

`PreviewsPanel.svelte`'s `handleOpenBrief()` awaited `openNewsBrief()` (`POST /news/brief/open`) and
navigated to the new conversation, but never re-touched the `newsBriefPreview` store that actually
backs the panel's rendered article list — that store is populated exactly once, in `onMount`
(`fetchNewsBriefPreview()`). Because `PreviewsPanel.svelte` lives in `+layout.svelte`, it persists
across the `goto()` navigation instead of remounting, so `onMount` never refires on a client-side route
change; only a full page reload re-ran it. The backend side was already correct — `/news/brief/open`
(`main.py:2303`) writes the fresh sections into `news_brief_cache` before it responds, so `GET
/news/brief/preview` was already serving current data. Nothing client-side was asking for it again.

Fixed by having `handleOpenBrief()` call `fetchNewsBriefPreview()` immediately after `openNewsBrief()`
resolves, before navigating away (`PreviewsPanel.svelte:21-29`) — no backend change needed. Verified via
`svelte-check` (0 errors).

### 7.16 Send a Single Live Feed Article to Chat for a Detailed Summary (2026-07-23)

An "Ask about this" button next to each Live Feed article sends just that one story into the currently
open conversation and prompts the model to look up more detail on it specifically, rather than making
the user retype the headline into chat. Backend `news_search` tool changes (the new `url` pin param)
are covered at §14.10, not duplicated here; this section covers only the UI/dispatch surface.

Built entirely on existing plumbing rather than new endpoints: Planner's Priority 3-news routing (§4.2)
is pure keyword match, so an instruction containing "news" already routes to `news_search` — the
synthetic instruction (`Tell me more about this news story: "<title>"`) is worded to guarantee this,
load-bearing, not cosmetic. `MCPToolDispatcher._derive_initial_query()` (§14.3) already prefers
`context["web_search_queries"][0]` over deriving a query from instruction text, giving a ready-made hook
to force the exact article title as the query rather than trusting the model's phrasing.
`currentConversationId` always holds a value, so there's always a target conversation — no
new-conversation branch needed.

`PreviewsPanel.svelte`'s `handleAskAboutArticle()` mirrors `ChatPanel.svelte`'s `handleSubmit()` pattern
(read `currentConversationId`/`isFirstTurnOfConversation`, push an optimistic user turn + empty
assistant turn into `chatHistoryStore` under one `task_id`, call `submitTask()`) rather than
`files.ts`'s `ingestFile()` pattern, which duplicates its own SSE parsing and only backfills chat
history once a task fully completes — the user is expected to watch this stream in live, not land on it
after the fact. `ChatPanel`'s existing reactive block already patches any task's assistant turn live
regardless of which component called `submitTask()`, so nothing in `ChatPanel.svelte` itself needed to
change. `submitTask()` is called with `context = { web_search_queries: [title], news_article_url: url }`
— the second key is new, consumed only by §14.10's pinning logic. After submitting, the handler
navigates to `/conversation/<id>` since there's no root route in this app (only `conversation/[id]`
mounts `ChatPanel`) — so a user elsewhere (e.g. Settings) still lands somewhere they can watch the reply
arrive.

The button itself is a small hover-revealed control per article, deliberately separate from the
existing `<a>` that opens the source URL in a new tab — chosen over converting that link into a
click-then-popover menu, so the pre-existing single-click-opens-source behavior stays untouched.

Verified via `npm run check` (0 errors). Live NewsAPI click-through not yet performed.

### 7.17 Settings "Local Area" Save Silently Gated by an Incomplete Topics Selection (2026-07-23)

A reported "the city field won't stick, it keeps showing Seattle" turned out to have no bug anywhere in
the actual persistence pipeline (frontend store → `PUT /news/preferences` → SQLite `news_preferences`
row → `GET /news/preferences` — confirmed no caching, no mismatch, no hardcoded default). The real cause
was UX: `main.py`'s `PUT /news/preferences` requires exactly 3 topics on every request (422 otherwise) —
a genuine backend contract — and `settings/+page.svelte` mirrored that gate but only surfaced it as a
small error *after* a click. A user who had never completed the one-time topic selection would have
every city-only save attempt silently fail, leaving the field empty — which then displays its own
`placeholder="Seattle"` hint, easily mistaken for a stuck real value.

Fixed, keeping the backend contract unchanged (the user chose this over decoupling city saves from the
topics requirement, which would have needed a backend change): the Save button is now `disabled`
whenever `selectedTopics.length !== 3`, with a persistent inline hint visible before any click
explaining the two are saved together in one request; the placeholder changed from `"Seattle"` to `"e.g.
Seattle"`; and the success message now echoes the actual saved city back
(`Saved — Local area set to "X".`). Verified via `npm run check` (0 errors); user tested and confirmed
live.

### 7.18 "Daily News Brief Refresh" No Longer Opens a Chat Conversation (2026-07-24)

Since §7.16 shipped the per-article "Ask about this" button, dumping the *entire* brief into a new Chat
Conversation on every "Daily News Brief Refresh" click had become redundant — a user who wants to
discuss a specific story now clicks that story directly, so auto-populating the chat transcript with
every headline was, per live feedback, "just noise without context."

`POST /news/brief/open` (`main.py`) previously wrote two `chat_turns` rows (a synthetic user instruction
+ the full brief markdown as the assistant reply, backing the visible Chat Conversation history) and,
when a `session_id` was supplied, also seeded `conversation_log` (Slot 6 working memory) with the same
exchange so a same-session follow-up question had brief content in context. Both writes are removed.
The endpoint now does exactly one thing: regenerate the brief and overwrite `news_brief_cache` (still not
idempotent within a day, same "Refresh means refresh" rationale as §7.14/§7.15) so `GET
/news/brief/preview` reflects it. `NewsBriefOpenRequest` (the `session_id` payload) and
`_NEWS_BRIEF_USER_INSTRUCTION` were deleted as dead code; `NewsBriefOpenResponse` now returns
`{"success": true}` instead of a `conversation_id`, since no conversation is created.

Frontend: `newsBrief.ts`'s `openNewsBrief()` returns a plain `boolean` instead of a `conversation_id`,
and posts with no body (the `SESSION_ID` payload is gone). `PreviewsPanel.svelte`'s `handleOpenBrief()`
still calls `fetchNewsBriefPreview()` after a successful refresh (§7.15's fix, still needed since the
panel's store only populates once on mount) but no longer calls `goto()` — the click now stays on
whatever page the user was on, since there is no conversation to navigate to. `handleAskAboutArticle()`
(§7.16, per-article chat) is unaffected — it never wrote to `chat_turns` directly and still runs through
the normal `submitTask()` pipeline. Backend regression tests updated in
`tests/test_main_news_endpoints.py`'s `TestBriefOpen` to assert `chat_turns`/`conversation_log` stay
empty across a refresh. Verified via `pytest tests/` (1160 passed) and `npm run check` (0 errors).

### 7.19 Collapsed the Duplicative World/National Sections into One "Top Stories" Section (2026-07-24)

`docs/daily-news-brief-plan.md` §2 flagged this live during the feature's original build (2026-07-22)
and shipped it anyway as a documented "known limitation": NewsAPI's top-headlines endpoint has no
"world" category, so the World section's `category=general` call (no `country` param) returned the same
US-outlet results as the National section's `country=home_country` call for a US `home_country` —
two calls, one set of headlines, shown twice under different labels. Live use in the Live Feed panel
surfaced this as a real, reported redundancy rather than an acceptable tradeoff.

Fixed in `backend/news_brief.py`'s `build_brief()`: the World and National calls are replaced with one
call, section key `top_stories` / label "Top Stories", using the same params National always used
(`{"country": home_country}`) — i.e. the World call was simply dropped, not replaced with something new,
since National's country-anchored query was already the more meaningful of the two. Cuts a brief with
all 3 topics selected from 6 NewsAPI calls to 5. `PreviewsPanel.svelte` needed no change — the Live Feed
panel renders sections generically by `section.label`/`section.key` from the backend response, no
per-section frontend logic existed to update. `settings/+page.svelte`'s Daily News Brief card description
updated to describe "Top Stories" instead of "World and National." Backend tests in
`tests/test_news_brief.py`'s `TestBuildBrief` updated for the single section (was two), plus a new test
asserting the call is still anchored to `home_country`. Verified via `pytest tests/` (1161 passed) and
`npm run check` (0 errors).

### 7.20 GitHub Watch Feed — the Live Feed Panel's "GitHub" Block Goes Live (2026-07-29)

§7.14's reserved "🐙 GitHub — Coming soon" block is wired up: a per-repo feed of the user's own
GitHub-watched repos' latest releases, the structural twin of the Daily News Brief block right above it
in the same panel. Full backend design (why classic PATs over fine-grained, `github_watch_cache` schema,
endpoint contracts) is not duplicated here — see this feature's build session and `sessions-log.md`
under 2026-07-29; this section covers only the UI surface. The companion on-demand `github_search`/
`github_read` MCP tools (chat-triggered public-repo crawling, unrelated to this panel) are documented at
§14.11, not here — this block never touches chat, same as the News Brief block's own §7.18 change.

**Store.** `$lib/stores/githubWatch.ts`, structurally identical to `newsBrief.ts` (§7.14): a
`githubWatchPreview` writable populated by `fetchGithubWatchPreview()` (`GET /api/github/watch/preview`,
read-only, never calls GitHub) and `openGithubWatch()` (`POST /api/github/watch/refresh`, always fetches
fresh — same "a button literally labeled Refresh should never silently reuse stale content" rationale
§7.14/§7.18 already established for the News block).

**Panel.** `PreviewsPanel.svelte`'s reserved GitHub block (§7.14) is replaced with a live one, same shape
as the News block (§7.14) rather than the still-reserved Hacker News block below it: a single underlined
"GitHub Watch Feed Refresh" link, then each watched repo as a row reusing the existing
`preview-news-article-*` CSS classes unchanged (repo name linked to its GitHub page, latest release
tag/name + date, or "no releases yet", or "unavailable" on a per-repo error) — no new CSS needed. No
"Ask about this" button (§7.16) — that pattern is specific to feeding a news article into chat context;
this feed has no chat involvement at all. `onMount` now fires both `fetchNewsBriefPreview()` and
`fetchGithubWatchPreview()`.

A whole-feed failure (most commonly `GITHUB_TOKEN` unset or invalid) comes back from the backend as a
single sentinel repo entry (`key: "_error"`) rather than an HTTP error — the panel special-cases that key
and renders its `error` message alone in the same italic "unavailable" style used elsewhere, rather than
one confusing-looking repo row named `_error`.

**Known gap, accepted as-is.** `repos.length === 0` means either "never refreshed" or "refreshed, but you
aren't watching any repos on GitHub" — both show the same "No watch feed generated yet — click the link
above to generate" message. Same class of minor ambiguity the News block doesn't hit (a brief always
produces at least a Top Stories section), left unfixed for now since it's cosmetic, not incorrect.

Verified via `npm run check` (0 errors) and a headless-browser pass (Playwright, installed standalone in
a scratchpad directory — not added as a project dependency) driving the actual running dev server:
confirmed the refresh link renders and behaves identically to the News Brief link, and that a missing
`GITHUB_TOKEN` degrades to the `"GITHUB_TOKEN not configured"` message with zero console errors rather
than a broken UI. Live-verified again after the user generated a real classic PAT and added repos to
their GitHub watch list — confirmed real repo rows with real release data render correctly.

### 7.21 Hacker News Live Feed Block, Per-Block Collapse, and "Ask about this" Parity (2026-07-30)

§7.14's remaining reserved block ("💬 Hacker News — Coming soon", positioned *after* the GitHub block) is
replaced with a live one, moved to sit *between* the News Brief and GitHub Watch blocks per an explicit
user requirement — unlike GitHub Watch (§7.20), this block *does* touch chat, via the new
`hacker_news_search` MCP tool (§14.13), so it was built with News Brief's fuller feature set as the
template, not GitHub Watch's chat-free one.

**Store.** `$lib/stores/hackerNews.ts`, structurally identical to `githubWatch.ts`/`newsBrief.ts`:
`hackerNewsPreview` populated by `fetchHackerNewsPreview()` (`GET /api/hacker-news/top/preview`,
read-only) and `openHackerNews()` (`POST /api/hacker-news/top/refresh`, always fetches fresh — same
"Refresh never silently reuses stale content" rationale). Backend data source is
`backend/hacker_news.py` — HN's public Firebase API (`topstories.json` + per-item lookup, no key), top
10 stories, `hacker_news_cache` table (schema v11→v12). Zero inference cost, zero chat involvement at
this layer — same "lives in the main backend process, not `mcp_server/`" split `github_watch.py`/
`news_brief.py` already establish; the separate chat-callable `hacker_news_search` MCP tool (§14.13) is
an intentionally independent implementation, not a wrapper around this one.

**Panel.** Same row shape as GitHub Watch (§7.20) — each story's title links to its external article URL
(or the HN discussion page for a self-post with no external link), with points/author as a subtitle line
— plus a per-story "Ask about this" button (hover-reveal, `preview-news-article-ask`), the same control
the News Brief block has (§7.16) and GitHub Watch correctly lacks. Sends an instruction naming "Hacker
News" explicitly (so Planner's P3-hacker-news gate fires, §14.13) plus `context.hn_story_url` to pin the
answer to the exact clicked story via `hacker_news_search`'s new `url` param.

**Per-block collapse/expand — new, and applied to all three Live Feed blocks, not just this one.** New
store `$lib/stores/previewBlocks.ts`: a `{news, github, hackerNews}` collapsed-state record, persisted to
`localStorage` (`lora-preview-blocks-collapsed`), defaulting all three to expanded — a sibling to
`previewsPanel.ts` (§7.14), which owns the *whole-panel* collapse and defaults to collapsed instead, since
that's the "don't eat space until the user opts in" default for a whole new panel, whereas per-block
collapse is a decluttering action once the panel is already open. Each of the three `preview-block`
sections gained a header row (title + a small chevron toggle button, `preview-block-collapse-btn`,
visually consistent with the panel's own `previews-collapse-btn`) — the refresh link and body render only
when that block isn't collapsed; the header always renders so a collapsed block stays identifiable and
reversible. The previously-unused `.preview-block-reserved`/`.preview-block-badge` CSS (only ever
referenced by the now-removed "Coming soon" placeholder) was deleted as dead code.

**Real bug found and fixed via live testing (see §14.13 for the tool-side detail): comment fabrication.**
The first live test of the "Ask about this" button surfaced the model inventing a specific, plausible
"one commenter noted..." paraphrase with zero real grounding — `hacker_news_search`'s pinned-story result
carried only a bare comment count at the time, never comment text. Root-caused by reproducing the exact
tool call directly (confirmed the raw `result_text` had no comment content at all) rather than accepting
the plausible-looking output at face value. Fixed at the tool layer (§14.13's `fetch_top_comments()`); no
frontend change was needed here since the panel just renders whatever `result_text` the model was given.

**Test coverage / verification.** `npm run check`: 0 errors throughout all three build passes. Backend:
1330 → 1371 passed across the panel, MCP tool, URL-pinning, and comment-fix work combined (full breakdown
at §14.13). Live-verified end to end through the real running stack at each pass: the three-block
ordering and collapse/expand behavior confirmed visually by the user; the "Ask about this" flow confirmed
via a direct `/task` call reproducing exactly what the button sends, both before the comment fix
(fabricated paraphrase, now understood) and after (real comments, real usernames, grounded).

### 7.22 Chat Turn "Save As": Direct-Write Generated-File Endpoint + In-Place Turn Editor (2026-08-03)

Michael asked for every assistant chat turn to be saveable as a file into `generated_files/`, with a
custom name, a `.md`/`.txt` choice, and the ability to edit the content before saving. Scoped and
planned first (an Explore-agent research pass mapped `ChatPanel.svelte`'s per-turn rendering, the
already-fully-built-but-write-only `generated_files/` read path, `mcp_server/file_ops.py`'s
sandboxing/versioning, and confirmed no editable-text or per-message-toolbar precedent existed
anywhere in the frontend), then built one approved phase at a time per Michael's established cadence.
Two scoping decisions were resolved via `AskUserQuestion` before planning: editing is **export-only**
(the draft never writes back to the live chat turn) and the file-type control is a fixed `.md`/`.txt`
dropdown, not free-text extensions.

**Phase 1 — Backend: `POST /files/generated`.** The direct-write counterpart to the model-driven
`file_op write_file` MCP tool, for a user-triggered action rather than an agent tool call — no MCP
round trip. Body: `{filename, extension: "txt"|"md", content}`. New `_sanitize_filename_stem()`
(`main.py`) collapses arbitrary input to `[A-Za-z0-9_-]`, blocking path traversal (`../../etc/passwd`
→ `etc-passwd`) and, by construction, ensuring the sanitized name never contains a space — which
matters because the endpoint parses `file_ops.write_file()`'s human-readable return string (`"OK:
wrote N characters to <name>"`) to learn the *actual* saved filename when a collision auto-versions it
(`name_2.ext`, …), and a space in the name would make that parse ambiguous. Real correctness gap
closed at startup, not just at the endpoint: `mcp_server/file_ops.py`'s sandbox root is resolved
independently (`LOCALIST_MCP_PROJECT_ROOT`, read inside the standalone `localist-mcp` process) from
`main.py`'s own `_state.generated_dir` (`LOCALIST_GENERATED_DIR`) — the two happen to coincide by
default but aren't guaranteed to if configured differently. Since `main.py` now imports `file_ops`
directly as a library (not over MCP) to reuse its sandboxing/never-overwrite logic, a one-line
`file_ops.set_project_root(generated_dir)` call was added right after `_state.generated_dir` is
resolved at startup — this only rebinds `file_ops`'s in-process module-global inside `main.py`'s own
process, leaving the separate `localist-mcp` process's copy (and its own `LOCALIST_MCP_PROJECT_ROOT`
resolution) untouched — guaranteeing the new endpoint always writes into the exact directory
`GET /files/generated`/`/files/download`/`DELETE /files` already read from. 7 new tests
(`tests/test_files_generated_endpoint.py`, fixture pattern mirrors §7.11's precedent): both
extensions, collision auto-versioning, path-traversal sanitization, empty-content/empty-filename
rejection, invalid-extension rejection. Full suite 1492 passed, 0 failed.

**Phase 2 — Frontend, first cut: `SaveAsButton.svelte` (superseded within the same session, see
below).** A small save icon overlaid top-right on a completed assistant bubble (hover-revealed),
opening a floating popover with a textarea pre-filled from `turn.content`, a filename input, and the
extension dropdown. `stores/files.ts` gained `saveGeneratedFile()`, mirroring `uploadFile()`'s
POST-then-refresh-the-list pattern. Live-verified against the real running dev server (backend,
`localist-mcp`, and frontend were all already up from a prior session) via a Playwright script —
`chromium-cli` wasn't registered in this environment, so Playwright was installed ad hoc into the
session's scratchpad directory only, the same recipe §79/§80 used, never touching the project's own
`package.json`. Confirmed the full round trip: popover opens, content prefilled, save succeeds, file
appears via `GET /files/generated`, no console errors.

**Redesign — `EditableTurnContent.svelte` replaces the popover entirely.** Michael's live feedback
after trying it: the popover's textarea was "not somewhere a user would want to draft their document
edits" — too small, and structurally the wrong shape, since the original intent was editing the
*actual chat turn paragraph* in place, with Save appearing top-right only once toggled into edit mode.
Rebuilt as a single component now responsible for a turn's entire view↔edit lifecycle, replacing both
`SaveAsButton.svelte` and the direct `<MarkdownRenderer>` call in `ChatPanel.svelte`: view mode renders
the markdown as before, with a hover-revealed pencil toggle (`.turn-edit-toggle`, same top-right
corner); clicking it swaps the rendered markdown for a full-width, auto-growing textarea (min 160px,
grows with content to a 560px cap via a resize handler mirroring the composer's own
`autoResizeTextarea()`, then becomes manually resizable/scrollable beyond that) with the filename input
and extension dropdown directly beneath it, inline — not a second popup step. The pencil is replaced by
a green checkmark (Save) and an X (Cancel) in the same corner only while editing. Editing remains
export-only exactly as scoped — `draft` is local component state, never written to `chatHistoryStore`;
Cancel discards it, and a completed Save shows an inline "Saved as `<name>`" confirmation with a Done
button that returns to view mode, the live turn unchanged either way. A stale-error UX rough edge
caught during this pass's own verification (a validation error like "Enter a file name" used to linger
after the user started retyping) was fixed by clearing `error` on input to either the name field or the
textarea.

**Real CSS bug caught by Michael, not by the automated verification pass.** The first redesign
verification only exercised a two-line response, which looked fine. A real long (6-paragraph) response
came back looking "long and skinny vertical — a noticeable departure from the landscape orientated box
it normally is." Root cause: `.bubble`/`.turn` are shrink-to-fit flex items with no explicit width
(sized by content, up to a `max-width` cap) — `.turn-edit-textarea`'s `width: 100%` can't resolve
against an undetermined ancestor width, so browsers fall back to the textarea's tiny default intrinsic
column width (~20ch) in that ambiguous case; wrapping a long response into that narrow width then drove
`scrollHeight` (and the JS-computed height, capped at 560px) far taller than the box was ever wide,
producing exactly the observed skinny column. Fixed with an explicit `width: 600px; max-width: 100%;
box-sizing: border-box` on the textarea instead of a percentage — a definite width gives shrink-to-fit
ancestors a concrete value to size around, sidestepping the percentage-in-shrink-to-fit ambiguity
entirely, while `max-width: 100%` keeps it responsive on narrower viewports. Re-verified against the
same 6-paragraph response via Playwright bounding-box measurements: bubble width recovered from the
collapsed narrow box to 634px (vs. 680px in plain view mode — a normal, minor difference, not a
regression), textarea correctly capped at 560px tall with an internal scrollbar.

**Verification.** `npm run check`: 0 errors, 0 warnings, throughout every iteration (initial popover,
redesign, and the width fix). Every pass live-verified against the real running dev server via
Playwright (installed ad hoc in the scratchpad, per the §79/§80 recipe) rather than mocked: hover
reveal, in-place edit swap, content prefill, inline validation (including the stale-error fix),
save-round-trip confirmed on disk through `GET /files/generated`, collision auto-versioning (`_2.md`)
preserved from Phase 1, export-only behavior explicitly confirmed (the rendered turn is unchanged after
an edit-then-cancel and after an edit-then-save), and the width-fix regression check on real long
content. No console/page errors at any pass. All test-created files were deleted from `generated_files/`
after each verification round, leaving no leftover artifacts.

**Open items:**
- No frontend automated test coverage for `EditableTurnContent.svelte`'s edit/save/cancel state
  machine — no frontend test framework exists in this repo (same precedent as §20.9); verified only via
  `svelte-check` plus live Playwright runs, not a checked-in regression suite.
- Each turn's editor is fully self-contained local component state — nothing prevents multiple turns
  from being in edit mode simultaneously. Not restricted, not tested; low risk given the single-user,
  one-conversation-at-a-time usage pattern, but worth a deliberate decision if it ever surfaces as a
  real point of confusion.

### 7.23 Compose Mode: Multi-Turn Document Accumulation (2026-08-04)

Michael asked to extend §7.22's per-turn editor into a "compose mode" — the assistant generates a long
document across several chat turns, and the user assembles/edits the whole accumulated artifact as one
document before saving. Scoped first, not built blind: an Explore-agent research pass confirmed no
existing backend concept fits this (Slot 6A's `active_artifacts`, §9, is a same-turn RAG-source-path
snapshot rebuilt every turn, not an accumulating document — repurposing it would break its actual
semantics; `workflow_id`, §18.10, correlates steps within one tool dispatch, not across separate chat
turns) and that `Turn`/`Task.metadata` was the only existing hook shaped right for a new
turn-grouping concept. Two product decisions were put to Michael via `AskUserQuestion` before planning,
since both meaningfully shaped the build: turns join the document via an explicit **manual "Add to
document"** control per turn (not auto-capture-while-active, so incidental replies never land in the
draft unasked), and the accumulating document lives in a **persistent side-by-side panel** (reusing
`PreviewsPanel.svelte`/`previewsPanel.ts`'s collapsible-column mechanism, §7.14), not a bottom drawer.
Built in three approved phases.

**Realized scope: zero backend changes.** The entire feature is frontend-only — turn content
concatenates client-side into a draft string, and saving reuses §7.22's exact `POST /files/generated` +
`saveGeneratedFile()` path. No new endpoint, no new `chat_turns` column, no server-side document state.

**Phase 1 — Panel shell.** New `stores/composeDocument.ts`: `{active, draft}`, scoped per
`conversation_id` (mirrors `chatHistoryStore`'s reset-on-conversation-switch pattern in
`conversation/[id]/+page.svelte`) and `localStorage`-persisted per conversation, so switching
conversations swaps in that conversation's own draft and a reload doesn't lose an in-progress document.
New `ComposeDocumentPanel.svelte` — structurally parallel to `PreviewsPanel.svelte` (same header/body
shape, same `--previews-w` sizing) but **not** folded into it: a genuinely separate concern (an
in-progress document vs. external live feeds), and unlike Previews' always-present collapsed-to-a-strip
default, this panel is contextual — "off" means the `#app-shell` grid's new 4th column goes to `0px`,
not a persistent discoverable strip, since there's nothing to keep visible across every route when no
compose session exists. `+layout.svelte` grew a 4th `gridTemplateColumns` track
(`$composeDocument.active ? 'var(--previews-w)' : '0px'`); `app.css`'s `#app-shell` initial rule updated
to match. New composer-row toggle button in `ChatPanel.svelte` (document icon, next to attach/pin,
highlighted `var(--accent)` while active). No route-gating — the panel deliberately stays open across
navigation away from `/conversation` (e.g. to Settings) while a session is active, rather than forcing
it closed, since the document isn't tied to `ChatPanel` being mounted.

**Phase 2 — Turn wiring.** `composeDocument.ts` gained `addedTaskIds: string[]` and
`addTurnToDocument(turnKey, content)` — idempotent per turn key (`ChatPanel.svelte`'s existing
`provKey(turn)`, i.e. `task_id` falling back to `timestamp`), and critically **always appends to
whatever the draft currently is**, never recomputed from the included-turns list — a prior hand-edit in
the panel can never be silently clobbered by a later addition. This also means there's no structured
"remove one turn's contribution" once added (a deliberate simplification, consistent with the manual-add
decision): undoing means editing the draft text directly, the same capability the panel needs anyway.
`EditableTurnContent.svelte` gained `composeActive`/`alreadyAdded`/`onAddToDocument` props — view mode
shows a "+" control next to the existing pencil when compose mode is on, becoming a disabled green
checkmark once that turn's been added.

**Real bug caught by live verification, fixed before the phase shipped:** the checkmark was initially
designed to stay permanently visible (not hover-gated like everything else in this corner), reasoned as
an "at-a-glance record of what's already in the document." Playwright verification against a real short
single-line answer ("The capital of France is Paris. (From my training)") showed the permanently-visible
checkmark overlapping the bubble's own last word — a real, common case for terse answers, not an edge
case. Fixed by making it hover-gated like the pencil toggle always was; only the resting color (green)
reads as "added" once revealed via hover, same accepted hover-reveal tradeoff §7.22 already established
for short content.

**Phase 3 — Clear/reset and a styling pass.** `composeDocument.ts` gained `clearComposeDraft()`
(resets both `draft` and `addedTaskIds`, keeps `active` — starts a fresh document in the same session
rather than accumulating forever, e.g. after already saving one document). `ComposeDocumentPanel.svelte`
gained a "Clear" link in the header (only shown once the draft has content) behind a two-step inline
confirm banner — same destructive-action convention as Sidebar's file/conversation delete (§7.11),
adapted to a banner rather than an in-row swap since the header has less room. Verified both dark and
light themes (all colors are existing CSS custom properties, no new hardcoded values) and the two-panel
case (Previews expanded alongside the Document panel at 1440px width) — chat column narrows but nothing
overlaps or clips.

**Verification, cumulative across all three phases.** `npm run check`: 0 errors, 0 warnings, at every
step. Every functional pass live-verified against the real running dev server via Playwright (ad hoc in
the scratchpad, per the §79/§80/§81 recipe): panel open/close, manual typing, save round-trip, reload
persistence (both `active` and `draft` survive), two real chat turns added independently with correct
separator formatting, idempotency (repeat-add is a no-op), hand-edit-after-add non-clobbering, confirm/
cancel/clear, and the light-theme + dual-panel layout checks. Zero console/page errors at any pass. All
test-created files deleted from `generated_files/` after each round; the directory's real pre-existing
contents confirmed unchanged throughout.

**Open items:**
- No frontend automated test coverage (same precedent as §7.22/§20.9 — no test framework in this repo).
- No structured way to remove a single already-added turn's contribution from the draft short of
  editing the text by hand — an accepted simplification of the manual-add design, not an oversight, but
  worth revisiting if it becomes a real friction point.
- The Document panel has no route-gating — it stays mounted and visible while navigating away from the
  conversation it belongs to. Intentional per Phase 1's reasoning, but not explicitly re-confirmed with
  Michael as a final product decision.

### 7.24 Font Size Accessibility Setting (2026-08-05)

First accessibility-scoped feature (scoped as a standalone control, not the start of a broader
accessibility initiative — the broader-push option was explicitly declined via `AskUserQuestion`). A new
Settings card, "Font Size," offering four presets (Small/Medium/Large/X-Large) as a segmented control,
placed directly under the existing Theme toggle.

**Realized scope: zero backend changes**, purely client-side — cloned `stores/theme.ts`'s exact pattern
rather than the SQLite-singleton-table pattern (`retentionSettings.ts`/§20.10) or the `.env`-persisted
pattern (`runtimeBackendSwitch.ts`/§16.5), since font size is cosmetic/per-device rendering with no
server-side behavioral implication (that split — `.env` for runtime infra, SQLite `*_settings` tables for
retention/assistant-name, `localStorage` for theme/streaming/episodic-approval/backend-url — was
confirmed against the actual current code before picking, not assumed). New `stores/fontSize.ts`:
`localStorage['localist-font-size']` (renamed from `lora-font-size`, see §7.25) + a `writable` store,
mirrored onto `document.documentElement`'s
`data-font-size` attribute (parallel to `data-theme`) both on `set()` and again in `+layout.svelte`'s
`onMount` for post-hydration correctness. `app.css` gained three `:root[data-font-size="sm|lg|xl"]`
override blocks (sibling to the existing `:root[data-theme="light"]` block) remapping the eight
`--text-*` typography tokens; `md` is the pre-existing default scale, untouched. `svelte-check`: 0
errors/warnings. Live-verified by Michael directly against the running dev server (segmented-control
active-state highlighting and no overflow/clipping at every preset, including X-Large).

**Known limitation, scoped and accepted up front, not discovered after the fact:** only 86 of the
frontend's 169 total `font-size` declarations use `var(--text-*)` tokens — the other 81 are hardcoded
px literals in component-local `<style>` blocks (heaviest in `Sidebar.svelte`, `ChatPanel.svelte`,
`PreviewsPanel.svelte`, `ComposeDocumentPanel.svelte`, `EpisodesPanel.svelte`, `settings/+page.svelte`).
This setting scales prose/headings/primary copy; sidebar labels and a good deal of chat/preview-panel
meta text stay fixed size. Michael explicitly accepted this gap rather than requesting the wider
migration in the same pass.

**Proposed future design (scoped, not built) — hardcoded-`font-size` token migration.** The 81 literal
values don't land on the 8 existing tokens (`xs:11 / sm:13 / base:14 / md:15 / lg:17 / xl:20 / 2xl:24 /
3xl:30`), so a naive nearest-token remap would visibly shift sizes even at the Medium default. Proposed
fix: add three new intermediate tokens sized to exactly match today's most common literals (zero visual
change at Medium), each gaining its own row in the three `sm`/`lg`/`xl` override blocks so it actually
scales:

| New token | Medium (= today's literal) | Absorbs |
|---|---|---|
| `--text-3xs` | 10px | 9.5px, 10px, 10.5px (13 call sites) |
| `--text-2sm` | 12px | 12px, 12.5px (15 call sites) |
| `--text-md-plus` | 16px | 16px (1 call site, `ComposeDocumentPanel.svelte`) |

The remaining literals (11/11.5 → `xs`, 13/13.5 → `sm`, 14 → `base`, 15 → `md`) already land on an
existing token within ≤0.5px, an accepted sub-pixel rounding tolerance. `html`'s root `font-size: 16px`
(the `rem` anchor, `app.css:157` at the time of this scoping) and the handful of `em`-relative
declarations are explicitly out of scope — the former is a deliberate fixed anchor independent of the
`--text-*` scale, the latter already scale proportionally with whatever token their ancestor uses.
Estimated ~80 mechanical edits across ~15 component files plus the 3 new token rows. **Deferred at
Michael's request** (2026-08-05) — no code changes made; this section exists so the mapping table and
the "why not just remap onto existing tokens" reasoning don't have to be re-derived if picked up later.

### 7.25 New Logo (Place-Marker Mark), Sidebar Nav-Icon Cleanup, and Remaining "LORA" → "Localist" Text (2026-08-05)

Michael supplied a new logo kit (monochrome place-marker mark, dark tile `#111318` / mark `#F5F4F0`;
`icon.svg`, `logomark-tile.svg`, `lockup-light.svg`, `lockup-dark.svg` + kit `README.md`) and asked to
scope its implementation. Auditing first, rather than assuming prior branding existed to replace, found
two placeholders instead of real branding: `app.html`'s `<link rel="icon" href="%sveltekit.assets%/
favicon.png">` was a dead link (`localist-ui` had no `static/` directory at all — 404 in both dev and
prod), and `Sidebar.svelte`'s wordmark rendered `<span class="brand-mark">L</span>` with **no CSS rule
for `.brand-mark` anywhere in the file** — an unstyled bare letter, not a mark.

**Assets.** New `localist-ui/static/brand/` holds the kit verbatim (all four SVGs + README) as the
canonical, servable copy.

**Favicon.** `app.html`'s dead link replaced with `<link rel="icon" type="image/svg+xml"
href="%sveltekit.assets%/brand/logomark-tile.svg">` plus a matching `apple-touch-icon` —
`logomark-tile.svg` was picked over `icon.svg` specifically because it's self-contained (tile baked
in), unlike `icon.svg`'s `fill="currentColor"` which needs a CSS color context a favicon `<link>`
doesn't provide.

**Sidebar mark.** `icon.svg`'s path data is inlined directly as markup in `Sidebar.svelte` (not
imported — `static/` isn't part of Vite's module graph, so a `?raw` import isn't available for it),
`fill="currentColor"`, replacing the bare `L`. The previously-nonexistent `.brand-mark` CSS rule was
added (20×20px, `flex-shrink: 0`, `color: var(--text-primary)`) — it now actually tracks both themes
automatically, the same pattern the kit's own README documents.

**Nav-icon-sq removal.** Follow-up ask: the five single-bold-letter nav badges (`C`/`M`/`E`/`F`/`S` next
to Chat/Memory/Episodes/Files/Settings) read as redundant against the adjacent text label and clashed
with the new minimalist mark. Removed all five `<span class="nav-icon-sq">` markup instances, the
now-orphaned `.nav-link.active .nav-icon-sq` override in `Sidebar.svelte`, and the base `.nav-icon-sq`
rule in `app.css` (confirmed unused anywhere else in the frontend before deleting).

**Remaining "LORA" text.** `app.html`'s `<title>`/meta description were the last user-visible "LORA"
strings in the UI — every per-route `<title>` (`Conversation — Localist`, `Settings — Localist`, etc.)
already said "Localist". Updated to `<title>Localist</title>` /
`"Localist — a local-first, agentic general assistant"`.

**`lora-*` → `localist-*` localStorage key rename.** At Michael's explicit request (single-user, an
in-place preferences reset on next load was an accepted tradeoff, not a bug) — every remaining
`lora-*` localStorage key was renamed to `localist-*`: `theme.ts` (`lora-theme`), `sidebar.ts`
(`lora-sidebar-width`/`lora-sidebar-collapsed`), `fontSize.ts` (`lora-font-size`, §7.24),
`composeDocument.ts` (`lora-compose-doc-` prefix, `lora-compose-panel-width`, §7.23),
`previewBlocks.ts` (`lora-preview-blocks-collapsed`, §7.21), `previewsPanel.ts`
(`lora-previews-panel-collapsed`, §7.14), `model.ts` (`lora-runtime-backend`), and
`settings/+page.svelte` (`lora-backend-url`, `lora-streaming`, `lora-episodic-approval`) — 8 files, 17
call sites, confirmed via full-repo grep both before (to enumerate) and after (to verify zero `lora-`
keys remained).

**Verification.** `svelte-check`: 0 errors/warnings after each step. Live-verified by Michael directly
against the running dev server: favicon renders correctly (confirmed via the served HTML's `<link>`
tags, not just file presence), sidebar mark renders correctly in both themes, nav spacing/alignment
holds with the letter badges gone, and — after a hard refresh — all settings (theme, sidebar
width/collapsed, font size) cleanly reset to defaults under the new key names with no leftover stale
values or console errors.

### 7.26 Pinned GitHub Repos — Release Tracking Without a GitHub "Watch" (2026-08-22)

§7.20's GitHub Watch Feed block only ever showed repos the user had clicked "Watch" on in GitHub —
which also subscribes them to every PR/issue email for that repo. Michael isn't a contributor to any of
the repos he wanted release visibility for (Ollama, an MLX fork, a Microsoft Foundry repo), so watching
them just for release notes meant unwanted notification noise. This adds a second, independent way to
track a repo's releases — pinning it by `owner/repo` slug directly, with no GitHub-side state change and
no emails. Full backend design (schema, endpoint contracts, merge/dedupe logic) is not duplicated here —
see this feature's build session and `sessions-log.md` under 2026-08-22; this section covers only the UI
surface.

**Store.** New `$lib/stores/githubWatchPins.ts`, kept separate from `githubWatch.ts` (same split as
`newsBrief.ts`/`newsPreferences.ts`: one store owns the read-only Live Feed preview/refresh pair, the
other owns the user-editable settings list). `loadGithubWatchPins()` (`GET /api/github/watch/pinned-repos`)
and `setGithubWatchPins()` (`PUT /api/github/watch/pinned-repos`, whole-list replace). Saving here does
not itself refresh the feed — same "takes effect on next generation" posture §7.18 established for the
News Brief — the user triggers that via the existing "GitHub Watch Feed Refresh" link.

**Settings card.** New "Pinned GitHub Repos" card on the Settings page, placed after "Daily News Brief".
Unlike that card's fixed chip-grid (§7.14) — there's no fixed pool of valid pinned repos, any
`owner/repo` slug is legal — this is a free-text list editor: a text input + "Add" button appending to a
local array (client-side regex check for early feedback, `owner/repo` shape only), each entry rendered
as a removable chip, gated "Save" button matching the `assistantNameUnchanged`-style disabled pattern
(disabled when the local list matches the loaded store state).

**Live Feed badge.** `GithubWatchRepo`'s TS interface (`githubWatch.ts`) gained a `source: 'watched' |
'pinned'` field matching the backend model. `PreviewsPanel.svelte`'s GitHub block (§7.20) originally
rendered a small 📌 badge next to a repo's label when `source === 'pinned'`, distinguishing it from an
actually-watched repo — reuses the existing `preview-news-article-*` row markup unchanged otherwise. The
badge was removed and an "Ask about this" button added in §7.27; the `source` field itself is unchanged
and still distinguishes the two at the data level.

**Caveat.** Pinned repos still require `GITHUB_TOKEN` to be configured, same as watched repos (the
release-lookup call is the same authenticated endpoint either way) — a user with zero watched repos but
several pinned ones still needs a token set.

**Verification.** `svelte-check`: 0 errors/warnings. Live-verified against the running dev server and
real GitHub data: pinned `ollama/ollama`, `jundot/omlx`, and `microsoft/foundry-local` via the new
Settings card, triggered a refresh, and confirmed all three resolved real release data and rendered with
the 📌 badge in the Live Feed panel (one iteration corrected `ml-explore/mlx` to `jundot/omlx` after
Michael flagged it as the wrong repo for "oMLX" — the pinned-repos feature itself is slug-agnostic, so
this was a one-line data fix via the Settings card, not a code change).

**Follow-up: link straight to Releases, not the repo homepage.** Both `repo_url` construction sites in
`build_watch_feed()` (`backend/github_watch.py`) originally pointed at the bare repo (GitHub's own
`html_url` for watched repos, a synthesized `https://github.com/{full_name}` for pinned ones) — clicking
a row took you to the repo's homepage, one extra click from the release info the row is actually about.
Changed both to `https://github.com/{full_name}/releases` (applied to watched entries too, for
consistency, at Michael's request — this block only ever shows release data, so both entry types should
land on the same page). Live-verified: all three pinned repos' links now open directly on their Releases
tab.

### 7.27 GitHub Watch Feed "Ask about this" + Pin Badge Removal (2026-08-23)

§7.20/§7.26 both explicitly noted the GitHub Watch Feed block had no "Ask about this" button and never
touched chat, unlike the News Brief (§7.16) and Hacker News (§7.21) blocks — `github_watch.py` itself has
no Planner/`MCPToolDispatcher` involvement. This closes that gap using backend plumbing that turned out
to already be in place: `MCPToolDispatcher._run_github_release()` (§14.12) already accepted a
caller-supplied `context["github_repo"]` ("owner/repo") and `context["github_tag"]` pin, and
`_priority3_github_release` (`planner.py`) already routes to `github_release` on `"release notes"` /
`"latest release"` phrasing — the same caller-supplied-pin convention `news_search`'s
`context["news_article_url"]` (§14.10) established, just never wired to a frontend caller for this tool
before now. No backend changes were needed.

**Frontend.** New `handleAskAboutRepo(repo: GithubWatchRepo)` in `PreviewsPanel.svelte`, same shape as
`handleAskAboutArticle`/`handleAskAboutHackerNewsStory`: sends `Summarize the release notes for the
latest release of {repo.label}` (deliberately containing both `"release notes"` and `"latest release"` so
the Planner gate fires regardless of future keyword-list edits) with
`context: { github_repo: repo.key, github_tag: repo.latest_release.tag_name }`. Pinning the tag as well as
the repo — not just "latest" — means the summary can't drift onto a newer release landing between the
click and the model's tool call. A per-row "Ask about this" button (reusing the existing
`preview-news-article-ask` styling) is shown for both `watched` and `pinned` entries alike whenever
`repo.latest_release` exists and `repo.error` doesn't — nothing to summarize otherwise.

**Pin badge removal.** At Michael's request, the 📌 emoji badge introduced in §7.26 was removed from the
GitHub block entirely (`.preview-pin-badge` CSS rule deleted along with it) — `source: 'watched' |
'pinned'` still exists on the data model, it just isn't surfaced visually anymore.

**Verification.** `svelte-check`: 0 errors/warnings.
