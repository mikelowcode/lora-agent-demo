# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Localist Framework is a local-first, agentic general assistant running entirely on macOS Apple
Silicon: persistent cross-session memory, live web fetch/search, indexed document retrieval, and a
deterministic priority engine that routes every query before any inference is spent. It is
inference-engine-agnostic (oMLX, Ollama/Ollama Cloud, Azure AI Foundry, swappable via one config
variable); embeddings always run locally regardless of the active chat backend.

## Commands

### Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mlx,ocr,chart,dev]"   # Apple Silicon: full local stack
# pip install -e ".[dev]"               # any OS: base install, Ollama/Foundry only
```
`backend/pyproject.toml` is the dependency source of truth (`src/localist/` layout, installed
editable). Base install is cross-platform with no MLX/Vision/PyMuPDF; `[mlx]` (local embeddings)
and `[ocr]` (local OCR — Apple Vision + PyMuPDF) are both Apple-Silicon-only extras, gated by
`platform_machine == 'arm64'` markers matching each feature's own runtime gate; `[chart]`
(matplotlib, for `generate_chart`) is cross-platform. See `THIRD_PARTY_LICENSES.md` for why `[mlx]`
and `[ocr]` pull in copyleft dependencies (GPLv3, AGPL-3.0 respectively).

Copy `backend/.env.example` to `backend/.env`. Only `LANGSEARCH_API_KEY` is required for full
functionality (`web_search`); everything else has a working default.

### Running all services
```bash
./start_localist.sh          # starts backend (8001), localist-mcp (8003), frontend (5173)
./start_localist.sh --stop   # stops all three from a separate terminal
```
This is the standard way to bring the system up — it also handles log tailing
(`[backend]`/`[mcp]`/`[frontend]` prefixes) and clean shutdown on Ctrl+C. A runtime backend (oMLX on
:8000, or Ollama on :11434, or Foundry) must already be reachable; it is managed separately.

### Backend tests
```bash
cd backend
source .venv/bin/activate
python -m pytest tests/ -v                          # full suite
python -m pytest tests/test_planner_phase3.py -v    # one file
python -m pytest tests/test_planner_phase3.py -k test_whats_up_with_question_mark_filtered -v   # one test
```
All tests mock inference and SQLite — no live oMLX/Ollama server or API keys required to run them.
Verified clean as of 2026-08-30: 1523 passed, 0 failed.

### Frontend
```bash
cd localist-ui
npm run dev      # Vite dev server, port 5173
npm run build
npm run check    # svelte-kit sync + svelte-check (type checking; no separate lint/test script exists)
```

## Architecture

Request flow: **Localist UI** (SvelteKit) → HTTP → **FastAPI backend**, port 8001
(`backend/src/localist/main.py`) → `ControllerAgent` → `Planner` (deterministic, priority-ordered
rule engine, P1–P6, no inference used for routing) → `RoutingPlan` → either `ConversationalAgent`
(answers, RAG, tool calls) or `WikiAgent` (raw document → structured wiki page ingestion). The
backend is a real installable package (`localist`, `backend/pyproject.toml`, `src/` layout,
installed editable) — see `backend/src/localist/`, not a flat script directory.

Tool calls go through `MCPToolDispatcher`, which opens an MCP session (SSE transport) to
**localist-mcp**, a standalone FastAPI/FastMCP server on port 8003
(`backend/src/localist/mcp_server/`) exposing `web_search` (LangSearch), `fetch_url`
(readability-lxml extraction), and the `file_op` tools (`read_file`/`write_file`/`append_file`,
sandboxed under `LOCALIST_MCP_PROJECT_ROOT`). The legacy in-process `ToolDispatcher` and the
standalone Fetcher microservice (former port 8002) are both retired — their logic now lives on
localist-mcp.

All inference goes through a `BaseRuntimeClient`-conforming runtime selected at startup via
`LOCALIST_RUNTIME_BACKEND` and constructed by `runtime_factory.py`: `OMLXRuntimeClient`,
`OllamaRuntimeClient` (also serves Ollama Cloud models over the same local daemon), or
`FoundryRuntimeClient`. The active backend can also be changed live, without a restart, via
`POST /settings/runtime-backend` (health-checks the target before swapping, persists the choice to
`.env`; see `docs/architecture/16-runtime-backend-layer.md` §16.5) — so never assume the backend
active at startup is still the one active when a request is handled; always resolve it from
`_state.runtime` at request time, never capture it once and hold onto it. The Settings UI's Runtime
Backend control is wired to this endpoint (§7.10, §16.6) — a live switch there is a real,
confirm-gated action, not a display preference. Vector embeddings, when enabled, run locally via
`EmbeddingEngine` (`mlx-community/embeddinggemma-300m-4bit`, 768-dim) — but `EmbeddingEngine` is
opt-in (`LOCALIST_EMBEDDING_ENGINE_ENABLED`, default `false` as of the OSS packaging pass; see
`backend/pyproject.toml`'s `[mlx]` extra and `start_localist.sh`'s first-run prompt), not the
silent default it once was. Three-tier precedence (§16.4): a runtime-backend embed source, if
configured and active, wins over `EmbeddingEngine`; `EmbeddingEngine`, if enabled and available,
wins over the true zero-config default — keyword-only (BM25) retrieval. Independent of the chat
backend only in the keyword-only/EmbeddingEngine case, not universally.

`MemoryManager` is the SQLite-backed store (WAL mode) for two independent memory types:
**episodic memory** (typed, sparse facts — preferences, corrections, decisions — extracted
explicitly on trigger phrases or implicitly after every exchange, with supersession rather than
duplication) and the **RAG corpus** (embedded wiki/raw documents). A separate **user profile**
(`wiki/users/michael.md`) is loaded once per session, embedded line-by-line, and injected only
where individual lines score above a similarity threshold — never wholesale.

`PromptBuilder` assembles a fixed multi-slot prompt (identity → persona → session files →
episodic memory + user profile → RAG → tool results → working state → instruction) designed
around KV-cache reuse: the leading slots are static across turns so the runtime can reuse cached
prefix compute. Full slot-by-slot detail lives in the architecture spec, not here.

The **Graph Retrieval Layer** (`build_graph.py`, `wiki_doc.py`) maintains an offline concept
relationship graph over wiki pages (SQLite node/edge tables) that supplements RAG for
cross-document reasoning; it's built by an explicit trigger, not live on every request.

### Two kinds of scripts outside `backend/tests/`
- `diagnostics/` — read-only, live-verification scripts run against real running services
  (not mocked, not part of the pytest suite, never collected by `pytest tests/`). Used to confirm
  or falsify a hypothesis about actual runtime behavior before a fix is built around it. A
  diagnostic reporting an unexpected or "failing" result *is* the finding, not a bug in the
  script — do not edit a diagnostic to make its output look better, and do not run or include
  these when asked to "run the tests."
- `sessions-log.md` — a chronological, per-architecture-section changelog of investigation and
  fix sessions, split out precisely so `LOCALIST-Architecture.md` doesn't have to carry history.
  Grep it for past incident context; don't read it in full.

## Architecture spec

The canonical architecture spec lives in `LOCALIST-Architecture.md` (index) +
`docs/architecture/NN-*.md` (one file per numbered section). Retired sections
are under `docs/architecture/archive/`.

- Read `LOCALIST-Architecture.md` first — it's a short index (summary, status,
  last-updated per section). Only open the specific `docs/architecture/NN-*.md`
  file you need; do not reconstruct or read the whole spec.
- Edits to substance go in the individual section file, never back into the
  index.
- When you change a section file, update its row in `LOCALIST-Architecture.md`:
  bump the last-updated date and revisit the status (authoritative / draft /
  retired) and one-paragraph summary if the change is material.
