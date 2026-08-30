# Localist Framework

A local-first, agentic general assistant built primarily for macOS Apple Silicon. Persistent memory across sessions, live web search/fetch, indexed document search, and a deterministic priority-based router — no inference spent deciding how to route a query.

Inference-engine-agnostic: ships with oMLX, Ollama (including Ollama Cloud models), and Azure AI Foundry, swappable via one config variable or live at runtime with no restart. Embeddings always run locally regardless of chat backend — via MLX EmbeddingGemma (Apple Silicon only) or, with the Ollama backend, via any locally-served Ollama embedding model (e.g. `nomic-embed-text`), which also makes the framework usable on non-Apple-Silicon hardware.

See `PRIVACY.md` for what stays local, what can leave your machine and when, and what a fresh clone actually contains; `THIRD_PARTY_LICENSES.md` for the dependency/model-weight license audit.

---

## Architecture

SvelteKit frontend → FastAPI backend (port 8001). The backend's `ControllerAgent` runs each task through `Planner` (a priority-ordered rule engine, plus an explicit `/chart`/`/research` slash-command bypass ahead of it) and dispatches to `ConversationalAgent` (answers/tools) or `WikiAgent` (document ingestion). Tool calls go through `MCPToolDispatcher` to **localist-mcp** (port 8003), a standalone MCP server exposing `web_search`, `fetch_url`, `file_op`, `generate_chart`, `news_search`, `github_search`/`github_read`/`github_release`, and `hacker_news_search` tools. All inference runs through a `BaseRuntimeClient` implementation selected via `LOCALIST_RUNTIME_BACKEND` — swappable live, without a restart, via the Settings UI or `POST /settings/runtime-backend`. Episodic memory and embeddings live in SQLite (WAL mode), surviving restarts.

```
Localist UI ──HTTP──► FastAPI :8001
                          │
              Planner Priority 0 — /chart, /research
                    (explicit tool bypass)
                          │
                    ControllerAgent → Planner → RoutingPlan
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                    ▼
 ConversationalAgent                    WikiAgent
   │        │                         (ingestion/diff)
   │        └── MCPToolDispatcher ──► localist-mcp :8003
   │                                    ├─ web_search (research loop upgrade)
   │                                    ├─ fetch_url
   │                                    ├─ file_op
   │                                    ├─ generate_chart
   │                                    ├─ news_search (NewsAPI → Brave fallback)
   │                                    ├─ github_search / github_read / github_release
   │                                    └─ hacker_news_search (Algolia HN Search)
   └── MemoryManager (SQLite episodic + RAG)
```

`ocr_extract` is also served by localist-mcp but bypasses this whole path — called directly by
`POST /chat/files` at upload time (Apple Vision framework + PyMuPDF), never planner-routed and never
reached via `MCPToolDispatcher`'s normal chat-turn dispatch; see Attachments below.

---

## Prerequisites

- Python 3.13, Node.js
- One runtime backend: oMLX (chat model on :8000, macOS Apple Silicon only), [Ollama](https://ollama.com) (local or Ollama Cloud, :11434, any OS), or Azure AI Foundry
- MLX EmbeddingGemma (the local embedding model, opt-in) requires Apple Silicon and the `[mlx]` extra; on other platforms, set `LOCALIST_EMBEDDING_MODEL` to an Ollama-served embedding model instead (or fall back to keyword-only retrieval)
- OCR'd PDF chat uploads (see Attachments below) require macOS on Apple Silicon (Apple Vision framework + PyMuPDF); on other platforms those uploads are cleanly rejected with an explanatory error, everything else in the app is unaffected. Image uploads work everywhere: Apple Silicon uses Vision framework, other platforms fall back to a configured Ollama vision-capable chat model (`LOCALIST_OLLAMA_VISION_MODEL`)

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mlx,ocr,chart,dev]"   # Apple Silicon: full local stack (MLX embeddings, OCR, chart tool)
# pip install -e ".[dev]"               # any OS: base install — Ollama/Foundry, keyword-only retrieval
```

`[mlx]` and `[ocr]` are Apple-Silicon-only (platform markers make them a no-op elsewhere); `[chart]`
is cross-platform. See `THIRD_PARTY_LICENSES.md` — both `[mlx]` and `[ocr]` pull in copyleft
dependencies (GPLv3, AGPL-3.0 respectively).

The local embedding model is opt-in, not automatic — `LOCALIST_EMBEDDING_ENGINE_ENABLED` defaults to `false`, so a fresh install runs in keyword-only retrieval mode with zero download. `./start_localist.sh` asks once, interactively, on first run (only when `.env` doesn't already set the key, this is Apple Silicon, and the `[mlx]` extra looks installed) whether to enable it; answering yes downloads `mlx-community/embeddinggemma-300m-4bit` (~400MB, one-time, needs internet access) on the next backend startup. Set `LOCALIST_EMBEDDING_ENGINE_ENABLED=true` in `backend/.env` yourself to skip that prompt. Without it (or on non-Apple-Silicon hardware, where it can't run at all), episodic memory and RAG retrieval still work in keyword-only mode, or via an Ollama-served embedding model instead (see `LOCALIST_EMBEDDING_MODEL` below).

Localist's own code is MIT-licensed (see `LICENSE`), but this downloaded model is not — EmbeddingGemma is built on Google's Gemma and distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), not MIT. See `THIRD_PARTY_LICENSES.md` for the full dependency and model-weight license audit.

## Running

```bash
./start_localist.sh         # starts backend, localist-mcp, and frontend
./start_localist.sh --stop  # stops all three
```

`backend/wiki/` starts empty — see `examples/` for a sample document you can ingest right away to try the raw→wiki pipeline without supplying your own documents first.

## Configuration

Copy `backend/.env.example` to `backend/.env`. Only an API key for the active `web_search` provider — `LANGSEARCH_API_KEY` by default, or `BRAVE_API_KEY` if you switch providers — is required for full functionality; everything else has a working default.

| Variable | Default | Description |
|---|---|---|
| `LOCALIST_RUNTIME_BACKEND` | `foundry` | `foundry`, `omlx`, or `ollama` — also swappable live at runtime, see below |
| `LOCALIST_CHAT_MODEL` | *(none)* | Chat model ID override — wins over any per-backend pin below; required for `ollama` if no pin is set either (fails fast at startup if unset) |
| `LOCALIST_CHAT_MODEL_OLLAMA` / `_OMLX` / `_FOUNDRY` | *(none)* | Per-backend chat model pin, used when `LOCALIST_CHAT_MODEL` is unset — lets each backend remember its own model choice independently, including across a live runtime-backend switch |
| `LOCALIST_EMBEDDING_MODEL` | *(none)* | Embedding model ID for the active backend (`foundry`/`ollama`); if set and found, takes precedence over MLX EmbeddingGemma |
| `SEARCH_PROVIDER` | `langsearch` | `web_search` provider: `langsearch` or `brave` |
| `LANGSEARCH_API_KEY` | *(none)* | Required when `SEARCH_PROVIDER=langsearch`; without it, `web_search` fails and falls back to corpus |
| `BRAVE_API_KEY` | *(none)* | Required when `SEARCH_PROVIDER=brave`; without it, `web_search` fails and falls back to corpus |
| `NEWSAPI_API_KEY` | *(none)* | Optional — powers `news_search` (falls back to Brave-backed `web_search` without it) and the Daily News Brief Live Feed panel (individual sections degrade to "unavailable" without it, no fallback). Free Developer tier only (100 req/day); not licensed for production use |
| `GITHUB_TOKEN` | *(none)* | Optional for `github_search`/`github_read`/`github_release` (works unauthenticated at a lower rate limit); **required** for the GitHub Watch Feed Live Feed panel, for watched *and* pinned repos alike (`GET /user/subscriptions` and the per-repo releases lookup both need an authenticated identity). A classic PAT with no scopes selected is sufficient — public data only, read-only, nothing is ever written back to GitHub |
| `LOCALIST_MCP_URL` | `http://localhost:8003` | localist-mcp server URL |
| `LOCALIST_OLLAMA_VISION_MODEL` | *(none)* | Vision-capable Ollama chat model (e.g. `llava`, `qwen2.5vl`) for image OCR on non-Apple-Silicon platforms — see Attachments above; no effect on Apple Silicon, where Vision framework is always used |
| `LOCALIST_EPISODIC_WRITE_APPROVAL` | `false` | Gate implicit memory writes behind approve/reject |
| `LOCALIST_RESEARCH_LOOP_ENABLED` | `false` | Upgrade `web_search` to a bounded search/evaluate/fetch/reformulate loop for price/spec-lookup queries — a request can also always force this loop directly with a leading `/research`, regardless of this flag (see Tools below) |
| `LOCALIST_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING` |

See `backend/.env.example` for the full list (embedding engine, wiki/raw directories, MCP project root, etc.).

**Live runtime-backend switching** — the active backend doesn't require a restart to change: the Settings UI (or `POST /settings/runtime-backend` directly) health-checks the target backend, swaps it in, and persists the choice to `.env`, all while the server keeps running. Each backend remembers its own chat-model pin (`LOCALIST_CHAT_MODEL_OLLAMA`/`_OMLX`/`_FOUNDRY` above), so switching back to a backend you'd previously configured doesn't lose that choice.

---

## How It Works

**Routing** — `Planner` evaluates priority rules (P0–P6) in order, no inference required: an explicit `/chart` or `/research` slash command → force that tool directly (P0, ahead of everything else); raw file/ingest → `WikiAgent`; diff keywords, or a pinned wiki page (see Attachments below) with any diff phrasing anywhere in the instruction → targeted wiki diff; memory keywords → episode write; tool signals (URL, file, search, chart) → tool dispatch; factual gaps → web search; corpus match → RAG; fallback → direct answer.

**Slash commands** — `/chart <data>` and `/research <question>` bypass the normal detection paths and force that tool directly, even on input that wouldn't otherwise trigger it (a bare `/chart` with no data still reaches the tool and degrades gracefully; `/research` runs the full search/evaluate/fetch/reformulate loop even when `LOCALIST_RESEARCH_LOOP_ENABLED` is off). An explicit, user-invoked escape hatch — normal (non-slash) instructions are routed exactly as before.

**Tools** — served over MCP/SSE by localist-mcp: `web_search` (LangSearch or Brave), `fetch_url` (readability-lxml extraction), sandboxed `file_op` (`read_file`/`write_file`/`append_file`, versioned on collision), `generate_chart` (bar/line/pie charts rendered server-side and as an interactive Chart.js widget in the UI), `news_search` (NewsAPI.org, purpose-built for news-shaped queries — headlines, breaking news — that `web_search` has no freshness/source concept for; falls back to the existing Brave-backed `web_search` on a NewsAPI miss or error), a public-GitHub crawl trio — `github_search` (repo/code search), `github_read` (README or file/directory contents via the Contents API), and `github_release` (a specific or latest release's notes, defaulting to latest; resolves a bare project name to owner/repo by chaining an internal `github_search` call, so "fetch the oMLX 0.5.3 release notes" is keyword-routable without a pasted URL) — and `hacker_news_search` (Algolia's HN Search API, public/no key; an optional `url` param pins the result to one already-known story and, only in that case, fetches its real top comments via Algolia's separate item API — added after live testing caught the model fabricating plausible-sounding commentary from a bare comment count with no real text behind it). File writes for not-yet-generated content are deferred until after the answer, then confirmed inline. A bounded research loop (search → evaluate → fetch → reformulate, capped at 3 iterations, with a relevance-aware gate that checks the candidate text actually answers the question asked rather than just containing pricing-shaped content) can upgrade `web_search` for price/spec-lookup queries a single search snippet can't resolve — automatically above a semantic-intent threshold when `LOCALIST_RESEARCH_LOOP_ENABLED` is on (off by default), or always on-demand via `/research`.

**Live Feed** — a collapsible right-side panel (collapsed by default to a slim vertical tab) surfacing daily-update content outside the normal chat flow, with three blocks — Daily News Brief, Hacker News, and GitHub Watch Feed, in that order — each independently collapsible/expandable (state persisted across reloads) on top of the whole-panel collapse. The Daily News Brief block shows the latest cached World/National/Local + 3 user-chosen special-interest topics (home country, local-area keyword, and topic picker configured under Settings), with a "Daily News Brief Refresh" link that always fetches a fresh brief from NewsAPI. The Hacker News block shows the current top 10 stories (HN's public Firebase API, no key), each linking straight to the original article (or the HN discussion page itself for a self-post with no external link); a "Hacker News Refresh" link always fetches fresh. The GitHub Watch Feed block mirrors that same refresh-link pattern: it lists the repos you watch (GitHub's native Watch feature, via `GET /user/subscriptions`) plus any repos you've *pinned* by `owner/repo` slug under Settings — pinning tracks a repo's releases independently of GitHub's Watch relationship, so it doesn't subscribe you to that repo's PR/issue emails the way clicking Watch on GitHub does. Both kinds are merged into one list (a pinned slug matching an already-watched repo is deduped, not shown twice), each row linking straight to that repo's Releases page and a small 📌 badge marking the pinned ones, cached in SQLite until the next explicit refresh; a missing `GITHUB_TOKEN` surfaces as a single inline "not configured" message rather than an error. Both the News Brief and Hacker News blocks additionally have a per-story "Ask about this" button that sends just that one item into the current chat conversation, pinned to it specifically (`news_search`'s/`hacker_news_search`'s `url` param) rather than trusting a fresh query to find the same story again — GitHub Watch has no such button, correctly, since it has no chat-callable tool behind it.

**Attachments** — the "+" button in the chat UI uploads a local file into an ephemeral, session-scoped cache injected into every subsequent prompt. Text files, images (including HEIC), and PDFs are all supported: images and PDFs are OCR'd to plain text once at upload time by a local `ocr_extract` MCP tool — entirely independent of whichever chat runtime backend is active, so the attach button works the same whether oMLX, Ollama, or Foundry is running. On Apple Silicon, both use Apple's Vision framework (images) and PyMuPDF (PDFs — text-layer extraction first, falling back to per-page rasterize+OCR for scanned PDFs); on other platforms, image uploads fall back to a configured Ollama vision-capable chat model (`LOCALIST_OLLAMA_VISION_MODEL`, prompted for verbatim text transcription — same OCR-only contract, not image captioning), while PDF uploads still require Apple Silicon. An "Extracting text…" state shows briefly (real OCR latency, not mocked) before the file lands in the same cache as a text upload — same budget, same prompt slot, no separate image handling anywhere downstream. A second, paperclip-icon control pins an *existing wiki page* into the same cache instead, so asking Assistant to propose a diff against a specific page hands it the real, current file content rather than the model's own (possibly stale) memory of it. Either kind bypasses Planner routing and wiki indexing entirely for as long as it's attached, and clears on backend restart.

**Chat turn editing & Compose Mode** — any completed assistant reply can be saved straight to disk. Hovering a reply reveals a pencil toggle that swaps the rendered markdown for a full-width editable textarea in place, with a filename + `.md`/`.txt` picker beneath it; editing is export-only (never rewrites the actual chat turn) and Save goes through a direct `POST /files/generated` endpoint — the same sandboxed, collision-versioned write the `file_op` tool above uses, just triggered by a user click instead of a model call, so no inference round trip. **Compose Mode** extends this across several turns: a document icon in the composer opens a persistent, drag-resizable side panel (280–800px, mirrors the left sidebar's own resize handle), and each turn gains an "Add to document" button that appends its content to one growing draft — always onto the end of whatever's there, never recomputed from the included turns, so a hand-edit is never clobbered by a later addition. The assembled document is then edited and saved as a single artifact through the same save flow. Both features are entirely frontend-side; neither adds backend state beyond the one write endpoint they share.

**Memory** — two SQLite-backed stores. Episodic memory captures typed facts (preferences, decisions, corrections, etc.) with confidence scores and a `pending → active → superseded/retracted` lifecycle; retrieval by subject, recency, or similarity — real cosine similarity where an embedding exists, hand-rolled Okapi BM25 keyword ranking otherwise (no embed_fn, or a stale corpus); retraction via semantic match, kept to a strict floor even in keyword-only mode since a false-positive retraction silently destroys the wrong memory. Every write scanned for prompt-injection/credential content before storing. A human-readable snapshot regenerates at `wiki/MEMORY.md`. The corpus (RAG) stores embeddings of wiki pages and documents, each following an OKF (Open Knowledge Framework)-aligned front-matter convention (`type` required; `title`/`description`/`resource`/`tags`/`timestamp` optional); the same cosine/BM25 split scores corpus retrieval, with results below a fixed relevance floor excluded only when the score is a genuine cosine similarity — a BM25 match is included on rank alone, since its raw score isn't on a comparable scale. A per-directory `index.md` and a dated `logs.md` changelog are deterministically regenerated from on-disk state after every write, never model-authored. `MEMORY.md`/`index.md`/`logs.md` are structural/generated files, always excluded from RAG indexing, the graph, and the attachment picker. A user profile (`wiki/users/user_name.md`) is embedded line-by-line and injected only where relevant (cosine ≥ 0.45).

**Episode browsing** — a three-pane `/episodes` view (Filters / Episode list / Episode detail) for searching past chat turns as a semantic event stream rather than a linear log: keyword or embedding-similarity search over `chat_turns`, filters by conversation, date range, and tool-result presence, and a detail pane that renders tool-result turns inline — charts, wiki diffs (reviewable and directly applicable from the pane), and research-loop step chains — alongside a read-only "Related Memory" overlay of episodic-memory facts semantically related to that turn's own content (not just facts stamped with that exact turn's id, which is sparse and usually empty). Supersedes the earlier, narrower `/history` tab.

**Prompt layout** — fixed 7-slot structure (identity, persona, episodic+profile, RAG, tool results, working memory, instruction) optimized for KV-cache reuse. Local working-memory budget is now sized from the active model's real context window (oMLX's reported `max_model_len`) rather than a fixed turn count; on oMLX specifically, working-memory turns are sent as discrete messages — mirroring oMLX's own client — instead of flattened into one string, for genuine cross-turn KV-cache reuse.

---

## Project Structure

```
localist/
├── backend/
│   ├── pyproject.toml            # Dependency source of truth (base + mlx/ocr/chart/dev extras)
│   ├── src/localist/              # Installable package (pip install -e .), src/ layout
│   │   ├── main.py                  # FastAPI entry, port 8001
│   │   ├── controller_agent.py      # Task orchestration
│   │   ├── planner.py               # Routing rules (P0–P6)
│   │   ├── conversational_agent.py  # Prompt assembly, RAG, tools
│   │   ├── wiki_agent.py            # Document ingestion / diff
│   │   ├── prompt_builder.py        # 7-slot prompt assembler
│   │   ├── mcp_tool_dispatcher.py   # MCP/SSE client to localist-mcp; research loop
│   │   ├── memory_manager.py        # Episodic + RAG memory
│   │   ├── bm25.py                  # Okapi BM25 keyword scorer (no-embedding fallback)
│   │   ├── episodic_extractor.py    # Episode extraction
│   │   ├── content_safety.py        # Pre-write content scanner
│   │   ├── embedding_engine.py      # Local embedding engine
│   │   ├── runtime_factory.py       # Backend selection (foundry/omlx/ollama), live-swappable
│   │   ├── chart_tool_schema.py     # generate_chart argument extraction/validation
│   │   ├── news_brief.py            # Daily News Brief: NewsAPI calls, formatting (Live Feed panel)
│   │   ├── github_watch.py          # GitHub Watch Feed: watched-repo releases (Live Feed panel)
│   │   ├── hacker_news.py           # Hacker News: top-stories feed (Live Feed panel)
│   │   └── mcp_server/              # localist-mcp — port 8003 (web_search, fetch_url, file_op, generate_chart, news_search, github_search/github_read/github_release, hacker_news_search, ocr_extract)
│   ├── wiki/                    # Persona, user profile, indexed pages, MEMORY.md/index.md/logs.md (gitignored, empty on a fresh clone)
│   └── tests/                   # Unit + integration tests by phase
├── diagnostics/                 # Read-only live-verification scripts (not part of the test suite)
├── examples/                    # Sample document + walkthrough for the raw→wiki ingestion pipeline
└── localist-ui/                 # Frontend
```

## Development

```bash
cd backend
source .venv/bin/activate
python -m pytest tests/ -v
```

Run `pre-commit install` once per clone (after `pip install -e ".[dev]"`) — it wires in a hook that blocks accidentally committing runtime/personal data (`backend/wiki/`, `backend/raw/`, database files, `.env`) or stray credentials, on top of what `.gitignore` already excludes. See `PRIVACY.md` for the full boundary this enforces.

Tests are organized by phase (memory substrate, routing, controller dispatch, extraction, tool dispatcher, integration, content safety, REST API) and mock inference/SQLite — no live server or API keys required.

---

## Roadmap

**Done**
- ✅ Localist CLI launcher; MCP migration off legacy dispatcher/Fetcher
- ✅ Identity continuity + user profile injection
- ✅ Graph retrieval layer (SQLite schema v6)
- ✅ Ollama runtime backend (incl. Cloud), real `/api/embed`, cross-platform embeddings
- ✅ Wiki diff review/apply UI with pre-write snapshots (30-day undo)
- ✅ Episodic memory hardening — cosine retrieval, write-approval gate, semantic retraction
- ✅ KaTeX math rendering in chat output
- ✅ oMLX multi-turn prompt caching, sized from real context window
- ✅ Live-switchable runtime backend with per-backend chat-model pinning
- ✅ `generate_chart` tool with interactive Chart.js rendering
- ✅ Bounded research loop with relevance-aware answer gate
- ✅ `/chart` and `/research` slash commands
- ✅ Chat attachments can pin an existing wiki page, not just upload a file
- ✅ OKF-aligned wiki front matter, generated `index.md`/`logs.md`
- ✅ Episode Browsing UI (`/episodes`) — semantic chat-history search, tool-result rendering
- ✅ `news_search` tool (NewsAPI, falls back to Brave `web_search`)
- ✅ Daily News Brief Live Feed panel, always-fresh regeneration
- ✅ Related Memory panel now does real semantic similarity, not exact-turn matching
- ✅ Hand-rolled BM25 keyword scoring (`backend/bm25.py`) replacing Jaccard, for corpus + episodic recall when no embedding is available
- ✅ Generalize the bullet/diff-marker collision edge case
- ✅ GitHub integration: Watch Feed Live Feed panel (watched-repo releases, plus repos pinned by `owner/repo` slug independently of GitHub's Watch relationship) + `github_search`/`github_read`/`github_release` crawl tools, the latter keyword-routable without a pasted URL
- ✅ Hacker News integration: top-stories Live Feed panel block + `hacker_news_search` crawl tool (Algolia HN Search, URL-pinning + real comment grounding); per-block Live Feed collapse/expand
- ✅ Local OCR service — chat image (incl. HEIC) and PDF uploads extracted to text at upload time via a local `ocr_extract` MCP tool (Apple Vision framework + PyMuPDF), independent of the active chat runtime backend
- ✅ Cross-platform image OCR — a configured Ollama vision-capable chat model as a second `OCRProvider`, so image uploads work on non-Apple-Silicon platforms too (PDFs still require Apple Silicon)
- ✅ Per-turn "Save as" — edit any assistant reply in place and save it to disk as `.md`/`.txt`, no model round trip
- ✅ Compose Mode — accumulate multiple turns into one document in a persistent, drag-resizable side panel, then save the assembled draft as a single file

**Open**
- ⬜ macOS `.app` packaging via PyInstaller + Tauri
