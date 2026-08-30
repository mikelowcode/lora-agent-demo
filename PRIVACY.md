# Privacy

Localist Framework is local-first by design. This document is a plain-language
account of what stays on your machine, what can leave it, and what a fresh
clone of this repository actually contains.

## What never leaves your machine

The following are excluded from this repository wholesale (see `.gitignore`)
and are never committed, regardless of how the code evolves:

| Path | Contents | Why it's excluded |
|---|---|---|
| `backend/wiki/` | Persona, user profile, all ingested/generated wiki pages, `MEMORY.md`, `index.md`, `logs.md` | Your persona configuration and everything the assistant has learned about you |
| `backend/raw/` | Documents you've fed the assistant for ingestion | Your own source material — could be anything |
| `backend/generated_files/` | Files the assistant has written to disk (chart exports, saved chat turns, Compose Mode documents) | Generated from your conversations |
| `*.db`, `*.sqlite`, `*.sqlite3` | Episodic memory and the RAG corpus (SQLite, WAL mode) | Every fact the assistant has stored about you, and every document it has indexed |
| `backend/.env` | API keys, runtime backend selection | Secrets |
| `sessions-log.md` | Development session history | Internal changelog, not runtime data, but kept local by the same convention |

None of this has ever been committed to this repository — `.gitignore` has
excluded all of it from the start of the project. This document exists to
make that boundary explicit, not to describe a change in behavior.

## What can leave your machine, and when

Localist only reaches the network when a tool call actually fires, and only
with the specific query, URL, or parameters that call needs — never your
wiki, memory, or raw documents wholesale:

- **`web_search`** (LangSearch or Brave) and **`fetch_url`** — the search
  query or the URL you asked it to fetch.
- **`news_search`** (NewsAPI, falling back to Brave-backed `web_search`) and
  the Daily News Brief Live Feed panel.
- **`github_search`** / **`github_read`** / **`github_release`** and the
  GitHub Watch Feed Live Feed panel — public GitHub data only, read-only,
  nothing is ever written back to GitHub.
- **`hacker_news_search`** (Algolia's public HN Search API).

**Inference** stays fully local when the active runtime backend is oMLX or a
local Ollama daemon. It leaves your machine only if you deliberately
configure Azure AI Foundry or Ollama Cloud as the active backend
(`LOCALIST_RUNTIME_BACKEND`) — an explicit, visible config choice, not a
default.

**Embeddings** always run locally when the embedding engine is enabled (MLX
EmbeddingGemma on Apple Silicon) or when a locally-served Ollama embedding
model is configured — never sent to a remote service, independent of which
chat backend is active.

## What a fresh clone actually contains

A clone of this repository ships with:

- An empty `backend/wiki/` (see `backend/wiki/.gitkeep` — pages are created
  as raw documents are ingested; the app runs fine with nothing there)
- An empty `backend/raw/`
- No database — created fresh on first run
- No `backend/.env` — you copy `backend/.env.example` yourself and fill in
  your own keys

Nothing from a previous user's runtime state — persona, memory, documents,
or credentials — ships with the repository or with any release of it.

See `examples/` for sample content you can use to try the assistant out
without supplying your own documents first.
