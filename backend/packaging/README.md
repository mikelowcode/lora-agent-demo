# Packaging — PyInstaller (Phase B of native `.app` packaging)

Freezes the base-only backend (`localist.main`) and `localist-mcp`
(`localist.mcp_server.main`) into standalone executables. No MLX
embeddings, no Vision-framework OCR, no PyMuPDF — see the OSS release
build order's step 7a for why (license footprint; not lost, just outside
this bundle — those stay available via `pip install -e ".[mlx,ocr]"`).

## Build from a clean, extras-free venv — not `backend/.venv`

This matters more than it looks. PyInstaller's static analysis bundles
anything syntactically importable anywhere in the source tree,
**regardless of whether that code path actually runs** — a deferred
`import mlx_embeddings` inside a function body is still found and
bundled, exactly like a module-level one. Building from `backend/.venv`
(which has every extra installed for day-to-day development) pulls in
MLX, PyMuPDF, OpenCV, transformers, pyarrow — over 500MB of dependencies
that were never supposed to ship, discovered the hard way during Phase B.

```bash
cd backend
python3 -m venv .venv-packaging      # separate from .venv, gitignored
source .venv-packaging/bin/activate
pip install -e ".[dev,packaging]"    # base + pytest/pre-commit + pyinstaller — no mlx/ocr/chart
```

## Build

```bash
cd backend/packaging
source ../.venv-packaging/bin/activate
pyinstaller localist-backend.spec
pyinstaller localist-mcp.spec
```

Output: `dist/localist-backend/` and `dist/localist-mcp/` — onedir builds
(not onefile: near-instant startup, and a missing-module error is far
easier to diagnose than in a single self-extracting executable; Phase C
can revisit onefile for final distribution if needed).

## Run

```bash
./dist/localist-backend/localist-backend   # port 8001
./dist/localist-mcp/localist-mcp           # port 8003
```

Both respect the same env vars as the source-tree app (`LOCALIST_RUNTIME_BACKEND`,
`LOCALIST_CHAT_MODEL`, etc. — see `backend/.env.example`), read from the
process environment (a frozen build has no `backend/.env` to load
automatically from the source tree; see `paths.py` below for where it
looks instead).

For local testing without writing to your real
`~/Library/Application Support/Localist`, set `LOCALIST_BACKEND_ROOT` to
a scratch directory first.

## Where a frozen build's data lives

`localist/paths.py` centralizes this (previously 13 independent
`Path(__file__)`-based computations scattered across the codebase — see
its module docstring for the full story). Two distinct roots:

- **`get_backend_root()`** — user-writable data: `wiki/`, `raw/`,
  `generated_files/`, the SQLite db, logs. Frozen default:
  `~/Library/Application Support/Localist` (created on first run).
  Override: `LOCALIST_BACKEND_ROOT`.
- **`get_resource_root()`** — read-only, ship-with-the-app resources:
  `SCHEMA.md` and `templates/`, which `WikiAgent` requires to exist
  (validated, not best-effort). Frozen default: PyInstaller's own bundle
  extraction directory (`sys._MEIPASS`), populated via each `.spec`
  file's `datas`. Override: `LOCALIST_RESOURCE_ROOT`.

Both fall back to the source tree's `backend/` directory when not frozen
— unchanged from before this centralization.

## What's verified so far (Phase B)

Both frozen executables start cleanly, respond on their real ports, and
a real raw-document ingestion (`POST /task` with `context.raw_path`) run
end-to-end against the frozen backend — Ollama as the runtime backend,
producing a real OKF-formatted wiki page on disk. Not yet
covered: Tauri sidecar spawning (Phase C), first-run config UX (Phase D),
code signing / Gatekeeper (Phase E).
