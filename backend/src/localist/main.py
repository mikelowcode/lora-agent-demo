"""
LORA — FastAPI Backend
======================
The HTTP boundary between the Svelte UI and the agent core.

Layer placement
---------------
  Svelte UI  →  FastAPI (this module)  →  ControllerAgent  →  Sub-agents / Runtime
                                       →  MemoryManager    →  SQLite (local file)

Architectural contract
----------------------
- This is the ONLY module that imports FastAPI, Pydantic, or anything HTTP-related.
- All agent logic lives in controller_agent.py, wiki_agent.py, conversational_agent.py.
- All model inference flows through the active RuntimeClient.
- This module constructs the runtime, MemoryManager, agents, and controller
  once at startup and holds them as app-level state.  No singleton is
  constructed per-request.
- Long-running synchronous calls (controller.handle_task, runtime.infer) are
  dispatched to a thread pool via asyncio.to_thread so they never block the
  event loop.

Endpoints
---------
  POST /task
      Submit a task.  Accepts a TaskRequest body, calls handle_task(),
      returns a TaskResponse.  The response always includes task_id and
      status so the caller can correlate.

  POST /task/stream
      Submit a task whose synthesis answer is streamed back token-by-token
      as Server-Sent Events (SSE).  Planning and sub-agent dispatch still
      run synchronously in the background; only the final synthesis call
      streams.  The event stream closes with a [DONE] sentinel.

  GET /health
      Calls runtime.health_check() and returns its dict.  Returns HTTP 200
      even when the runtime is unreachable so the UI can display a degraded
      state rather than a hard error page.  A separate "healthy" boolean in
      the body signals true service health to automated monitors.

  GET /agents
      Returns the list of registered agent names.  Useful for the UI to
      show which capabilities are active without parsing log files.

  GET /memory/stats
      Returns MemoryManager statistics: document counts, DB size, embedding
      coverage, cache state.  Always HTTP 200 — degraded values are shown
      when the MemoryManager is not initialised.

  POST /memory/reembed
      Manually re-embed the wiki/raw corpus with the active embedding model
      and clear MemoryManager's corpus_stale flag (docs/architecture/
      16-runtime-backend-layer.md §16.4). Idempotent; safe to call whether
      or not the corpus is currently stale.

  GET /files/raw
      List all .md/.txt and OCR-eligible (image/PDF, see mcp_server/ocr.py's
      OCR_MIME_BY_EXTENSION) files in the raw/ directory.

  GET /files/wiki
      List all .md files in the wiki/ directory.

  GET /files/content?path=<absolute_path>
      Return the plain-text content of a file.  Only paths inside raw_dir
      or wiki_dir are permitted — anything else returns HTTP 403.  Not
      meant for OCR-eligible raw files (image/PDF) — reads as UTF-8 text
      and fails with HTTP 500 on binary content; use /files/download for
      those instead.

  POST /files/upload
      Accept a multipart .md/.txt or OCR-eligible (image/PDF) file upload
      and save it to raw/ unchanged (no OCR at upload time — WikiAgent
      extracts text from OCR-eligible raw files lazily at ingest time).
      Immediately indexes .md/.txt files in MemoryManager; OCR-eligible
      files are skipped until ingested into a wiki page.

Running locally
---------------
  uvicorn main:app --reload --host 127.0.0.1 --port 8001

Environment / configuration
----------------------------
All tuneable values are in the ``Settings`` class (pydantic-settings).  They
can be overridden via environment variables or a .env file:

  LOCALIST_RUNTIME_BACKEND             Runtime backend: "foundry" | "omlx" (default "foundry")
  LOCALIST_CHAT_MODEL                  Chat model ID override (wins over any per-backend pin below)
  LOCALIST_CHAT_MODEL_OMLX             Per-backend chat model pin for "omlx"
  LOCALIST_CHAT_MODEL_OLLAMA           Per-backend chat model pin for "ollama"
  LOCALIST_CHAT_MODEL_FOUNDRY          Per-backend chat model pin for "foundry"
  LOCALIST_FOUNDRY_URL                 Override auto-resolved Foundry base URL (foundry only)
  LOCALIST_OMLX_URL                    oMLX server base URL (omlx only, default http://localhost:8000)
  LOCALIST_OLLAMA_URL                  Ollama server base URL (ollama only, default http://localhost:11434)
  LOCALIST_LOG_LEVEL                   Root log level (default INFO)
  LOCALIST_WIKI_DIR                    Absolute path to the wiki directory
  LOCALIST_RAW_DIR                     Absolute path to the raw files directory
  LOCALIST_GENERATED_DIR               Absolute path to the generated files directory
  LOCALIST_SCHEMA_PATH                 Absolute path to SCHEMA.md
  LOCALIST_TEMPLATES_DIR               Absolute path to the templates directory
  LOCALIST_AUTO_APPLY                  Whether WikiAgent writes to disk immediately (bool)
  LOCALIST_STREAM_TIMEOUT              Streaming timeout in seconds (float)
  LOCALIST_REQUEST_TIMEOUT             Non-streaming timeout in seconds (float)
  LOCALIST_MEMORY_DB                   Absolute path to the SQLite memory DB file.
                                       Defaults to <project_root>/localist_memory.db
  LOCALIST_EMBEDDING_MODEL             Runtime-backend embedding model ID (foundry/ollama
                                       only; omlx does not yet wire this through). Empty
                                       string (default) = not configured, falls back to
                                       EmbeddingEngine.
  LOCALIST_EMBEDDING_ENGINE_ENABLED    Load the standalone MLX-LM EmbeddingEngine at
                                       startup (bool, default True).  Set False to run
                                       in keyword-only mode without loading the model.
                                       Ignored when LOCALIST_EMBEDDING_MODEL is set and
                                       found by the active runtime backend — the runtime
                                       backend's own embed() is used instead.  Also a
                                       no-op on non-Apple-Silicon platforms regardless of
                                       its value, since EmbeddingEngine's mlx_lm
                                       dependency only runs on Apple Silicon.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
# backend/src/localist/main.py -> backend/.env (3 parents up)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import asyncio
import datetime
import json
import logging
import mimetypes
import os
import platform
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project imports (agent core + runtime)
# ---------------------------------------------------------------------------

from .base_runtime_client import BaseRuntimeClient
from .build_graph import build_graph
from .context_profile import check_local_ram_headroom, profile_for
from .controller_agent import ControllerAgent, TaskStatus, _MEMORY_MD_PATH
from .conversational_agent import ConversationalAgent
from .embedding_engine import EmbeddingEngine
from . import github_watch
from . import hacker_news
from .mcp_server.ocr import get_upload_root as _ocr_upload_root
from .mcp_server.ocr import OCR_MIME_BY_EXTENSION as _OCR_MIME_BY_EXTENSION
from .mcp_server.file_ops import write_file as _write_generated_file
from .mcp_server.file_ops import set_project_root as _set_generated_file_root
from .mcp_tool_dispatcher import MCPToolDispatcher
from .memory_manager import MemoryManager, EpisodicMemoryWriter, EpisodicMemoryReader
from . import news_brief
from .runtime_factory import available_backends, create_runtime
from . import session_files
from . import wiki_maintenance_log
from .warmup import run_cache_warmup as _run_cache_warmup
from .wiki_agent import WikiAgent, read_text_file, sweep_expired_snapshots
from .wiki_doc import META_WIKI_FILENAMES

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    All configuration for the Localist Framework backend.
    Override any field via environment variable or .env file.
    """
    model_config = SettingsConfigDict(env_prefix="LOCALIST_", env_file=".env", extra="ignore")

    # Runtime backend selection
    runtime_backend: str = "foundry"

    # Model ID — chat only; embeddings are handled by EmbeddingEngine (MLX-LM)
    # unless embedding_model below is set and found by the active runtime
    # backend, in which case the runtime backend's own embed() is used instead.
    chat_model:       str | None = None

    # Per-backend chat-model pins (LOCALIST_CHAT_MODEL_OMLX / _OLLAMA / _FOUNDRY).
    # Used by _resolve_chat_model() when chat_model above is unset; lets a
    # live runtime-backend switch remember which model to use per backend
    # instead of carrying one backend's model id into another's client.
    chat_model_omlx:    str | None = None
    chat_model_ollama:  str | None = None
    chat_model_foundry: str | None = None

    # Runtime-backend embedding model ID (foundry/ollama only; empty string =
    # not configured, falls back to EmbeddingEngine).
    embedding_model:  str = ""

    # Foundry network (foundry backend only)
    foundry_url:      str | None = None

    # oMLX network (omlx backend only)
    omlx_url:         str = "http://localhost:8000"

    # Ollama network (ollama backend only)
    ollama_url:       str = "http://localhost:11434"

    # Shared network timeouts
    stream_timeout:   float = 60.0
    request_timeout:  float = 30.0

    # Paths — resolved at startup; defaults are relative to project root
    wiki_dir:        str | None = None
    raw_dir:         str | None = None
    generated_dir:   str | None = None
    schema_path:     str | None = None
    templates_dir:   str | None = None

    # MemoryManager
    memory_db:                str | None = None   # None → <project_root>/localist_memory.db

    # EmbeddingEngine — standalone MLX-LM embedding, backend-agnostic.
    # Set False to skip model load and run MemoryManager in keyword-only mode.
    embedding_engine_enabled: bool = True

    # Agent behaviour
    auto_apply:      bool = False

    # Episodic memory — write-approval gate for model_extracted (implicit)
    # episodes. When True, those writes are staged as "pending" instead of
    # going live immediately, and must be approved via
    # POST /memory/episodes/{id}/approve (or rejected via .../reject).
    # Explicit (user-said "remember that...") episodes are never gated.
    episodic_write_approval: bool = False

    # Logging
    log_level:       str = "INFO"


# ---------------------------------------------------------------------------
# App-level state (constructed once at startup)
# ---------------------------------------------------------------------------

class AppState:
    """Holds singletons that live for the lifetime of the process."""

    def __init__(self) -> None:
        self.runtime:           BaseRuntimeClient | None = None
        self.controller:        ControllerAgent   | None = None
        self.wiki_agent:        WikiAgent         | None = None
        self.memory_manager:    MemoryManager     | None = None
        self.embedding_engine:  EmbeddingEngine   | None = None
        self.settings:          Settings          | None = None
        # Name of the embedding model actually backing embed_fn, resolved by
        # _derive_active_embedding_model_name() alongside _configure_embedding_
        # source() — see planner.py's _TUNED_EMBEDDING_MODEL guard. None means
        # keyword-only mode (no embedding source at all).
        self.active_embedding_model_name: str | None = None
        # Resolved at startup by lifespan()
        self.wiki_dir:          Path | None = None
        self.raw_dir:           Path | None = None
        self.generated_dir:     Path | None = None
        self.schema_path:       Path | None = None
        self.templates_dir:     Path | None = None


_state = AppState()


# ---------------------------------------------------------------------------
# Embedding source selection (pulled out of lifespan() for testability)
# ---------------------------------------------------------------------------

def _configure_embedding_source(
    settings: Settings,
    runtime:  BaseRuntimeClient,
    health:   dict,
) -> tuple[Any, EmbeddingEngine | None]:
    """
    Decide and construct which embedding source lifespan() should use, in
    three-tier precedence order:

      1. The active runtime backend's own embed() — used when
         settings.embedding_model is set AND health["embed_model_found"]
         (from the health check already run in lifespan()) is truthy.
         Platform-agnostic.
      2. EmbeddingEngine (standalone MLX-LM) — attempted only when enabled
         AND this platform is Apple Silicon, since mlx_lm cannot run
         elsewhere.
      3. Neither — MemoryManager falls back to keyword-only retrieval.

    Tiers 1 and 2 are mutually exclusive: loading both would hold two
    embedding models in memory for no benefit.

    Returns
    -------
    tuple[Callable | None, EmbeddingEngine | None]
        (embed_fn, embedding_engine). embedding_engine is the constructed
        EmbeddingEngine instance whenever tier 2 was attempted (whether or
        not it ended up available), so the caller can stash it on app
        state; otherwise None.

    This is a standalone function (not inlined in lifespan()) purely so the
    branch selection can be unit-tested without running the full startup
    sequence (real runtime construction, directory indexing, graph build,
    etc.) — lifespan() itself is not exercised by the test suite.
    """
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")

    if settings.embedding_model and health.get("embed_model_found"):
        logger.info(
            "Runtime-backend embeddings ready — backend=%s model=%s. "
            "EmbeddingEngine will not be loaded.",
            settings.runtime_backend.upper(), settings.embedding_model,
        )
        return runtime.embed, None

    if settings.embedding_engine_enabled and is_apple_silicon:
        embedding_engine = EmbeddingEngine()
        if embedding_engine.available:
            logger.info("EmbeddingEngine ready — embeddings enabled.")
            return embedding_engine.embed, embedding_engine
        logger.warning(
            "EmbeddingEngine failed to load — MemoryManager will run "
            "in keyword-only mode.  Install mlx-lm and retry."
        )
        return None, embedding_engine

    if settings.embedding_engine_enabled and not is_apple_silicon:
        logger.info(
            "EmbeddingEngine skipped — mlx_lm requires Apple Silicon, this platform "
            "is %s/%s. MemoryManager will run in keyword-only mode.",
            platform.system(), platform.machine(),
        )
        return None, None

    logger.info(
        "EmbeddingEngine disabled (LOCALIST_EMBEDDING_ENGINE_ENABLED=false) — "
        "MemoryManager will run in keyword-only mode."
    )
    return None, None


def _derive_active_embedding_model_name(
    settings:         Settings,
    embed_fn:         Any,
    embedding_engine: EmbeddingEngine | None,
) -> str | None:
    """
    Name the embedding model actually backing `embed_fn`, mirroring
    _configure_embedding_source()'s three-tier precedence:

      1. Runtime-backend embed (embedding_engine is None, embed_fn is not)
         -> settings.embedding_model.
      2. EmbeddingEngine (embedding_engine is not None) -> its model_path,
         but only if it loaded successfully (embedding_engine.available);
         a construction attempt that failed to load names no model.
      3. Keyword-only (embed_fn is None, embedding_engine is None) -> None.

    Consumed by Planner's _TUNED_EMBEDDING_MODEL guard (docs/architecture/
    16-runtime-backend-layer.md §16.4) so a mismatched embedding model
    disables semantic gating instead of silently producing thresholds with
    no validated meaning.
    """
    if embedding_engine is not None:
        return embedding_engine.model_path if embedding_engine.available else None
    if embed_fn is not None:
        return settings.embedding_model
    return None


# ---------------------------------------------------------------------------
# Controller construction (extracted so lifespan() and a live runtime-backend
# switch share one code path — see _build_controller() below).
# ---------------------------------------------------------------------------

def _build_controller(
    settings:              Settings,
    runtime:               BaseRuntimeClient,
    memory_manager:        MemoryManager,
    embed_fn:              Any,
    project_root:          Path,
    templates_dir:         Path,
    embedding_model_name:  str | None = None,
) -> tuple[WikiAgent, ConversationalAgent, ControllerAgent]:
    """
    Construct WikiAgent, ConversationalAgent, and ControllerAgent for a given
    runtime, and warm its persona cache. Used both at startup (lifespan())
    and by a live runtime-backend switch, since ControllerAgent captures its
    runtime by value (via its own Synthesizer/_RulePlanner) — there is no way
    to rebind an existing ControllerAgent to a new runtime, only to build a
    fresh one.
    """
    wiki_agent = WikiAgent(
        runtime        = runtime,
        project_root   = project_root,
        memory_manager = memory_manager,
    )
    conversational_agent = ConversationalAgent(
        runtime        = runtime,
        memory_manager = memory_manager,
        project_root   = project_root,
    )
    controller = ControllerAgent(
        runtime                 = runtime,
        agents                  = [wiki_agent, conversational_agent],
        memory_manager          = memory_manager,
        embed_fn                = embed_fn,
        embedding_model_name    = embedding_model_name,
        episodic_write_approval = settings.episodic_write_approval,
    )
    _run_cache_warmup(controller, runtime, templates_dir)
    return wiki_agent, conversational_agent, controller


# ---------------------------------------------------------------------------
# Live runtime-backend switching support
# ---------------------------------------------------------------------------

# backend/src/localist/main.py -> backend/ (3 parents up)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Serializes any sequence that reads-then-swaps _state.runtime/wiki_agent/
# controller, so two concurrent switch/pin requests can't interleave their
# health-check → rebuild → swap steps. An asyncio.Lock, not threading.Lock —
# held across `await asyncio.to_thread(...)` below, and a plain threading.Lock
# there would block the whole event loop on a second requester's acquire
# instead of just queuing it behind the first (docs/architecture/
# 16-runtime-backend-layer.md §16.5).
_runtime_switch_lock = asyncio.Lock()

_CHAT_MODEL_SETTINGS_FIELD: dict[str, str] = {
    "omlx":    "chat_model_omlx",
    "ollama":  "chat_model_ollama",
    "foundry": "chat_model_foundry",
}

_CHAT_MODEL_ENV_KEY: dict[str, str] = {
    "omlx":    "LOCALIST_CHAT_MODEL_OMLX",
    "ollama":  "LOCALIST_CHAT_MODEL_OLLAMA",
    "foundry": "LOCALIST_CHAT_MODEL_FOUNDRY",
}


def _resolve_chat_model(settings: Settings, backend: str) -> str | None:
    """
    Resolve the chat model to use for `backend`, one source of truth shared
    by lifespan() and the runtime-backend endpoints.

    Precedence: settings.chat_model (global override) > the per-backend pin
    for `backend` > None (falls through to runtime_factory.py's own
    hardcoded per-backend default).
    """
    if settings.chat_model:
        return settings.chat_model
    field = _CHAT_MODEL_SETTINGS_FIELD.get(backend.strip().lower())
    return getattr(settings, field) if field else None


def _write_env_var(project_root: Path, key: str, value: str) -> None:
    """
    Set `key=value` in project_root/.env, preserving every other line
    (comments, blank lines, unrelated keys) byte-for-byte. Replaces the
    existing `key=...` line if present, otherwise appends a new one.
    Written atomically via a temp file + os.replace() so a crash or
    concurrent read never observes a half-written .env.
    """
    env_path = project_root / ".env"
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines(keepends=True)

    new_line = f"{key}={value}\n"
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("#") and stripped.split("=", 1)[0].strip() == key:
            lines[i] = new_line
            break
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line)

    fd, tmp_path = tempfile.mkstemp(dir=str(project_root), prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.writelines(lines)
        os.replace(tmp_path, env_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Construct the runtime, MemoryManager, agents, and controller once at
    startup.  Runs in the main thread before the first request is accepted.
    """
    settings = Settings()
    _state.settings = settings

    # Configure logging
    logging.basicConfig(
        level   = getattr(logging, settings.log_level.upper(), logging.INFO),
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    # The MCP SDK's SSE transport logs every raw JSON-RPC message (full tool
    # schemas, full results) at DEBUG — useless noise even under
    # LOCALIST_LOG_LEVEL=DEBUG. Keep warnings/errors from it, drop the rest.
    logging.getLogger("mcp.client.sse").setLevel(logging.INFO)
    # httpcore logs per-socket connection-pool trace events
    # (connect_tcp.started, send_request_headers.complete, ...) at DEBUG —
    # noise, not signal. "httpcore" covers its http11/http2/connection/proxy/
    # socks children since none of them set their own level. httpx itself
    # only logs one INFO line per request, which stays — that's signal.
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logger.info("LORA backend starting up.")

    project_root = _PROJECT_ROOT

    # -- Runtime -------------------------------------------------------------

    runtime = create_runtime(
        backend         = settings.runtime_backend,
        chat_model      = _resolve_chat_model(settings, settings.runtime_backend),
        embedding_model = settings.embedding_model,
        foundry_url     = settings.foundry_url,
        omlx_url        = settings.omlx_url,
        ollama_url      = settings.ollama_url,
        request_timeout = settings.request_timeout,
        stream_timeout  = settings.stream_timeout,
    )
    _state.runtime = runtime

    health = runtime.health_check()
    if health["reachable"]:
        logger.info(
            "%s runtime reachable at %s — chat_found=%s  embed_found=%s",
            settings.runtime_backend.upper(),
            health["base_url"],
            health["chat_model_found"],
            health["embed_model_found"],
        )
    else:
        logger.warning(
            "%s runtime NOT reachable at startup (%s). "
            "Requests will fail until the service is running.",
            settings.runtime_backend.upper(),
            health.get("base_url"),
        )

    if getattr(runtime, "is_local", True):
        local_profile = profile_for(runtime)
        logger.info(
            "LOCAL_PROFILE working_memory_tokens=%d (max_model_len=%s)",
            local_profile.working_memory_tokens,
            getattr(runtime, "max_model_len", "unknown"),
        )
        ram = check_local_ram_headroom()
        if ram["warning"]:
            logger.warning(
                "LOCAL_PROFILE RAM headroom check: %s — this machine's current "
                "load matches the swap-under-load condition measured 2026-07-19 "
                "(see diagnostics/reports/local_working_memory_ram_findings.md); "
                "expect swap activity under the working-memory ceiling.",
                ram["message"],
            )
        else:
            logger.info("LOCAL_PROFILE RAM headroom check: %s", ram["message"])

    # -- Resolve path defaults -----------------------------------------------

    wiki_dir      = Path(settings.wiki_dir)      if settings.wiki_dir      else project_root / "wiki"
    raw_dir       = Path(settings.raw_dir)       if settings.raw_dir       else project_root / "raw"
    generated_dir = Path(settings.generated_dir) if settings.generated_dir else project_root / "generated_files"
    schema_path   = Path(settings.schema_path)   if settings.schema_path   else project_root / "SCHEMA.md"
    templates_dir = Path(settings.templates_dir) if settings.templates_dir else project_root / "templates"
    memory_db     = Path(settings.memory_db)     if settings.memory_db     else project_root / "localist_memory.db"

    # -- Embedding source selection -------------------------------------------
    # See _configure_embedding_source() for the three-tier precedence rules
    # (runtime-backend embed / EmbeddingEngine-if-Apple-Silicon / keyword-only).

    embed_fn, embedding_engine = _configure_embedding_source(settings, runtime, health)
    if embedding_engine is not None:
        _state.embedding_engine = embedding_engine
    _state.active_embedding_model_name = _derive_active_embedding_model_name(
        settings, embed_fn, embedding_engine,
    )

    memory_manager = MemoryManager(
        db_path               = memory_db,
        embed_fn              = embed_fn,
        embedding_model_name  = _state.active_embedding_model_name,
    )
    _state.memory_manager = memory_manager

    # Seed the document index from disk on startup.  index_directory() is
    # idempotent — unchanged files are skipped via content-hash comparison.
    # This ensures the index is always current even after pages were written
    # while the server was down.
    if wiki_dir.exists():
        n_wiki = memory_manager.index_directory(
            wiki_dir, doc_type="wiki", embed=bool(embed_fn), exclude=META_WIKI_FILENAMES,
        )
        logger.info("MemoryManager: indexed %d wiki pages from %s.", n_wiki, wiki_dir)
    else:
        logger.warning("MemoryManager: wiki_dir does not exist yet (%s) — skipping seed.", wiki_dir)

    if raw_dir.exists():
        n_raw = memory_manager.index_directory(raw_dir, doc_type="raw", embed=bool(embed_fn))
        logger.info("MemoryManager: indexed %d raw files from %s.", n_raw, raw_dir)
    else:
        logger.info("MemoryManager: raw_dir does not exist yet (%s) — skipping seed.", raw_dir)

    stats = memory_manager.stats()
    logger.info(
        "MemoryManager ready — wiki=%d  raw=%d  db_size=%.1f KB  embeddings=%.0f%%",
        stats["wiki_docs"], stats["raw_docs"],
        stats["db_size_kb"], stats["embeddings_pct"],
    )

    if wiki_dir.exists():
        try:
            graph_summary = build_graph(wiki_dir, memory_manager)
            logger.info(
                "Graph rebuilt at startup — nodes=%d edges=%d resolved=%d unresolved=%d",
                graph_summary["nodes"], graph_summary["edges"],
                graph_summary["resolved"], graph_summary["unresolved"],
            )
        except Exception as exc:
            logger.warning("Graph build failed at startup (non-fatal): %s", exc)
    else:
        logger.info("Graph build skipped — wiki_dir does not exist yet (%s).", wiki_dir)

    if wiki_dir.exists():
        try:
            reconcile_summary = memory_manager.reconcile_wiki(wiki_dir)
            logger.info(
                "Wiki reconciled at startup — reindexed=%d orphans_removed=%d%s",
                reconcile_summary["reindexed"],
                reconcile_summary["orphans_removed"],
                f" ({', '.join(reconcile_summary['orphan_names'])})"
                    if reconcile_summary["orphan_names"] else "",
            )
        except Exception as exc:
            logger.warning("Wiki reconciliation failed at startup (non-fatal): %s", exc)
    else:
        logger.info("Wiki reconciliation skipped — wiki_dir does not exist yet (%s).", wiki_dir)

    if wiki_dir.exists():
        try:
            pruned = sweep_expired_snapshots(wiki_dir)
            for p in pruned:
                wiki_maintenance_log.log_snapshot_pruned(p.name, str(p))
            logger.info("Wiki snapshot TTL sweep at startup — pruned=%d", len(pruned))
        except Exception as exc:
            logger.warning("Wiki snapshot TTL sweep failed at startup (non-fatal): %s", exc)
    else:
        logger.info("Wiki snapshot TTL sweep skipped — wiki_dir does not exist yet (%s).", wiki_dir)

    try:
        sweep_result = await asyncio.to_thread(memory_manager.sweep_expired_memory)
        logger.info(
            "Retention sweep at startup — chat_turns_deleted=%d episodes_retracted=%d",
            sweep_result["chat_turns_deleted"], sweep_result["episodes_retracted"],
        )
        if sweep_result["episodes_retracted"] > 0:
            writer = EpisodicMemoryWriter(
                db_path=memory_manager._db_path, memory_md_path=_MEMORY_MD_PATH,
            )
            await asyncio.to_thread(writer.regenerate_memory_md)
    except Exception as exc:
        logger.warning("Retention sweep failed at startup (non-fatal): %s", exc)

    # -- Store resolved paths in state so endpoints can inject them ----------

    _state.wiki_dir      = wiki_dir
    _state.raw_dir       = raw_dir
    _state.generated_dir = generated_dir
    _state.schema_path   = schema_path
    _state.templates_dir = templates_dir

    # file_ops resolves its own sandbox root independently (from
    # LOCALIST_MCP_PROJECT_ROOT, for the standalone localist-mcp process) —
    # sync its in-process copy to generated_dir so POST /files/generated
    # below always writes into the exact directory GET /files/generated,
    # /files/download, and DELETE /files already read from, even if the two
    # env vars are configured to point elsewhere.
    _set_generated_file_root(generated_dir)

    # -- Agents + Controller --------------------------------------------------

    logger.info(
        "Episodic write-approval gate: %s",
        "ON" if settings.episodic_write_approval else "OFF",
    )

    wiki_agent, conversational_agent, controller = _build_controller(
        settings, runtime, memory_manager, embed_fn, project_root, templates_dir,
        _state.active_embedding_model_name,
    )
    _state.wiki_agent = wiki_agent
    _state.controller = controller

    logger.info(
        "ControllerAgent ready — agents: %s",
        [wiki_agent.name, conversational_agent.name],
    )

    yield  # — application runs —

    logger.info("LORA backend shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "LORA — Local Reasoning Agent",
    description = "Multi-agent research system — WikiAgent + corpus-aware ConversationalAgent (RAG).",
    version     = "0.4.0",
    lifespan    = lifespan,
)

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://127.0.0.1:5173",
                         "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic)
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    """
    Payload accepted by POST /task and POST /task/stream.

    ``instruction`` is the only required field.  ``context`` is passed
    through verbatim to the controller and then on to each agent:

        {
            "query":      "What do we know about attention mechanisms?",
            "raw_path":   "/abs/path/to/paper.md",
            "auto_apply": false
        }
    """
    task_id:            str              = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction:        str              = Field(..., min_length=1)
    context:            dict[str, Any]   = Field(default_factory=dict)
    metadata:           dict[str, Any]   = Field(default_factory=dict)
    conversation_id:    str              = Field(..., min_length=1)
    conversation_title: str | None       = Field(default=None)


class TaskResponse(BaseModel):
    """Serialised ControllerResult returned by POST /task."""
    task_id:  str
    status:   str
    answer:   str
    sources:  list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any]       = Field(default_factory=dict)
    error:    str | None           = None


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    healthy:           bool
    reachable:         bool
    backend:           str
    base_url:          str
    models:            list[str]  = Field(default_factory=list)
    chat_model_found:  bool       = False
    embed_model_found: bool       = False
    error:             str | None = None


class AgentsResponse(BaseModel):
    """Response body for GET /agents."""
    agents: list[str]


class RuntimeBackendSwitchRequest(BaseModel):
    """
    Payload accepted by POST /settings/runtime-backend.

    chat_model is optional — if provided, it also becomes that backend's
    persisted pin (see _resolve_chat_model()), not a one-shot override.
    """
    backend:    str
    chat_model: str | None = None


class RuntimeBackendSwitchResponse(BaseModel):
    """Response body for POST /settings/runtime-backend."""
    backend:          str
    chat_model:       str | None = None
    persisted:        bool
    reachable:        bool
    base_url:         str
    models:           list[str]  = Field(default_factory=list)
    chat_model_found: bool       = False
    error:            str | None = None
    warning:          str | None = None


class RuntimeBackendModelsResponse(BaseModel):
    """Response body for GET /settings/runtime-backend/{backend}/models."""
    reachable:        bool
    base_url:         str
    models:           list[str]  = Field(default_factory=list)
    chat_model_found: bool       = False
    error:            str | None = None


class ChatModelPinRequest(BaseModel):
    """Payload accepted by POST /settings/runtime-backend/{backend}/chat-model."""
    chat_model: str = Field(..., min_length=1)


class ChatModelPinResponse(BaseModel):
    """Response body for POST /settings/runtime-backend/{backend}/chat-model."""
    backend:      str
    chat_model:   str
    persisted:    bool
    applied_live: bool


class MemoryStatsResponse(BaseModel):
    """Response body for GET /memory/stats."""
    db_path:          str
    db_size_kb:       float
    wiki_docs:        int
    raw_docs:         int
    conv_log_rows:    int
    cache_valid:      int
    cache_invalid:    int
    embeddings_pct:   float
    corpus_stale:     bool   # True when the wiki/raw corpus needs POST /memory/reembed
    chat_turns_stale: bool   # True when chat_turns needs POST /memory/reembed-chat-turns
    available:        bool   # False when MemoryManager is not initialised


class ReembedCorpusResponse(BaseModel):
    """
    Response body for POST /memory/reembed — the manual, explicitly-
    triggered wiki/raw corpus re-embed (docs/architecture/
    16-runtime-backend-layer.md §16.4's confirmed embedding-provenance
    follow-up). Episodes get no equivalent endpoint — a genuine embedding-
    model mismatch there is auto-corrected at startup, not left pending.
    """
    reembedded: int
    total:      int
    model:      str | None


class ReembedChatTurnsResponse(BaseModel):
    """
    Response body for POST /memory/reembed-chat-turns — the chat_turns
    counterpart to ReembedCorpusResponse/POST /memory/reembed. chat_turns
    can grow arbitrarily large under the "forever" eviction preset, so a
    detected embedding-model mismatch never re-embeds it automatically.
    """
    reembedded: int
    total:      int
    model:      str | None


class EpisodeItem(BaseModel):
    """A single episode record returned by GET /memory/episodes."""
    id:              int
    episode_type:    str
    subject:         str
    content:         str
    confidence:      float
    source:          str
    task_id:         str | None = None
    project_context: str | None = None
    status:          str
    created_at:      float
    last_accessed:   float | None = None

class EpisodesResponse(BaseModel):
    """Response body for GET /memory/episodes."""
    episodes: list[EpisodeItem]
    total:    int
    offset:   int
    limit:    int


class RelatedEpisodesResponse(BaseModel):
    """Response body for GET /memory/episodes/related."""
    episodes: list[EpisodeItem]


class EpisodeApprovalResponse(BaseModel):
    """
    Response body for POST /memory/episodes/{id}/approve and .../reject.

    updated=False (rather than a 404/409) means the id doesn't exist or
    wasn't in "pending" status (already resolved, or never staged) —
    kept idempotent and simple for a single-user local app.
    """
    episode_id: int
    status:     Literal["active", "retracted"]
    updated:    bool


class FileEntry(BaseModel):
    """Metadata for a single file in raw/, wiki/, or generated_files/."""
    name:     str    # stem without extension
    filename: str    # filename with extension, e.g. "my-doc.md"
    path:     str    # absolute path — passed as context.raw_path on ingest
    size:     int    # bytes
    modified: str    # ISO-8601 UTC timestamp
    type:     Literal["raw", "wiki", "generated"]


class FilesResponse(BaseModel):
    """Response body for GET /files/raw and GET /files/wiki."""
    files: list[FileEntry]


class FileContentResponse(BaseModel):
    """Response body for GET /files/content."""
    path:    str
    content: str


class FileDeleteResponse(BaseModel):
    """Response body for DELETE /files."""
    path:    str
    deleted: bool


class SaveGeneratedFileRequest(BaseModel):
    """Request body for POST /files/generated — e.g. a chat turn's "Save as"."""
    filename:  str
    extension: Literal["txt", "md"]
    content:   str


class NewsPreferencesResponse(BaseModel):
    """Response body for GET/PUT /news/preferences."""
    home_country: str
    local_query:  str | None = None
    topics:       list[str]      = Field(default_factory=list)
    topic_pool:   dict[str, str] = Field(default_factory=dict)  # key -> display label


class NewsPreferencesRequest(BaseModel):
    """Payload accepted by PUT /news/preferences."""
    home_country: str = Field(min_length=2, max_length=2)
    local_query:  str | None = None
    topics:       list[str]


class NewsBriefSection(BaseModel):
    """One section (World/National/Local, or one topic) of a Daily News Brief."""
    key:      str
    label:    str
    articles: list[dict[str, Any]] = Field(default_factory=list)
    error:    str | None = None


class NewsBriefPreviewResponse(BaseModel):
    """
    Response body for GET /news/brief/preview.

    Read-only — never triggers a NewsAPI call. `available=False` means no
    brief has been generated yet today (either never generated, or the
    cached one is from a previous day).
    """
    available:  bool
    brief_date: str | None = None
    sections:   list[NewsBriefSection] = Field(default_factory=list)


class NewsBriefOpenResponse(BaseModel):
    """Response body for POST /news/brief/open — always a fresh generation."""
    success: bool = True


class GithubWatchRepo(BaseModel):
    """
    One entry in the GitHub Watch Feed — either a repo the user has
    actually clicked "Watch" on in GitHub (source="watched") or a repo
    they've pinned for release-only tracking, independent of GitHub's
    Watch/subscriptions relationship (source="pinned").
    """
    key:             str
    label:           str
    repo_url:        str
    latest_release:  dict[str, Any] | None = None
    error:           str | None = None
    source:          Literal["watched", "pinned"] = "watched"


class GithubWatchPreviewResponse(BaseModel):
    """
    Response body for GET /github/watch/preview.

    Read-only — never calls GitHub. `available=False` means the watch feed
    has never been generated yet (unlike the news brief, there's no
    same-day staleness check — watched-repo releases aren't day-keyed).
    """
    available:    bool
    generated_at: float | None = None
    repos:        list[GithubWatchRepo] = Field(default_factory=list)


class GithubWatchOpenResponse(BaseModel):
    """Response body for POST /github/watch/refresh — always a fresh fetch."""
    success: bool = True


class PinnedGithubReposResponse(BaseModel):
    """Response body for GET/PUT /github/watch/pinned-repos."""
    repos: list[str] = Field(default_factory=list)


class PinnedGithubReposRequest(BaseModel):
    """
    Payload accepted by PUT /github/watch/pinned-repos — full-list
    replace, same semantics as PUT /news/preferences's topics field.
    """
    repos: list[str]


class HackerNewsStory(BaseModel):
    """One story's entry in the Hacker News Live Feed."""
    key:    str
    title:  str
    url:    str
    hn_url: str
    score:  int | None = None
    by:     str | None = None
    error:  str | None = None


class HackerNewsPreviewResponse(BaseModel):
    """
    Response body for GET /hacker-news/top/preview.

    Read-only — never calls Hacker News. `available=False` means the top-
    stories feed has never been generated yet (no same-day staleness check,
    same as GithubWatchPreviewResponse — top stories aren't day-keyed).
    """
    available:    bool
    generated_at: float | None = None
    stories:      list[HackerNewsStory] = Field(default_factory=list)


class HackerNewsOpenResponse(BaseModel):
    """Response body for POST /hacker-news/top/refresh — always a fresh fetch."""
    success: bool = True


class RetentionSettingsResponse(BaseModel):
    """Response body for GET/PUT /settings/retention."""
    eviction_preset: str | None = None


class RetentionSettingsRequest(BaseModel):
    """Payload accepted by PUT /settings/retention."""
    eviction_preset: Literal["7d", "30d", "90d", "forever"]


class AssistantNameResponse(BaseModel):
    """Response body for GET/PUT /settings/assistant-name."""
    assistant_name: str


class AssistantNameRequest(BaseModel):
    """Payload accepted by PUT /settings/assistant-name."""
    assistant_name: str


class ChatTurnItem(BaseModel):
    """A single chat_turns record returned by GET /chat/history."""
    id:                 int
    task_id:            str
    role:               str
    content:            str
    sources:            list[dict[str, Any]] = Field(default_factory=list)
    status_message:     str | None = None
    metadata:           dict[str, Any]       = Field(default_factory=dict)
    conversation_id:    str
    conversation_title: str | None = None
    created_at:         float
    score:              float | None = None  # only set in mode="semantic" results


class ChatHistoryResponse(BaseModel):
    """Response body for GET /chat/history."""
    turns:  list[ChatTurnItem]
    total:  int
    offset: int
    limit:  int


class ConversationSummary(BaseModel):
    """One row per distinct conversation, for the sidebar list."""
    conversation_id:    str
    conversation_title: str | None = None
    last_created_at:    float
    first_created_at:   float


class ConversationListResponse(BaseModel):
    """Response body for GET /chat/history/conversations."""
    conversations: list[ConversationSummary]


class ConversationDeleteResponse(BaseModel):
    """Response body for DELETE /chat/history/conversations/{conversation_id}."""
    conversation_id: str
    turns_deleted:   int


class ApplyDiffRequest(BaseModel):
    """
    Payload accepted by POST /wiki/apply-diff.

    task_id identifies the chat turn whose persisted metadata.pending_diffs
    entry should be marked "applied" on success (see
    MemoryManager.mark_diff_applied()) — the round-tripped page_name/diff
    are what actually gets written; task_id only updates the durable
    review-then-apply UI state.
    """
    task_id:   str = Field(..., min_length=1)
    page_name: str = Field(..., min_length=1)
    diff:      str = Field(..., min_length=1)


class ApplyDiffResponse(BaseModel):
    """Response body for POST /wiki/apply-diff (success only — failures raise HTTPException)."""
    success:   bool = True
    page_name: str


class PinWikiPageRequest(BaseModel):
    """Payload accepted by POST /chat/pin-wiki-page."""
    stem: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _require_controller() -> ControllerAgent:
    if _state.controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialised.")
    return _state.controller


def _require_runtime() -> BaseRuntimeClient:
    if _state.runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not initialised.")
    return _state.runtime


def _require_wiki_agent() -> WikiAgent:
    if _state.wiki_agent is None:
        raise HTTPException(status_code=503, detail="WikiAgent not initialised.")
    return _state.wiki_agent


def _require_memory_manager() -> MemoryManager:
    if _state.memory_manager is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")
    return _state.memory_manager


def _enrich_context(context: dict[str, Any]) -> dict[str, Any]:
    """
    Merge app-level path defaults into the caller-supplied context dict.
    Caller-supplied values always win — this only fills gaps.
    """
    defaults: dict[str, Any] = {}

    if _state.wiki_dir:
        defaults["wiki_dir"] = str(_state.wiki_dir)
    if _state.raw_dir:
        defaults["raw_dir"] = str(_state.raw_dir)
    if _state.schema_path:
        defaults["schema_path"] = str(_state.schema_path)
    if _state.templates_dir:
        defaults["templates_dir"] = str(_state.templates_dir)
    if _state.settings:
        defaults["auto_apply"] = _state.settings.auto_apply

    return {**defaults, **context}


def _persist_chat_turn(
    role:               str,
    content:            str,
    task_id:            str,
    conversation_id:    str,
    sources:            list[dict[str, Any]] | None = None,
    status_message:     str | None = None,
    metadata:           dict[str, Any] | None = None,
    conversation_title: str | None = None,
) -> None:
    """
    Best-effort write of one chat turn to the chat_turns table.

    No-ops silently when no memory_manager is configured. Never raises —
    a chat_turns write failure must not break the actual task response,
    since the source of truth for an in-flight answer is the SSE stream /
    TaskResponse, not this table.

    Parameters
    ----------
    conversation_id :
        Groups turns by conversation. Required.
    conversation_title :
        Optional human-readable title for the conversation.
    """
    if _state.memory_manager is None:
        return
    try:
        _state.memory_manager.add_chat_turn(
            task_id            = task_id,
            role               = role,
            content            = content,
            conversation_id    = conversation_id,
            sources            = sources,
            status_message     = status_message,
            metadata           = metadata,
            conversation_title = conversation_title,
        )
    except Exception:
        logger.warning("Failed to persist chat turn (role=%s, task_id=%s).", role, task_id, exc_info=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/task",
    response_model       = TaskResponse,
    summary              = "Submit a task (blocking)",
    response_description = "The completed task result.",
)
async def post_task(request: TaskRequest) -> TaskResponse:
    """
    Submit an instruction to the LORA multi-agent system.

    The call blocks until the full pipeline (plan → dispatch → synthesize)
    completes.  Use POST /task/stream to receive tokens incrementally.
    """
    controller = _require_controller()

    task_dict = {
        "task_id":     request.task_id,
        "instruction": request.instruction,
        "context":     {**_enrich_context(request.context), "conversation_id": request.conversation_id},
        "metadata":    request.metadata,
    }

    _persist_chat_turn(
        "user", request.instruction, request.task_id, request.conversation_id,
        conversation_title = request.conversation_title,
    )

    try:
        result: dict[str, Any] = await asyncio.to_thread(
            controller.handle_task, task_dict
        )
    except Exception as exc:
        logger.exception("Unhandled error in POST /task for task %s.", request.task_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _persist_chat_turn(
        "assistant", result.get("answer", ""), request.task_id, request.conversation_id,
        sources  = result.get("sources"),
        metadata = result.get("metadata"),
    )

    return TaskResponse(**result)


@app.post(
    "/task/stream",
    summary = "Submit a task and stream the synthesis answer (SSE)",
)
async def post_task_stream(request: TaskRequest) -> StreamingResponse:
    """
    Submit a task and receive the synthesis answer as a Server-Sent Events stream.

    Event format:

        data: {"type": "status",        "message": "Planning..."}
        data: {"type": "token",         "token": "The"}
        data: {"type": "sources",       "sources": [...]}
        data: {"type": "done",          "task_id": "...", "status": "complete"}
        data: {"type": "task_complete", "task_id": "..."}
        data: [DONE]

    'done' fires as soon as the visible answer is ready (may precede
    background memory writes by up to ~20-30s). 'task_complete' fires only
    after the full pipeline — including post-answer episodic/working-state
    hooks — has finished, and always precedes [DONE]. Clients that submit
    a new task while a prior one's background writes are still running can
    cause overlapping calls against a single-instance local model backend,
    so the client should gate the next submission on 'task_complete', not
    'done'.
    """
    controller = _require_controller()
    runtime    = _require_runtime()

    task_dict = {
        "task_id":     request.task_id,
        "instruction": request.instruction,
        "context":     {**_enrich_context(request.context), "conversation_id": request.conversation_id},
        "metadata":    request.metadata,
    }

    return StreamingResponse(
        _stream_task(
            controller, runtime, task_dict, request.task_id,
            request.conversation_id, request.conversation_title,
        ),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Runtime service health check",
)
async def get_health() -> HealthResponse:
    """
    Check whether the active runtime backend is reachable and the configured
    models are available.  Always returns HTTP 200.
    """
    runtime = _require_runtime()
    raw: dict[str, Any] = await asyncio.to_thread(runtime.health_check)

    # Mirrors the embed_fn precedence lifespan() establishes at startup: the
    # runtime backend's own embed() wins when an embedding_model is configured
    # and the health check confirms it's actually present, since in that case
    # lifespan() never loads EmbeddingEngine at all (_state.embedding_engine
    # stays None). Only fall back to EmbeddingEngine's own availability when
    # the runtime-backend path isn't the one actually wired to MemoryManager.
    settings = _state.settings
    if settings is not None and settings.embedding_model and raw.get("embed_model_found"):
        embed_available = True
    else:
        embedding_engine = _state.embedding_engine
        embed_available  = embedding_engine is not None and embedding_engine.available

    return HealthResponse(
        healthy           = bool(raw.get("reachable") and raw.get("chat_model_found")),
        reachable         = bool(raw.get("reachable", False)),
        backend           = settings.runtime_backend if settings is not None else "",
        base_url          = str(raw.get("base_url", "")),
        models            = raw.get("models", []),
        chat_model_found  = bool(raw.get("chat_model_found", False)),
        embed_model_found = embed_available,
        error             = raw.get("error"),
    )


@app.get(
    "/agents",
    response_model = AgentsResponse,
    summary        = "List registered agents",
)
async def get_agents() -> AgentsResponse:
    """Return the names of all agents currently registered with the controller."""
    controller = _require_controller()
    return AgentsResponse(agents=list(controller._agents.keys()))


def _validate_backend_name(backend: str) -> str:
    """Normalize and reject an unknown backend name before anything is touched."""
    normalized = backend.strip().lower()
    if normalized not in available_backends():
        raise HTTPException(
            status_code = 422,
            detail      = (
                f"Unknown runtime backend: {backend!r}. "
                f"Supported backends: {', '.join(available_backends())}."
            ),
        )
    return normalized


def _require_settings() -> Settings:
    if _state.settings is None:
        raise HTTPException(status_code=503, detail="Settings not initialised.")
    return _state.settings


def _create_and_check_backend(settings: Settings, backend: str) -> tuple[BaseRuntimeClient, dict]:
    """Build a runtime client for `backend` and health-check it. Blocking — run via asyncio.to_thread."""
    chat_model = _resolve_chat_model(settings, backend)
    candidate_runtime = create_runtime(
        backend         = backend,
        chat_model      = chat_model,
        embedding_model = settings.embedding_model,
        foundry_url     = settings.foundry_url,
        omlx_url        = settings.omlx_url,
        ollama_url      = settings.ollama_url,
        request_timeout = settings.request_timeout,
        stream_timeout  = settings.stream_timeout,
    )
    return candidate_runtime, candidate_runtime.health_check()


@app.post(
    "/settings/runtime-backend",
    response_model = RuntimeBackendSwitchResponse,
    summary        = "Live-switch the active runtime backend",
)
async def switch_runtime_backend(request: RuntimeBackendSwitchRequest) -> RuntimeBackendSwitchResponse:
    """
    Live-switch the active runtime backend. The target backend must answer
    health_check() successfully before anything is mutated — an unreachable
    target leaves the current backend running untouched.

    An optional chat_model on the request pins that backend's chat model as
    a side effect (persisted Settings field + .env), independent of whether
    the switch itself succeeds.
    """
    backend  = _validate_backend_name(request.backend)
    settings = _require_settings()

    if request.chat_model:
        setattr(settings, _CHAT_MODEL_SETTINGS_FIELD[backend], request.chat_model)
        _write_env_var(_PROJECT_ROOT, _CHAT_MODEL_ENV_KEY[backend], request.chat_model)

    chat_model = _resolve_chat_model(settings, backend)

    async with _runtime_switch_lock:
        candidate_runtime, health = await asyncio.to_thread(
            _create_and_check_backend, settings, backend,
        )

        if not health.get("reachable"):
            raise HTTPException(
                status_code = 502,
                detail      = (
                    f"Runtime backend {backend!r} is not reachable at "
                    f"{health.get('base_url')!r} — current backend left untouched."
                ),
            )

        memory_manager = _require_memory_manager()
        # Deliberately NOT re-derived from candidate_runtime — a chat-backend switch changes
        # inference only, never the embedding source. Re-coupling this would risk silently
        # dropping a working embedder when switching to a backend that doesn't wire embeddings
        # (oMLX today, per §16.4's open gap), even though nothing about embeddings was supposed
        # to change. See docs/architecture/16-runtime-backend-layer.md §16.5.
        embed_fn = memory_manager.embed_fn
        # Read fresh from _state, not captured once — the embedding source
        # itself isn't re-derived here either (see comment above), but this
        # follows the same "always resolve from _state at request time" rule
        # as _state.runtime, in case that ever changes.
        embedding_model_name = _state.active_embedding_model_name

        wiki_agent, _conversational_agent, controller = await asyncio.to_thread(
            _build_controller,
            settings, candidate_runtime, memory_manager, embed_fn,
            _PROJECT_ROOT, _state.templates_dir, embedding_model_name,
        )

        _state.runtime    = candidate_runtime
        _state.wiki_agent = wiki_agent
        _state.controller = controller
        settings.runtime_backend = backend

        persisted: bool = True
        warning:   str | None = None
        try:
            _write_env_var(_PROJECT_ROOT, "LOCALIST_RUNTIME_BACKEND", backend)
        except OSError as exc:
            persisted = False
            warning = (
                f"Runtime switched to {backend!r} in-process, but writing .env failed "
                f"({exc}) — it will revert to the previous backend on next restart."
            )

    return RuntimeBackendSwitchResponse(
        backend          = backend,
        chat_model       = chat_model,
        persisted        = persisted,
        reachable        = bool(health.get("reachable", False)),
        base_url         = str(health.get("base_url", "")),
        models           = health.get("models", []),
        chat_model_found = bool(health.get("chat_model_found", False)),
        error            = health.get("error"),
        warning          = warning,
    )


@app.get(
    "/settings/runtime-backend/{backend}/models",
    response_model = RuntimeBackendModelsResponse,
    summary        = "List models available on a runtime backend without switching to it",
)
async def get_runtime_backend_models(backend: str) -> RuntimeBackendModelsResponse:
    """
    Build a throwaway client for `backend`, health-check it, and return its
    reported models. Never touches _state — not even a read — so this is
    safe to call for the currently-inactive backend(s) as a "what's
    available there" lookup.
    """
    normalized = _validate_backend_name(backend)
    settings   = _require_settings()

    _candidate_runtime, health = await asyncio.to_thread(
        _create_and_check_backend, settings, normalized,
    )

    return RuntimeBackendModelsResponse(
        reachable        = bool(health.get("reachable", False)),
        base_url         = str(health.get("base_url", "")),
        models           = health.get("models", []),
        chat_model_found = bool(health.get("chat_model_found", False)),
        error            = health.get("error"),
    )


@app.post(
    "/settings/runtime-backend/{backend}/chat-model",
    response_model = ChatModelPinResponse,
    summary        = "Pin a chat model for a specific runtime backend",
)
async def set_runtime_backend_chat_model(
    backend: str, request: ChatModelPinRequest,
) -> ChatModelPinResponse:
    """
    Persist a chat-model pin for `backend`. Always writes the Settings field
    and .env, regardless of which backend is currently active. If `backend`
    is the active backend, also live-rebuilds the controller against it so
    the pin takes effect immediately rather than only on the next switch.
    """
    normalized = _validate_backend_name(backend)
    settings   = _require_settings()

    setattr(settings, _CHAT_MODEL_SETTINGS_FIELD[normalized], request.chat_model)
    _write_env_var(_PROJECT_ROOT, _CHAT_MODEL_ENV_KEY[normalized], request.chat_model)

    applied_live = False
    if normalized == settings.runtime_backend.strip().lower():
        async with _runtime_switch_lock:
            candidate_runtime, health = await asyncio.to_thread(
                _create_and_check_backend, settings, normalized,
            )

            if not health.get("reachable"):
                raise HTTPException(
                    status_code = 502,
                    detail      = (
                        f"Chat model pin saved for {normalized!r}, but it is not "
                        f"reachable at {health.get('base_url')!r} — live rebuild skipped."
                    ),
                )

            memory_manager = _require_memory_manager()
            # Deliberately NOT re-derived from candidate_runtime — a chat-backend switch changes
            # inference only, never the embedding source. Re-coupling this would risk silently
            # dropping a working embedder when switching to a backend that doesn't wire embeddings
            # (oMLX today, per §16.4's open gap), even though nothing about embeddings was supposed
            # to change. See docs/architecture/16-runtime-backend-layer.md §16.5.
            embed_fn = memory_manager.embed_fn
            # Read fresh from _state, not captured once — the embedding source
            # itself isn't re-derived here either (see comment above), but this
            # follows the same "always resolve from _state at request time" rule
            # as _state.runtime, in case that ever changes.
            embedding_model_name = _state.active_embedding_model_name

            wiki_agent, _conversational_agent, controller = await asyncio.to_thread(
                _build_controller,
                settings, candidate_runtime, memory_manager, embed_fn,
                _PROJECT_ROOT, _state.templates_dir, embedding_model_name,
            )

            _state.runtime    = candidate_runtime
            _state.wiki_agent = wiki_agent
            _state.controller = controller
            applied_live = True

    return ChatModelPinResponse(
        backend      = normalized,
        chat_model   = request.chat_model,
        persisted    = True,
        applied_live = applied_live,
    )


@app.get(
    "/memory/stats",
    response_model = MemoryStatsResponse,
    summary        = "MemoryManager statistics",
)
async def get_memory_stats() -> MemoryStatsResponse:
    """
    Return MemoryManager statistics.

    Always HTTP 200.  When the MemoryManager is not initialised (e.g. startup
    failure), all numeric fields are 0 and ``available`` is False.
    """
    mm = _state.memory_manager
    if mm is None:
        return MemoryStatsResponse(
            db_path        = "",
            db_size_kb     = 0.0,
            wiki_docs      = 0,
            raw_docs       = 0,
            conv_log_rows  = 0,
            cache_valid    = 0,
            cache_invalid  = 0,
            embeddings_pct = 0.0,
            corpus_stale   = False,
            chat_turns_stale = False,
            available      = False,
        )

    raw: dict[str, Any] = await asyncio.to_thread(mm.stats)
    return MemoryStatsResponse(
        db_path        = raw["db_path"],
        db_size_kb     = raw["db_size_kb"],
        wiki_docs      = raw["wiki_docs"],
        raw_docs       = raw["raw_docs"],
        conv_log_rows  = raw["conv_log_rows"],
        cache_valid    = raw["cache_valid"],
        cache_invalid  = raw["cache_invalid"],
        embeddings_pct = raw["embeddings_pct"],
        corpus_stale   = mm._corpus_stale,
        chat_turns_stale = mm._chat_turns_stale,
        available      = True,
    )


@app.post(
    "/memory/reembed",
    response_model = ReembedCorpusResponse,
    summary        = "Manually re-embed the wiki/raw corpus with the active embedding model",
)
async def reembed_corpus() -> ReembedCorpusResponse:
    """
    Explicit, manually-triggered corpus re-embed — the counterpart to
    episodes' automatic startup re-embed (docs/architecture/
    16-runtime-backend-layer.md §16.4). Wiki/raw corpora can be arbitrarily
    large, so unlike episodes a detected embedding-model mismatch never
    triggers this automatically; call it after switching embedding models
    to clear MemoryManager's corpus_stale flag and restore embedding-based
    re-ranking in query_corpus().

    Idempotent — safe to call whether or not the corpus is currently
    flagged stale (a "just refresh it" operation).
    """
    mm = _state.memory_manager
    if mm is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")
    if mm.embed_fn is None:
        raise HTTPException(
            status_code = 409,
            detail      = "No embedding source configured — nothing to re-embed with.",
        )

    result = await asyncio.to_thread(mm.reembed_corpus)
    return ReembedCorpusResponse(**result)


@app.post(
    "/memory/reembed-chat-turns",
    response_model = ReembedChatTurnsResponse,
    summary        = "Manually re-embed chat_turns with the active embedding model",
)
async def reembed_chat_turns() -> ReembedChatTurnsResponse:
    """
    Explicit, manually-triggered chat_turns re-embed — the chat_turns
    counterpart to POST /memory/reembed. chat_turns can grow arbitrarily
    large (the "forever" eviction preset), so unlike episodes a detected
    embedding-model mismatch never triggers this automatically; call it
    after switching embedding models to clear MemoryManager's
    chat_turns_stale flag and restore embedding-based search in
    GET /chat/history?mode=semantic.

    Idempotent — safe to call whether or not chat_turns is currently
    flagged stale (a "just refresh it" operation).
    """
    mm = _state.memory_manager
    if mm is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")
    if mm.embed_fn is None:
        raise HTTPException(
            status_code = 409,
            detail      = "No embedding source configured — nothing to re-embed with.",
        )

    result = await asyncio.to_thread(mm.reembed_chat_turns)
    return ReembedChatTurnsResponse(**result)


@app.get(
    "/memory/episodes",
    response_model = EpisodesResponse,
    summary        = "List stored episodes",
)
async def get_memory_episodes(
    status:          str      = "active",
    project_context: str | None = None,
    episode_type:    str | None = None,
    task_id:         str | None = None,
    limit:           int      = 50,
    offset:          int      = 0,
) -> EpisodesResponse:
    """
    Return a paginated list of episodes from the episodic memory store.

    Query parameters
    ----------------
    status          : "active" (default) | "pending" | "superseded" |
                      "retracted" | "all". "pending" surfaces episodes
                      staged by the episodic_write_approval gate awaiting
                      POST /memory/episodes/{id}/approve or .../reject.
    project_context : filter by project context string
    episode_type    : filter by episode type
    task_id         : filter by originating task_id — the Episode Browsing
                      UI's per-turn "related memory" overlay uses this.
    limit           : max results (default 50, max 200)
    offset          : pagination offset (default 0)
    """
    mm = _state.memory_manager
    if mm is None:
        return EpisodesResponse(episodes=[], total=0, offset=offset, limit=limit)

    rows: list[dict] = await asyncio.to_thread(
        mm.list_episodes,
        status          = status,
        project_context = project_context,
        episode_type    = episode_type,
        task_id         = task_id,
        limit           = limit,
        offset          = offset,
    )
    # Total is the full matching-row count (mm.count_episodes()), not
    # len(rows) — the latter is silently capped by `limit` and would make
    # e.g. ?status=pending&limit=1 (used for a pending-count badge) always
    # report 0 or 1 regardless of how many pending episodes actually exist.
    total: int = await asyncio.to_thread(
        mm.count_episodes,
        status          = status,
        project_context = project_context,
        episode_type    = episode_type,
        task_id         = task_id,
    )

    return EpisodesResponse(
        episodes = [EpisodeItem(**row) for row in rows],
        total    = total,
        offset   = offset,
        limit    = limit,
    )


@app.get(
    "/memory/episodes/related",
    response_model = RelatedEpisodesResponse,
    summary        = "Find episodes semantically related to a chat turn",
)
async def get_related_episodes(
    content: str,
    task_id: str | None = None,
    limit:   int = 5,
) -> RelatedEpisodesResponse:
    """
    Mode 3 (semantic similarity, §2.6) lookup backing the Episode Browsing
    UI's "Related Memory" panel (EpisodeAnnotations.svelte). Replaces the
    old task_id exact-match query — episodes are sparse by design (§2.1),
    so most selected turns have zero same-task_id episodes and the old
    query read as "no related memory" for the common case.

    `content` is the selected chat turn's own content, scored against every
    active episode via EpisodicMemoryReader.by_similarity() — real cosine
    similarity where an embedding exists, keyword/Jaccard fallback
    otherwise, exactly as Mode 3 already behaves elsewhere. `task_id`, when
    supplied, excludes episodes written from that exact turn: they were
    extracted from this very content, so are trivially "related" and were
    already what the old query surfaced — the case this fixes is
    everything else.

    min_score=0.45 mirrors the same Mode 3 threshold controller_agent.py
    uses for episodic recall during a live turn (_execute_plan's Step 5).
    """
    mm = _state.memory_manager
    if mm is None:
        return RelatedEpisodesResponse(episodes=[])

    reader = EpisodicMemoryReader(db_path=mm._db_path, embed_fn=mm.embed_fn)
    records = await asyncio.to_thread(
        reader.by_similarity,
        content,
        top_n           = limit,
        min_score       = 0.45,
        exclude_task_id = task_id,
    )

    return RelatedEpisodesResponse(
        episodes = [EpisodeItem(**asdict(record)) for record in records],
    )


@app.post(
    "/memory/episodes/{episode_id}/approve",
    response_model = EpisodeApprovalResponse,
    summary        = "Approve a pending episode (write-approval gate)",
)
async def approve_memory_episode(episode_id: int) -> EpisodeApprovalResponse:
    """
    Transition a pending episode to active — the "yes" path of the
    episodic_write_approval gate. Once active, the episode is eligible for
    by_recency()/by_similarity() and appears in MEMORY.md immediately.
    Also retriggers the Phase B graph hook (memory-graph-inference-plan
    §8.9), which the implicit-extraction post-response hook deliberately
    skipped while the episode was pending — see
    ControllerAgent._write_episode_graph_node()'s docstring for why a
    pending episode gets no graph node until this point.

    Idempotent: approving an id that's already active/retracted, or that
    doesn't exist, returns updated=False rather than an error.
    """
    mm = _state.memory_manager
    if mm is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")

    writer = EpisodicMemoryWriter(
        db_path=getattr(mm, "_db_path", None), memory_md_path=_MEMORY_MD_PATH,
    )
    count = await asyncio.to_thread(writer.approve, episode_id)
    if count > 0:
        subject = await asyncio.to_thread(writer.get_episode_subject, episode_id)
        controller = _state.controller
        if controller is not None and subject is not None:
            # Reaches into ControllerAgent's private hook — same precedent
            # as GET /agents reading controller._agents directly elsewhere
            # in this file. Best-effort and self-contained: the hook
            # catches and logs its own exceptions, so a broken graph write
            # can never fail this endpoint or leave the approval half-done.
            await asyncio.to_thread(
                controller._write_episode_graph_node, episode_id, subject,
            )
    return EpisodeApprovalResponse(
        episode_id = episode_id,
        status     = "active",
        updated    = count > 0,
    )


@app.post(
    "/memory/episodes/{episode_id}/reject",
    response_model = EpisodeApprovalResponse,
    summary        = "Reject a pending episode (write-approval gate)",
)
async def reject_memory_episode(episode_id: int) -> EpisodeApprovalResponse:
    """
    Transition a pending episode to retracted — the "no" path of the
    episodic_write_approval gate. A rejected episode never becomes live
    memory.

    Idempotent: rejecting an id that's already active/retracted, or that
    doesn't exist, returns updated=False rather than an error.
    """
    mm = _state.memory_manager
    if mm is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")

    writer = EpisodicMemoryWriter(
        db_path=getattr(mm, "_db_path", None), memory_md_path=_MEMORY_MD_PATH,
    )
    count = await asyncio.to_thread(writer.reject, episode_id)
    return EpisodeApprovalResponse(
        episode_id = episode_id,
        status     = "retracted",
        updated    = count > 0,
    )


@app.post(
    "/memory/episodes/{episode_id}/reactivate",
    response_model = EpisodeApprovalResponse,
    summary        = "Reactivate a retracted episode",
)
async def reactivate_memory_episode(episode_id: int) -> EpisodeApprovalResponse:
    """
    Transition a retracted episode back to active — the reversal path for
    retract()/retract_by_id()/reject() and the global-retention TTL sweep
    (MemoryManager.sweep_expired_memory()), all of which land a row at
    status='retracted'. If a different episode is currently active for the
    same (subject, episode_type), that row is superseded first — see
    EpisodicMemoryWriter.reactivate()'s docstring for why this differs from
    approve(), which does not resolve that edge case. Also retriggers the
    Phase B graph hook, same as approve() — safe to call unconditionally
    since it's an upsert.

    Idempotent: reactivating an id that's already active/pending, or that
    doesn't exist, returns updated=False rather than an error.
    """
    mm = _state.memory_manager
    if mm is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")

    writer = EpisodicMemoryWriter(
        db_path=getattr(mm, "_db_path", None), memory_md_path=_MEMORY_MD_PATH,
    )
    count = await asyncio.to_thread(writer.reactivate, episode_id)
    if count > 0:
        subject = await asyncio.to_thread(writer.get_episode_subject, episode_id)
        controller = _state.controller
        if controller is not None and subject is not None:
            await asyncio.to_thread(
                controller._write_episode_graph_node, episode_id, subject,
            )
    return EpisodeApprovalResponse(
        episode_id = episode_id,
        status     = "active",
        updated    = count > 0,
    )


# ---------------------------------------------------------------------------
# File management endpoints
# ---------------------------------------------------------------------------

def _file_entry(p: "Path", type: Literal["raw", "wiki", "generated"]) -> FileEntry:
    """Build a FileEntry from a Path."""
    stat = p.stat()
    return FileEntry(
        name     = p.stem,
        filename = p.name,
        path     = str(p.resolve()),
        size     = stat.st_size,
        modified = datetime.datetime.fromtimestamp(
            stat.st_mtime, tz=datetime.timezone.utc
        ).isoformat(),
        type     = type,
    )


_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize_filename_stem(raw: str) -> str:
    """
    Turn arbitrary user input (e.g. a "Save as" filename field) into a safe
    generated_files/ filename stem: whitespace and anything else collapses
    to '-', repeats collapse, leading/trailing '-' are stripped, and the
    result is capped at 150 chars. No spaces survive, which keeps
    POST /files/generated's parse of file_ops.write_file()'s return string
    (" ... to <name>") unambiguous even when the name was auto-versioned.
    """
    collapsed = _FILENAME_SANITIZE_RE.sub("-", raw.strip())
    collapsed = re.sub(r"-{2,}", "-", collapsed).strip("-")
    return collapsed[:150]


@app.get(
    "/files/raw",
    response_model = FilesResponse,
    summary        = "List raw files",
)
async def get_files_raw() -> FilesResponse:
    """
    Return metadata for every .md/.txt or OCR-eligible (image/PDF) file in
    the raw/ directory.
    """
    if _state.raw_dir is None:
        raise HTTPException(status_code=503, detail="raw_dir not configured.")
    raw_dir = _state.raw_dir
    if not raw_dir.exists():
        return FilesResponse(files=[])
    files = [
        _file_entry(p, "raw")
        for p in sorted(raw_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {".md", ".txt", *_OCR_MIME_BY_EXTENSION}
    ]
    return FilesResponse(files=files)


@app.get(
    "/files/wiki",
    response_model = FilesResponse,
    summary        = "List wiki pages",
)
async def get_files_wiki() -> FilesResponse:
    """
    Return metadata for every .md content page in the wiki/ directory.

    Excludes META_WIKI_FILENAMES (index.md, logs.md, MEMORY.md) — these are
    structural/generated files, never a page a user would pin as a diff
    target.
    """
    if _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="wiki_dir not configured.")
    wiki_dir = _state.wiki_dir
    if not wiki_dir.exists():
        return FilesResponse(files=[])
    files = [
        _file_entry(p, "wiki")
        for p in sorted(wiki_dir.iterdir())
        if p.is_file() and p.suffix == ".md" and p.name not in META_WIKI_FILENAMES
    ]
    return FilesResponse(files=files)


@app.get(
    "/files/generated",
    response_model = FilesResponse,
    summary        = "List generated files",
)
async def get_files_generated() -> FilesResponse:
    """Return metadata for every file in the generated_files/ directory."""
    if _state.generated_dir is None:
        raise HTTPException(status_code=503, detail="generated_dir not configured.")
    generated_dir = _state.generated_dir
    if not generated_dir.exists():
        return FilesResponse(files=[])
    files = [
        _file_entry(p, "generated")
        for p in sorted(generated_dir.iterdir())
        if p.is_file() and not p.name.startswith(".")
    ]
    return FilesResponse(files=files)


@app.get(
    "/files/content",
    response_model = FileContentResponse,
    summary        = "Read file content",
)
async def get_file_content(path: str) -> FileContentResponse:
    """
    Return the plain-text content of a file by absolute path.

    Only paths inside raw_dir or wiki_dir are permitted — anything else
    returns HTTP 403.
    """
    if _state.raw_dir is None or _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="Directories not configured.")

    target = Path(path).resolve()
    allowed_roots = [
        _state.raw_dir.resolve(),
        _state.wiki_dir.resolve(),
    ]
    if _state.generated_dir is not None:
        allowed_roots.append(_state.generated_dir.resolve())
    if not any(str(target).startswith(str(root)) for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside permitted directories.",
        )
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        content = await asyncio.to_thread(target.read_text, "utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read file: {exc}") from exc

    return FileContentResponse(path=str(target), content=content)


@app.get(
    "/files/download",
    summary = "Download a file",
)
async def get_file_download(path: str) -> FileResponse:
    """
    Stream a file back with a Content-Disposition: attachment header so the
    browser saves it (Safari's Downloads queue) instead of navigating to it.

    Same allowed-roots gate as /files/content, but path containment is
    checked with is_relative_to() rather than a raw string prefix — a
    prefix check would let /data/wiki_evil slip through for an allowed
    root of /data/wiki.
    """
    if _state.raw_dir is None or _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="Directories not configured.")

    target = Path(path).resolve()
    allowed_roots = [
        _state.raw_dir.resolve(),
        _state.wiki_dir.resolve(),
    ]
    if _state.generated_dir is not None:
        allowed_roots.append(_state.generated_dir.resolve())
    if not any(target.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside permitted directories.",
        )
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(
        path=target,
        filename=target.name,
        media_type=media_type,
    )


@app.delete(
    "/files",
    response_model = FileDeleteResponse,
    summary        = "Delete a file",
)
async def delete_file(path: str) -> FileDeleteResponse:
    """
    Delete a file from raw/, wiki/, or generated_files/ by absolute path.

    Same allowed-roots gate as /files/content and /files/download. Also
    purges any document_index row for the path — a no-op for generated
    files (never indexed), but necessary for raw/wiki so a deleted file
    doesn't linger in RAG retrieval.
    """
    if _state.raw_dir is None or _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="Directories not configured.")

    target = Path(path).resolve()
    allowed_roots = [
        _state.raw_dir.resolve(),
        _state.wiki_dir.resolve(),
    ]
    if _state.generated_dir is not None:
        allowed_roots.append(_state.generated_dir.resolve())
    if not any(target.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside permitted directories.",
        )
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        await asyncio.to_thread(target.unlink)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete file: {exc}") from exc

    if _state.memory_manager is not None:
        try:
            await asyncio.to_thread(_state.memory_manager.remove_document, target)
        except Exception as exc:
            logger.warning("remove_document failed for deleted file %s: %s", target, exc)

    return FileDeleteResponse(path=str(target), deleted=True)


@app.post(
    "/files/upload",
    response_model = FileEntry,
    summary        = "Upload a raw file",
)
async def post_file_upload(file: UploadFile = File(...)) -> FileEntry:
    """
    Accept a multipart file upload and save it to raw/.

    .md and .txt files are accepted as-is, plus the same OCR-eligible
    extensions .chat/files accepts (images incl. HEIC, PDF) — the raw
    bytes are saved unchanged here; WikiAgent extracts text from them via
    ocr_extract lazily at ingest time (§22 follow-up), not at upload time,
    so the original file remains the canonical raw source on disk exactly
    like .md/.txt already are. If a file with the same name already
    exists it is overwritten. Returns the FileEntry for the saved file.
    """
    if _state.raw_dir is None:
        raise HTTPException(status_code=503, detail="raw_dir not configured.")

    filename = file.filename or "upload.md"
    suffix   = Path(filename).suffix.lower()
    if suffix not in {".md", ".txt", *_OCR_MIME_BY_EXTENSION}:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type, got: {suffix}",
        )

    raw_dir = _state.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename

    try:
        contents = await file.read()
        await asyncio.to_thread(dest.write_bytes, contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        await file.close()

    # Index the newly uploaded file immediately so ConversationalAgent can find it
    # without waiting for the next server restart seed pass.
    if _state.memory_manager is not None:
        try:
            await asyncio.to_thread(
                _state.memory_manager.index_document,
                dest,
                "raw",
                None,
                False,
            )
        except Exception as exc:
            logger.warning(
                "MemoryManager.index_document failed for upload %s: %s", filename, exc
            )

    return _file_entry(dest, "raw")


@app.post(
    "/files/generated",
    response_model = FileEntry,
    summary        = "Save content as a new generated file",
)
async def post_file_generated(body: SaveGeneratedFileRequest) -> FileEntry:
    """
    Write user-supplied content (e.g. a "Save as" on an edited chat turn)
    into generated_files/ as a new file — the direct-write counterpart to
    the model-driven file_op write_file MCP tool, for actions the user
    triggers themselves rather than the agent.

    Reuses file_ops.write_file()'s sandboxing and never-overwrite/auto-
    versioning behavior (name_2.ext ... name_10.ext) via the in-process
    root synced to _state.generated_dir at startup, so the saved file
    immediately shows up through the existing GET /files/generated,
    /files/download, and DELETE /files endpoints.
    """
    if _state.generated_dir is None:
        raise HTTPException(status_code=503, detail="generated_dir not configured.")

    stem = _sanitize_filename_stem(body.filename)
    if not stem:
        raise HTTPException(
            status_code=400,
            detail="filename must contain at least one letter, digit, '-', or '_'.",
        )
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty.")

    target_name = f"{stem}.{body.extension}"
    try:
        result = await asyncio.to_thread(_write_generated_file, target_name, body.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # write_file() returns "OK: wrote N characters to <actual_name>" — the
    # actual name may differ from target_name if it was auto-versioned
    # because target_name already existed.
    actual_name = result.rsplit(" ", 1)[-1]
    return _file_entry(_state.generated_dir / actual_name, "generated")


# ---------------------------------------------------------------------------
# Review-then-apply wiki diffs
# ---------------------------------------------------------------------------

@app.post(
    "/wiki/apply-diff",
    response_model = ApplyDiffResponse,
    summary        = "Apply a previously-proposed wiki diff",
)
async def post_wiki_apply_diff(body: ApplyDiffRequest) -> ApplyDiffResponse:
    """
    Write a diff WikiAgent previously proposed (surfaced to the chat UI via
    a turn's metadata.pending_diffs) directly to disk — no fresh model
    call, no re-routing through the Planner.

    Content-based matching in apply_unified_diff() is the staleness check:
    if the target page changed on disk since the diff was proposed, the
    match legitimately fails and this raises 409 rather than corrupting
    the page. A missing target page raises 404.

    On success, best-effort marks the originating chat turn's persisted
    pending_diffs entry as "applied" (MemoryManager.mark_diff_applied())
    so a page reload reflects the write; failure to do so is logged but
    does not fail the request — the disk write already succeeded.
    """
    wiki_agent = _require_wiki_agent()
    if _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="wiki_dir not configured.")

    result = await asyncio.to_thread(
        wiki_agent.apply_pending_diff, body.page_name, body.diff, _state.wiki_dir,
    )

    if result.status != TaskStatus.COMPLETE:
        status_code = 404 if result.output.get("error_kind") == "not_found" else 409
        raise HTTPException(status_code=status_code, detail=result.error)

    if _state.memory_manager is not None:
        try:
            await asyncio.to_thread(
                _state.memory_manager.mark_diff_applied, body.task_id, body.page_name,
            )
        except Exception as exc:
            logger.warning(
                "mark_diff_applied failed for task_id=%s page_name=%s: %s",
                body.task_id, body.page_name, exc,
            )

    return ApplyDiffResponse(success=True, page_name=body.page_name)


# ---------------------------------------------------------------------------
# Chat file attachments (session-scoped, ephemeral, no wiki ingestion)
# ---------------------------------------------------------------------------

async def _extract_text_via_ocr(filename: str, raw: bytes, mime_type: str) -> str:
    """
    Write raw bytes to a temp file under the ocr_extract sandbox root
    (mcp_server.ocr.get_upload_root() — imported directly so this process
    and localist-mcp always agree on the same directory, rather than
    duplicating LOCALIST_MCP_UPLOAD_ROOT resolution here) and OCR it via
    MCPToolDispatcher, cleaning the temp file up afterward regardless of
    outcome. Raises HTTPException(422) on any OCR failure — unsupported
    platform, unreadable file, or near-empty extraction — same status code
    the UTF-8-decode-failure branch below already uses for binary files.
    """
    upload_root = _ocr_upload_root()
    tmp_name = f"{uuid.uuid4().hex}{os.path.splitext(filename)[1].lower()}"
    tmp_path = upload_root / tmp_name
    tmp_path.write_bytes(raw)

    try:
        dispatcher = MCPToolDispatcher(runtime=_require_runtime())
        results = await asyncio.to_thread(
            dispatcher.dispatch,
            ["ocr_extract"],
            "",
            {"ocr_file_path": tmp_name, "ocr_mime_type": mime_type},
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    result = results[0]
    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=f"'{filename}' could not be read — {result.result}",
        )
    return result.result


@app.post("/chat/files")
async def attach_chat_file(file: UploadFile = File(...)):
    """
    Upload a file into the ephemeral session file cache.

    Text files are decoded as UTF-8 directly. Images (incl. HEIC) and PDFs
    are routed through the local ocr_extract MCP tool instead — text is
    extracted once at upload time, independent of whichever chat runtime
    backend is active, then cached exactly like a text upload (same
    session_files budget, no separate image cache). See
    docs/architecture/22-local-ocr-service.md.

    Returns 200 + {filename, token_estimate} on success.
    Returns 400 + {detail} on rejection (type, size, or budget).
    Returns 422 on encoding failure (binary/non-UTF-8 text file) or OCR
    failure (unsupported/unreadable image or PDF).
    """
    raw = await file.read()
    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext in _OCR_MIME_BY_EXTENSION:
        content = await _extract_text_via_ocr(file.filename, raw, _OCR_MIME_BY_EXTENSION[ext])
    else:
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=422,
                detail=f"'{file.filename}' could not be read as UTF-8 text. Binary files are not supported.",
            )

    error = session_files.add_file(file.filename, content)
    if error:
        raise HTTPException(status_code=400, detail=error)

    return {
        "filename":       file.filename,
        "token_estimate": len(content) // 4,
    }


@app.delete("/chat/files/{filename}")
async def detach_chat_file(filename: str):
    """
    Remove a file from the ephemeral session file cache by filename.

    Returns 200 + {removed: true} if found and removed.
    Returns 404 if the filename was not in the cache.
    """
    removed = session_files.remove_file(filename)
    if not removed:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in session files.")
    return {"removed": True}


@app.post("/chat/pin-wiki-page")
async def pin_wiki_page(body: PinWikiPageRequest):
    """
    Pin an existing wiki page into the ephemeral session file cache by stem.

    Reads the page straight off disk (wiki_dir/{stem}.md) rather than via
    the graph index, since the graph is only rebuilt on an explicit trigger
    and can lag behind real files. Returns 200 + {filename, token_estimate,
    source} on success. Returns 404 if no such page exists on disk.
    Returns 400 + {detail} on rejection (size or budget), same as
    POST /chat/files.
    """
    if _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="wiki_dir not configured.")

    filename = f"{body.stem}.md"
    if filename in META_WIKI_FILENAMES:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is a structural wiki file, not a pinnable page.",
        )

    page_path = _state.wiki_dir / filename
    if not page_path.is_file():
        raise HTTPException(status_code=404, detail=f"Wiki page '{body.stem}' not found.")

    content = await asyncio.to_thread(read_text_file, page_path)
    error = session_files.add_file(filename, content, source="wiki_pin")
    if error:
        raise HTTPException(status_code=400, detail=error)

    return {
        "filename":       filename,
        "token_estimate": len(content) // 4,
        "source":         "wiki_pin",
    }


# ---------------------------------------------------------------------------
# Retention settings  (Settings tab — global TTL for chat_turns + episodes)
# ---------------------------------------------------------------------------

@app.get(
    "/settings/retention",
    response_model = RetentionSettingsResponse,
    summary        = "Read the global retention preset",
)
async def get_retention_settings() -> RetentionSettingsResponse:
    """
    Return the current global retention preset.

    ``eviction_preset`` is None when the user has never set one.
    """
    mm = _require_memory_manager()
    preset = await asyncio.to_thread(mm.get_retention_preset)
    return RetentionSettingsResponse(eviction_preset=preset)


@app.put(
    "/settings/retention",
    response_model = RetentionSettingsResponse,
    summary        = "Set the global retention preset",
)
async def put_retention_settings(
    request: RetentionSettingsRequest,
) -> RetentionSettingsResponse:
    """
    Set the global retention preset.

    Governs both chat_turns (hard-deleted) and episodes (soft-retracted via
    status='retracted') once the TTL sweep runs. The sweep itself only runs
    at backend startup (MemoryManager.sweep_expired_memory(), called from
    lifespan()) — this endpoint only persists the preference and returns the
    value re-read from the database to confirm the write landed.
    """
    mm = _require_memory_manager()
    await asyncio.to_thread(mm.set_retention_preset, request.eviction_preset)
    preset = await asyncio.to_thread(mm.get_retention_preset)
    return RetentionSettingsResponse(eviction_preset=preset)


# ---------------------------------------------------------------------------
# Assistant name  (Settings tab — user-configurable spoken identity)
# ---------------------------------------------------------------------------

@app.get(
    "/settings/assistant-name",
    response_model = AssistantNameResponse,
    summary        = "Read the configured assistant name",
)
async def get_assistant_name_setting() -> AssistantNameResponse:
    """
    Return the current assistant name, defaulting to "Localist" when the
    user has never set one.
    """
    mm = _require_memory_manager()
    name = await asyncio.to_thread(mm.get_assistant_name)
    return AssistantNameResponse(assistant_name=name)


@app.put(
    "/settings/assistant-name",
    response_model = AssistantNameResponse,
    summary        = "Set the assistant name",
)
async def put_assistant_name_setting(
    request: AssistantNameRequest,
) -> AssistantNameResponse:
    """
    Set the assistant name.

    Persists to MemoryManager.assistant_settings, then invalidates
    ControllerAgent's persona cache so the very next request re-fetches the
    persona doc with the new name substituted in, rather than serving a
    persona that still says the old one until the process restarts.
    PromptBuilder's identity slot (Slot 1a) needs no separate invalidation —
    it reads the name fresh on every build() call, never caches it.
    """
    mm = _require_memory_manager()
    try:
        await asyncio.to_thread(mm.set_assistant_name, request.assistant_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if _state.controller is not None:
        _state.controller.invalidate_persona_cache()
    name = await asyncio.to_thread(mm.get_assistant_name)
    return AssistantNameResponse(assistant_name=name)


# ---------------------------------------------------------------------------
# Daily News Brief  (docs/daily-news-brief-plan.md)
# ---------------------------------------------------------------------------

def _today_str() -> str:
    """Local calendar date, 'YYYY-MM-DD' — same local-time convention the
    [CURRENT DATETIME] prompt slot uses (prompt_builder.py), not UTC."""
    return datetime.datetime.now().astimezone().date().isoformat()


@app.get(
    "/news/preferences",
    response_model = NewsPreferencesResponse,
    summary        = "Read Daily News Brief preferences",
)
async def get_news_preferences() -> NewsPreferencesResponse:
    """Return current preferences, or defaults if the user has never set any."""
    mm = _require_memory_manager()
    prefs = await asyncio.to_thread(mm.get_news_preferences)
    if prefs is None:
        prefs = {"home_country": "us", "local_query": None, "topics": []}
    return NewsPreferencesResponse(
        home_country = prefs["home_country"],
        local_query  = prefs["local_query"],
        topics       = prefs["topics"],
        topic_pool   = news_brief.NEWS_TOPIC_LABELS,
    )


@app.put(
    "/news/preferences",
    response_model = NewsPreferencesResponse,
    summary        = "Set Daily News Brief preferences",
)
async def put_news_preferences(request: NewsPreferencesRequest) -> NewsPreferencesResponse:
    """
    Set home_country/local_query/topics. Does not touch an already-cached
    brief — changes take effect on the next generation (§4/§5).
    """
    mm = _require_memory_manager()

    if len(request.topics) != 3:
        raise HTTPException(
            status_code=422, detail="topics must have exactly 3 entries."
        )
    invalid = [t for t in request.topics if t not in news_brief.NEWS_TOPIC_POOL]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown topic key(s): {invalid}. "
                   f"Valid: {sorted(news_brief.NEWS_TOPIC_POOL)}",
        )

    await asyncio.to_thread(
        mm.set_news_preferences,
        request.home_country.lower(), request.local_query, request.topics,
    )
    prefs = await asyncio.to_thread(mm.get_news_preferences)
    return NewsPreferencesResponse(
        home_country = prefs["home_country"],
        local_query  = prefs["local_query"],
        topics       = prefs["topics"],
        topic_pool   = news_brief.NEWS_TOPIC_LABELS,
    )


@app.get(
    "/news/brief/preview",
    response_model = NewsBriefPreviewResponse,
    summary        = "Read today's cached Daily News Brief, if any",
)
async def get_news_brief_preview() -> NewsBriefPreviewResponse:
    """
    Read-only, for the header button's hover popover. Never calls NewsAPI —
    only ever reads news_brief_cache (docs/daily-news-brief-plan.md §6).
    """
    mm = _require_memory_manager()
    cache = await asyncio.to_thread(mm.get_news_brief_cache)
    if cache is None or cache["brief_date"] != _today_str():
        return NewsBriefPreviewResponse(available=False)
    return NewsBriefPreviewResponse(
        available  = True,
        brief_date = cache["brief_date"],
        sections   = [NewsBriefSection(**s) for s in cache["content"]],
    )


@app.post(
    "/news/brief/open",
    response_model = NewsBriefOpenResponse,
    summary        = "Refresh today's Daily News Brief for the Live Feed panel",
)
async def post_news_brief_open() -> NewsBriefOpenResponse:
    """
    The Previews panel's "Daily News Brief Refresh" link handler.

    Always fetches a fresh brief and writes it to news_brief_cache (so
    GET /news/brief/preview reflects it) — deliberately not idempotent
    within a day, same rationale as before: pressing a link literally
    labeled "Refresh" should never silently reuse stale content.

    Does NOT touch chat_turns or conversation_log. An earlier revision
    also opened the brief as a synthetic chat_turns conversation and
    seeded conversation_log (Slot 6 working memory) with the full
    markdown, so the Chat Conversation UI displayed it and a same-session
    follow-up question had context. Removed 2026-07-24: once the
    per-article "Ask about this" button (§7.16) shipped, dumping the
    *entire* unscoped brief into a chat turn on every refresh became
    redundant noise — a user who wants to discuss a specific story now
    clicks that story instead.
    """
    mm = _require_memory_manager()
    today = _today_str()

    prefs = await asyncio.to_thread(mm.get_news_preferences)
    if prefs is None:
        prefs = {"home_country": "us", "local_query": None, "topics": []}

    sections = await news_brief.build_brief(
        prefs["home_country"], prefs["local_query"], prefs["topics"],
    )
    await asyncio.to_thread(mm.set_news_brief_cache, today, sections, None)

    return NewsBriefOpenResponse()


@app.get(
    "/github/watch/preview",
    response_model = GithubWatchPreviewResponse,
    summary        = "Read the cached GitHub Watch Feed, if any",
)
async def get_github_watch_preview() -> GithubWatchPreviewResponse:
    """
    Read-only, for the Previews panel's GitHub section. Never calls
    GitHub — only ever reads github_watch_cache.
    """
    mm = _require_memory_manager()
    cache = await asyncio.to_thread(mm.get_github_watch_cache)
    if cache is None:
        return GithubWatchPreviewResponse(available=False)
    return GithubWatchPreviewResponse(
        available    = True,
        generated_at = cache["generated_at"],
        repos        = [GithubWatchRepo(**r) for r in cache["content"]],
    )


@app.post(
    "/github/watch/refresh",
    response_model = GithubWatchOpenResponse,
    summary        = "Refresh the GitHub Watch Feed for the Previews panel",
)
async def post_github_watch_refresh() -> GithubWatchOpenResponse:
    """
    The Previews panel's GitHub section refresh handler.

    Always fetches a fresh feed and writes it to github_watch_cache (so
    GET /github/watch/preview reflects it) — same "a refresh always hits
    the network" rationale as POST /news/brief/open. github_watch.build_watch_feed()
    never raises (a missing GITHUB_TOKEN or a failed repo listing degrades
    to a single error entry — see its docstring), so there's no try/except
    needed here, same as news_brief.build_brief().
    """
    mm = _require_memory_manager()
    pinned = await asyncio.to_thread(mm.get_pinned_github_repos)
    repos = await github_watch.build_watch_feed(pinned_full_names=pinned)
    await asyncio.to_thread(mm.set_github_watch_cache, repos)

    return GithubWatchOpenResponse()


_PINNED_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_PINNED_REPOS_MAX = 20


@app.get(
    "/github/watch/pinned-repos",
    response_model = PinnedGithubReposResponse,
    summary        = "Read the user's pinned GitHub repos",
)
async def get_pinned_github_repos() -> PinnedGithubReposResponse:
    """
    Local-only, no GitHub call — pinned repos are release-tracked
    independent of GitHub's Watch/subscriptions relationship (no email
    noise), merged into the feed by POST /github/watch/refresh.
    """
    mm = _require_memory_manager()
    repos = await asyncio.to_thread(mm.get_pinned_github_repos)
    return PinnedGithubReposResponse(repos=repos)


@app.put(
    "/github/watch/pinned-repos",
    response_model = PinnedGithubReposResponse,
    summary        = "Set the user's pinned GitHub repos",
)
async def put_pinned_github_repos(
    request: PinnedGithubReposRequest,
) -> PinnedGithubReposResponse:
    """
    Full-list replace. Does not touch an already-cached watch feed —
    changes take effect on the next POST /github/watch/refresh, same
    "changes take effect on the next generation" posture as
    PUT /news/preferences.
    """
    mm = _require_memory_manager()

    if len(request.repos) > _PINNED_REPOS_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"repos must have at most {_PINNED_REPOS_MAX} entries, "
                   f"got {len(request.repos)}.",
        )
    invalid = [r for r in request.repos if not _PINNED_REPO_RE.match(r)]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid repo slug(s), expected 'owner/repo': {invalid}",
        )
    lowered = [r.lower() for r in request.repos]
    if len(lowered) != len(set(lowered)):
        raise HTTPException(
            status_code=422, detail="repos must not contain duplicates."
        )

    await asyncio.to_thread(mm.set_pinned_github_repos, request.repos)
    repos = await asyncio.to_thread(mm.get_pinned_github_repos)
    return PinnedGithubReposResponse(repos=repos)


@app.get(
    "/hacker-news/top/preview",
    response_model = HackerNewsPreviewResponse,
    summary        = "Read the cached Hacker News top-stories feed, if any",
)
async def get_hacker_news_preview() -> HackerNewsPreviewResponse:
    """
    Read-only, for the Previews panel's Hacker News section. Never calls
    Hacker News — only ever reads hacker_news_cache.
    """
    mm = _require_memory_manager()
    cache = await asyncio.to_thread(mm.get_hacker_news_cache)
    if cache is None:
        return HackerNewsPreviewResponse(available=False)
    return HackerNewsPreviewResponse(
        available    = True,
        generated_at = cache["generated_at"],
        stories      = [HackerNewsStory(**s) for s in cache["content"]],
    )


@app.post(
    "/hacker-news/top/refresh",
    response_model = HackerNewsOpenResponse,
    summary        = "Refresh the Hacker News top-stories feed for the Previews panel",
)
async def post_hacker_news_refresh() -> HackerNewsOpenResponse:
    """
    The Previews panel's Hacker News section refresh handler.

    Always fetches a fresh feed and writes it to hacker_news_cache (so
    GET /hacker-news/top/preview reflects it) — same "a refresh always
    hits the network" rationale as POST /github/watch/refresh.
    hacker_news.build_top_stories() never raises (a failed top-story
    listing degrades to a single error entry — see its docstring), so no
    try/except is needed here, same as github_watch.build_watch_feed().
    """
    mm = _require_memory_manager()
    stories = await hacker_news.build_top_stories()
    await asyncio.to_thread(mm.set_hacker_news_cache, stories)

    return HackerNewsOpenResponse()


@app.get(
    "/chat/history",
    response_model = ChatHistoryResponse,
    summary        = "List chat_turns, optionally full-text filtered",
)
async def get_chat_history(
    q:               str | None = None,
    limit:           int         = 50,
    offset:          int         = 0,
    conversation_id: str | None = None,
    mode:            Literal["keyword", "semantic"] = "keyword",
    min_score:       float       = 0.3,
    date_from:       float | None = None,
    date_to:         float | None = None,
    has_tool_result: bool         = False,
) -> ChatHistoryResponse:
    """
    Return a paginated list of chat_turns, newest first.

    Query parameters
    ----------------
    q               : optional search string. mode="keyword" matches via
                      chat_turns_fts; mode="semantic" scores via cosine
                      similarity over the stored embedding column.
    limit           : max results (default 50, max 200)
    offset          : pagination offset (default 0)
    conversation_id : optional conversation_id filter — when provided, restricts
                      results to one conversation; when omitted, searches/lists
                      across all conversations.
    mode            : "keyword" (default) or "semantic". Silently falls back to
                      keyword search when no embed_fn is configured or
                      chat_turns is flagged stale — see MemoryManager.get_chat_turns().
    min_score       : mode="semantic" only — minimum cosine score to include
                      a result (default 0.3).
    date_from       : optional inclusive lower bound on created_at (unix seconds).
    date_to         : optional inclusive upper bound on created_at (unix seconds).
    has_tool_result : when true, restricts to turns carrying a chart, pending_diffs,
                      or workflow_id in metadata — see MemoryManager.get_chat_turns().

    Read-only — no eviction/deletion happens here.
    """
    mm = _require_memory_manager()
    limit = min(limit, 200)

    rows, total = await asyncio.to_thread(
        mm.get_chat_turns, query=q, limit=limit, offset=offset,
        conversation_id=conversation_id, mode=mode, min_score=min_score,
        date_from=date_from, date_to=date_to, has_tool_result=has_tool_result,
    )

    return ChatHistoryResponse(
        turns  = [ChatTurnItem(**row) for row in rows],
        total  = total,
        offset = offset,
        limit  = limit,
    )


@app.get(
    "/chat/history/conversations",
    response_model = ConversationListResponse,
    summary        = "List distinct conversations, newest first",
)
async def get_conversations() -> ConversationListResponse:
    """
    Return one summary row per distinct conversation_id, ordered by
    last_created_at descending — used to populate the Chat tab's
    conversation sub-list in the sidebar.

    Read-only.
    """
    mm = _require_memory_manager()
    rows = await asyncio.to_thread(mm.get_conversations)
    return ConversationListResponse(
        conversations = [ConversationSummary(**row) for row in rows]
    )


@app.delete(
    "/chat/history/conversations/{conversation_id}",
    response_model = ConversationDeleteResponse,
    summary        = "Delete a conversation and all its turns",
)
async def delete_conversation(conversation_id: str) -> ConversationDeleteResponse:
    """
    Permanently delete every chat_turns row for conversation_id (and its FTS
    entries, via the chat_turns_ad trigger). Used by the sidebar's per-row
    delete button.
    """
    mm = _require_memory_manager()
    turns_deleted = await asyncio.to_thread(mm.delete_conversation, conversation_id)
    if turns_deleted == 0:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    return ConversationDeleteResponse(conversation_id=conversation_id, turns_deleted=turns_deleted)


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

async def _stream_task(
    controller:         ControllerAgent,
    runtime:            BaseRuntimeClient,
    task_dict:          dict[str, Any],
    task_id:            str,
    conversation_id:    str,
    conversation_title: str | None = None,
) -> AsyncIterator[str]:
    """
    Async generator that drives the streaming endpoint.

    Pipeline
    --------
    1.  Emit a "Planning..." status event.
    2.  Run the full handle_task() pipeline in a thread pool.
    3.  Emit a "Streaming answer…" status event.
    4.  Replay the completed answer word-by-word as token events.
    5.  Emit sources, done, and the [DONE] sentinel.
    """

    def _sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    yield _sse({"type": "status", "message": "Planning task…", "task_id": task_id})

    _persist_chat_turn(
        "user", task_dict["instruction"], task_id, conversation_id,
        conversation_title = conversation_title,
    )

    # Route in a separate thread — some priority branches call embed_fn / infer().
    try:
        plan = await asyncio.to_thread(
            controller.route_task,
            task_dict["instruction"],
            task_dict.get("context", {}),
        )
    except Exception as exc:
        logger.exception("Error during routing for task %s.", task_id)
        yield _sse({"type": "error", "message": str(exc), "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    yield _sse({
        "type":    "status",
        "message": f"Routed to {plan.agent}",
        "task_id": task_id,
    })

    # Execute the precomputed plan with real per-token streaming.
    #
    # Bridge design: asyncio.Queue + loop.call_soon_threadsafe
    # --------------------------------------------------------
    # ConversationalAgent.run() calls on_token(chunk) from a worker thread
    # (via asyncio.to_thread).  call_soon_threadsafe schedules a put_nowait
    # on the asyncio.Queue from that thread so we can await items on the
    # event loop side without crossing the thread boundary per-get.  This
    # avoids the overhead of wrapping every queue.get() in asyncio.to_thread.
    # For agents that never call on_token (e.g. WikiAgent), the queue stays
    # empty and the drain loop terminates immediately once the producer task
    # is done — no stall.

    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_token(chunk: str) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, {"_kind": "token", "chunk": chunk})

    def on_status(message: str) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, {"_kind": "status", "message": message})

    def on_answer_ready(result_dict: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(
            event_queue.put_nowait, {"_kind": "answer_ready", "result": result_dict}
        )

    def _drain_item(item: dict[str, Any]) -> str:
        if item["_kind"] == "status":
            return _sse({"type": "status", "message": item["message"], "task_id": task_id})
        return _sse({"type": "token", "token": item["chunk"]})

    producer_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(
        asyncio.to_thread(
            controller.handle_task_with_plan,
            task_dict,
            plan,
            on_token=on_token,
            on_status=on_status,
            on_answer_ready=on_answer_ready,
        )
    )

    # Tracks whether sources+done were already emitted via on_answer_ready.
    # When True, the post-hook [DONE] sentinel closes the stream without
    # re-emitting those events; failure events are only logged, not sent.
    answer_ready_emitted = False

    # Drain events while producer runs
    while not producer_task.done():
        try:
            item = await asyncio.wait_for(event_queue.get(), timeout=0.05)
            if item["_kind"] == "answer_ready":
                # Answer is complete — emit sources+done immediately so the
                # client unblocks before memory hooks finish.
                answer_ready_emitted = True
                rd = item["result"]
                yield _sse({"type": "sources", "sources": rd.get("sources", [])})
                yield _sse({
                    "type":     "done",
                    "task_id":  task_id,
                    "status":   rd.get("status", "complete"),
                    "metadata": rd.get("metadata", {}),
                    "answer":   rd.get("answer", ""),
                })
                _persist_chat_turn(
                    "assistant", rd.get("answer", ""), task_id, conversation_id,
                    sources  = rd.get("sources"),
                    metadata = rd.get("metadata"),
                )
            elif not answer_ready_emitted:
                # Relay token/status events only before 'done' is sent; silently
                # drop post-done hook status events (e.g. "Updating working memory…")
                # to avoid flickering the task status back to 'planning'.
                yield _drain_item(item)
        except asyncio.TimeoutError:
            pass

    # Collect result / surface exception
    try:
        result: dict[str, Any] = await producer_task
    except Exception as exc:
        if answer_ready_emitted:
            # Error occurred in post-answer hooks — answer already sent, so do
            # not attempt to emit an error event over an already-closed stream.
            logger.exception(
                "Error in post-answer hooks for task %s (answer already sent).", task_id
            )
        else:
            logger.exception("Error during planning/dispatch for task %s.", task_id)
            yield _sse({"type": "error", "message": str(exc), "task_id": task_id})
        # The pipeline (including post-answer hooks) has stopped running one
        # way or another — signal task_complete so the client can re-enable
        # input rather than waiting indefinitely.
        yield _sse({"type": "task_complete", "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    # Drain any events queued between the last poll and task completion.
    # Skip answer_ready (already handled) and post-done hook events.
    while not event_queue.empty():
        item = event_queue.get_nowait()
        if item["_kind"] not in ("answer_ready",) and not answer_ready_emitted:
            yield _drain_item(item)

    if answer_ready_emitted:
        # sources+done were already sent early; the pipeline (including
        # post-answer episodic/working-state hooks) has now actually
        # finished, since we're past `await producer_task`. Signal that
        # distinctly so the client can tell "answer visible" apart from
        # "fully done" and re-enable input only now.
        yield _sse({"type": "task_complete", "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    if result.get("status") == "failed":
        yield _sse({
            "type":    "error",
            "message": result.get("error", "Task failed during planning or dispatch."),
            "task_id": task_id,
        })
        yield _sse({"type": "task_complete", "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    _persist_chat_turn(
        "assistant", result.get("answer", ""), task_id, conversation_id,
        sources  = result.get("sources"),
        metadata = result.get("metadata"),
    )

    yield _sse({"type": "sources",  "sources": result.get("sources", [])})
    yield _sse({
        "type":     "done",
        "task_id":  task_id,
        "status":   result.get("status", "complete"),
        "metadata": result.get("metadata", {}),
        "answer":   result.get("answer", ""),
    })
    # No on_answer_ready path was taken (e.g. non-conversational agent) —
    # 'done' above already reflects the fully-resolved pipeline, but emit
    # task_complete too so the client's completion signal is uniform across
    # both paths.
    yield _sse({"type": "task_complete", "task_id": task_id})
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code = 500,
        content     = {
            "detail": str(exc),
            "path":   str(request.url.path),
        },
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host      = "127.0.0.1",
        port      = 8001,
        reload    = True,
        log_level = "info",
    )