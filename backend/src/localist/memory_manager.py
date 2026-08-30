"""
Localist — SQLite MemoryManager
============================
Persistent, local-first replacement for the in-process MemoryManager
currently defined in controller_agent.py.

Layer placement
---------------
  ControllerAgent / ResearchAgent  →  MemoryManager  →  SQLite (local file)
                                                       →  EmbeddingGemma (optional)

Architectural contract
----------------------
- Pure Python module.  No FastAPI, no HTTP, no agent logic.
- Drop-in replacement for the in-process MemoryManager in controller_agent.py.
  The ControllerAgent API (add / add_agent_result / format_for_prompt /
  get_context_window / clear) is fully preserved so no call sites change.
- ResearchAgent gains a new retrieval path: query_corpus() replaces the
  full filesystem walk in _load_corpus() when a MemoryManager is available.
- All data lives in a single SQLite file.  No external services.  The file
  persists across server restarts, page refreshes, and process kills.
- Embeddings are optional.  When an embed callable is not supplied, keyword
  overlap scoring is used for retrieval (same strategy as the current
  ResearchAgent).  Embeddings can be enabled at any time without a schema
  migration — the column is always present, just NULL until populated.
- Thread-safe: every public method acquires a threading.Lock before any
  DB write.  Reads use a separate connection per call (WAL mode allows
  concurrent readers alongside one writer).

Database file location
----------------------
Defaults to  <project_root>/localist_memory.db
Override via the ``db_path`` constructor argument or the LOCALIST_MEMORY_DB
environment variable (the latter is read by main.py's Settings class —
add  memory_db: str | None = None  to Settings and pass it through).

Schema — three tables
----------------------

  conversation_log
  ----------------
  Append-only log of every agent turn within a task session.
  Mirrors the existing in-process MemoryManager._entries list but survives
  restarts.  Used by Synthesizer.format_for_prompt().

    id          INTEGER PRIMARY KEY AUTOINCREMENT
    task_id     TEXT    NOT NULL          — groups entries by task
    role        TEXT    NOT NULL          — "user" | "agent" | "system"
    content     TEXT    NOT NULL
    meta_json   TEXT    DEFAULT '{}'      — JSON-encoded metadata dict
    created_at  REAL    NOT NULL          — unix timestamp (time.time())

  document_index
  --------------
  One row per unique document (wiki page or raw file).  Updated whenever
  WikiAgent writes a new page.  ResearchAgent queries this table instead
  of walking the filesystem on every call.

    id           INTEGER PRIMARY KEY AUTOINCREMENT
    name         TEXT    NOT NULL          — page stem (kebab-case)
    path         TEXT    NOT NULL UNIQUE   — absolute path on disk
    doc_type     TEXT    NOT NULL          — "wiki" | "raw"
    content      TEXT    NOT NULL          — full UTF-8 text
    token_set    TEXT    NOT NULL DEFAULT '' — space-separated lowercase tokens
                                             (pre-computed for fast keyword scoring)
    embedding    BLOB    DEFAULT NULL      — packed float32 array (struct.pack)
                                             NULL until embed() is called
    content_hash TEXT    NOT NULL DEFAULT '' — sha256[:16] for change detection
    indexed_at   REAL    NOT NULL          — unix timestamp of last index/update

  retrieval_cache
  ---------------
  Optional query-level cache.  Maps a query string + top-N to a JSON array
  of (name, score) pairs.  Invalidated whenever document_index changes.
  Cheap hit rate gain for repeated identical sub-queries within ResearchAgent.

    id          INTEGER PRIMARY KEY AUTOINCREMENT
    query_hash  TEXT    NOT NULL          — sha256 of (query + str(top_n) + str(doc_type))
    top_n       INTEGER NOT NULL
    result_json TEXT    NOT NULL          — JSON: [{name, path, doc_type, score}]
    created_at  REAL    NOT NULL
    valid       INTEGER NOT NULL DEFAULT 1  — 0 = invalidated on index mutation

Embedding storage
-----------------
Embeddings are stored as raw bytes using struct.pack(f">{n}f", *vector).
Big-endian float32, one float per dimension (768 floats × 4 bytes = 3072 bytes).
This is faster to serialise/deserialise than JSON and avoids numpy/sqlite3
adapter complexity.  Cosine similarity is computed in pure Python on retrieval
(same as the existing _cosine_similarity helper) — fine for corpora up to
a few thousand documents.  At much larger scale, swap in a sqlite-vec extension
or a dedicated vector store without changing the public API.

Integration points
------------------
1. main.py lifespan — construct MemoryManager once, store on _state, pass to
   agents that need it:

     from .memory_manager import MemoryManager
     memory_manager = MemoryManager(db_path=..., embed_fn=runtime.embed)
     _state.memory_manager = memory_manager

2. ControllerAgent — replace the per-request in-process MemoryManager with the
   persistent one.  Pass it into the controller constructor:

     controller = ControllerAgent(
         runtime        = runtime,
         agents         = [wiki_agent, research_agent],
         memory_manager = memory_manager,   # new optional param
     )

   Inside ControllerAgent._execute(), replace:
     memory = MemoryManager()
   with:
     memory = self._memory_manager or MemoryManager()

3. WikiAgent — after writing a page, call:
     memory_manager.index_document(path, doc_type="wiki", embed=True)
   This keeps the index current without a full corpus reload.

4. ResearchAgent — replace _load_corpus() with:
     docs = memory_manager.query_corpus(query, max_results=max_src)
   Pass use_embeddings through context as before; the manager handles both
   keyword and embedding retrieval transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
import math
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import bm25
from . import content_safety
from . import wiki_maintenance_log
from .paths import get_backend_root
from .wiki_doc import META_WIKI_FILENAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION   = 15         # increment when schema changes require migration
_EMBEDDING_DIM    = 768        # EmbeddingGemma-300M-4bit output dimension
_EMBEDDING_FORMAT = ">768f"    # big-endian float32 × 768

# Soft cap: keep at most this many conversation_log rows per task_id.
# Older rows are deleted (FIFO) when the cap is exceeded.
_CONV_LOG_CAP_PER_TASK = 200

# embedding_provenance store keys — see _check_embedding_provenance().
_PROVENANCE_STORES: tuple[str, ...] = ("corpus", "episodes", "chat_turns")

# Episodes are expected to stay small enough to re-embed synchronously at
# startup with no bounding/pagination (docs/architecture/
# 16-runtime-backend-layer.md §16.4). This is a "revisit if it fires live"
# tripwire, not a hard limit — re-embedding still runs regardless.
_EPISODES_REEMBED_WARN_ROW_COUNT = 500

# get_chat_turns(mode="semantic") does a full-table cosine scan (chat_turns
# has no token_set column to cheaply pre-filter with, unlike document_index)
# — this just logs a warning past this many rows rather than bounding
# anything, so a "forever"-eviction-preset install with a long history is
# visible in logs rather than silently slow.
_CHAT_TURNS_SEMANTIC_SCAN_WARN_ROW_COUNT = 2000

# Closed set of valid retention_settings.eviction_preset values.
_RETENTION_PRESETS: frozenset[str] = frozenset({"7d", "30d", "90d", "forever"})

# Default assistant_settings.assistant_name when no row exists yet — the
# spoken identity PromptBuilder falls back to, and what the Settings UI
# shows before the user has ever changed it.
_DEFAULT_ASSISTANT_NAME = "Localist"
_ASSISTANT_NAME_MAX_LEN = 60

# Seconds-per-preset for the retention sweep (sweep_expired_memory()). "forever"
# is intentionally absent — it means "no sweep", not "sweep with an infinite TTL".
_RETENTION_PRESET_SECONDS: dict[str, int] = {
    "7d":  7 * 86400,
    "30d": 30 * 86400,
    "90d": 90 * 86400,
}


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _pack_embedding(vector: list[float]) -> bytes:
    """Pack a float list into big-endian float32 bytes."""
    return struct.pack(f">{len(vector)}f", *vector)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack big-endian float32 bytes into a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f">{n}f", blob))


def _normalize_doc_path(doc_path: "Path | str") -> str:
    """
    Normalize a graph_nodes doc_path for storage/lookup.

    Real wiki/raw corpus paths are resolved against the filesystem
    (str(Path(doc_path).resolve())) so doc_path is stable regardless of the
    caller's relative-path spelling. Synthetic non-filesystem identifiers
    (e.g. "episode://<id>", introduced for episode graph nodes) must NOT be
    passed through Path.resolve() — resolve() treats any string as a
    relative filesystem path and prepends the process's current working
    directory, so a synthetic id would silently vary across launch
    contexts (start_localist.sh vs. pytest vs. a shell in a different
    directory) instead of staying a stable, opaque key. Recognized by a
    "://" scheme marker and passed through unchanged.
    """
    doc_path_str = str(doc_path)
    if "://" in doc_path_str:
        return doc_path_str
    return str(Path(doc_path_str).resolve())


def _tokenize(text: str) -> set[str]:
    """Word-level token set for keyword overlap scoring."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _content_hash(text: str) -> str:
    """First 16 hex chars of the SHA-256 digest — used for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _query_hash(query: str, top_n: int, doc_type: str | None) -> str:
    """Cache key for the retrieval_cache table."""
    raw = f"{query}||{top_n}||{doc_type}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Document result type (mirrors _Document in research_agent.py)
# ---------------------------------------------------------------------------

class DocumentResult:
    """
    A document returned by query_corpus().

    Mirrors the public interface of research_agent._Document so that
    ResearchAgent can use either path without changing its internal logic.
    The to_source_dict() method matches the existing shape consumed by the
    Synthesizer's _collect_sources() helper.
    """

    __slots__ = ("name", "path", "doc_type", "content", "relevance_score", "scored_by_embedding")

    def __init__(
        self,
        name:                str,
        path:                Path,
        doc_type:            str,
        content:             str,
        relevance_score:     float = 0.0,
        scored_by_embedding: bool  = True,
    ) -> None:
        self.name                = name
        self.path                = path
        self.doc_type            = doc_type
        self.content             = content
        self.relevance_score     = relevance_score
        # False when relevance_score came from BM25 (bm25.py) rather than
        # embedding cosine similarity — raw BM25 is unbounded and not on a
        # scale comparable to cosine, so a caller applying a fixed absolute
        # relevance floor (e.g. controller_agent.py's `>= 0.55` RAG-source
        # gate) should skip that floor for these and trust BM25's relative
        # ranking instead. Defaults True since most DocumentResult
        # construction sites (get_all_documents(), the retrieval-cache
        # hydration path before this field existed) never scored via BM25
        # in the first place.
        self.scored_by_embedding = scored_by_embedding

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "name":            self.name,
            "path":            str(self.path),
            "type":            self.doc_type,
            "relevance_score": round(self.relevance_score, 4),
        }


class GraphEdgeResult:
    """
    A single graph_edges row, resolved for read-time consumption.

    For outgoing queries, node_title/node_doc_path are None when
    target_resolved is False (the edge points to a page that doesn't
    exist). For incoming queries, target_resolved is always True by
    construction and node_title/node_doc_path are always populated —
    they describe the SOURCE side of the edge in that case (see
    get_backlinks() docstring for why the field names stay fixed but the
    populated side changes per query direction).
    """

    __slots__ = (
        "link_text", "target_path", "target_resolved",
        "node_title", "node_doc_path",
    )

    def __init__(
        self,
        link_text:       str,
        target_path:     str,
        target_resolved: bool,
        node_title:      str | None,
        node_doc_path:   str | None,
    ) -> None:
        self.link_text       = link_text
        self.target_path     = target_path
        self.target_resolved = target_resolved
        self.node_title      = node_title
        self.node_doc_path   = node_doc_path


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    SQLite-backed persistent memory for Localist.

    Responsibilities
    ----------------
    1. Conversation log  — persist and retrieve per-task agent turns.
    2. Document index    — index wiki pages and raw files for fast retrieval.
    3. Corpus retrieval  — return ranked DocumentResults for a query without
                          walking the filesystem.
    4. Embedding cache   — store and reuse 768-dim EmbeddingGemma vectors.
    5. Retrieval cache   — cache query→results mappings, invalidated on writes.

    Parameters
    ----------
    db_path :
        Path to the SQLite file.  Created (with parent dirs) if absent.
        Defaults to  <backend root>/localist_memory.db.
    embed_fn :
        Optional callable that accepts a text string and returns a list[float]
        of length 768.  When provided, embeddings are computed and stored on
        index_document() and used for cosine re-ranking in query_corpus().
        When absent, keyword overlap scoring is used exclusively.
    embedding_model_name :
        Name of the embedding model actually producing embed_fn's vectors
        (e.g. "mlx-community/embeddinggemma-300m-4bit"), or None when no
        embedding source is configured at all. Compared at construction
        time against the 'corpus' and 'episodes' rows in the
        embedding_provenance table (see _check_embedding_provenance()) —
        the same detect-and-fail-safe pattern as Planner's
        _TUNED_EMBEDDING_MODEL guard, applied to stored vectors instead of
        threshold constants. docs/architecture/16-runtime-backend-layer.md
        §16.4.
    """

    def __init__(
        self,
        db_path:               Path | str | None = None,
        embed_fn:               Callable[[str], list[float]] | None = None,
        embedding_model_name:   str | None = None,
    ) -> None:
        if db_path is None:
            db_path = get_backend_root() / "localist_memory.db"
        self._db_path             = Path(db_path)
        self._embed_fn            = embed_fn
        self._embedding_model_name = embedding_model_name
        self._lock                = threading.Lock()

        # Set when the 'corpus' store's stored embedding_provenance model
        # disagrees with self._embedding_model_name (see
        # _check_embedding_provenance()). query_corpus() falls back to
        # keyword-only scoring while this is True — same fail-safe posture
        # as self._embed_fn being None — until POST /memory/reembed
        # (reembed_corpus()) explicitly clears it. Episodes get no
        # equivalent flag: a mismatch there is auto-corrected in place
        # below, never left stale.
        self._corpus_stale = False

        # Same stale-flag posture as _corpus_stale, for the same reason:
        # chat_turns can grow arbitrarily large under the "forever" eviction
        # preset, so — unlike episodes — a provenance mismatch is never
        # auto-corrected at startup. get_chat_turns(mode="semantic") falls
        # back to keyword/FTS scoring while this is True, until
        # reembed_chat_turns() (POST /memory/reembed-chat-turns) clears it.
        self._chat_turns_stale = False

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        if self._embedding_model_name is not None:
            self._check_embedding_provenance()

        logger.info(
            "MemoryManager initialised — db=%s  embeddings=%s  corpus_stale=%s  "
            "chat_turns_stale=%s",
            self._db_path,
            "enabled" if embed_fn else "disabled (keyword-only)",
            self._corpus_stale,
            self._chat_turns_stale,
        )

    @property
    def embed_fn(self) -> Callable[[str], list[float]] | None:
        """
        The embedding function this instance is currently configured with.

        Deliberately read-only and not reassignable from outside the class — see
        docs/architecture/16-runtime-backend-layer.md §16.5: a live runtime-backend switch must
        never change which embedding source is in use, only which chat-inference backend is.
        """
        return self._embed_fn

    # -----------------------------------------------------------------------
    # Database initialisation
    # -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        Open a new connection to the database.

        WAL mode allows concurrent reads while a write is in progress.
        foreign_keys and journal_mode are set on every connection.
        """
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")   # safe with WAL; faster than FULL
        return conn

    def _init_db(self) -> None:
        """
        Create tables and indexes if they do not already exist.

        The schema_version table holds a single row.  If the stored version
        is less than _SCHEMA_VERSION, _migrate() is called before returning.
        This is the extension point for future schema changes.
        """
        with self._lock:
            conn = self._connect()
            try:
                # The schema_version check must happen before any other DDL —
                # the fresh-install script below and _migrate()'s incremental
                # blocks are mutually exclusive paths, so we need to know
                # which one applies before running either. Create just
                # schema_version first (idempotent) so it can be queried.
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version  INTEGER NOT NULL
                    );
                """)

                row = conn.execute("SELECT version FROM schema_version").fetchone()

                if row is None:
                    # Genuinely fresh database — safe to run the full DDL in
                    # one shot, since every column referenced by every index
                    # below (e.g. chat_turns.conversation_id) is created
                    # earlier in this same script.
                    conn.executescript("""
                        CREATE TABLE IF NOT EXISTS schema_version (
                            version  INTEGER NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS conversation_log (
                            id          INTEGER PRIMARY KEY AUTOINCREMENT,
                            task_id     TEXT    NOT NULL,
                            role        TEXT    NOT NULL,
                            content     TEXT    NOT NULL,
                            meta_json   TEXT    NOT NULL DEFAULT '{}',
                            created_at  REAL    NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_conv_task
                            ON conversation_log(task_id, created_at);

                        CREATE TABLE IF NOT EXISTS document_index (
                            id           INTEGER PRIMARY KEY AUTOINCREMENT,
                            name         TEXT    NOT NULL,
                            path         TEXT    NOT NULL UNIQUE,
                            doc_type     TEXT    NOT NULL,
                            content      TEXT    NOT NULL,
                            token_set    TEXT    NOT NULL DEFAULT '',
                            embedding    BLOB    DEFAULT NULL,
                            content_hash TEXT    NOT NULL DEFAULT '',
                            indexed_at   REAL    NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_doc_name
                            ON document_index(name);
                        CREATE INDEX IF NOT EXISTS idx_doc_type
                            ON document_index(doc_type);

                        CREATE TABLE IF NOT EXISTS retrieval_cache (
                            id          INTEGER PRIMARY KEY AUTOINCREMENT,
                            query_hash  TEXT    NOT NULL UNIQUE,
                            top_n       INTEGER NOT NULL,
                            result_json TEXT    NOT NULL,
                            created_at  REAL    NOT NULL,
                            valid       INTEGER NOT NULL DEFAULT 1
                        );
                        CREATE INDEX IF NOT EXISTS idx_cache_hash
                            ON retrieval_cache(query_hash, valid);

                        CREATE TABLE IF NOT EXISTS embedding_provenance (
                            store TEXT NOT NULL PRIMARY KEY,
                            model TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS episodes (
                            id              INTEGER PRIMARY KEY AUTOINCREMENT,
                            episode_type    TEXT    NOT NULL,
                            subject         TEXT    NOT NULL,
                            content         TEXT    NOT NULL,
                            confidence      REAL    NOT NULL DEFAULT 1.0,
                            source          TEXT    NOT NULL,
                            task_id         TEXT,
                            conversation_id TEXT,
                            project_context TEXT,
                            status          TEXT    NOT NULL DEFAULT 'active',
                            created_at      REAL    NOT NULL,
                            last_accessed   REAL,
                            embedding       BLOB
                        );
                        CREATE INDEX IF NOT EXISTS idx_episodes_type_status
                            ON episodes (episode_type, status);
                        CREATE INDEX IF NOT EXISTS idx_episodes_subject
                            ON episodes (subject, status);
                        CREATE INDEX IF NOT EXISTS idx_episodes_project
                            ON episodes (project_context, status);
                        CREATE INDEX IF NOT EXISTS idx_episodes_created
                            ON episodes (created_at);

                        CREATE TABLE IF NOT EXISTS graph_nodes (
                            id              INTEGER PRIMARY KEY AUTOINCREMENT,
                            doc_path        TEXT    NOT NULL UNIQUE,
                            node_type       TEXT,
                            title           TEXT,
                            source_doc_path TEXT    NOT NULL,
                            created_at      REAL    NOT NULL,
                            updated_at      REAL    NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc_path
                            ON graph_nodes(doc_path);

                        CREATE TABLE IF NOT EXISTS graph_edges (
                            id              INTEGER PRIMARY KEY AUTOINCREMENT,
                            source_node_id  INTEGER NOT NULL REFERENCES graph_nodes(id),
                            target_path     TEXT    NOT NULL,
                            target_node_id  INTEGER REFERENCES graph_nodes(id),
                            target_resolved INTEGER NOT NULL DEFAULT 0,
                            link_text       TEXT    NOT NULL,
                            source_doc_path TEXT    NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_graph_edges_source
                            ON graph_edges(source_node_id);
                        CREATE INDEX IF NOT EXISTS idx_graph_edges_target_path
                            ON graph_edges(target_path);
                        CREATE INDEX IF NOT EXISTS idx_graph_edges_resolved
                            ON graph_edges(target_resolved);

                        CREATE TABLE IF NOT EXISTS working_state (
                            mem_key                TEXT    PRIMARY KEY,
                            current_focus          TEXT,
                            open_loops_json        TEXT    NOT NULL DEFAULT '[]',
                            recent_decisions_json  TEXT    NOT NULL DEFAULT '[]',
                            updated_at             REAL    NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS chat_turns (
                            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                            task_id             TEXT    NOT NULL,
                            role                TEXT    NOT NULL,
                            content             TEXT    NOT NULL,
                            sources_json        TEXT    NOT NULL DEFAULT '[]',
                            status_message      TEXT,
                            metadata_json       TEXT    NOT NULL DEFAULT '{}',
                            conversation_id     TEXT    NOT NULL DEFAULT 'legacy',
                            conversation_title  TEXT,
                            embedding           BLOB,
                            created_at          REAL    NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_chat_turns_created
                            ON chat_turns(created_at);
                        CREATE INDEX IF NOT EXISTS idx_chat_turns_task
                            ON chat_turns(task_id);
                        CREATE INDEX IF NOT EXISTS idx_chat_turns_conversation
                            ON chat_turns(conversation_id, created_at);

                        CREATE VIRTUAL TABLE IF NOT EXISTS chat_turns_fts USING fts5(
                            content,
                            content='chat_turns',
                            content_rowid='id'
                        );

                        CREATE TRIGGER IF NOT EXISTS chat_turns_ai AFTER INSERT ON chat_turns BEGIN
                            INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);
                        END;
                        CREATE TRIGGER IF NOT EXISTS chat_turns_ad AFTER DELETE ON chat_turns BEGIN
                            INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) VALUES ('delete', old.id, old.content);
                        END;
                        CREATE TRIGGER IF NOT EXISTS chat_turns_au AFTER UPDATE ON chat_turns BEGIN
                            INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) VALUES ('delete', old.id, old.content);
                            INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);
                        END;

                        CREATE TABLE IF NOT EXISTS retention_settings (
                            id              INTEGER PRIMARY KEY CHECK (id = 1),
                            eviction_preset TEXT
                        );

                        CREATE TABLE IF NOT EXISTS news_preferences (
                            id              INTEGER PRIMARY KEY CHECK (id = 1),
                            home_country    TEXT    NOT NULL DEFAULT 'us',
                            local_query     TEXT,
                            topics_json     TEXT    NOT NULL DEFAULT '[]',
                            updated_at      REAL    NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS news_brief_cache (
                            id              INTEGER PRIMARY KEY CHECK (id = 1),
                            brief_date      TEXT    NOT NULL,
                            content_json    TEXT    NOT NULL,
                            conversation_id TEXT,
                            generated_at    REAL    NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS github_watch_cache (
                            id              INTEGER PRIMARY KEY CHECK (id = 1),
                            content_json    TEXT    NOT NULL,
                            generated_at    REAL    NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS hacker_news_cache (
                            id              INTEGER PRIMARY KEY CHECK (id = 1),
                            content_json    TEXT    NOT NULL,
                            generated_at    REAL    NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS assistant_settings (
                            id              INTEGER PRIMARY KEY CHECK (id = 1),
                            assistant_name  TEXT
                        );

                        CREATE TABLE IF NOT EXISTS pinned_github_repos (
                            id              INTEGER PRIMARY KEY CHECK (id = 1),
                            full_names_json TEXT    NOT NULL DEFAULT '[]',
                            updated_at      REAL    NOT NULL
                        );
                    """)

                    conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (_SCHEMA_VERSION,),
                    )
                    conn.commit()
                    logger.debug("schema_version initialised to %d.", _SCHEMA_VERSION)
                elif row["version"] < _SCHEMA_VERSION:
                    self._migrate(conn, from_version=row["version"])

                # Self-heal: a DB can report schema_version == _SCHEMA_VERSION
                # while still missing a column an earlier migration was
                # supposed to add, if that migration bumped the version
                # without the ALTER actually landing on this DB file — the
                # `row["version"] < _SCHEMA_VERSION` gate above never
                # re-triggers _migrate() once that's happened. Same class of
                # drift already handled for the embedding_provenance table in
                # _check_embedding_provenance(); observed in the wild for
                # chat_turns.embedding specifically (2026-07-22). Runs
                # unconditionally (not gated by embedding_model_name) since
                # add_chat_turn()'s INSERT always names this column
                # regardless of whether an embed_fn is configured — a
                # keyword-only setup hits the same missing-column error
                # otherwise. PRAGMA table_info + ADD COLUMN IF missing is a
                # cheap no-op on a healthy DB.
                cols = {c[1] for c in conn.execute("PRAGMA table_info(chat_turns)").fetchall()}
                if "embedding" not in cols:
                    logger.warning(
                        "MemoryManager: chat_turns missing 'embedding' column "
                        "despite schema_version=%d — self-healing via ALTER TABLE.",
                        _SCHEMA_VERSION,
                    )
                    conn.execute("ALTER TABLE chat_turns ADD COLUMN embedding BLOB;")
                    conn.commit()

                # Same self-heal, same rationale, for news_preferences/
                # news_brief_cache (schema v10, docs/daily-news-brief-plan.md
                # §4) — hit live during this feature's own build (2026-07-22):
                # a `--reload` cycle landed mid-edit between the
                # _SCHEMA_VERSION bump and the matching `_migrate()` block,
                # stamping schema_version=10 on a real dev DB before the
                # CREATE TABLEs had actually landed in the file. CREATE TABLE
                # IF NOT EXISTS is a cheap no-op on a healthy DB.
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS news_preferences (
                        id              INTEGER PRIMARY KEY CHECK (id = 1),
                        home_country    TEXT    NOT NULL DEFAULT 'us',
                        local_query     TEXT,
                        topics_json     TEXT    NOT NULL DEFAULT '[]',
                        updated_at      REAL    NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS news_brief_cache (
                        id              INTEGER PRIMARY KEY CHECK (id = 1),
                        brief_date      TEXT    NOT NULL,
                        content_json    TEXT    NOT NULL,
                        conversation_id TEXT,
                        generated_at    REAL    NOT NULL
                    );
                """)
                conn.commit()

                # Same self-heal, same rationale, for github_watch_cache
                # (schema v11, GitHub Watch Feed).
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS github_watch_cache (
                        id              INTEGER PRIMARY KEY CHECK (id = 1),
                        content_json    TEXT    NOT NULL,
                        generated_at    REAL    NOT NULL
                    );
                """)
                conn.commit()

                # Same self-heal, same rationale, for hacker_news_cache
                # (schema v12, Hacker News Live Feed panel).
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS hacker_news_cache (
                        id              INTEGER PRIMARY KEY CHECK (id = 1),
                        content_json    TEXT    NOT NULL,
                        generated_at    REAL    NOT NULL
                    );
                """)
                conn.commit()

                # Same self-heal, same rationale, for the chat_history_settings
                # → retention_settings rename (schema v13, global retention TTL).
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()}
                if "chat_history_settings" in tables and "retention_settings" not in tables:
                    conn.execute("ALTER TABLE chat_history_settings RENAME TO retention_settings")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS retention_settings (
                        id              INTEGER PRIMARY KEY CHECK (id = 1),
                        eviction_preset TEXT
                    );
                """)
                if "episodes" in tables:
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes (created_at)"
                    )
                conn.commit()

                # Same self-heal, same rationale, for pinned_github_repos
                # (schema v15, pinned GitHub repos independent of Watch).
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS pinned_github_repos (
                        id              INTEGER PRIMARY KEY CHECK (id = 1),
                        full_names_json TEXT    NOT NULL DEFAULT '[]',
                        updated_at      REAL    NOT NULL
                    );
                """)
                conn.commit()

            finally:
                conn.close()

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """
        Apply incremental schema migrations.

        Called only when the stored schema_version < _SCHEMA_VERSION.
        Add elif blocks here for each new version.
        """
        logger.info(
            "Migrating MemoryManager schema from v%d to v%d.",
            from_version,
            _SCHEMA_VERSION,
        )

        if from_version < 2:
            logger.info("Applying migration v1→v2: creating episodes table.")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_type    TEXT    NOT NULL,
                    subject         TEXT    NOT NULL,
                    content         TEXT    NOT NULL,
                    confidence      REAL    NOT NULL DEFAULT 1.0,
                    source          TEXT    NOT NULL,
                    task_id         TEXT,
                    conversation_id TEXT,
                    project_context TEXT,
                    status          TEXT    NOT NULL DEFAULT 'active',
                    created_at      REAL    NOT NULL,
                    last_accessed   REAL,
                    embedding       BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_type_status
                    ON episodes (episode_type, status);
                CREATE INDEX IF NOT EXISTS idx_episodes_subject
                    ON episodes (subject, status);
                CREATE INDEX IF NOT EXISTS idx_episodes_project
                    ON episodes (project_context, status);
            """)

        if from_version < 3:
            logger.info("Applying migration v2→v3: creating graph_nodes and graph_edges tables.")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_path        TEXT    NOT NULL UNIQUE,
                    node_type       TEXT,
                    title           TEXT,
                    source_doc_path TEXT    NOT NULL,
                    created_at      REAL    NOT NULL,
                    updated_at      REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc_path
                    ON graph_nodes(doc_path);

                CREATE TABLE IF NOT EXISTS graph_edges (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_node_id  INTEGER NOT NULL REFERENCES graph_nodes(id),
                    target_path     TEXT    NOT NULL,
                    target_node_id  INTEGER REFERENCES graph_nodes(id),
                    target_resolved INTEGER NOT NULL DEFAULT 0,
                    link_text       TEXT    NOT NULL,
                    source_doc_path TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_graph_edges_source
                    ON graph_edges(source_node_id);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_target_path
                    ON graph_edges(target_path);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_resolved
                    ON graph_edges(target_resolved);
            """)

        if from_version < 4:
            logger.info("Applying migration v3→v4: creating working_state table.")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS working_state (
                    mem_key                TEXT    PRIMARY KEY,
                    current_focus          TEXT,
                    open_loops_json        TEXT    NOT NULL DEFAULT '[]',
                    recent_decisions_json  TEXT    NOT NULL DEFAULT '[]',
                    turn_summaries_json    TEXT    NOT NULL DEFAULT '[]',
                    updated_at             REAL    NOT NULL
                );
            """)

        if from_version < 5:
            logger.info("Applying migration v4→v5: dropping turn_summaries_json column.")
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(working_state)"
            ).fetchall()}
            if "turn_summaries_json" in cols:
                conn.executescript(
                    "ALTER TABLE working_state DROP COLUMN turn_summaries_json;"
                )

        if from_version < 6:
            logger.info("Applying migration v5→v6: creating chat_turns, chat_turns_fts, and chat_history_settings.")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id         TEXT    NOT NULL,
                    role            TEXT    NOT NULL,
                    content         TEXT    NOT NULL,
                    sources_json    TEXT    NOT NULL DEFAULT '[]',
                    status_message  TEXT,
                    metadata_json   TEXT    NOT NULL DEFAULT '{}',
                    created_at      REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chat_turns_created
                    ON chat_turns(created_at);
                CREATE INDEX IF NOT EXISTS idx_chat_turns_task
                    ON chat_turns(task_id);

                CREATE VIRTUAL TABLE IF NOT EXISTS chat_turns_fts USING fts5(
                    content,
                    content='chat_turns',
                    content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS chat_turns_ai AFTER INSERT ON chat_turns BEGIN
                    INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS chat_turns_ad AFTER DELETE ON chat_turns BEGIN
                    INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) VALUES ('delete', old.id, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS chat_turns_au AFTER UPDATE ON chat_turns BEGIN
                    INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) VALUES ('delete', old.id, old.content);
                    INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);
                END;

                CREATE TABLE IF NOT EXISTS chat_history_settings (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    eviction_preset TEXT
                );
            """)

        if from_version < 7:
            logger.info("Applying migration v6→v7: adding conversation_id/conversation_title to chat_turns.")
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(chat_turns)"
            ).fetchall()}
            if "conversation_id" not in cols:
                conn.executescript(
                    "ALTER TABLE chat_turns ADD COLUMN conversation_id TEXT NOT NULL DEFAULT 'legacy';"
                )
            if "conversation_title" not in cols:
                conn.executescript(
                    "ALTER TABLE chat_turns ADD COLUMN conversation_title TEXT;"
                )
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_chat_turns_conversation
                    ON chat_turns(conversation_id, created_at);
            """)

        if from_version < 8:
            logger.info("Applying migration v7→v8: creating embedding_provenance table.")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS embedding_provenance (
                    store TEXT NOT NULL PRIMARY KEY,
                    model TEXT NOT NULL
                );
            """)

        if from_version < 9:
            logger.info("Applying migration v8→v9: adding embedding column to chat_turns.")
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(chat_turns)"
            ).fetchall()}
            if "embedding" not in cols:
                conn.executescript(
                    "ALTER TABLE chat_turns ADD COLUMN embedding BLOB;"
                )

        if from_version < 10:
            logger.info(
                "Applying migration v9→v10: creating news_preferences and "
                "news_brief_cache tables."
            )
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS news_preferences (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    home_country    TEXT    NOT NULL DEFAULT 'us',
                    local_query     TEXT,
                    topics_json     TEXT    NOT NULL DEFAULT '[]',
                    updated_at      REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS news_brief_cache (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    brief_date      TEXT    NOT NULL,
                    content_json    TEXT    NOT NULL,
                    conversation_id TEXT,
                    generated_at    REAL    NOT NULL
                );
            """)

        if from_version < 11:
            logger.info("Applying migration v10→v11: creating github_watch_cache table.")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS github_watch_cache (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    content_json    TEXT    NOT NULL,
                    generated_at    REAL    NOT NULL
                );
            """)

        if from_version < 12:
            logger.info("Applying migration v11→v12: creating hacker_news_cache table.")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS hacker_news_cache (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    content_json    TEXT    NOT NULL,
                    generated_at    REAL    NOT NULL
                );
            """)

        if from_version < 13:
            logger.info(
                "Applying migration v12→v13: renaming chat_history_settings to "
                "retention_settings (now governs both chat_turns and episodes "
                "TTL) and indexing episodes(created_at) for the retention sweep."
            )
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()}
            if "chat_history_settings" in tables and "retention_settings" not in tables:
                conn.execute("ALTER TABLE chat_history_settings RENAME TO retention_settings")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retention_settings (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    eviction_preset TEXT
                );
            """)
            if "episodes" in tables:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes (created_at)"
                )

        if from_version < 14:
            logger.info(
                "Applying migration v13→v14: creating assistant_settings table "
                "(configurable assistant name, default %r).",
                _DEFAULT_ASSISTANT_NAME,
            )
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS assistant_settings (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    assistant_name  TEXT
                );
            """)

        if from_version < 15:
            logger.info(
                "Applying migration v14→v15: creating pinned_github_repos table."
            )
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pinned_github_repos (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    full_names_json TEXT    NOT NULL DEFAULT '[]',
                    updated_at      REAL    NOT NULL
                );
            """)

        conn.execute(
            "UPDATE schema_version SET version = ?", (_SCHEMA_VERSION,)
        )
        conn.commit()
        logger.info("Migration complete. Schema is now v%d.", _SCHEMA_VERSION)

    # -----------------------------------------------------------------------
    # Embedding provenance  (docs/architecture/16-runtime-backend-layer.md §16.4)
    #
    # Same detect-and-fail-safe pattern as Planner's _TUNED_EMBEDDING_MODEL
    # guard, applied to stored vectors: cosine similarity between a query
    # embedding and a stored document/episode/turn embedding is only
    # meaningful if both came from the same model's geometry.
    # embedding_provenance tracks, per store ('corpus' | 'episodes' |
    # 'chat_turns'), which model actually produced the vectors currently on
    # disk.
    # -----------------------------------------------------------------------

    def _check_embedding_provenance(self) -> None:
        """
        Called once at construction time (only when embedding_model_name is
        not None — no embedding source configured means nothing to check).

        For each store in _PROVENANCE_STORES, compares the stored
        embedding_provenance row (if any) against self._embedding_model_name:

          - No row, and the store has zero embedded rows: nothing to compare
            against yet — defer. 'corpus' seeds its own row the first time
            index_document() actually writes an embedding (see
            _maybe_seed_corpus_provenance()); 'episodes' seeds on the next
            process restart that finds embedded rows present (or whenever
            episode-embedding provenance tracking is added to the writer
            path — not yet, so it seeds one boot later than corpus, at worst);
            'chat_turns' seeds the first time add_chat_turn() writes an
            embedding.
          - No row, but the store already has embedded rows: the pre-
            existing-database migration case — provenance tracking shipped
            after data was already embedded under whatever model was active
            at the time. Seeded silently, no warning, no re-embed. Treating
            this as a mismatch would trigger a surprise re-embed for every
            existing user on their very next boot.
          - Row present and it matches: no-op.
          - Row present and it disagrees: a genuine mismatch. Handled per-
            store below — 'corpus' and 'chat_turns' are flagged stale
            (manual refresh via reembed_corpus()/reembed_chat_turns()); both
            can grow arbitrarily large, so neither is ever re-embedded
            automatically. 'episodes' is auto-corrected immediately (small,
            bounded cost).
        """
        with self._lock:
            conn = self._connect()
            try:
                # Self-heal: a DB can report schema_version == _SCHEMA_VERSION
                # while missing this table if an earlier buggy migration run
                # bumped the version without actually creating it — _init_db's
                # `row["version"] < _SCHEMA_VERSION` gate never re-triggers
                # _migrate() in that state, so this guard is the only recovery
                # path. IF NOT EXISTS makes it a no-op on a healthy DB.
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS embedding_provenance (
                        store TEXT NOT NULL PRIMARY KEY,
                        model TEXT NOT NULL
                    );
                """)
                conn.commit()

                for store in _PROVENANCE_STORES:
                    if store == "corpus":
                        table = "document_index"
                    elif store == "chat_turns":
                        table = "chat_turns"
                    else:
                        table = "episodes"
                    row = conn.execute(
                        "SELECT model FROM embedding_provenance WHERE store = ?",
                        (store,),
                    ).fetchone()

                    if row is None:
                        has_data = conn.execute(
                            f"SELECT 1 FROM {table} WHERE embedding IS NOT NULL LIMIT 1"
                        ).fetchone() is not None
                        if has_data:
                            conn.execute(
                                "INSERT INTO embedding_provenance (store, model) VALUES (?, ?)",
                                (store, self._embedding_model_name),
                            )
                            conn.commit()
                            logger.info(
                                "MemoryManager: seeded embedding_provenance for %r store "
                                "at %r — pre-existing embedded data found with no "
                                "provenance row (first boot after provenance tracking "
                                "shipped); trusted silently, no re-embed triggered.",
                                store, self._embedding_model_name,
                            )
                        continue

                    stored_model = row["model"]
                    if stored_model == self._embedding_model_name:
                        continue

                    if store == "corpus":
                        logger.warning(
                            "MemoryManager: corpus embeddings were produced by %r, but "
                            "the active embedding model is now %r — disabling "
                            "embedding-based corpus re-ranking (falling back to "
                            "keyword-only scoring) until POST /memory/reembed is run. "
                            "Cosine similarity is not portable across embedding models; "
                            "see docs/architecture/16-runtime-backend-layer.md §16.4.",
                            stored_model, self._embedding_model_name,
                        )
                        self._corpus_stale = True
                        self._invalidate_cache(conn)
                        conn.commit()
                    elif store == "chat_turns":
                        logger.warning(
                            "MemoryManager: chat_turns embeddings were produced by %r, "
                            "but the active embedding model is now %r — disabling "
                            "embedding-based chat history search (falling back to "
                            "keyword/FTS scoring) until POST /memory/reembed-chat-turns "
                            "is run. Cosine similarity is not portable across embedding "
                            "models; see docs/architecture/16-runtime-backend-layer.md "
                            "§16.4.",
                            stored_model, self._embedding_model_name,
                        )
                        self._chat_turns_stale = True
                        conn.commit()
                    else:
                        self._reembed_episodes_locked(conn, stored_model)
            finally:
                conn.close()

    def _reembed_episodes_locked(self, conn: sqlite3.Connection, stored_model: str) -> None:
        """
        Re-embed every episodes row in place with the currently active
        embed_fn, then update the 'episodes' provenance row.

        Must be called with self._lock already held (only caller today is
        _check_embedding_provenance()) — threading.Lock is not reentrant.

        Provenance is only advanced to the new model if every row succeeded;
        a partial failure leaves both the surviving stale rows and the old
        provenance value in place (so the mismatch is detected again, and
        retried, on the next boot) rather than claiming a clean re-embed
        that didn't fully happen.
        """
        rows = conn.execute("SELECT id, subject, content FROM episodes").fetchall()
        if len(rows) > _EPISODES_REEMBED_WARN_ROW_COUNT:
            logger.warning(
                "MemoryManager: episodes re-embed triggered for %d rows — episodes "
                "was assumed to stay small enough to re-embed synchronously at "
                "startup with no bounding/pagination; that assumption may need "
                "revisiting.", len(rows),
            )
        logger.warning(
            "MemoryManager: episodes embeddings were produced by %r, but the "
            "active embedding model is now %r — auto re-embedding %d episode "
            "row(s) now (small, bounded cost; see docs/architecture/"
            "16-runtime-backend-layer.md §16.4).",
            stored_model, self._embedding_model_name, len(rows),
        )

        reembedded = 0
        for row in rows:
            try:
                vector = self._embed_fn(f"{row['subject']}. {row['content']}")
                blob   = _pack_embedding(vector)
            except Exception as exc:
                logger.warning(
                    "MemoryManager: episodes re-embed failed for id=%d (%s) — "
                    "leaving its previous embedding in place.", row["id"], exc,
                )
                continue
            conn.execute("UPDATE episodes SET embedding = ? WHERE id = ?", (blob, row["id"]))
            reembedded += 1

        if rows and reembedded < len(rows):
            logger.warning(
                "MemoryManager: episodes re-embed incomplete (%d/%d succeeded) — "
                "leaving 'episodes' provenance at %r; will retry on next boot.",
                reembedded, len(rows), stored_model,
            )
            conn.commit()
            return

        conn.execute(
            """
            INSERT INTO embedding_provenance (store, model) VALUES ('episodes', ?)
            ON CONFLICT(store) DO UPDATE SET model = excluded.model
            """,
            (self._embedding_model_name,),
        )
        conn.commit()
        logger.info(
            "MemoryManager: episodes re-embed complete — %d/%d row(s) updated to %r.",
            reembedded, len(rows), self._embedding_model_name,
        )

    def _maybe_seed_corpus_provenance(self, conn: sqlite3.Connection) -> None:
        """
        Seed the 'corpus' embedding_provenance row the first time this
        process actually writes a corpus embedding, if no row exists yet.

        Cheap (INSERT OR IGNORE) and safe to call on every embedded write
        from index_document() — only the very first call after a fresh DB
        (or after _check_embedding_provenance()'s migration-seed case
        already handled it) actually inserts a row. Must be called with
        self._lock already held, inside the same transaction as the write.
        """
        if self._embedding_model_name is None:
            return
        conn.execute(
            "INSERT OR IGNORE INTO embedding_provenance (store, model) VALUES ('corpus', ?)",
            (self._embedding_model_name,),
        )

    def _maybe_seed_chat_turns_provenance(self, conn: sqlite3.Connection) -> None:
        """
        Seed the 'chat_turns' embedding_provenance row the first time this
        process actually writes a chat_turns embedding, if no row exists yet.

        Same pattern as _maybe_seed_corpus_provenance() — cheap
        (INSERT OR IGNORE), safe to call on every embedded write from
        add_chat_turn(). Must be called with self._lock already held,
        inside the same transaction as the write.
        """
        if self._embedding_model_name is None:
            return
        conn.execute(
            "INSERT OR IGNORE INTO embedding_provenance (store, model) VALUES ('chat_turns', ?)",
            (self._embedding_model_name,),
        )

    def reembed_corpus(self) -> dict[str, Any]:
        """
        Manually re-embed every document_index row with the currently
        active embed_fn, update the 'corpus' embedding_provenance row,
        flush the retrieval cache, and clear self._corpus_stale.

        The explicit, potentially-expensive counterpart to episodes'
        automatic startup re-embed — wiki/raw corpora can be arbitrarily
        large, so unlike episodes this is never triggered automatically on
        a detected mismatch (docs/architecture/16-runtime-backend-layer.md
        §16.4). Exposed as POST /memory/reembed.

        Idempotent — safe to call whether or not the corpus is currently
        flagged stale (a "just refresh it" operation, not gated on the flag).

        Raises
        ------
        RuntimeError
            If no embed_fn is configured — there is nothing to re-embed with.
        """
        if self._embed_fn is None:
            raise RuntimeError("reembed_corpus: no embed_fn configured — cannot re-embed.")

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT id, content FROM document_index").fetchall()
                reembedded = 0
                for row in rows:
                    try:
                        vec = self._embed_fn(row["content"][:500])
                    except Exception as exc:
                        logger.warning(
                            "reembed_corpus: embed failed for document id=%d (%s) — "
                            "leaving its previous embedding in place.", row["id"], exc,
                        )
                        continue
                    if len(vec) != _EMBEDDING_DIM:
                        logger.warning(
                            "reembed_corpus: embed returned dim=%d, expected %d for "
                            "document id=%d — skipping.",
                            len(vec), _EMBEDDING_DIM, row["id"],
                        )
                        continue
                    conn.execute(
                        "UPDATE document_index SET embedding = ? WHERE id = ?",
                        (_pack_embedding(vec), row["id"]),
                    )
                    reembedded += 1

                self._invalidate_cache(conn)
                conn.execute(
                    """
                    INSERT INTO embedding_provenance (store, model) VALUES ('corpus', ?)
                    ON CONFLICT(store) DO UPDATE SET model = excluded.model
                    """,
                    (self._embedding_model_name,),
                )
                conn.commit()
            finally:
                conn.close()

        self._corpus_stale = False
        logger.info(
            "reembed_corpus: complete — %d/%d document(s) re-embedded to %r.",
            reembedded, len(rows), self._embedding_model_name,
        )
        return {
            "reembedded": reembedded,
            "total":      len(rows),
            "model":      self._embedding_model_name,
        }

    def reembed_chat_turns(self) -> dict[str, Any]:
        """
        Manually re-embed every chat_turns row with the currently active
        embed_fn, update the 'chat_turns' embedding_provenance row, and
        clear self._chat_turns_stale.

        Same rationale as reembed_corpus(): chat history can grow
        arbitrarily large (the "forever" eviction preset), so unlike
        episodes this is never triggered automatically on a detected
        mismatch. Exposed as POST /memory/reembed-chat-turns.

        Idempotent — safe to call whether or not chat_turns is currently
        flagged stale.

        Raises
        ------
        RuntimeError
            If no embed_fn is configured — there is nothing to re-embed with.
        """
        if self._embed_fn is None:
            raise RuntimeError("reembed_chat_turns: no embed_fn configured — cannot re-embed.")

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT id, content FROM chat_turns").fetchall()
                reembedded = 0
                for row in rows:
                    try:
                        vec = self._embed_fn(row["content"][:500])
                    except Exception as exc:
                        logger.warning(
                            "reembed_chat_turns: embed failed for chat_turns id=%d (%s) — "
                            "leaving its previous embedding in place.", row["id"], exc,
                        )
                        continue
                    if len(vec) != _EMBEDDING_DIM:
                        logger.warning(
                            "reembed_chat_turns: embed returned dim=%d, expected %d for "
                            "chat_turns id=%d — skipping.",
                            len(vec), _EMBEDDING_DIM, row["id"],
                        )
                        continue
                    conn.execute(
                        "UPDATE chat_turns SET embedding = ? WHERE id = ?",
                        (_pack_embedding(vec), row["id"]),
                    )
                    reembedded += 1

                conn.execute(
                    """
                    INSERT INTO embedding_provenance (store, model) VALUES ('chat_turns', ?)
                    ON CONFLICT(store) DO UPDATE SET model = excluded.model
                    """,
                    (self._embedding_model_name,),
                )
                conn.commit()
            finally:
                conn.close()

        self._chat_turns_stale = False
        logger.info(
            "reembed_chat_turns: complete — %d/%d row(s) re-embedded to %r.",
            reembedded, len(rows), self._embedding_model_name,
        )
        return {
            "reembedded": reembedded,
            "total":      len(rows),
            "model":      self._embedding_model_name,
        }

    # -----------------------------------------------------------------------
    # Conversation log  (ControllerAgent / Synthesizer API)
    # -----------------------------------------------------------------------

    def add(
        self,
        role:     str,
        content:  str,
        metadata: dict[str, Any] | None = None,
        task_id:  str = "global",
    ) -> None:
        """
        Append one entry to the conversation log.

        Parameters
        ----------
        role :
            "user" | "agent" | "system"
        content :
            The text content of the entry.
        metadata :
            Optional dict stored as JSON.  Useful for agent name, subtask_id, etc.
        task_id :
            Groups entries by task.  Use the task's UUID.  Defaults to "global"
            for entries that are not task-scoped (e.g. system messages).
        """
        meta_json = json.dumps(metadata or {})
        now       = time.time()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO conversation_log
                        (task_id, role, content, meta_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (task_id, role, content, meta_json, now),
                )
                conn.commit()
                self._evict_conversation_log(conn, task_id)
            finally:
                conn.close()

    def add_agent_result(
        self,
        result:  Any,           # AgentResult — typed Any to avoid circular import
        task_id: str = "global",
    ) -> None:
        """
        Convenience wrapper matching the existing MemoryManager API.

        Serialises result.output as the content string, preserving the same
        format the Synthesizer's format_for_prompt() already expects.
        """
        self.add(
            role     = "agent",
            content  = str(result.output),
            metadata = {
                "agent":      result.agent_name,
                "subtask_id": result.subtask_id,
            },
            task_id  = task_id,
        )

    def get_context_window(
        self,
        task_id:    str            = "global",
        limit:      int | None     = 50,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent ``limit`` entries for a task, optionally
        capped by a token budget.

        Parameters
        ----------
        task_id :
            Groups entries by task. Defaults to "global".
        limit :
            Maximum number of rows to fetch from the DB before token
            trimming is applied. Defaults to 50. None means unbounded —
            every row for `task_id` is fetched (used by the cloud
            ContextProfile, context_profile.py, which removes the
            turn-count cap entirely and relies on `max_tokens` alone).
        max_tokens :
            When provided, entries are trimmed (oldest first) until the
            total estimated token count of all remaining entries is at or
            below this value. Token count is estimated as
            ``len(content) // 4`` per entry (1 token ≈ 4 characters).
            Truncation never cuts mid-entry.
            When None (default), no token trimming is applied and all
            ``limit`` entries are returned as before.

        Returns
        -------
        list[dict[str, Any]]
            Dicts with keys: role, content, metadata.
            Ordered chronologically (oldest first).
        """
        conn = self._connect()
        try:
            if limit is None:
                rows = conn.execute(
                    """
                    SELECT role, content, meta_json
                    FROM   conversation_log
                    WHERE  task_id = ?
                    ORDER  BY created_at DESC
                    """,
                    (task_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT role, content, meta_json
                    FROM   conversation_log
                    WHERE  task_id = ?
                    ORDER  BY created_at DESC
                    LIMIT  ?
                    """,
                    (task_id, limit),
                ).fetchall()
        finally:
            conn.close()

        # Reverse to chronological order (oldest first)
        entries = [
            {
                "role":     row["role"],
                "content":  row["content"],
                "metadata": json.loads(row["meta_json"]),
            }
            for row in reversed(rows)
        ]

        # Apply token ceiling: drop oldest entries until budget is met
        if max_tokens is not None:
            while entries:
                total_chars = sum(len(e["content"]) for e in entries)
                estimated_tokens = total_chars // 4
                if estimated_tokens <= max_tokens:
                    break
                entries.pop(0)   # drop the oldest entry

        return entries

    def format_for_prompt(
        self,
        task_id: str = "global",
        limit:   int = 50,
    ) -> str:
        """
        Flatten the conversation log into a single prompt-ready string.

        Matches the existing MemoryManager.format_for_prompt() output exactly
        so the Synthesizer needs no changes.
        """
        entries = self.get_context_window(task_id=task_id, limit=limit)
        return "\n".join(
            f"[{e['role'].upper()}] {e['content']}" for e in entries
        )

    def clear(self, task_id: str | None = None) -> None:
        """
        Delete conversation log entries.

        If task_id is given, only entries for that task are deleted.
        If task_id is None, ALL conversation log entries are deleted.
        The document index is NOT affected by clear().
        """
        with self._lock:
            conn = self._connect()
            try:
                if task_id is not None:
                    conn.execute(
                        "DELETE FROM conversation_log WHERE task_id = ?",
                        (task_id,),
                    )
                    logger.debug("Cleared conversation log for task_id=%s.", task_id)
                else:
                    conn.execute("DELETE FROM conversation_log")
                    logger.debug("Cleared all conversation log entries.")
                conn.commit()
            finally:
                conn.close()

    def _evict_conversation_log(
        self,
        conn:    sqlite3.Connection,
        task_id: str,
    ) -> None:
        """
        Delete the oldest rows for task_id when the per-task cap is exceeded.

        Called inside an existing write transaction — no lock re-entry needed.
        Silently skips if the count is within the cap.
        """
        count = conn.execute(
            "SELECT COUNT(*) FROM conversation_log WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0]

        if count > _CONV_LOG_CAP_PER_TASK:
            excess = count - _CONV_LOG_CAP_PER_TASK
            conn.execute(
                """
                DELETE FROM conversation_log
                WHERE  id IN (
                    SELECT id FROM conversation_log
                    WHERE  task_id = ?
                    ORDER  BY created_at ASC
                    LIMIT  ?
                )
                """,
                (task_id, excess),
            )
            logger.debug(
                "Evicted %d old conversation_log rows for task_id=%s.",
                excess, task_id,
            )

    # -----------------------------------------------------------------------
    # Chat turns  (Chat History Tab — write + read/list; no eviction yet)
    # -----------------------------------------------------------------------

    def add_chat_turn(
        self,
        task_id:            str,
        role:               str,
        content:            str,
        conversation_id:    str,
        sources:            list[dict[str, Any]] | None = None,
        status_message:     str | None = None,
        metadata:           dict[str, Any] | None = None,
        conversation_title: str | None = None,
    ) -> None:
        """
        Append one entry to the chat_turns table.

        Parameters
        ----------
        task_id :
            Groups turns by task.
        role :
            "user" | "assistant"
        content :
            The text content of the turn.
        conversation_id :
            Groups turns by conversation.
        sources :
            Optional list of source dicts, stored as JSON. Defaults to [].
        status_message :
            Optional final status string for this turn. Not populated by any
            current call site — reserved for future use.
        metadata :
            Optional dict stored as JSON. Defaults to {}.
        conversation_title :
            Optional human-readable title for the conversation.

        Embeds `content` (truncated to 500 chars, same convention as
        index_document()/reembed_corpus()) when embed_fn is configured —
        this is what get_chat_turns(mode="semantic") searches over. Embed
        failures degrade to a NULL embedding for this row rather than
        blocking the write; the row is still findable via keyword/FTS.
        """
        sources_json  = json.dumps(sources or [])
        metadata_json = json.dumps(metadata or {})
        now           = time.time()

        embedding_blob: bytes | None = None
        if self._embed_fn is not None and content:
            try:
                vec = self._embed_fn(content[:500])
                if len(vec) == _EMBEDDING_DIM:
                    embedding_blob = _pack_embedding(vec)
                else:
                    logger.warning(
                        "add_chat_turn: embed returned dim=%d, expected %d — skipping.",
                        len(vec), _EMBEDDING_DIM,
                    )
            except Exception as exc:
                logger.warning("add_chat_turn: embed failed for task_id=%s — %s", task_id, exc)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO chat_turns
                        (task_id, role, content, sources_json, status_message,
                         metadata_json, conversation_id, conversation_title,
                         embedding, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (task_id, role, content, sources_json, status_message,
                     metadata_json, conversation_id, conversation_title,
                     embedding_blob, now),
                )
                if embedding_blob is not None:
                    self._maybe_seed_chat_turns_provenance(conn)
                conn.commit()
            finally:
                conn.close()

    def mark_diff_applied(self, task_id: str, page_name: str) -> bool:
        """
        Mark one entry in a chat_turn's persisted metadata_json
        pending_diffs list as applied, in place — the review-then-apply
        wiki-diff UI's only persisted state transition (Discard is
        deliberately ephemeral/client-only; no backend call happens for it).
        Called after POST /wiki/apply-diff succeeds, so a page reload shows
        "applied" rather than re-offering Apply on a diff already written
        to disk.

        Looks up the most recent assistant chat_turn row for `task_id`
        (there is exactly one per task_id — see main.py's _persist_chat_turn
        call sites), and sets pending_diffs[i]["status"] = "applied" for
        every entry whose page_name matches.

        Returns True if a matching row and pending_diffs entry were found
        and updated, False otherwise (no matching row, no pending_diffs key,
        or no matching page_name — non-fatal either way; the disk write
        itself already happened independently of this call).
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, metadata_json FROM chat_turns
                    WHERE task_id = ? AND role = 'assistant'
                    ORDER BY id DESC LIMIT 1
                    """,
                    (task_id,),
                ).fetchone()
                if row is None:
                    return False

                metadata = json.loads(row["metadata_json"])
                diffs = metadata.get("pending_diffs")
                if not diffs:
                    return False

                found = False
                for entry in diffs:
                    if entry.get("page_name") == page_name:
                        entry["status"] = "applied"
                        found = True

                if not found:
                    return False

                conn.execute(
                    "UPDATE chat_turns SET metadata_json = ? WHERE id = ?",
                    (json.dumps(metadata), row["id"]),
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def get_chat_turns(
        self,
        query:           str | None = None,
        limit:           int = 50,
        offset:          int = 0,
        conversation_id: str | None = None,
        mode:            str = "keyword",
        min_score:       float = 0.3,
        date_from:       float | None = None,
        date_to:         float | None = None,
        has_tool_result: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Return a page of chat_turns rows plus the total matching count.

        Read-only — no eviction, no writes.

        Parameters
        ----------
        query :
            Optional search string. When None or empty, returns an
            unfiltered page ordered by created_at DESC (mode is irrelevant
            in this case). When non-empty and mode="keyword" (default),
            searches chat_turns_fts and orders by bm25() ascending (best
            match first — FTS5's bm25() returns more-negative scores for
            better matches, so ascending puts the best match on top). The
            raw query is wrapped as a single quoted FTS5 phrase (embedded
            double quotes doubled) so punctuation-heavy input — "what's",
            hyphens, FTS5 operator keywords — is treated as literal text
            instead of MATCH syntax. If the sanitised query still raises
            an FTS5 syntax error, an empty result set is returned rather
            than propagating the exception.
        limit :
            Max rows to return (capped at 200, matching list_episodes()).
        offset :
            Row offset for pagination.
        conversation_id :
            Optional filter restricting results to a single conversation.
        mode :
            "keyword" (default) or "semantic". "semantic" scores every
            (optionally conversation-scoped) row against `query` via cosine
            similarity over the stored `embedding` column — see
            _get_chat_turns_semantic(). Silently falls back to "keyword"
            when `query` is empty, embed_fn is unavailable, or
            self._chat_turns_stale is True — same fail-safe posture as
            query_corpus()/EpisodicMemoryReader.by_similarity().
        min_score :
            "semantic" mode only. Rows scoring below this cosine threshold
            are excluded. Default 0.3, matching the low-end of
            by_similarity()'s documented guidance for cosine-scale
            thresholds.
        date_from / date_to :
            Optional inclusive `created_at` bounds (unix timestamp,
            seconds) for the Episode Browsing UI's date-range filter.
        has_tool_result :
            When True, restricts to turns whose metadata_json carries a
            renderable tool artifact — a "chart", "pending_diffs", or
            "workflow_id" key (see controller_agent.py's
            _build_conversational_result()/_build_wiki_diff_result(),
            the only three writers of those keys). A cheap LIKE-based
            substring check on the stored JSON text, not a real JSON
            predicate — sqlite's json1 extension is not assumed to be
            compiled in, and these three key names are distinctive enough
            in practice not to need it.

        Returns
        -------
        tuple[list[dict], int]
            (rows, total_count). total_count reflects the full matching
            set (unfiltered table count, FTS match count, or
            score-above-threshold count in semantic mode) — not just the
            current page. Each row dict has keys: id, task_id, role,
            content, sources, status_message, metadata, conversation_id,
            conversation_title, created_at, and (semantic mode only) score.
        """
        limit = min(limit, 200)

        if (
            mode == "semantic"
            and query
            and self._embed_fn is not None
            and not self._chat_turns_stale
        ):
            try:
                return self._get_chat_turns_semantic(
                    query, limit, offset, conversation_id, min_score,
                    date_from, date_to, has_tool_result,
                )
            except Exception as exc:
                logger.warning(
                    "get_chat_turns: semantic search failed (%s) — "
                    "falling back to keyword search.", exc,
                )

        # Extra filter clauses shared by both the unfiltered and FTS
        # branches below — `col` is the column prefix ("" for the plain
        # table, "c." for the chat_turns_fts JOIN).
        def _extra_clauses(col: str) -> tuple[list[str], list[Any]]:
            clauses: list[str] = []
            params:  list[Any] = []
            if conversation_id is not None:
                clauses.append(f"{col}conversation_id = ?")
                params.append(conversation_id)
            if date_from is not None:
                clauses.append(f"{col}created_at >= ?")
                params.append(date_from)
            if date_to is not None:
                clauses.append(f"{col}created_at <= ?")
                params.append(date_to)
            if has_tool_result:
                clauses.append(
                    f'({col}metadata_json LIKE \'%"chart"%\' OR '
                    f'{col}metadata_json LIKE \'%"pending_diffs"%\' OR '
                    f'{col}metadata_json LIKE \'%"workflow_id"%\')'
                )
            return clauses, params

        conn = self._connect()
        try:
            if not query:
                clauses, params = _extra_clauses("")
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                total = conn.execute(
                    f"SELECT COUNT(*) FROM chat_turns {where}", params,
                ).fetchone()[0]
                rows = conn.execute(
                    f"""
                    SELECT id, task_id, role, content, sources_json,
                           status_message, metadata_json, conversation_id,
                           conversation_title, created_at
                    FROM   chat_turns
                    {where}
                    ORDER  BY created_at DESC
                    LIMIT  ? OFFSET ?
                    """,
                    [*params, limit, offset],
                ).fetchall()
            else:
                fts_query = '"' + query.replace('"', '""') + '"'
                clauses, params = _extra_clauses("c.")
                extra_where = "".join(f" AND {c}" for c in clauses)
                try:
                    total = conn.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM   chat_turns_fts
                        JOIN   chat_turns c ON c.id = chat_turns_fts.rowid
                        WHERE  chat_turns_fts MATCH ? {extra_where}
                        """,
                        [fts_query, *params],
                    ).fetchone()[0]
                    rows = conn.execute(
                        f"""
                        SELECT c.id, c.task_id, c.role, c.content, c.sources_json,
                               c.status_message, c.metadata_json, c.conversation_id,
                               c.conversation_title, c.created_at
                        FROM   chat_turns_fts
                        JOIN   chat_turns c ON c.id = chat_turns_fts.rowid
                        WHERE  chat_turns_fts MATCH ? {extra_where}
                        ORDER  BY bm25(chat_turns_fts) ASC
                        LIMIT  ? OFFSET ?
                        """,
                        [fts_query, *params, limit, offset],
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    logger.debug(
                        "get_chat_turns: FTS5 query failed for %r — %s", query, exc,
                    )
                    return [], 0
        finally:
            conn.close()

        result = [
            {
                "id":                 row["id"],
                "task_id":            row["task_id"],
                "role":               row["role"],
                "content":            row["content"],
                "sources":            json.loads(row["sources_json"]),
                "status_message":     row["status_message"],
                "metadata":           json.loads(row["metadata_json"]),
                "conversation_id":    row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "created_at":         row["created_at"],
            }
            for row in rows
        ]
        return result, total

    def _get_chat_turns_semantic(
        self,
        query:           str,
        limit:           int,
        offset:          int,
        conversation_id: str | None,
        min_score:       float,
        date_from:       float | None = None,
        date_to:         float | None = None,
        has_tool_result: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        mode="semantic" implementation for get_chat_turns().

        Full-table cosine scan, same approach as EpisodicMemoryReader.
        _score_all_active() — chat_turns has no token_set column to
        cheaply pre-filter with the way document_index does, so every row
        with a stored embedding is scored directly and rows below
        min_score are dropped before pagination. See
        _CHAT_TURNS_SEMANTIC_SCAN_WARN_ROW_COUNT.

        date_from/date_to/has_tool_result narrow the candidate set via SQL
        before scoring — same semantics as get_chat_turns()'s own
        parameters, applied here too so filters compose correctly with
        semantic search.

        Raises on embed_fn failure — callers (get_chat_turns()) are
        expected to catch and fall back to keyword search.
        """
        query_vec = self._embed_fn(query)

        clauses: list[str] = []
        params:  list[Any] = []
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if date_from is not None:
            clauses.append("created_at >= ?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("created_at <= ?")
            params.append(date_to)
        if has_tool_result:
            clauses.append(
                '(metadata_json LIKE \'%"chart"%\' OR '
                'metadata_json LIKE \'%"pending_diffs"%\' OR '
                'metadata_json LIKE \'%"workflow_id"%\')'
            )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT id, task_id, role, content, sources_json,
                       status_message, metadata_json, conversation_id,
                       conversation_title, created_at, embedding
                FROM   chat_turns
                {where}
                """,
                params,
            ).fetchall()
        finally:
            conn.close()

        if len(rows) > _CHAT_TURNS_SEMANTIC_SCAN_WARN_ROW_COUNT:
            logger.warning(
                "get_chat_turns: semantic scan over %d chat_turns rows — "
                "unindexed full-table scan; consider whether the configured "
                "eviction preset should be revisited for this install.",
                len(rows),
            )

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            if row["embedding"] is None:
                continue
            doc_vec = _unpack_embedding(row["embedding"])
            score   = _cosine_similarity(query_vec, doc_vec)
            if score >= min_score:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)

        total = len(scored)
        page  = scored[offset : offset + limit]

        result = [
            {
                "id":                 row["id"],
                "task_id":            row["task_id"],
                "role":               row["role"],
                "content":            row["content"],
                "sources":            json.loads(row["sources_json"]),
                "status_message":     row["status_message"],
                "metadata":           json.loads(row["metadata_json"]),
                "conversation_id":    row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "created_at":         row["created_at"],
                "score":              score,
            }
            for score, row in page
        ]
        return result, total

    def get_conversations(self) -> list[dict[str, Any]]:
        """
        Return one summary row per distinct conversation_id in chat_turns.

        Read-only — no eviction, no writes.

        Returns
        -------
        list[dict]
            Rows ordered by last_created_at DESC. Each row dict has keys:
            conversation_id, conversation_title, last_created_at,
            first_created_at.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT
                    t1.conversation_id,
                    (SELECT t2.conversation_title
                       FROM   chat_turns t2
                       WHERE  t2.conversation_id = t1.conversation_id
                         AND  t2.conversation_title IS NOT NULL
                       LIMIT  1)                      AS conversation_title,
                    MAX(t1.created_at)                AS last_created_at,
                    MIN(t1.created_at)                AS first_created_at
                FROM   chat_turns t1
                GROUP  BY t1.conversation_id
                ORDER  BY last_created_at DESC
                """
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "conversation_id":    row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "last_created_at":    row["last_created_at"],
                "first_created_at":   row["first_created_at"],
            }
            for row in rows
        ]

    def delete_conversation(self, conversation_id: str) -> int:
        """
        Delete every chat_turns row for a conversation_id.

        The chat_turns_ad trigger keeps chat_turns_fts in sync automatically.
        Returns the number of turns deleted (0 if the conversation_id didn't
        exist — callers surface that as a 404).
        """
        with self._lock:
            conn = self._connect()
            try:
                deleted = conn.execute(
                    "DELETE FROM chat_turns WHERE conversation_id = ?",
                    (conversation_id,),
                ).rowcount
                conn.commit()
                return deleted
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Retention settings  (Settings tab — global TTL for chat_turns + episodes)
    # -----------------------------------------------------------------------

    def get_retention_preset(self) -> str | None:
        """
        Read the current global retention preset.

        Returns None when no row exists yet — this is the expected default
        state (the user has never set a preset), not an error.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT eviction_preset FROM retention_settings WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()

        return row["eviction_preset"] if row is not None else None

    def set_retention_preset(self, preset: str) -> None:
        """
        Insert or update the global retention preset (row id=1).

        Parameters
        ----------
        preset :
            Must be one of _RETENTION_PRESETS.

        Raises
        ------
        ValueError
            If preset is not one of the allowed values.
        """
        if preset not in _RETENTION_PRESETS:
            raise ValueError(
                f"preset {preset!r} not in allowed set. "
                f"Valid: {sorted(_RETENTION_PRESETS)}"
            )

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO retention_settings (id, eviction_preset)
                    VALUES (1, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        eviction_preset = excluded.eviction_preset
                    """,
                    (preset,),
                )
                conn.commit()
                logger.info("set_retention_preset: preset=%r.", preset)
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Assistant name  (Settings tab — user-configurable spoken identity)
    # -----------------------------------------------------------------------

    def get_assistant_name(self) -> str:
        """
        Read the current assistant name.

        Unlike get_retention_preset(), always returns a usable string — the
        default (_DEFAULT_ASSISTANT_NAME) when no row exists yet or the
        stored value is NULL, never None. Callers (PromptBuilder call sites)
        should never have to null-check this.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT assistant_name FROM assistant_settings WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()

        if row is None or not row["assistant_name"]:
            return _DEFAULT_ASSISTANT_NAME
        return row["assistant_name"]

    def set_assistant_name(self, name: str) -> None:
        """
        Insert or update the assistant name (row id=1).

        Parameters
        ----------
        name :
            Non-empty after stripping whitespace, at most
            _ASSISTANT_NAME_MAX_LEN characters.

        Raises
        ------
        ValueError
            If name is empty/whitespace-only or exceeds the length cap.
        """
        stripped = name.strip()
        if not stripped:
            raise ValueError("assistant_name must not be empty.")
        if len(stripped) > _ASSISTANT_NAME_MAX_LEN:
            raise ValueError(
                f"assistant_name must be at most {_ASSISTANT_NAME_MAX_LEN} "
                f"characters (got {len(stripped)})."
            )

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO assistant_settings (id, assistant_name)
                    VALUES (1, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        assistant_name = excluded.assistant_name
                    """,
                    (stripped,),
                )
                conn.commit()
                logger.info("set_assistant_name: name=%r.", stripped)
            finally:
                conn.close()

    def sweep_expired_memory(self) -> dict[str, int]:
        """
        Enforce the global retention preset: hard-delete chat_turns and
        soft-retract episodes older than the configured TTL.

        No-ops (returns zero counts) when no preset is set or the preset is
        "forever" — an absent/forever preset means "keep everything", not
        "sweep with an infinite TTL". chat_turns rows are deleted outright
        (no lifecycle column exists on that table); episodes are marked
        status='retracted' rather than deleted, matching the existing
        approve/reject/retract lifecycle so expired episodes stay reversible
        and are automatically excluded from every status='active' retrieval
        path (by_subject, by_recency, _score_all_active, get_by_ids).

        Does NOT regenerate MEMORY.md itself — that snapshot is owned by
        EpisodicMemoryWriter (a separate class; see approve()/reject() in
        main.py for the same split), which this method has no reference to.
        Callers should invoke EpisodicMemoryWriter(...).regenerate_memory_md()
        afterward when episodes_retracted > 0 (main.py's lifespan() does this
        for the startup sweep).
        """
        preset = self.get_retention_preset()
        if preset is None or preset == "forever":
            return {"chat_turns_deleted": 0, "episodes_retracted": 0}

        cutoff = time.time() - _RETENTION_PRESET_SECONDS[preset]

        with self._lock:
            conn = self._connect()
            try:
                chat_turns_deleted = conn.execute(
                    "DELETE FROM chat_turns WHERE created_at < ?", (cutoff,)
                ).rowcount
                episodes_retracted = conn.execute(
                    """
                    UPDATE episodes
                    SET    status = 'retracted'
                    WHERE  status = 'active'
                      AND  created_at < ?
                    """,
                    (cutoff,),
                ).rowcount
                conn.commit()
                logger.info(
                    "sweep_expired_memory: preset=%r cutoff=%.0f "
                    "chat_turns_deleted=%d episodes_retracted=%d.",
                    preset, cutoff, chat_turns_deleted, episodes_retracted,
                )
                return {
                    "chat_turns_deleted": chat_turns_deleted,
                    "episodes_retracted": episodes_retracted,
                }
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Daily News Brief — preferences and same-day cache
    # (docs/daily-news-brief-plan.md §4/§5/§6)
    # -----------------------------------------------------------------------

    def get_news_preferences(self) -> dict[str, Any] | None:
        """
        Read the current news_preferences row.

        Returns None when no row exists yet (the user has never set
        preferences) — the caller (main.py's GET /news/preferences)
        supplies defaults in that case, same posture as
        get_retention_preset().
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT home_country, local_query, topics_json "
                "FROM news_preferences WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return {
            "home_country": row["home_country"],
            "local_query":  row["local_query"],
            "topics":       json.loads(row["topics_json"]),
        }

    def set_news_preferences(
        self,
        home_country: str,
        local_query:  str | None,
        topics:       list[str],
    ) -> None:
        """
        Insert or update news_preferences (row id=1).

        Parameters
        ----------
        topics :
            Must have exactly 3 entries. Validating each entry against the
            actual topic pool (docs/daily-news-brief-plan.md §3) is the
            caller's responsibility — this method only enforces shape;
            `news_brief.py`'s `NEWS_TOPIC_POOL` is the domain constant that
            owns which keys are actually valid, checked at the API layer
            (main.py's `PUT /news/preferences`), not here.

        Raises
        ------
        ValueError
            If `topics` does not have exactly 3 entries.
        """
        if len(topics) != 3:
            raise ValueError(f"topics must have exactly 3 entries, got {len(topics)}")

        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO news_preferences
                        (id, home_country, local_query, topics_json, updated_at)
                    VALUES (1, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        home_country = excluded.home_country,
                        local_query  = excluded.local_query,
                        topics_json  = excluded.topics_json,
                        updated_at   = excluded.updated_at
                    """,
                    (home_country, local_query, json.dumps(topics), now),
                )
                conn.commit()
                logger.info(
                    "set_news_preferences: home_country=%r local_query=%r topics=%r.",
                    home_country, local_query, topics,
                )
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Pinned GitHub Repos — release-only tracking independent of GitHub's
    # Watch/subscriptions relationship (schema v15)
    # -----------------------------------------------------------------------

    def get_pinned_github_repos(self) -> list[str]:
        """
        Read the pinned repo list.

        Returns [] when no row exists yet — unlike get_news_preferences(),
        there is no None-vs-empty distinction to make at the API layer: an
        empty pinned list and "never configured" both render identically
        (no pinned entries in the merged GitHub Watch Feed).
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT full_names_json FROM pinned_github_repos WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return []
        return json.loads(row["full_names_json"])

    def set_pinned_github_repos(self, full_names: list[str]) -> None:
        """
        Insert or update pinned_github_repos (row id=1). Whole-list
        overwrite, same semantics as set_news_preferences().

        Parameters
        ----------
        full_names :
            "owner/repo" slugs. Format validation (owner/repo shape) and
            dedupe are the caller's responsibility (main.py's
            PUT /github/watch/pinned-repos) — this method only enforces a
            max-count guard, same "shape here, domain rules at the API
            layer" split set_news_preferences() uses for topics.

        Raises
        ------
        ValueError
            If `full_names` has more than 20 entries.
        """
        if len(full_names) > 20:
            raise ValueError(
                f"full_names must have at most 20 entries, got {len(full_names)}"
            )

        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO pinned_github_repos (id, full_names_json, updated_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        full_names_json = excluded.full_names_json,
                        updated_at      = excluded.updated_at
                    """,
                    (json.dumps(full_names), now),
                )
                conn.commit()
                logger.info(
                    "set_pinned_github_repos: full_names=%r.", full_names
                )
            finally:
                conn.close()

    def get_news_brief_cache(self) -> dict[str, Any] | None:
        """
        Read the current news_brief_cache row.

        Returns None when no brief has ever been generated. There is only
        ever one row — a same-day cache, not a history of past briefs (a
        new generation overwrites it via set_news_brief_cache()).
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT brief_date, content_json, conversation_id, generated_at "
                "FROM news_brief_cache WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return {
            "brief_date":      row["brief_date"],
            "content":         json.loads(row["content_json"]),
            "conversation_id": row["conversation_id"],
            "generated_at":    row["generated_at"],
        }

    def set_news_brief_cache(
        self,
        brief_date:      str,
        content:         dict[str, Any],
        conversation_id: str | None,
    ) -> None:
        """
        Insert or update news_brief_cache (row id=1) after a fresh
        generation. Always overwrites the whole row — see get_news_brief_cache().
        """
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO news_brief_cache
                        (id, brief_date, content_json, conversation_id, generated_at)
                    VALUES (1, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        brief_date      = excluded.brief_date,
                        content_json    = excluded.content_json,
                        conversation_id = excluded.conversation_id,
                        generated_at    = excluded.generated_at
                    """,
                    (brief_date, json.dumps(content), conversation_id, now),
                )
                conn.commit()
                logger.info(
                    "set_news_brief_cache: brief_date=%r conversation_id=%r.",
                    brief_date, conversation_id,
                )
            finally:
                conn.close()

    def get_github_watch_cache(self) -> dict[str, Any] | None:
        """
        Read the current github_watch_cache row.

        Returns None when the watch feed has never been generated. There is
        only ever one row — unlike news_brief_cache there's no brief_date
        column, since watched-repo releases aren't day-keyed the way a news
        brief is; a new refresh overwrites it via set_github_watch_cache().
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT content_json, generated_at FROM github_watch_cache WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return {
            "content":      json.loads(row["content_json"]),
            "generated_at": row["generated_at"],
        }

    def set_github_watch_cache(self, content: list[dict[str, Any]]) -> None:
        """
        Insert or update github_watch_cache (row id=1) after a fresh
        refresh. Always overwrites the whole row — see get_github_watch_cache().
        """
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO github_watch_cache (id, content_json, generated_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content_json = excluded.content_json,
                        generated_at = excluded.generated_at
                    """,
                    (json.dumps(content), now),
                )
                conn.commit()
                logger.info("set_github_watch_cache: repos=%d.", len(content))
            finally:
                conn.close()

    def get_hacker_news_cache(self) -> dict[str, Any] | None:
        """
        Read the current hacker_news_cache row.

        Returns None when the top-stories feed has never been generated.
        There is only ever one row — same "not day-keyed" reasoning as
        github_watch_cache; a new refresh overwrites it via
        set_hacker_news_cache().
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT content_json, generated_at FROM hacker_news_cache WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return {
            "content":      json.loads(row["content_json"]),
            "generated_at": row["generated_at"],
        }

    def set_hacker_news_cache(self, content: list[dict[str, Any]]) -> None:
        """
        Insert or update hacker_news_cache (row id=1) after a fresh
        refresh. Always overwrites the whole row — see get_hacker_news_cache().
        """
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO hacker_news_cache (id, content_json, generated_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content_json = excluded.content_json,
                        generated_at = excluded.generated_at
                    """,
                    (json.dumps(content), now),
                )
                conn.commit()
                logger.info("set_hacker_news_cache: stories=%d.", len(content))
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Document index  (WikiAgent → index; ResearchAgent → query)
    # -----------------------------------------------------------------------

    def index_document(
        self,
        path:        Path | str,
        doc_type:    str,            # "wiki" | "raw"
        content:     str | None = None,
        embed:       bool       = True,
    ) -> None:
        """
        Add or update a document in the index.

        Parameters
        ----------
        path :
            Absolute path to the file on disk.
        doc_type :
            "wiki" for pages in wiki/, "raw" for files in raw/.
        content :
            File text.  If None, the file is read from disk.  Supply it
            directly if you already have the text in memory (e.g. from
            WikiAgent after writing to disk) to avoid a redundant read.
        embed :
            Whether to compute and store an embedding for this document.
            Requires embed_fn to have been supplied at construction time.
            Set False when bulk-indexing many documents and you want to
            embed them in a separate batch pass.

        Behaviour on collision (same path):
            If the content hash is unchanged, the row is left as-is.
            If the content changed, the row is updated and the retrieval
            cache is invalidated.
        """
        path = Path(path).resolve()
        if content is None:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("index_document: cannot read %s — %s", path, exc)
                return

        name         = path.stem
        token_set    = " ".join(sorted(_tokenize(content)))
        c_hash       = _content_hash(content)
        now          = time.time()

        # Compute embedding before acquiring the lock — it can be slow.
        embedding_blob: bytes | None = None
        if embed and self._embed_fn is not None:
            try:
                # Embed the first ~500 chars — consistent with the existing
                # ResearchAgent strategy that keeps embedding calls cheap.
                vec = self._embed_fn(content[:500])
                if len(vec) == _EMBEDDING_DIM:
                    embedding_blob = _pack_embedding(vec)
                else:
                    logger.warning(
                        "index_document: embed returned dim=%d, expected %d — skipping.",
                        len(vec), _EMBEDDING_DIM,
                    )
            except Exception as exc:
                logger.warning("index_document: embed failed for %s — %s", path.name, exc)

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT id, content_hash FROM document_index WHERE path = ?",
                    (str(path),),
                ).fetchone()

                if existing is not None:
                    if existing["content_hash"] == c_hash:
                        logger.debug(
                            "index_document: %s unchanged (hash match), skipping.",
                            path.name,
                        )
                        return
                    # Content changed — update and invalidate cache.
                    conn.execute(
                        """
                        UPDATE document_index
                        SET    name=?, doc_type=?, content=?, token_set=?,
                               embedding=?, content_hash=?, indexed_at=?
                        WHERE  path=?
                        """,
                        (name, doc_type, content, token_set,
                         embedding_blob, c_hash, now, str(path)),
                    )
                    logger.info("index_document: updated %s (%s).", path.name, doc_type)
                else:
                    conn.execute(
                        """
                        INSERT INTO document_index
                            (name, path, doc_type, content, token_set,
                             embedding, content_hash, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (name, str(path), doc_type, content, token_set,
                         embedding_blob, c_hash, now),
                    )
                    logger.info("index_document: indexed %s (%s).", path.name, doc_type)

                if embedding_blob is not None:
                    self._maybe_seed_corpus_provenance(conn)

                conn.commit()
                # Invalidate retrieval cache on any index mutation.
                self._invalidate_cache(conn)
            finally:
                conn.close()

    def index_directory(
        self,
        directory: Path | str,
        doc_type:  str,
        embed:     bool = True,
        extensions: set[str] = frozenset({".md", ".txt"}),
        exclude:   frozenset[str] = frozenset(),
    ) -> int:
        """
        Bulk-index all matching files in a directory.

        Skips files whose content hash matches the stored value (idempotent).
        Returns the number of files newly indexed or updated.

        Parameters
        ----------
        directory :
            Path to walk (non-recursive — top-level files only, matching
            the existing _load_corpus() behaviour).
        doc_type :
            "wiki" or "raw".
        embed :
            Whether to compute embeddings.  Pass False for the first bulk
            import of a large directory and run a separate embed pass later.
        extensions :
            File extensions to include.
        exclude :
            Filenames (e.g. wiki_doc.META_WIKI_FILENAMES) to skip
            regardless of extension — structural/generated files that
            should never become a RAG-eligible document_index row.
            Defaults to empty, so existing callers (raw_dir) are unaffected.
        """
        directory = Path(directory).resolve()
        if not directory.exists():
            logger.warning("index_directory: %s does not exist.", directory)
            return 0

        count = 0
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix.lower() in extensions and p.name not in exclude:
                try:
                    content = p.read_text(encoding="utf-8")
                except Exception as exc:
                    logger.warning("index_directory: cannot read %s — %s", p, exc)
                    continue
                self.index_document(p, doc_type=doc_type, content=content, embed=embed)
                count += 1

        logger.info(
            "index_directory: indexed %d files from %s (%s).",
            count, directory, doc_type,
        )
        return count

    def remove_document(self, path: Path | str) -> None:
        """
        Remove a document from the index by path.

        Also invalidates the retrieval cache.
        """
        path = Path(path).resolve()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "DELETE FROM document_index WHERE path = ?", (str(path),)
                )
                conn.commit()
                self._invalidate_cache(conn)
                logger.info("remove_document: removed %s.", path.name)
            finally:
                conn.close()

    def reconcile_wiki(self, wiki_dir: Path) -> dict:
        """
        Reconcile document_index against the wiki/ directory on disk.

        Composes index_directory() and remove_document() — does not
        duplicate their logic:

        1. Re-index wiki_dir via index_directory().  Already idempotent via
           content_hash — unchanged files are a no-op, changed/new files are
           updated/inserted (retrieval cache invalidation is handled inside
           index_directory() / index_document() as a side effect).
        2. Orphan detection: for every "wiki" row whose on-disk path no
           longer exists (file deleted or renamed since the last run),
           remove the row and append an entry to the wiki maintenance
           audit log.

        Returns
        -------
        dict with keys:
            reindexed       : int        — count from index_directory()
            orphans_removed : int
            orphan_names    : list[str]
        """
        reindexed = self.index_directory(
            wiki_dir, doc_type="wiki", embed=True, exclude=META_WIKI_FILENAMES,
        )

        orphan_names: list[str] = []
        for doc in self.get_all_documents(doc_type="wiki"):
            if not Path(doc.path).exists():
                self.remove_document(doc.path)
                orphan_names.append(doc.name)
                wiki_maintenance_log.log_orphan_removed(doc.name, str(doc.path))

        return {
            "reindexed":       reindexed,
            "orphans_removed": len(orphan_names),
            "orphan_names":    orphan_names,
        }

    # -----------------------------------------------------------------------
    # Link graph  (build_graph.py API)
    # -----------------------------------------------------------------------

    def upsert_graph_node(
        self,
        doc_path:  Path | str,
        node_type: str | None,
        title:     str | None,
    ) -> int:
        """
        Insert or update a graph_nodes row identified by doc_path.
        Returns the row's id.

        On insert:           created_at = updated_at = now.
        On conflict (same doc_path already exists): node_type/title/updated_at
        are overwritten; created_at is preserved from the original insert so
        the first-seen timestamp is stable across rebuilds.

        doc_path is normalized via _normalize_doc_path() — a real filesystem
        path is resolve()'d as before; a synthetic scheme like "episode://42"
        (see upsert_graph_node_for_episode()) is stored verbatim.
        """
        doc_path_str = _normalize_doc_path(doc_path)
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO graph_nodes
                        (doc_path, node_type, title, source_doc_path,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_path) DO UPDATE SET
                        node_type  = excluded.node_type,
                        title      = excluded.title,
                        updated_at = excluded.updated_at
                    """,
                    (doc_path_str, node_type, title, doc_path_str, now, now),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT id FROM graph_nodes WHERE doc_path = ?",
                    (doc_path_str,),
                ).fetchone()
                return row["id"]
            finally:
                conn.close()

    def upsert_graph_edge(
        self,
        source_node_id:  int,
        source_doc_path: Path | str,
        target_path:     str,
        target_node_id:  int | None,
        target_resolved: bool,
        link_text:       str,
    ) -> None:
        """
        Insert or update a graph_edges row by the natural key
        (source_doc_path, target_path).

        On update, target_node_id / target_resolved / link_text are refreshed
        so a previously-unresolved link can resolve on a later rebuild once
        the target page exists.

        source_doc_path is normalized via _normalize_doc_path() — see
        upsert_graph_node() for the filesystem-vs-synthetic-id rationale.
        """
        source_doc_path_str = _normalize_doc_path(source_doc_path)
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """SELECT id FROM graph_edges
                       WHERE source_doc_path = ? AND target_path = ?""",
                    (source_doc_path_str, target_path),
                ).fetchone()
                if existing is not None:
                    conn.execute(
                        """UPDATE graph_edges
                           SET source_node_id  = ?,
                               target_node_id  = ?,
                               target_resolved = ?,
                               link_text       = ?
                           WHERE source_doc_path = ? AND target_path = ?""",
                        (source_node_id, target_node_id,
                         1 if target_resolved else 0, link_text,
                         source_doc_path_str, target_path),
                    )
                else:
                    conn.execute(
                        """INSERT INTO graph_edges
                               (source_node_id, source_doc_path, target_path,
                                target_node_id, target_resolved, link_text)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (source_node_id, source_doc_path_str, target_path,
                         target_node_id, 1 if target_resolved else 0, link_text),
                    )
                conn.commit()
            finally:
                conn.close()

    def clear_graph_for_doc(self, doc_path: Path | str) -> None:
        """
        Delete all graph_edges rows where source_doc_path == doc_path.

        Intended for per-document partial rebuilds (e.g. a future WikiAgent
        post-ingest hook). The full offline rebuild script uses
        clear_graph_edges() instead.

        doc_path is normalized via _normalize_doc_path() — see
        upsert_graph_node() for the filesystem-vs-synthetic-id rationale.
        """
        doc_path_str = _normalize_doc_path(doc_path)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "DELETE FROM graph_edges WHERE source_doc_path = ?",
                    (doc_path_str,),
                )
                conn.commit()
            finally:
                conn.close()

    def clear_graph_edges(self) -> None:
        """
        Delete ALL graph_edges rows.

        Used by the full-corpus rebuild script between the node-upsert pass
        and the edge-upsert pass, so stale edges (from since-removed [[...]]
        links) don't survive the rebuild. Does NOT delete graph_nodes rows.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM graph_edges")
                conn.commit()
            finally:
                conn.close()

    def resolve_node_by_stem(self, stem: str) -> dict | None:
        """
        Look up a single graph_nodes row whose doc_path stem equals ``stem``.

        Does exact-match only — caller is responsible for normalising the stem
        before calling. Returns a dict with keys ``id``, ``doc_path``, ``title``
        if exactly one row matches, else ``None`` (zero or multiple matches).

        Excludes node_type='episode' rows. This is the name-resolution path
        used by Planner P3c ("what links to X") to turn free-text into a wiki
        page — episode nodes use the synthetic "episode://<id>" doc_path
        scheme (see upsert_graph_node_for_episode()), whose Path(...).stem
        is not a meaningful user-facing name and must never surface as a
        P3c resolution candidate.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, doc_path, title FROM graph_nodes "
                    "WHERE node_type IS NULL OR node_type != 'episode'"
                ).fetchall()
            finally:
                conn.close()

        matches = [
            {"id": row["id"], "doc_path": row["doc_path"], "title": row["title"]}
            for row in rows
            if Path(row["doc_path"]).stem == stem
        ]
        return matches[0] if len(matches) == 1 else None

    def get_backlinks(
        self, node_id: int, include_episode_sources: bool = True,
    ) -> list[GraphEdgeResult]:
        """
        Return all graph_edges rows where ``target_node_id`` equals ``node_id``.

        For each edge, the source node's ``title`` and ``doc_path`` are fetched
        and exposed as ``node_title``/``node_doc_path`` — the interesting "other
        side" of a backlinks query is the source page, not the target.
        ``target_resolved`` is always ``True`` for every row this method returns
        (structural guarantee: an edge found by ``target_node_id = ?`` is, by
        construction, always resolved).

        include_episode_sources :
            When True (default), backlinks whose source is an episode node
            (episode → wiki "episodic association" edges, see
            upsert_graph_node_for_episode()) are included alongside authored
            [[wikilink]] backlinks — this is what the Phase A RAG-assist
            neighborhood wants, so "N conversations referenced this concept"
            can surface. Pass False to restrict to wiki-authored backlinks
            only — used by Planner P3c, whose "pure structural query"
            contract predates episode nodes and must not silently start
            mixing inferred episodic associations into an explicit
            "what links to X" answer without a deliberate design decision.

        Returns an empty list when no backlinks exist.
        """
        episode_filter = (
            "" if include_episode_sources
            else " AND (n.node_type IS NULL OR n.node_type != 'episode')"
        )
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT e.link_text, e.target_path,
                           n.title    AS node_title,
                           n.doc_path AS node_doc_path
                    FROM   graph_edges e
                    JOIN   graph_nodes n ON n.id = e.source_node_id
                    WHERE  e.target_node_id = ?{episode_filter}
                    """,
                    (node_id,),
                ).fetchall()
            finally:
                conn.close()

        return [
            GraphEdgeResult(
                link_text       = row["link_text"],
                target_path     = row["target_path"],
                target_resolved = True,
                node_title      = row["node_title"],
                node_doc_path   = row["node_doc_path"],
            )
            for row in rows
        ]

    def get_outgoing_links(self, node_id: int) -> list[GraphEdgeResult]:
        """
        Return all graph_edges rows where ``source_node_id`` equals ``node_id``.

        Includes both resolved and unresolved edges — unresolved edges (where the
        target page does not yet exist in the graph) are never silently dropped.
        For resolved edges, ``node_title``/``node_doc_path`` are populated from
        the target node. For unresolved edges, both are ``None``.

        Returns an empty list when no outgoing links exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT e.link_text, e.target_path, e.target_resolved,
                           n.title    AS node_title,
                           n.doc_path AS node_doc_path
                    FROM   graph_edges e
                    LEFT JOIN graph_nodes n ON n.id = e.target_node_id
                    WHERE  e.source_node_id = ?
                    """,
                    (node_id,),
                ).fetchall()
            finally:
                conn.close()

        return [
            GraphEdgeResult(
                link_text       = row["link_text"],
                target_path     = row["target_path"],
                target_resolved = bool(row["target_resolved"]),
                node_title      = row["node_title"],
                node_doc_path   = row["node_doc_path"],
            )
            for row in rows
        ]

    def list_graph_node_stems(self) -> list[str]:
        """
        Return the stem of every wiki ``doc_path`` currently stored in
        ``graph_nodes``.

        Used by Planner P3c to feed a candidate-stem list into
        ``resolve_graph_target()`` without touching the filesystem — also
        reused by the Phase B episode→wiki edge-resolution hook (same
        stem/token-overlap pipeline, applied to episode subject text
        instead of a P3c instruction). Excludes node_type='episode' rows
        for the same reason as resolve_node_by_stem(): their doc_path stem
        is a synthetic id, not a real page name. Returns an empty list when
        the graph has not been built yet.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT doc_path FROM graph_nodes "
                    "WHERE node_type IS NULL OR node_type != 'episode'"
                ).fetchall()
            finally:
                conn.close()

        return [Path(row["doc_path"]).stem for row in rows]

    def get_graph_node_by_doc_path(self, doc_path: Path | str) -> dict | None:
        """
        Look up a single graph_nodes row by exact doc_path.

        Unlike resolve_node_by_stem() (free-text name resolution, tolerant
        of stem ambiguity, episode-excluded), this is an exact-match lookup
        for when the caller already holds a real, resolved document path —
        e.g. Phase A's RAG-assist trigger, which has query_corpus()'s
        top-ranked result path in hand and just needs to know whether it is
        also a graph node. doc_path is resolved via _normalize_doc_path()
        exactly as upsert_graph_node() stores it, so a real filesystem path
        passed in either raw or already-resolved form matches correctly.

        Returns None if no row matches.
        """
        doc_path_str = _normalize_doc_path(doc_path)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, doc_path, title FROM graph_nodes WHERE doc_path = ?",
                    (doc_path_str,),
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        return {"id": row["id"], "doc_path": row["doc_path"], "title": row["title"]}

    def upsert_graph_node_for_episode(self, episode_id: int, title: str) -> int:
        """
        Upsert a graph_nodes row for an episode (Phase B — the B2 "full node
        model" decision in memory-graph-inference-plan: episodes get real
        graph_nodes rows, node_type='episode', rather than being bolted on
        as edge-targets only. This lets an episode node carry its own
        backlinks/outgoing edges — e.g. a wiki page's neighborhood can cite
        "N conversations referenced this concept" as a first-class graph
        citizen, and a future correction-episode could point at the
        decision episode it supersedes.

        doc_path uses the synthetic "episode://<id>" scheme — episodes have
        no filesystem path, and the episode's own row id is already a
        stable, permanent, unique key (unlike `subject`, which can repeat
        across unrelated episode rows and would collide on graph_nodes'
        UNIQUE(doc_path) constraint, silently merging distinct episodes
        into one node).

        Returns the row's id.
        """
        return self.upsert_graph_node(
            doc_path  = f"episode://{episode_id}",
            node_type = "episode",
            title     = title,
        )

    # -----------------------------------------------------------------------
    # Corpus retrieval  (ResearchAgent API)
    # -----------------------------------------------------------------------

    def query_corpus(
        self,
        query:          str,
        max_results:    int  = 5,
        doc_type:       str | None = None,   # None = all; "wiki" | "raw" to filter
        use_embeddings: bool = True,
    ) -> list[DocumentResult]:
        """
        Return the top-N most relevant documents for a query.

        Strategy
        --------
        1. Score every indexed document by keyword overlap (Jaccard).
        2. If use_embeddings=True and embed_fn is available, re-rank the
           top 2*max_results keyword candidates by embedding cosine similarity
           and return the top max_results of those.
        3. Fall back to keyword ranking if embedding fails or is unavailable.

        The retrieval_cache is checked first.  On a cache hit (same query +
        max_results and cache is still valid), documents are fetched by name
        from the index rather than re-scoring the whole table.  This makes
        repeated identical sub-queries within a single ResearchAgent run free.

        Parameters
        ----------
        query :
            The sub-query string from ResearchAgent.
        max_results :
            Maximum number of DocumentResults to return.
        doc_type :
            Optional filter.  "wiki" returns only wiki pages; "raw" only
            raw files; None returns both.
        use_embeddings :
            Whether to attempt embedding-based re-ranking.

        Returns
        -------
        list[DocumentResult]
            Sorted by relevance_score descending.
        """
        q_hash = _query_hash(query, max_results, doc_type)

        # -- Cache check --
        cached = self._check_cache(q_hash)
        if cached is not None:
            logger.debug("query_corpus: cache hit for query '%s...'.", query[:40])
            return self._hydrate_cache_result(cached, doc_type)

        # -- Load all documents from the index (token_set + embedding) --
        conn = self._connect()
        try:
            where_clause = "WHERE doc_type = ?" if doc_type else ""
            params       = (doc_type,) if doc_type else ()
            rows = conn.execute(
                f"""
                SELECT id, name, path, doc_type, content, token_set, embedding
                FROM   document_index
                {where_clause}
                """,
                params,
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            logger.debug("query_corpus: index is empty — returning [].")
            return []

        # Stage 1 — BM25 keyword prefilter. Raw, unbounded score (bm25.py
        # deliberately doesn't normalize — see its module docstring); fine
        # for ranking/top-2N selection here since every row is on the same
        # scale for this call, but not safe to compare against an absolute
        # cosine-tuned threshold downstream — see scored_by_embedding
        # below. Scored over `rows`' raw content (not the deduplicated
        # token_set column, which loses the term frequency BM25 needs —
        # see bm25.tokenize()), keyed by row id so scores map back onto
        # the right row even if `name` collides. Computed once and reused
        # below for any un-embedded doc in the rerank pool, rather than
        # re-scored in isolation there.
        bm25_scores = bm25.score_documents(
            query, [(row["id"], row["content"]) for row in rows],
        )
        # (score, row, scored_by_embedding) — the third element tracks
        # which scoring path produced `score` so callers downstream (e.g.
        # controller_agent.py's RAG-source filters) know whether a fixed
        # absolute threshold means anything for this particular result.
        scored: list[tuple[float, sqlite3.Row, bool]] = [
            (bm25_scores[row["id"]], row, False) for row in rows
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Embedding re-rank on top-2N candidates. self._corpus_stale is the
        # same fail-safe-to-keyword-only posture as self._embed_fn being
        # None — see _check_embedding_provenance() / reembed_corpus().
        if use_embeddings and self._embed_fn is not None and not self._corpus_stale:
            pool = scored[: max_results * 2]
            try:
                query_vec = self._embed_fn(query)
                re_scored: list[tuple[float, sqlite3.Row, bool]] = []
                for _, row, _ in pool:
                    if row["embedding"] is not None:
                        doc_vec = _unpack_embedding(row["embedding"])
                        score   = _cosine_similarity(query_vec, doc_vec)
                        by_embedding = True
                    else:
                        # Same raw BM25 score computed above.
                        score = bm25_scores[row["id"]]
                        by_embedding = False
                    re_scored.append((score, row, by_embedding))
                re_scored.sort(key=lambda x: x[0], reverse=True)
                scored = re_scored
            except Exception as exc:
                logger.warning(
                    "query_corpus: embedding re-rank failed (%s); using keyword scores.", exc
                )

        top = scored[:max_results]

        # Build DocumentResult list
        results = [
            DocumentResult(
                name                = row["name"],
                path                = Path(row["path"]),
                doc_type            = row["doc_type"],
                content             = row["content"],
                relevance_score     = score,
                scored_by_embedding = by_embedding,
            )
            for score, row, by_embedding in top
        ]

        # Write to cache (store name+score pairs — content is re-fetched from index)
        cache_payload = [
            {
                "name":                r.name,
                "path":                str(r.path),
                "doc_type":            r.doc_type,
                "score":               r.relevance_score,
                "scored_by_embedding": r.scored_by_embedding,
            }
            for r in results
        ]
        self._write_cache(q_hash, max_results, cache_payload)

        logger.debug(
            "query_corpus: returning %d results for query '%s...'.",
            len(results), query[:40],
        )
        return results

    def get_all_documents(
        self,
        doc_type: str | None = None,
    ) -> list[DocumentResult]:
        """
        Return every document in the index, unsorted.

        Used by WikiAgent's build_wiki_context() to get the full wiki page
        list without scoring.  Replaces the _load_wiki_pages() filesystem
        walk when a MemoryManager is available.

        Parameters
        ----------
        doc_type :
            Optional filter: "wiki", "raw", or None for all.
        """
        conn = self._connect()
        try:
            if doc_type:
                rows = conn.execute(
                    "SELECT name, path, doc_type, content FROM document_index WHERE doc_type = ?",
                    (doc_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT name, path, doc_type, content FROM document_index",
                ).fetchall()
        finally:
            conn.close()

        return [
            DocumentResult(
                name     = row["name"],
                path     = Path(row["path"]),
                doc_type = row["doc_type"],
                content  = row["content"],
            )
            for row in rows
        ]

    def document_count(self, doc_type: str | None = None) -> int:
        """Return the number of indexed documents, optionally filtered by type."""
        conn = self._connect()
        try:
            if doc_type:
                return conn.execute(
                    "SELECT COUNT(*) FROM document_index WHERE doc_type = ?",
                    (doc_type,),
                ).fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM document_index"
            ).fetchone()[0]
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Retrieval cache helpers
    # -----------------------------------------------------------------------

    def _check_cache(self, q_hash: str) -> list[dict] | None:
        """Return the cached result list if a valid entry exists, else None."""
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT result_json FROM retrieval_cache
                WHERE  query_hash = ? AND valid = 1
                """,
                (q_hash,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        try:
            return json.loads(row["result_json"])
        except json.JSONDecodeError:
            return None

    def _write_cache(
        self,
        q_hash:  str,
        top_n:   int,
        payload: list[dict],
    ) -> None:
        """Upsert a cache entry (non-blocking — cache misses are harmless)."""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(
                        """
                        INSERT INTO retrieval_cache
                            (query_hash, top_n, result_json, created_at, valid)
                        VALUES (?, ?, ?, ?, 1)
                        ON CONFLICT(query_hash) DO UPDATE SET
                            result_json = excluded.result_json,
                            created_at  = excluded.created_at,
                            valid       = 1
                        """,
                        (q_hash, top_n, json.dumps(payload), time.time()),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            logger.debug("_write_cache: failed (non-fatal) — %s", exc)

    def _invalidate_cache(self, conn: sqlite3.Connection) -> None:
        """
        Mark all retrieval cache entries invalid.

        Called inside an existing write transaction whenever the document
        index changes so stale results are never returned.
        """
        conn.execute("UPDATE retrieval_cache SET valid = 0")

    def _hydrate_cache_result(
        self,
        cached:   list[dict],
        doc_type: str | None,
    ) -> list[DocumentResult]:
        """
        Reconstruct DocumentResults from a cache hit.

        Fetches full content from the index by path (content may have changed
        if the doc was updated after the cache entry was written, but since
        _invalidate_cache() is called on every mutation, a valid cache entry
        always corresponds to current content — the path lookup is safe).
        """
        if not cached:
            return []

        paths   = [r["path"] for r in cached]
        placeholders = ",".join("?" * len(paths))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT name, path, doc_type, content
                FROM   document_index
                WHERE  path IN ({placeholders})
                """,
                paths,
            ).fetchall()
        finally:
            conn.close()

        # Build a map for O(1) lookup
        row_map = {row["path"]: row for row in rows}

        results: list[DocumentResult] = []
        for entry in cached:
            row = row_map.get(entry["path"])
            if row is None:
                continue   # document was deleted since cache was written
            if doc_type and row["doc_type"] != doc_type:
                continue
            results.append(DocumentResult(
                name                = row["name"],
                path                = Path(row["path"]),
                doc_type            = row["doc_type"],
                content             = row["content"],
                relevance_score     = entry["score"],
                # Default True for cache rows written before this field
                # existed — the safer fallback, since it re-applies the
                # (pre-existing) absolute-threshold behavior rather than
                # silently bypassing it for old entries of unknown origin.
                scored_by_embedding = entry.get("scored_by_embedding", True),
            ))
        return results

    # -----------------------------------------------------------------------
    # Housekeeping / diagnostics
    # -----------------------------------------------------------------------

    def purge_cache(self) -> None:
        """Delete all retrieval cache entries (valid and invalid)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM retrieval_cache")
                conn.commit()
                logger.info("purge_cache: retrieval cache cleared.")
            finally:
                conn.close()

    def stats(self) -> dict[str, Any]:
        """
        Return a summary dict suitable for the GET /health endpoint.

        Keys
        ----
        db_path         — absolute path to the SQLite file
        db_size_kb      — file size on disk
        wiki_docs       — count of indexed wiki pages
        raw_docs        — count of indexed raw files
        conv_log_rows   — total conversation log entries
        cache_valid     — number of valid retrieval cache entries
        cache_invalid   — number of invalidated cache entries
        embeddings_pct  — percentage of documents with an embedding (0–100)
        """
        conn = self._connect()
        try:
            wiki_count  = conn.execute(
                "SELECT COUNT(*) FROM document_index WHERE doc_type='wiki'"
            ).fetchone()[0]
            raw_count   = conn.execute(
                "SELECT COUNT(*) FROM document_index WHERE doc_type='raw'"
            ).fetchone()[0]
            conv_count  = conn.execute(
                "SELECT COUNT(*) FROM conversation_log"
            ).fetchone()[0]
            cache_valid = conn.execute(
                "SELECT COUNT(*) FROM retrieval_cache WHERE valid=1"
            ).fetchone()[0]
            cache_inv   = conn.execute(
                "SELECT COUNT(*) FROM retrieval_cache WHERE valid=0"
            ).fetchone()[0]
            total_docs  = wiki_count + raw_count
            embedded    = conn.execute(
                "SELECT COUNT(*) FROM document_index WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        finally:
            conn.close()

        emb_pct = round(100 * embedded / total_docs, 1) if total_docs else 0.0
        db_kb   = round(self._db_path.stat().st_size / 1024, 1) if self._db_path.exists() else 0.0

        return {
            "db_path":       str(self._db_path),
            "db_size_kb":    db_kb,
            "wiki_docs":     wiki_count,
            "raw_docs":      raw_count,
            "conv_log_rows": conv_count,
            "cache_valid":   cache_valid,
            "cache_invalid": cache_inv,
            "embeddings_pct": emb_pct,
        }

    def list_episodes(
        self,
        status:          str = "active",
        project_context: str | None = None,
        episode_type:    str | None = None,
        task_id:         str | None = None,
        limit:           int = 100,
        offset:          int = 0,
    ) -> list[dict]:
        """
        Return a page of episodes as plain dicts for API serialisation.

        Does NOT update last_accessed — this is a neutral UI read.
        Results ordered by created_at DESC (newest first).

        Parameters
        ----------
        status :
            Filter by status column. "active" | "retracted" | "all".
            "all" returns every status.
        project_context :
            Optional filter by project_context column.
        episode_type :
            Optional filter by episode_type column.
        task_id :
            Optional filter by task_id column — the Episode Browsing UI's
            per-turn "related memory" overlay (episode-browsing-ui-plan.md
            Phase 6) uses this to find episodes stamped with the same
            task_id as a selected chat_turns row (implicit extraction
            passes task_id=task.task_id, see controller_agent.py's
            process_implicit_extraction call).
        limit :
            Max rows to return (capped at 200).
        offset :
            Row offset for pagination.
        """
        limit = min(limit, 200)
        conditions = []
        params: list = []

        if status != "all":
            conditions.append("status = ?")
            params.append(status)
        if project_context is not None:
            conditions.append("project_context = ?")
            params.append(project_context)
        if episode_type is not None:
            conditions.append("episode_type = ?")
            params.append(episode_type)
        if task_id is not None:
            conditions.append("task_id = ?")
            params.append(task_id)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT id, episode_type, subject, content, confidence,
                   source, task_id, project_context, status,
                   created_at, last_accessed
            FROM episodes
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with sqlite3.connect(str(self._db_path), timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def count_episodes(
        self,
        status:          str = "active",
        project_context: str | None = None,
        episode_type:    str | None = None,
        task_id:         str | None = None,
    ) -> int:
        """
        Return the total count of episodes matching the same filters as
        list_episodes(), ignoring limit/offset.

        Exists because GET /memory/episodes's `total` field needs the true
        matching-row count (e.g. for a pending-count badge that must show
        the real number even when the caller only wants a 1-row page) —
        list_episodes() alone can't provide that, since its result is
        already sliced by LIMIT/OFFSET by the time it returns.

        Parameters
        ----------
        status :
            Filter by status column. "all" returns every status.
        project_context :
            Optional filter by project_context column.
        episode_type :
            Optional filter by episode_type column.
        task_id :
            Optional filter by task_id column — see list_episodes().
        """
        conditions = []
        params: list = []

        if status != "all":
            conditions.append("status = ?")
            params.append(status)
        if project_context is not None:
            conditions.append("project_context = ?")
            params.append(project_context)
        if episode_type is not None:
            conditions.append("episode_type = ?")
            params.append(episode_type)
        if task_id is not None:
            conditions.append("task_id = ?")
            params.append(task_id)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT COUNT(*) FROM episodes {where}"

        with sqlite3.connect(str(self._db_path), timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            return conn.execute(sql, params).fetchone()[0]

    def __repr__(self) -> str:
        return (
            f"MemoryManager("
            f"db={self._db_path.name!r}, "
            f"embeddings={'on' if self._embed_fn else 'off'})"
        )


# ---------------------------------------------------------------------------
# EpisodicMemoryWriter
# ---------------------------------------------------------------------------

# Closed set of valid episode types. Adding a type is an architectural
# decision — do not expand this set without updating LOCALIST-Architecture.md.
VALID_EPISODE_TYPES: frozenset[str] = frozenset({
    "preference",
    "correction",
    "decision",
    "workflow",
    "project_fact",
    "task_completion",
    "naming_convention",
})

VALID_SOURCES: frozenset[str] = frozenset({
    "explicit",
    "model_extracted",
})

# Status lifecycle:
#   pending   -> active     (EpisodicMemoryWriter.approve())
#   pending   -> retracted  (EpisodicMemoryWriter.reject())
#   active    -> superseded (insert() superseding an older same-subject row)
#   active    -> retracted  (retract() / retract_by_id())
# "pending" is a staged, not-yet-real episode written when a caller (e.g.
# process_implicit_extraction() under episodic_write_approval) wants a
# model-inferred write to wait for manual review before it becomes live
# memory. All read paths (by_recency(), by_similarity(), best_match(),
# _write_memory_md()) filter on `status = 'active'`, so pending rows are
# excluded from retrieval/replay by construction — no separate filtering
# needed there.
VALID_STATUSES: frozenset[str] = frozenset({
    "active",
    "pending",
    "superseded",
    "retracted",
})


class EpisodicMemoryWriter:
    """
    Writes episodes to the `episodes` table in `lora_memory.db`.

    Responsibilities
    ----------------
    - insert()  : Write a new active episode, superseding any existing
                  active record with the same (subject, episode_type).
    - retract() : Mark an active episode as retracted by (subject, episode_type).
    - _supersede_existing() : Internal helper — marks conflicting active
                              records as superseded before a new insert.

    Architecture notes
    ------------------
    - This class owns no connection state. It receives a db_path and
      opens/closes connections per operation.
    - Thread-safety: a threading.Lock is acquired for every write. The same
      lock pattern used by MemoryManager is used here.
    - The episodes table follows an immutable audit trail: records are never
      deleted. Status transitions (active → superseded, active → retracted)
      are the only mutations.
    - When an ``embed_fn`` is supplied, every insert() embeds
      "{subject}. {content}" and stores the resulting vector in the
      `embedding` BLOB column so EpisodicMemoryReader.by_similarity() can
      do real cosine matching. Embedding failures are logged and swallowed
      — the row is still written with embedding=NULL and by_similarity()
      falls back to keyword scoring for it.
    - When ``memory_md_path`` is supplied, every insert()/retract() call
      regenerates a human-readable Markdown snapshot of all active
      episodes at that path (see _write_memory_md()).

    Parameters
    ----------
    db_path :
        Path to the SQLite file. Must be the same file used by MemoryManager.
    embed_fn :
        Optional ``Callable[[str], list[float]]`` (e.g. ``EmbeddingEngine.embed``).
        When None, episodes are written without embeddings and
        by_similarity() falls back to keyword-only scoring for them.
    memory_md_path :
        Optional path to a Markdown file that is regenerated after every
        write so the user has a single scrollable, human-readable view of
        active episodic memory. When None, no Markdown file is written.
    """

    def __init__(
        self,
        db_path:        Path | str,
        embed_fn:       "Callable[[str], list[float]] | None" = None,
        memory_md_path: Path | str | None = None,
    ) -> None:
        self._db_path       = Path(db_path)
        self._lock          = threading.Lock()
        self._embed_fn      = embed_fn
        self._memory_md_path = Path(memory_md_path) if memory_md_path is not None else None

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _supersede_existing(
        self,
        conn:         sqlite3.Connection,
        subject:      str,
        episode_type: str,
    ) -> int:
        """
        Mark all active records with (subject, episode_type) as superseded.

        Returns the number of rows updated.
        Called inside an existing write transaction before inserting a new
        active record.
        """
        cursor = conn.execute(
            """
            UPDATE episodes
            SET    status = 'superseded'
            WHERE  subject      = ?
              AND  episode_type = ?
              AND  status       = 'active'
            """,
            (subject, episode_type),
        )
        return cursor.rowcount

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def insert(
        self,
        episode_type:    str,
        subject:         str,
        content:         str,
        source:          str,
        confidence:      float      = 1.0,
        task_id:         str | None = None,
        conversation_id: str | None = None,
        project_context: str | None = "general",
        initial_status:  str        = "active",
    ) -> int | None:
        """
        Insert a new episode, active by default.

        If an active record with the same (subject, episode_type) already
        exists, it is marked 'superseded' before the new record is inserted.
        Both records are retained for audit.

        Before writing, `subject` and `content` are scanned by
        content_safety.scan_content() — episodes are replayed back into
        every future [EPISODIC MEMORY] prompt block and MEMORY.md, so
        content matching a known threat pattern (prompt injection,
        credential/secret-looking text, invisible Unicode) is rejected
        rather than stored. See content_safety.py for the pattern list.

        Parameters
        ----------
        episode_type :
            Must be one of VALID_EPISODE_TYPES.
        subject :
            What the episode is about. Used for exact-match retrieval and
            deduplication. Keep concise (< 80 chars recommended).
        content :
            The full text of the episode.
        source :
            Must be one of VALID_SOURCES.
        confidence :
            Float in [0, 1]. Defaults to 1.0 for explicit episodes;
            model-extracted episodes should supply a calibrated value.
        task_id :
            Optional task UUID linking the episode to a specific task session.
        conversation_id :
            Optional conversation UUID for traceability.
        project_context :
            Scope label (e.g. "general", "lora-app-demo"). Defaults to
            "general" so cross-project episodes are naturally grouped.
        initial_status :
            "active" (default) or "pending". "pending" stages the episode
            for manual review (see EpisodicMemoryWriter.approve()/reject())
            instead of making it live memory immediately — used by
            process_implicit_extraction() when episodic_write_approval is
            enabled. A pending insert does NOT supersede any existing
            active row with the same (subject, episode_type): an unreviewed
            model guess should never silently retire a confirmed fact.
            Superseding happens only once the pending row is approved into
            'active' status via a subsequent insert (or manually).

        Returns
        -------
        int | None
            The row id (``id`` column) of the newly inserted episode, or
            None if `subject`/`content` was rejected by the content safety
            scanner and nothing was written.

        Raises
        ------
        ValueError
            If episode_type, source, or initial_status is not in its
            respective valid set.
        """
        if episode_type not in VALID_EPISODE_TYPES:
            raise ValueError(
                f"episode_type {episode_type!r} not in VALID_EPISODE_TYPES. "
                f"Valid: {sorted(VALID_EPISODE_TYPES)}"
            )
        if source not in VALID_SOURCES:
            raise ValueError(
                f"source {source!r} not in VALID_SOURCES. "
                f"Valid: {sorted(VALID_SOURCES)}"
            )
        if initial_status not in {"active", "pending"}:
            raise ValueError(
                f"initial_status {initial_status!r} not in {{'active', 'pending'}}."
            )
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"confidence {confidence!r} out of range; must be in [0.0, 1.0]."
            )

        flagged_category = content_safety.scan_content(subject) or content_safety.scan_content(content)
        if flagged_category is not None:
            logger.warning(
                "insert: rejected by content safety scanner — "
                "category=%r episode_type=%r source=%r.",
                flagged_category, episode_type, source,
            )
            logger.debug(
                "insert: rejected content — subject=%r content=%r.",
                subject, content,
            )
            return None

        now = time.time()

        embedding_blob: bytes | None = None
        if self._embed_fn is not None:
            try:
                # Embed the first ~500 chars — consistent with
                # MemoryManager.index_document()'s embedding-truncation
                # convention (keeps embedding calls cheap). Generous
                # headroom here: subject is capped at 80 chars and content
                # is meant to be "one sentence preferred" per the episodes
                # schema, so this is not expected to bind in practice.
                vector = self._embed_fn(f"{subject}. {content}"[:500])
                embedding_blob = _pack_embedding(vector)
            except Exception as exc:
                logger.warning(
                    "insert: embed_fn failed for subject=%r (%s) — "
                    "storing without embedding.", subject, exc,
                )

        with self._lock:
            conn = self._connect()
            try:
                # A pending insert must not supersede an existing active
                # row for the same (subject, episode_type) — see the
                # initial_status docstring above. Supersession still runs
                # for ordinary active inserts exactly as before.
                if initial_status == "active":
                    superseded = self._supersede_existing(conn, subject, episode_type)
                    if superseded:
                        logger.debug(
                            "insert: superseded %d existing episode(s) for subject=%r type=%r.",
                            superseded, subject, episode_type,
                        )
                cursor = conn.execute(
                    """
                    INSERT INTO episodes
                        (episode_type, subject, content, confidence, source,
                         task_id, conversation_id, project_context,
                         status, created_at, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (episode_type, subject, content, confidence, source,
                     task_id, conversation_id, project_context, initial_status,
                     now, embedding_blob),
                )
                row_id = cursor.lastrowid
                conn.commit()
                logger.info(
                    "insert: episode id=%d type=%r subject=%r source=%r "
                    "status=%r embedded=%s.",
                    row_id, episode_type, subject, source, initial_status,
                    embedding_blob is not None,
                )
                self._write_memory_md(conn)
                return row_id
            finally:
                conn.close()

    def retract(
        self,
        subject:      str,
        episode_type: str,
    ) -> int:
        """
        Mark all active episodes with (subject, episode_type) as retracted.

        Unlike supersede — which is an automatic side-effect of insert — retract
        is an explicit operation signalling that the episode is wrong or no
        longer applicable. Retracted records are retained for audit.

        Parameters
        ----------
        subject :
            Exact subject string as stored.
        episode_type :
            Exact episode_type string as stored.

        Returns
        -------
        int
            Number of rows updated (0 if no matching active record exists).
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    UPDATE episodes
                    SET    status = 'retracted'
                    WHERE  subject      = ?
                      AND  episode_type = ?
                      AND  status       = 'active'
                    """,
                    (subject, episode_type),
                )
                count = cursor.rowcount
                conn.commit()
                logger.info(
                    "retract: retracted %d episode(s) for subject=%r type=%r.",
                    count, subject, episode_type,
                )
                if count:
                    self._write_memory_md(conn)
                return count
            finally:
                conn.close()

    def retract_by_id(self, episode_id: int) -> int:
        """
        Mark a single active episode as retracted by primary key.

        Retracting by id is strictly more precise than retract() (which
        matches on (subject, episode_type) — subject being a truncated,
        model-normalized content string, not a stable key). Intended for
        the semantic-match retraction path in episodic_extractor.py, which
        identifies the episode to retract via
        EpisodicMemoryReader.best_match() rather than a string match, so
        there is no ambiguity to resolve here — just retract the row that
        was already identified as the best match.

        Parameters
        ----------
        episode_id :
            Primary key of the episode to retract.

        Returns
        -------
        int
            1 if a matching active record was retracted, 0 if the id does
            not exist or is already inactive.
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    UPDATE episodes
                    SET    status = 'retracted'
                    WHERE  id     = ?
                      AND  status = 'active'
                    """,
                    (episode_id,),
                )
                count = cursor.rowcount
                conn.commit()
                logger.info(
                    "retract_by_id: retracted %d episode(s) for id=%d.",
                    count, episode_id,
                )
                if count:
                    self._write_memory_md(conn)
                return count
            finally:
                conn.close()

    def approve(self, episode_id: int) -> int:
        """
        Transition a pending episode to active — the write-approval gate's
        "yes" path (see episodic_write_approval / process_implicit_
        extraction()'s require_approval param). Once active, the episode is
        real memory: it's eligible for by_recency()/by_similarity()/
        best_match() and appears in MEMORY.md immediately.

        Does not run _supersede_existing() — a pending row approved after
        a newer active fact was written for the same (subject,
        episode_type) can end up alongside it as a second active row; this
        edge case is not resolved here (out of scope — see the write-
        approval task notes).

        Parameters
        ----------
        episode_id :
            Primary key of the episode to approve.

        Returns
        -------
        int
            1 if a matching pending record was approved, 0 if the id does
            not exist or is not currently pending (already approved,
            rejected, or was never staged).
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    UPDATE episodes
                    SET    status = 'active'
                    WHERE  id     = ?
                      AND  status = 'pending'
                    """,
                    (episode_id,),
                )
                count = cursor.rowcount
                conn.commit()
                logger.info(
                    "approve: approved %d episode(s) for id=%d.",
                    count, episode_id,
                )
                if count:
                    self._write_memory_md(conn)
                return count
            finally:
                conn.close()

    def reject(self, episode_id: int) -> int:
        """
        Transition a pending episode to retracted — the write-approval
        gate's "no" path. A rejected episode never becomes live memory.

        Parameters
        ----------
        episode_id :
            Primary key of the episode to reject.

        Returns
        -------
        int
            1 if a matching pending record was rejected, 0 if the id does
            not exist or is not currently pending.
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    UPDATE episodes
                    SET    status = 'retracted'
                    WHERE  id     = ?
                      AND  status = 'pending'
                    """,
                    (episode_id,),
                )
                count = cursor.rowcount
                conn.commit()
                logger.info(
                    "reject: rejected %d episode(s) for id=%d.",
                    count, episode_id,
                )
                if count:
                    # A rejected row was pending, never active, so it was
                    # never in MEMORY.md — regenerating here is a no-op in
                    # practice, kept only for symmetry with approve()/
                    # retract_by_id().
                    self._write_memory_md(conn)
                return count
            finally:
                conn.close()

    def reactivate(self, episode_id: int) -> int:
        """
        Transition a retracted episode back to active — the reversal path
        for retract()/retract_by_id()/reject() and the global-retention TTL
        sweep (MemoryManager.sweep_expired_memory()), all of which land a
        row at status='retracted'. No distinction is made between "was
        active, then retracted" and "was pending, then rejected" — both are
        status='retracted' today, and reactivate() treats them the same.

        Unlike approve() (which does NOT run _supersede_existing() and
        documents that a second active row for the same (subject,
        episode_type) is an unresolved edge case), reactivate() DOES run
        it: if a different episode is currently active for this row's
        (subject, episode_type) — e.g. a correction was written after this
        one was retracted — that newer active row is marked superseded
        first, so reactivating never results in two simultaneously active
        rows for the same subject. This is a deliberate choice: approving a
        pending row is accepting a new, previously-unreviewed guess (safer
        to leave ambiguity for a human to resolve), while reactivating is a
        deliberate choice to restore this exact row as canonical.

        Parameters
        ----------
        episode_id :
            Primary key of the episode to reactivate.

        Returns
        -------
        int
            1 if a matching retracted record was reactivated, 0 if the id
            does not exist or is not currently retracted.
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT subject, episode_type FROM episodes
                    WHERE  id = ? AND status = 'retracted'
                    """,
                    (episode_id,),
                ).fetchone()
                if row is None:
                    return 0

                superseded = self._supersede_existing(conn, row["subject"], row["episode_type"])
                cursor = conn.execute(
                    """
                    UPDATE episodes
                    SET    status = 'active'
                    WHERE  id     = ?
                      AND  status = 'retracted'
                    """,
                    (episode_id,),
                )
                count = cursor.rowcount
                conn.commit()
                logger.info(
                    "reactivate: reactivated %d episode(s) for id=%d "
                    "(superseded %d conflicting active row(s)).",
                    count, episode_id, superseded,
                )
                if count:
                    self._write_memory_md(conn)
                return count
            finally:
                conn.close()

    def get_episode_subject(self, episode_id: int) -> str | None:
        """
        Look up the ``subject`` of a single episode by id.

        Purpose-built for the approve() call site in main.py's
        POST /memory/episodes/{id}/approve endpoint: approve() itself
        returns only a count (unchanged, to avoid touching its existing
        contract), but the caller needs `subject` to feed
        ControllerAgent._write_episode_graph_node() and retrigger the
        Phase B graph hook now that the episode is live. Not used by
        insert()/approve()/reject() themselves. Returns None if no row
        with that id exists.
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT subject FROM episodes WHERE id = ?",
                    (episode_id,),
                ).fetchone()
            finally:
                conn.close()
        return row["subject"] if row is not None else None

    # -----------------------------------------------------------------------
    # Markdown snapshot
    # -----------------------------------------------------------------------

    def regenerate_memory_md(self) -> None:
        """
        Public entry point to (re)build MEMORY.md on demand — opens its own
        connection and lock, unlike _write_memory_md() which expects to run
        inside an existing insert()/retract() transaction. Intended for
        startup (so the file exists even before the next write) and for
        one-off backfill scripts. No-op if memory_md_path was not supplied.
        """
        if self._memory_md_path is None:
            return
        with self._lock:
            conn = self._connect()
            try:
                self._write_memory_md(conn)
            finally:
                conn.close()

    def _write_memory_md(self, conn: sqlite3.Connection) -> None:
        """
        Regenerate the human-readable Markdown snapshot of active episodic
        memory at ``self._memory_md_path``. No-op if that path is None.

        Reads all `status = 'active'` rows (fresh SELECT — cheap for the
        episode-count scale this system targets), groups them by the
        calendar date of `created_at`, newest date first, and within a date
        newest-first. This mirrors exactly what the Memory UI tab shows
        today (type badge, content, confidence, project_context, source,
        date) so the file and the UI never disagree.

        Called with the write lock already held by the caller (insert()/
        retract()), on the same connection, so this does not re-acquire
        self._lock or open a second connection.
        """
        if self._memory_md_path is None:
            return

        try:
            rows = conn.execute(
                """
                SELECT * FROM episodes
                WHERE  status = 'active'
                ORDER  BY created_at DESC
                """
            ).fetchall()

            from collections import OrderedDict
            groups: "OrderedDict[str, list[sqlite3.Row]]" = OrderedDict()
            for row in rows:
                day = datetime.fromtimestamp(row["created_at"]).strftime("%B %-d, %Y")
                groups.setdefault(day, []).append(row)

            lines: list[str] = [
                "# Memory",
                "",
                "_Auto-generated from Localist's episodic memory store — do not "
                "edit by hand, changes will be overwritten on the next write. "
                f"Last updated: {datetime.now().strftime('%B %-d, %Y %H:%M')}._",
                "",
            ]
            if not rows:
                lines.append("_No active memories yet._")
            for day, day_rows in groups.items():
                lines.append(f"## {day}")
                lines.append("")
                for row in day_rows:
                    conf_pct = round(row["confidence"] * 100)
                    lines.append(
                        f"- **{row['episode_type']}** ({conf_pct}%, "
                        f"{row['project_context'] or 'general'}, {row['source']}) — "
                        f"{row['content']}"
                    )
                lines.append("")

            self._memory_md_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_md_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            logger.warning(
                "_write_memory_md: failed to write %s (%s) — continuing.",
                self._memory_md_path, exc,
            )


# ---------------------------------------------------------------------------
# EpisodeRecord — return type for all EpisodicMemoryReader retrieval methods
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    id:              int
    episode_type:    str
    subject:         str
    content:         str
    confidence:      float
    source:          str
    task_id:         str | None
    conversation_id: str | None
    project_context: str | None
    status:          str
    created_at:      float
    last_accessed:   float | None


# ---------------------------------------------------------------------------
# WorkingStateRecord — return type for WorkingStateStore.get()
# ---------------------------------------------------------------------------

@dataclass
class WorkingStateRecord:
    mem_key:          str
    current_focus:    str | None
    open_loops:       list[str]
    recent_decisions: list[str]
    updated_at:       float


# ---------------------------------------------------------------------------
# EpisodicMemoryReader
# ---------------------------------------------------------------------------

class EpisodicMemoryReader:
    """
    Reads episodes from the `episodes` table in `lora_memory.db`.

    Implements the three retrieval modes defined in §2.6 of
    LOCALIST-Architecture.md:

      Mode 1 — Exact subject match
        By subject string. Returns up to 5 active records ordered by
        confidence DESC, created_at DESC.

      Mode 2 — Type-filtered recency
        High-priority types (preference, correction, decision, workflow)
        for a given project_context. Returns up to 5 active records ordered
        by last_accessed DESC, confidence DESC. Used for session priming.

      Mode 3 — Semantic similarity
        Open-ended query. Cosine ranking over the `embedding` column when
        embeddings are present; falls back to keyword overlap scoring when
        they are absent.

    Side effect — last_accessed update
        Every record returned by any retrieval method has its `last_accessed`
        field updated to the current Unix timestamp. by_subject() and
        by_recency() write this in a single batched UPDATE inside the same
        lock as their SELECT. by_similarity() and best_match() score via
        the read-only _score_all_active() helper first, then apply the
        batched UPDATE in a separate, immediately-following lock scope
        (see _touch_records()) — same net effect, not one atomic
        transaction.

    Parameters
    ----------
    db_path :
        Path to the SQLite file. Must be the same file used by MemoryManager.
    embed_fn :
        Optional ``Callable[[str], list[float]]`` (e.g. ``EmbeddingEngine.embed``).
        When supplied, by_similarity() embeds the query and does real cosine
        comparison against any row that has a stored embedding, falling back
        to keyword overlap for un-embedded rows. When None, by_similarity()
        runs keyword-only, as before.
    """

    # Types pulled for session priming (Mode 2).
    _PRIME_TYPES: tuple[str, ...] = (
        "preference", "correction", "decision", "workflow"
    )

    def __init__(
        self,
        db_path:  Path | str,
        embed_fn: "Callable[[str], list[float]] | None" = None,
    ) -> None:
        self._db_path  = Path(db_path)
        self._lock     = threading.Lock()
        self._embed_fn = embed_fn

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _row_to_record(self, row: sqlite3.Row) -> EpisodeRecord:
        return EpisodeRecord(
            id              = row["id"],
            episode_type    = row["episode_type"],
            subject         = row["subject"],
            content         = row["content"],
            confidence      = row["confidence"],
            source          = row["source"],
            task_id         = row["task_id"],
            conversation_id = row["conversation_id"],
            project_context = row["project_context"],
            status          = row["status"],
            created_at      = row["created_at"],
            last_accessed   = row["last_accessed"],
        )

    def _touch_last_accessed(
        self,
        conn: sqlite3.Connection,
        ids:  list[int],
    ) -> float | None:
        """
        Batch-update last_accessed for the given row ids.
        Called inside an existing write transaction.
        Returns the timestamp written, or None when ids is empty.
        """
        if not ids:
            return None
        now = time.time()
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"UPDATE episodes SET last_accessed = ? WHERE id IN ({placeholders})",
            [now, *ids],
        )
        return now

    # -----------------------------------------------------------------------
    # Public API — three retrieval modes
    # -----------------------------------------------------------------------

    def by_subject(self, subject: str) -> list[EpisodeRecord]:
        """
        Mode 1 — Exact subject match.

        Returns up to 5 active episodes whose subject equals `subject`,
        ordered by confidence DESC then created_at DESC.

        Parameters
        ----------
        subject :
            Exact string to match against the `subject` column.

        Returns
        -------
        list[EpisodeRecord]
            May be empty if no active record matches.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM episodes
                    WHERE  subject = ?
                      AND  status  = 'active'
                    ORDER  BY confidence DESC, created_at DESC
                    LIMIT  5
                    """,
                    (subject,),
                ).fetchall()

                records = [self._row_to_record(r) for r in rows]
                now = self._touch_last_accessed(conn, [r.id for r in records])
                if now is not None:
                    for rec in records:
                        rec.last_accessed = now
                conn.commit()
                return records
            finally:
                conn.close()

    def by_recency(
        self,
        project_context: str = "general",
    ) -> list[EpisodeRecord]:
        """
        Mode 2 — Type-filtered recency (session priming).

        Returns up to 5 active episodes of types
        (preference, correction, decision, workflow) for the given
        project_context, ordered by last_accessed DESC then confidence DESC.

        Parameters
        ----------
        project_context :
            Scopes retrieval to a project. Defaults to "general".

        Returns
        -------
        list[EpisodeRecord]
            May be empty.
        """
        placeholders = ",".join("?" * len(self._PRIME_TYPES))

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT * FROM episodes
                    WHERE  episode_type IN ({placeholders})
                      AND  status          = 'active'
                      AND  project_context = ?
                    ORDER  BY last_accessed DESC, confidence DESC
                    LIMIT  5
                    """,
                    (*self._PRIME_TYPES, project_context),
                ).fetchall()

                records = [self._row_to_record(r) for r in rows]
                now = self._touch_last_accessed(conn, [r.id for r in records])
                if now is not None:
                    for rec in records:
                        rec.last_accessed = now
                conn.commit()
                return records
            finally:
                conn.close()

    def _score_all_active(
        self, query: str,
    ) -> list[tuple[float, EpisodeRecord, bool]]:
        """
        Shared scoring core for by_similarity() and best_match().

        Scores every active episode against `query`, across every
        episode_type (unlike by_recency(), which is scoped to
        _PRIME_TYPES) — this is what makes project_fact / task_completion /
        naming_convention episodes reachable at all.

        If an embed_fn was injected at construction time, the query is
        embedded once and compared via true cosine similarity against every
        row that has a stored embedding (see EpisodicMemoryWriter.insert(),
        which populates the `embedding` column when it also holds an
        embed_fn). Rows without a stored embedding — or every row, if no
        embed_fn is available — fall back to BM25 keyword scoring (bm25.py),
        computed once over every active episode fetched this call (mirrors
        query_corpus()'s stage-1 prefilter — same per-call IDF/avg-length
        scoping, no persisted corpus stats). BM25 scores are raw and
        unbounded (bm25.py deliberately doesn't normalize them — see its
        module docstring), so each result carries a `scored_by_embedding`
        flag alongside its score: by_similarity() uses it to skip its
        min_score floor for BM25-scored episodes (trusting BM25's relative
        ranking instead, the same policy query_corpus() applies — see
        DocumentResult.scored_by_embedding); best_match() deliberately does
        NOT skip its floor for BM25 scores, since a false-positive
        retraction is a correctness risk a false-negative recall miss
        isn't (see best_match()'s docstring).

        Returns every active episode with its score, sorted descending.
        Callers are responsible for any min_score filtering, top_n
        slicing, and last_accessed side effects — this method only reads,
        never writes.

        Parameters
        ----------
        query :
            Free-text search string.

        Returns
        -------
        list[tuple[float, EpisodeRecord, bool]]
            (score, record, scored_by_embedding) for every active episode,
            sorted by score descending. May be empty.
        """
        query_vec: list[float] | None = None
        if self._embed_fn is not None:
            try:
                query_vec = self._embed_fn(query)
            except Exception as exc:
                logger.warning(
                    "_score_all_active: embed_fn failed for query %r (%s) — "
                    "falling back to keyword scoring.", query[:40], exc,
                )

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM episodes
                    WHERE  status = 'active'
                    """
                ).fetchall()

                records = [self._row_to_record(row) for row in rows]
                bm25_scores = bm25.score_documents(
                    query,
                    [(r.id, f"{r.subject} {r.content}") for r in records],
                )

                scored: list[tuple[float, EpisodeRecord, bool]] = []
                for row, record in zip(rows, records):
                    score: float | None = None
                    by_embedding = False

                    if query_vec is not None and row["embedding"] is not None:
                        try:
                            doc_vec = _unpack_embedding(row["embedding"])
                            score = _cosine_similarity(query_vec, doc_vec)
                            by_embedding = True
                        except Exception as exc:
                            logger.debug(
                                "_score_all_active: cosine failed for episode "
                                "id=%s (%s) — falling back to keyword.",
                                record.id, exc,
                            )
                            score = None

                    if score is None:
                        score = bm25_scores.get(record.id, 0.0)
                        by_embedding = False

                    scored.append((score, record, by_embedding))

                scored.sort(key=lambda x: x[0], reverse=True)
                return scored
            finally:
                conn.close()

    def _touch_records(self, records: list[EpisodeRecord]) -> None:
        """
        Update last_accessed for the given records, in place, in their own
        lock/connection scope. Shared tail step for by_similarity() and
        best_match() after they've picked their result set from
        _score_all_active(). No-op for an empty list.
        """
        if not records:
            return
        with self._lock:
            conn = self._connect()
            try:
                now = self._touch_last_accessed(conn, [rec.id for rec in records])
                if now is not None:
                    for rec in records:
                        rec.last_accessed = now
                conn.commit()
            finally:
                conn.close()

    def by_similarity(
        self,
        query:           str,
        top_n:           int   = 5,
        min_score:       float = 0.0,
        exclude_task_id: str | None = None,
    ) -> list[EpisodeRecord]:
        """
        Mode 3 — Semantic similarity.

        Scores all active episodes against `query` via _score_all_active(),
        filters to those scoring >= min_score, and returns the top_n.

        Parameters
        ----------
        query :
            Free-text search string.
        top_n :
            Maximum records to return. Default 5.
        min_score :
            Minimum score threshold, applied only to cosine-scored episodes.
            An episode scored via BM25 keyword fallback (no embedding, or
            no embed_fn available) skips this floor entirely — BM25 scores
            are raw and unbounded (see bm25.py's module docstring), not on
            a fixed scale comparable to cosine similarity, so an absolute
            threshold tuned for one doesn't mean anything for the other.
            Trusting BM25's relative ranking instead (same policy
            query_corpus() applies to RAG sources) means top_n still
            reflects "most relevant of what's here" in keyword-only mode,
            rather than everything failing a threshold it was never
            calibrated against.
        exclude_task_id :
            When supplied, episodes whose `task_id` equals this value are
            dropped before the top_n slice. Used by the Related Memory
            lookup (backend/main.py's GET /memory/episodes/related) to
            keep a turn's own episode(s) — trivially related, since they
            were extracted from the query text itself — out of its own
            "related" results.

        Returns
        -------
        list[EpisodeRecord]
            Sorted by relevance score descending. May be empty.
        """
        scored  = self._score_all_active(query)
        filtered = [
            (score, rec) for score, rec, by_embedding in scored
            if (not by_embedding or score >= min_score)
            and (exclude_task_id is None or rec.task_id != exclude_task_id)
        ]
        records = [rec for _, rec in filtered[:top_n]]

        self._touch_records(records)

        logger.debug(
            "by_similarity: returning %d records for query '%s...'.",
            len(records), query[:40],
        )
        return records

    def best_match(
        self,
        query:     str,
        min_score: float = 0.55,
    ) -> tuple[float, EpisodeRecord] | None:
        """
        Returns the single highest-scoring active episode above min_score,
        or None if nothing scores that high (or no active episodes exist).

        Built on the same cosine-with-keyword-fallback scoring as
        by_similarity() (via _score_all_active()), but returns at most one
        record together with its score. Intended for callers — like
        retraction — that need to identify *the* episode a query refers to
        rather than a ranked list of candidates. min_score defaults higher
        than by_similarity()'s typical recall thresholds (0.45 elsewhere in
        the codebase) since a caller acting on a single best_match() result
        has no second chance to disambiguate among candidates the way a
        ranked list would allow.

        Unlike by_similarity(), this floor is applied unconditionally —
        even to a BM25-scored (keyword-fallback) top candidate, despite
        BM25's raw score not being on a scale actually calibrated against
        0.55. by_similarity() skips its floor for BM25 scores and trusts
        ranking instead, which is the right tradeoff for recall (a low-
        confidence-but-plausible episode surfacing among several candidates
        is a minor miss at worst); best_match() is used for retraction,
        where a false positive silently destroys the wrong memory — the
        floor stays a deliberately blunt "no confident single match" guard
        here, at the cost of degrading to the exact (subject, episode_type)
        fallback loop more often in keyword-only mode than by_similarity()'s
        equivalent case would.

        Parameters
        ----------
        query :
            Free-text search string.
        min_score :
            Minimum score the top candidate must clear. Default 0.55.

        Returns
        -------
        tuple[float, EpisodeRecord] | None
            (score, record) for the highest scorer, or None.
        """
        scored = self._score_all_active(query)
        if not scored:
            return None

        top_score, top_record, _ = scored[0]
        if top_score < min_score:
            return None

        self._touch_records([top_record])
        return top_score, top_record

    def get_by_ids(self, ids: list[int]) -> list[EpisodeRecord]:
        """
        Exact-id batch lookup, restricted to active episodes.

        Used by Mode 4 (graph-neighbor expansion, controller_agent.py Step
        5) to resolve sibling episode ids discovered via the episode
        graph-node backlink traversal (see 08-graph-retrieval-layer.md §8.9)
        back into full EpisodeRecords. Not one of the three query-shaped
        retrieval modes in §2.6 — a plain id lookup, no scoring involved.

        Parameters
        ----------
        ids :
            Episode primary keys to fetch. Empty input returns [] with no
            query executed.

        Returns
        -------
        list[EpisodeRecord]
            Only rows with status='active'; order is not guaranteed to
            match `ids`. Shorter than `ids` if some are missing/inactive.
        """
        if not ids:
            return []

        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(ids))
                rows = conn.execute(
                    f"""
                    SELECT * FROM episodes
                    WHERE  id IN ({placeholders})
                      AND  status = 'active'
                    """,
                    ids,
                ).fetchall()

                records = [self._row_to_record(r) for r in rows]
                now = self._touch_last_accessed(conn, [r.id for r in records])
                if now is not None:
                    for rec in records:
                        rec.last_accessed = now
                conn.commit()
                return records
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# WorkingStateStore — Slot 6A Tier 2 persistent storage
# ---------------------------------------------------------------------------

class WorkingStateStore:
    """
    Reads and writes the `working_state` table in `lora_memory.db`.

    One row per mem_key (the same session-grouping key used by
    conversation_log and get_context_window). Tier 2 fields only —
    current_focus, open_loops, recent_decisions.
    The two list fields are stored as JSON-encoded strings; this class
    encodes/decodes at the boundary so callers always deal in Python lists.

    Architecture notes
    ------------------
    - No persistent connection. Opens/closes per call (WAL concurrent reads).
    - Threading lock acquired for all writes (upsert, clear). Reads are
      lock-free — WAL mode permits concurrent readers.
    - Same _connect() PRAGMA set as EpisodicMemoryWriter.
    - This class is storage only. No inference, no controller wiring.

    Parameters
    ----------
    db_path :
        Path to the SQLite file. Must be the same file used by MemoryManager.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._lock    = threading.Lock()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get(self, mem_key: str) -> WorkingStateRecord | None:
        """
        Read the working state for a session.

        Returns None when no row exists for mem_key — callers can
        distinguish "never written" from "written but empty".

        Parameters
        ----------
        mem_key :
            Session-grouping key (same key used by conversation_log).

        Returns
        -------
        WorkingStateRecord | None
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM working_state WHERE mem_key = ?", (mem_key,)
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        return WorkingStateRecord(
            mem_key          = row["mem_key"],
            current_focus    = row["current_focus"],
            open_loops       = json.loads(row["open_loops_json"]),
            recent_decisions = json.loads(row["recent_decisions_json"]),
            updated_at       = row["updated_at"],
        )

    def upsert(
        self,
        mem_key:          str,
        current_focus:    str | None,
        open_loops:       list[str],
        recent_decisions: list[str],
    ) -> None:
        """
        Insert or update the working state for a session.

        If a row already exists for mem_key, all fields are overwritten
        (ON CONFLICT DO UPDATE). updated_at is always set to time.time().

        Parameters
        ----------
        mem_key :
            Session-grouping key.
        current_focus :
            Current task focus string, or None when unset.
        open_loops :
            List of unresolved topic strings.
        recent_decisions :
            List of recent decision strings.
        """
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO working_state
                        (mem_key, current_focus, open_loops_json,
                         recent_decisions_json, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(mem_key) DO UPDATE SET
                        current_focus         = excluded.current_focus,
                        open_loops_json       = excluded.open_loops_json,
                        recent_decisions_json = excluded.recent_decisions_json,
                        updated_at            = excluded.updated_at
                    """,
                    (
                        mem_key,
                        current_focus,
                        json.dumps(open_loops),
                        json.dumps(recent_decisions),
                        now,
                    ),
                )
                conn.commit()
                logger.info(
                    "upsert: working_state written for mem_key=%r.", mem_key
                )
            finally:
                conn.close()

    def clear(self, mem_key: str) -> int:
        """
        Delete the working state row for a session.

        Parameters
        ----------
        mem_key :
            Session-grouping key to delete.

        Returns
        -------
        int
            Number of rows deleted (0 if no row existed for mem_key).
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM working_state WHERE mem_key = ?", (mem_key,)
                )
                count = cursor.rowcount
                conn.commit()
                logger.info(
                    "clear: deleted %d working_state row(s) for mem_key=%r.",
                    count, mem_key,
                )
                return count
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Episodic summarization contract  (§2.7 of LOCALIST-Architecture.md)
# ---------------------------------------------------------------------------

# Priority order for bullet ranking. Lower index = higher priority.
# This ordering is load-bearing: it determines which episodes reach the
# model when the 5-bullet cap forces a cut.
_EPISODE_TYPE_PRIORITY: dict[str, int] = {
    "correction":        0,
    "decision":          1,
    "preference":        2,
    "workflow":          3,
    "project_fact":      4,
    "naming_convention": 5,
    "task_completion":   6,
}

# Approximation: one token ≈ 4 characters (conservative for English prose).
# Used to enforce the 20-token-per-bullet ceiling without importing a
# tokeniser. The ceiling is a budget constraint, not a display preference.
_CHARS_PER_TOKEN = 4
_MAX_TOKENS_PER_BULLET = 20
_MAX_BULLET_CHARS = _MAX_TOKENS_PER_BULLET * _CHARS_PER_TOKEN  # 80 chars


def format_episodic_summary(
    episodes:          list["EpisodeRecord"],
    max_bullets:       int   = 5,
    min_confidence:    float = 0.7,
) -> str:
    """
    Format a list of EpisodeRecords into the canonical episodic memory block
    for prompt injection, as specified in §2.7 of LOCALIST-Architecture.md.

    Contract rules enforced
    -----------------------
    1. Only `active` episodes are eligible (callers should pre-filter, but
       this function enforces it defensively).
    2. Only episodes with confidence >= min_confidence (default 0.7) are
       included.
    3. Episodes are sorted by type priority (correction first, task_completion
       last), then by confidence descending as a tiebreaker.
    4. At most `max_bullets` (default 5) bullets are emitted.
    5. Each bullet's content is hard-truncated at 80 characters (≈20 tokens)
       with an ellipsis if truncated. The type annotation and confidence
       score are appended after the content and are NOT counted toward the
       80-char limit.
    6. If no episodes survive filtering, an empty string is returned (no
       label, no placeholder — the caller omits the slot entirely).

    Output format
    -------------
    [EPISODIC MEMORY]
    - {content} ({episode_type}, {confidence:.1f})
    - {content} ({episode_type}, {confidence:.1f})
    ...

    The `[EPISODIC MEMORY]` label and the inline annotation are mandatory.
    The confidence score is formatted to one decimal place.

    Parameters
    ----------
    episodes :
        List of EpisodeRecord objects. Typically the output of one of the
        EpisodicMemoryReader retrieval methods.
    max_bullets :
        Maximum number of bullets to emit. Default 5.
    min_confidence :
        Minimum confidence threshold. Records below this are excluded.
        Default 0.7.

    Returns
    -------
    str
        The formatted block, or "" if no eligible episodes exist.
    """
    # Step 1: filter — active status and confidence threshold
    eligible = [
        ep for ep in episodes
        if ep.status == "active" and ep.confidence >= min_confidence
    ]

    if not eligible:
        return ""

    # Step 2: sort — type priority ASC (lower = higher priority),
    #                confidence DESC as tiebreaker
    eligible.sort(
        key=lambda ep: (
            _EPISODE_TYPE_PRIORITY.get(ep.episode_type, 99),
            -ep.confidence,
        )
    )

    # Step 3: cap at max_bullets
    top = eligible[:max_bullets]

    # Step 4: format each bullet
    lines = ["[EPISODIC MEMORY]"]
    for ep in top:
        content = ep.content
        if len(content) > _MAX_BULLET_CHARS:
            content = content[: _MAX_BULLET_CHARS - 1] + "…"
        lines.append(
            f"- {content} ({ep.episode_type}, {ep.confidence:.1f})"
        )

    return "\n".join(lines)
