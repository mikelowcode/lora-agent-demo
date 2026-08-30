"""
Tests for chat_turns semantic search (v8->v9 migration adding the
`embedding` column, add_chat_turn()'s embed-on-write path, 'chat_turns'
embedding-provenance tracking, get_chat_turns(mode="semantic"), and
reembed_chat_turns()) — the Episode Browsing UI plan's Phase 1
(episode-browsing-ui-plan.md).

Mirrors the patterns already established in test_embedding_provenance.py
('corpus' store) and test_chat_turns_schema.py (chat_turns write/read
paths) rather than inventing new conventions.
"""

import logging
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localist.memory_manager import (
    MemoryManager,
    _pack_embedding,
    _unpack_embedding,
    _EMBEDDING_DIM,
    _SCHEMA_VERSION,
)


_TUNED = "mlx-community/embeddinggemma-300m-4bit"
_OTHER = "nomic-embed-text"


def _stub_embed_fn(vector_value: float = 0.1):
    return MagicMock(side_effect=lambda text: [vector_value] * _EMBEDDING_DIM)


def _create_v8_db(path: Path) -> None:
    """Create a v8-schema SQLite database (chat_turns with no embedding column)."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (8);
        CREATE TABLE document_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            doc_type TEXT NOT NULL,
            content TEXT NOT NULL,
            token_set TEXT NOT NULL DEFAULT '',
            embedding BLOB DEFAULT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            indexed_at REAL NOT NULL
        );
        CREATE TABLE retrieval_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_hash TEXT NOT NULL UNIQUE,
            top_n INTEGER NOT NULL,
            result_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            valid INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE embedding_provenance (
            store TEXT NOT NULL PRIMARY KEY,
            model TEXT NOT NULL
        );
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_type TEXT NOT NULL,
            subject TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            source TEXT NOT NULL,
            task_id TEXT,
            conversation_id TEXT,
            project_context TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at REAL NOT NULL,
            last_accessed REAL,
            embedding BLOB
        );
        CREATE TABLE graph_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_path TEXT NOT NULL UNIQUE,
            node_type TEXT,
            title TEXT,
            source_doc_path TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_node_id INTEGER NOT NULL REFERENCES graph_nodes(id),
            target_path TEXT NOT NULL,
            target_node_id INTEGER REFERENCES graph_nodes(id),
            target_resolved INTEGER NOT NULL DEFAULT 0,
            link_text TEXT NOT NULL,
            source_doc_path TEXT NOT NULL
        );
        CREATE TABLE working_state (
            mem_key TEXT PRIMARY KEY,
            current_focus TEXT,
            open_loops_json TEXT NOT NULL DEFAULT '[]',
            recent_decisions_json TEXT NOT NULL DEFAULT '[]',
            updated_at REAL NOT NULL
        );
        CREATE TABLE chat_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources_json TEXT NOT NULL DEFAULT '[]',
            status_message TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            conversation_id TEXT NOT NULL DEFAULT 'legacy',
            conversation_title TEXT,
            created_at REAL NOT NULL
        );
        CREATE VIRTUAL TABLE chat_turns_fts USING fts5(
            content, content='chat_turns', content_rowid='id'
        );
        CREATE TRIGGER chat_turns_ai AFTER INSERT ON chat_turns BEGIN
            INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);
        END;
        CREATE TRIGGER chat_turns_ad AFTER DELETE ON chat_turns BEGIN
            INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) VALUES ('delete', old.id, old.content);
        END;
        CREATE TRIGGER chat_turns_au AFTER UPDATE ON chat_turns BEGIN
            INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) VALUES ('delete', old.id, old.content);
            INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);
        END;
        CREATE TABLE chat_history_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            eviction_preset TEXT
        );
    """)
    conn.close()


def _set_provenance(path: Path, store: str, model: str) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO embedding_provenance (store, model) VALUES (?, ?)", (store, model)
    )
    conn.commit()
    conn.close()


def _get_provenance(path: Path, store: str) -> str | None:
    conn = sqlite3.connect(str(path))
    row = conn.execute(
        "SELECT model FROM embedding_provenance WHERE store = ?", (store,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _insert_chat_turn_raw(
    path: Path, *, content: str, embedded: bool, conversation_id: str = "conv-1",
    created_at: float | None = None,
) -> None:
    conn = sqlite3.connect(str(path))
    blob = _pack_embedding([0.1] * _EMBEDDING_DIM) if embedded else None
    conn.execute(
        """
        INSERT INTO chat_turns (task_id, role, content, conversation_id, embedding, created_at)
        VALUES ('t', 'user', ?, ?, ?, ?)
        """,
        (content, conversation_id, blob, created_at if created_at is not None else time.time()),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Migration v8->v9 — embedding column on chat_turns
# ---------------------------------------------------------------------------

class TestChatTurnsEmbeddingMigration:

    def test_fresh_db_chat_turns_has_embedding_column(self, tmp_path):
        path = tmp_path / "fresh.db"
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chat_turns)").fetchall()}
        conn.close()

        assert "embedding" in cols

    def test_v8_migration_adds_embedding_column(self, tmp_path):
        path = tmp_path / "migrate.db"
        _create_v8_db(path)

        conn = sqlite3.connect(str(path))
        cols_before = {row[1] for row in conn.execute("PRAGMA table_info(chat_turns)").fetchall()}
        conn.close()
        assert "embedding" not in cols_before

        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        version_after = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        cols_after = {row[1] for row in conn.execute("PRAGMA table_info(chat_turns)").fetchall()}
        conn.close()

        assert version_after == _SCHEMA_VERSION
        assert "embedding" in cols_after

    def test_v8_migration_preserves_existing_rows(self, tmp_path):
        path = tmp_path / "migrate.db"
        _create_v8_db(path)
        conn = sqlite3.connect(str(path))
        conn.execute(
            "INSERT INTO chat_turns (task_id, role, content, created_at) VALUES ('t', 'user', 'hi', ?)",
            (time.time(),),
        )
        conn.commit()
        conn.close()

        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        row = conn.execute("SELECT content, embedding FROM chat_turns").fetchone()
        conn.close()
        assert row[0] == "hi"
        assert row[1] is None


# ---------------------------------------------------------------------------
# add_chat_turn() — embed-on-write path
# ---------------------------------------------------------------------------

class TestAddChatTurnEmbedding:

    def test_embeds_content_when_embed_fn_configured(self, tmp_path):
        embed_fn = _stub_embed_fn(0.5)
        mm = MemoryManager(
            db_path=tmp_path / "embed.db", embed_fn=embed_fn, embedding_model_name=_TUNED,
        )
        mm.add_chat_turn(task_id="t", role="user", content="hello world", conversation_id="c")

        conn = sqlite3.connect(str(mm._db_path))
        blob = conn.execute("SELECT embedding FROM chat_turns").fetchone()[0]
        conn.close()

        assert blob is not None
        assert _unpack_embedding(blob) == pytest.approx([0.5] * _EMBEDDING_DIM)
        embed_fn.assert_called_once_with("hello world")

    def test_no_embed_fn_leaves_embedding_null(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "no_embed.db")
        mm.add_chat_turn(task_id="t", role="user", content="hello", conversation_id="c")

        conn = sqlite3.connect(str(mm._db_path))
        blob = conn.execute("SELECT embedding FROM chat_turns").fetchone()[0]
        conn.close()
        assert blob is None

    def test_content_truncated_to_500_chars_before_embedding(self, tmp_path):
        embed_fn = _stub_embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "trunc.db", embed_fn=embed_fn, embedding_model_name=_TUNED,
        )
        long_content = "x" * 600
        mm.add_chat_turn(task_id="t", role="user", content=long_content, conversation_id="c")

        embed_fn.assert_called_once_with("x" * 500)

    def test_embed_failure_leaves_null_embedding_write_still_succeeds(self, tmp_path):
        embed_fn = MagicMock(side_effect=RuntimeError("boom"))
        mm = MemoryManager(
            db_path=tmp_path / "fail.db", embed_fn=embed_fn, embedding_model_name=_TUNED,
        )
        mm.add_chat_turn(task_id="t", role="user", content="hello", conversation_id="c")

        conn = sqlite3.connect(str(mm._db_path))
        row = conn.execute("SELECT content, embedding FROM chat_turns").fetchone()
        conn.close()
        assert row[0] == "hello"
        assert row[1] is None

    def test_first_embedded_write_seeds_chat_turns_provenance(self, tmp_path):
        embed_fn = _stub_embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "seed.db", embed_fn=embed_fn, embedding_model_name=_TUNED,
        )
        assert _get_provenance(mm._db_path, "chat_turns") is None

        mm.add_chat_turn(task_id="t", role="user", content="hello", conversation_id="c")

        assert _get_provenance(mm._db_path, "chat_turns") == _TUNED

    def test_no_embed_fn_row_shape_unchanged_no_score_key(self, tmp_path):
        """Regression guard: keyword-mode callers must not see a new
        'score' key leak into row dicts — only mode="semantic" adds it."""
        mm = MemoryManager(db_path=tmp_path / "shape.db")
        mm.add_chat_turn(task_id="t", role="user", content="hello", conversation_id="c")

        rows, _ = mm.get_chat_turns()
        assert "score" not in rows[0]


# ---------------------------------------------------------------------------
# 'chat_turns' embedding-provenance mismatch handling
# ---------------------------------------------------------------------------

class TestChatTurnsProvenanceMismatch:

    def test_mismatch_warns_and_sets_chat_turns_stale(self, tmp_path, caplog):
        path = tmp_path / "mismatch.db"
        MemoryManager(db_path=path)  # materialise fresh schema
        _insert_chat_turn_raw(path, content="hello", embedded=True)
        _set_provenance(path, "chat_turns", _OTHER)

        with caplog.at_level(logging.WARNING, logger="localist.memory_manager"):
            mm = MemoryManager(
                db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_TUNED,
            )

        assert mm._chat_turns_stale is True
        assert "chat_turns embeddings were produced by" in caplog.text
        assert _OTHER in caplog.text
        assert _TUNED in caplog.text
        # Provenance still describes what's on disk until reembed_chat_turns() runs.
        assert _get_provenance(path, "chat_turns") == _OTHER

    def test_no_reembed_triggered_automatically_unlike_episodes(self, tmp_path):
        path = tmp_path / "mismatch.db"
        MemoryManager(db_path=path)
        _insert_chat_turn_raw(path, content="hello", embedded=True)
        _set_provenance(path, "chat_turns", _OTHER)

        embed_fn = _stub_embed_fn()
        MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_TUNED)

        embed_fn.assert_not_called()

    def test_pre_existing_embeddings_seeded_silently(self, tmp_path, caplog):
        path = tmp_path / "migrate.db"
        MemoryManager(db_path=path)
        _insert_chat_turn_raw(path, content="hello", embedded=True)
        assert _get_provenance(path, "chat_turns") is None

        with caplog.at_level(logging.WARNING, logger="localist.memory_manager"):
            mm = MemoryManager(
                db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_TUNED,
            )

        assert _get_provenance(path, "chat_turns") == _TUNED
        assert mm._chat_turns_stale is False
        assert caplog.text == ""

    def test_get_chat_turns_semantic_falls_back_to_keyword_when_stale(self, tmp_path):
        path = tmp_path / "mismatch.db"
        MemoryManager(db_path=path)
        _insert_chat_turn_raw(path, content="apple pie recipe", embedded=True)
        _set_provenance(path, "chat_turns", _OTHER)

        embed_fn = _stub_embed_fn()
        mm = MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_TUNED)
        assert mm._chat_turns_stale is True
        embed_fn.reset_mock()

        rows, total = mm.get_chat_turns(query="apple pie", mode="semantic")

        # Falls back to FTS keyword search — embed_fn must not be called.
        embed_fn.assert_not_called()
        assert total == 1
        assert rows[0]["content"] == "apple pie recipe"


# ---------------------------------------------------------------------------
# reembed_chat_turns() — manual refresh
# ---------------------------------------------------------------------------

class TestReembedChatTurns:

    def test_clears_staleness_and_updates_provenance(self, tmp_path):
        path = tmp_path / "mismatch.db"
        MemoryManager(db_path=path)
        _insert_chat_turn_raw(path, content="doc-a", embedded=True)
        _insert_chat_turn_raw(path, content="doc-b", embedded=True)
        _set_provenance(path, "chat_turns", _OTHER)

        embed_fn = _stub_embed_fn(0.42)
        mm = MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_TUNED)
        assert mm._chat_turns_stale is True

        result = mm.reembed_chat_turns()

        assert result == {"reembedded": 2, "total": 2, "model": _TUNED}
        assert mm._chat_turns_stale is False
        assert _get_provenance(path, "chat_turns") == _TUNED

        conn = sqlite3.connect(str(path))
        blobs = [row[0] for row in conn.execute("SELECT embedding FROM chat_turns").fetchall()]
        conn.close()
        for blob in blobs:
            assert _unpack_embedding(blob) == pytest.approx([0.42] * _EMBEDDING_DIM)

    def test_raises_without_embed_fn(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "keyword_only.db")
        with pytest.raises(RuntimeError, match="no embed_fn configured"):
            mm.reembed_chat_turns()

    def test_idempotent_when_not_stale(self, tmp_path):
        embed_fn = _stub_embed_fn(0.7)
        mm = MemoryManager(
            db_path=tmp_path / "fresh.db", embed_fn=embed_fn, embedding_model_name=_TUNED,
        )
        mm.add_chat_turn(task_id="t", role="user", content="hi", conversation_id="c")
        assert mm._chat_turns_stale is False

        result = mm.reembed_chat_turns()

        assert result["reembedded"] == 1
        assert mm._chat_turns_stale is False


# ---------------------------------------------------------------------------
# get_chat_turns(mode="semantic") — scoring, ranking, pagination, fallback
# ---------------------------------------------------------------------------

class TestGetChatTurnsSemantic:

    @pytest.fixture()
    def embed_fn(self):
        """
        Deterministic embed_fn: cosine similarity to the query is driven by
        keyword overlap, so ranking assertions don't depend on a real model.
        Each distinct keyword maps to one orthogonal-ish dimension.
        """
        vocab = {}

        def embed(text: str) -> list[float]:
            vec = [0.0] * _EMBEDDING_DIM
            for word in text.lower().split():
                idx = vocab.setdefault(word, len(vocab) % _EMBEDDING_DIM)
                vec[idx] += 1.0
            return vec or [0.0] * _EMBEDDING_DIM

        return MagicMock(side_effect=embed)

    @pytest.fixture()
    def mm(self, tmp_path, embed_fn) -> MemoryManager:
        return MemoryManager(
            db_path=tmp_path / "semantic.db", embed_fn=embed_fn, embedding_model_name=_TUNED,
        )

    def test_semantic_mode_ranks_by_cosine_similarity(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="zebra zebra zebra", conversation_id="c")
        mm.add_chat_turn(task_id="t", role="user", content="zebra mentioned once", conversation_id="c")
        mm.add_chat_turn(task_id="t", role="user", content="totally unrelated content", conversation_id="c")

        rows, total = mm.get_chat_turns(query="zebra", mode="semantic", min_score=0.0)

        assert rows[0]["content"] == "zebra zebra zebra"
        assert rows[0]["score"] >= rows[1]["score"]

    def test_min_score_filters_low_scoring_rows(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="zebra habitat facts", conversation_id="c")
        mm.add_chat_turn(task_id="t", role="user", content="completely different topic entirely", conversation_id="c")

        rows, total = mm.get_chat_turns(query="zebra", mode="semantic", min_score=0.9)

        assert total <= 1
        assert all(r["score"] >= 0.9 for r in rows)

    def test_semantic_pagination(self, mm):
        for i in range(5):
            mm.add_chat_turn(task_id="t", role="user", content=f"widget topic {i}", conversation_id="c")

        page1, total1 = mm.get_chat_turns(query="widget", mode="semantic", min_score=0.0, limit=2, offset=0)
        page2, total2 = mm.get_chat_turns(query="widget", mode="semantic", min_score=0.0, limit=2, offset=4)

        assert total1 == total2 == 5
        assert len(page1) == 2
        assert len(page2) == 1

    def test_conversation_id_scopes_semantic_search(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="zebra facts", conversation_id="conv-a")
        mm.add_chat_turn(task_id="t", role="user", content="zebra facts", conversation_id="conv-b")

        rows, total = mm.get_chat_turns(
            query="zebra", mode="semantic", min_score=0.0, conversation_id="conv-a",
        )

        assert total == 1
        assert all(r["conversation_id"] == "conv-a" for r in rows)

    def test_rows_without_embedding_are_excluded(self, mm, tmp_path):
        mm.add_chat_turn(task_id="t", role="user", content="zebra facts", conversation_id="c")
        _insert_chat_turn_raw(mm._db_path, content="zebra facts unembedded", embedded=False)

        rows, total = mm.get_chat_turns(query="zebra", mode="semantic", min_score=0.0)

        assert total == 1

    def test_empty_query_ignores_semantic_mode_returns_unfiltered_page(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="anything", conversation_id="c")

        rows, total = mm.get_chat_turns(query=None, mode="semantic")

        assert total == 1

    def test_no_embed_fn_falls_back_to_keyword(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "keyword_only.db")
        mm.add_chat_turn(task_id="t", role="user", content="zebra facts", conversation_id="c")

        rows, total = mm.get_chat_turns(query="zebra", mode="semantic")

        assert total == 1
        assert rows[0]["content"] == "zebra facts"

    def test_semantic_mode_respects_date_range(self, mm):
        _insert_chat_turn_raw(mm._db_path, content="zebra facts old", embedded=True, created_at=100.0)
        mm.add_chat_turn(task_id="t", role="user", content="zebra facts new", conversation_id="c")

        rows, total = mm.get_chat_turns(
            query="zebra", mode="semantic", min_score=0.0, date_from=time.time() - 5,
        )

        assert total == 1
        assert rows[0]["content"] == "zebra facts new"

    def test_semantic_mode_respects_has_tool_result(self, mm):
        mm.add_chat_turn(task_id="t1", role="assistant", content="zebra chart", conversation_id="c", metadata={"chart": {}})
        mm.add_chat_turn(task_id="t2", role="assistant", content="zebra plain", conversation_id="c", metadata={})

        rows, total = mm.get_chat_turns(
            query="zebra", mode="semantic", min_score=0.0, has_tool_result=True,
        )

        assert total == 1
        assert rows[0]["content"] == "zebra chart"

    def test_semantic_row_shape_includes_score(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="zebra facts", conversation_id="c")

        rows, _ = mm.get_chat_turns(query="zebra", mode="semantic", min_score=0.0)

        assert "score" in rows[0]
        assert isinstance(rows[0]["score"], float)
