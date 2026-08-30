"""
Tests for the v5→v6 chat_turns/chat_turns_fts/chat_history_settings schema
(Chat History Tab feature, chat_history_settings later renamed to
retention_settings in v12→v13 — see TestRetentionSettings), plus the
add_chat_turn() write path.
"""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from localist.memory_manager import MemoryManager, _SCHEMA_VERSION


# ---------------------------------------------------------------------------
# _create_v5_db — helper for the v5→v6 migration test
# ---------------------------------------------------------------------------

def _create_v5_db(path: Path) -> None:
    """Create a v5-schema SQLite database (no chat_turns objects) for migration tests."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (5);
        CREATE TABLE conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            meta_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
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
    """)
    conn.close()


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}


def _insert_chat_turn_raw(
    db_path: Path,
    *,
    conversation_id:    str,
    conversation_title: str | None,
    created_at:         float,
    content:            str = "turn",
) -> None:
    """
    Insert one chat_turns row with an explicit created_at, bypassing
    add_chat_turn()'s time.time() timestamp.

    Needed for get_conversations() tests that assert on relative ordering
    (MIN/MAX(created_at), ORDER BY last_created_at) across rows inserted
    within the same test — sequential add_chat_turn() calls don't offer
    enough control over timestamp spacing for that.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO chat_turns
            (task_id, role, content, conversation_id, conversation_title, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("t", "user", content, conversation_id, conversation_title, created_at),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# TestChatTurnsFreshSchema — fresh-DB path
# ---------------------------------------------------------------------------

class TestChatTurnsFreshSchema:

    def test_fresh_db_has_chat_turns_objects(self, tmp_path):
        path = tmp_path / "fresh.db"
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master"
        ).fetchall()}
        conn.close()

        assert "chat_turns" in names
        assert "chat_turns_fts" in names
        assert "retention_settings" in names
        assert "chat_turns_ai" in names
        assert "chat_turns_ad" in names
        assert "chat_turns_au" in names

    def test_chat_turns_fts_stays_in_sync_via_triggers(self, tmp_path):
        path = tmp_path / "fts.db"
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        now = time.time()
        cursor = conn.execute(
            """
            INSERT INTO chat_turns (task_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ("task-1", "user", "hello searchable world", now),
        )
        conn.commit()
        row_id = cursor.lastrowid

        match = conn.execute(
            "SELECT rowid FROM chat_turns_fts WHERE chat_turns_fts MATCH ?",
            ("searchable",),
        ).fetchone()
        assert match is not None
        assert match["rowid"] == row_id

        conn.execute("DELETE FROM chat_turns WHERE id = ?", (row_id,))
        conn.commit()

        match_after_delete = conn.execute(
            "SELECT rowid FROM chat_turns_fts WHERE chat_turns_fts MATCH ?",
            ("searchable",),
        ).fetchone()
        assert match_after_delete is None
        conn.close()

    def test_retention_settings_has_no_default_row(self, tmp_path):
        path = tmp_path / "settings.db"
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        count = conn.execute(
            "SELECT COUNT(*) FROM retention_settings"
        ).fetchone()[0]
        conn.close()

        assert count == 0


# ---------------------------------------------------------------------------
# TestChatTurnsMigration — v5→v6 migration path
# ---------------------------------------------------------------------------

class TestChatTurnsMigration:

    def test_v5_migration_creates_chat_turns_objects(self, tmp_path):
        path = tmp_path / "migrate.db"
        _create_v5_db(path)

        # Precondition: truly v5, none of the chat_turns objects exist yet.
        conn = sqlite3.connect(str(path))
        version_before = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        tables_before = _tables(conn)
        conn.close()
        assert version_before == 5
        assert "chat_turns" not in tables_before
        assert "chat_history_settings" not in tables_before

        # Open with MemoryManager → triggers _migrate(from_version=5) →
        # v5→v6→...→current (chat_turns objects land at v6; v6→v7 then adds
        # conversation_id/conversation_title on top, per §12.3;
        # chat_history_settings itself is later renamed to retention_settings
        # at v12→v13, see TestRetentionSettings).
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        version_after = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        names_after = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master"
        ).fetchall()}
        conn.close()

        assert version_after == _SCHEMA_VERSION
        assert "chat_turns" in names_after
        assert "chat_turns_fts" in names_after
        assert "retention_settings" in names_after
        assert "chat_history_settings" not in names_after
        assert "chat_turns_ai" in names_after
        assert "chat_turns_ad" in names_after
        assert "chat_turns_au" in names_after


# ---------------------------------------------------------------------------
# TestAddChatTurn — write path (memory_manager.MemoryManager.add_chat_turn)
# ---------------------------------------------------------------------------

class TestAddChatTurn:

    @pytest.fixture()
    def mm(self, tmp_path) -> MemoryManager:
        return MemoryManager(db_path=tmp_path / "add_chat_turn.db")

    def _rows(self, mm: MemoryManager) -> list[sqlite3.Row]:
        conn = sqlite3.connect(str(mm._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM chat_turns ORDER BY id").fetchall()
        conn.close()
        return rows

    def test_insert_creates_row_with_expected_fields(self, mm):
        mm.add_chat_turn(
            task_id         = "task-1",
            role            = "user",
            content         = "hello world",
            sources         = [{"name": "doc-a"}],
            metadata        = {"agent": "wiki"},
            conversation_id = "conv-1",
        )

        rows = self._rows(mm)
        assert len(rows) == 1
        row = rows[0]
        assert row["task_id"] == "task-1"
        assert row["role"] == "user"
        assert row["content"] == "hello world"
        assert row["status_message"] is None
        assert isinstance(row["created_at"], float)
        assert row["created_at"] > 0

    def test_sources_and_metadata_round_trip_through_json(self, mm):
        mm.add_chat_turn(
            task_id         = "task-2",
            role            = "assistant",
            content         = "the answer",
            sources         = [{"name": "doc-a", "type": "wiki"}, {"name": "doc-b"}],
            metadata        = {"agent": "conversational", "latency_ms": 42},
            conversation_id = "conv-2",
        )

        row = self._rows(mm)[0]
        assert json.loads(row["sources_json"]) == [
            {"name": "doc-a", "type": "wiki"}, {"name": "doc-b"},
        ]
        assert json.loads(row["metadata_json"]) == {
            "agent": "conversational", "latency_ms": 42,
        }

    def test_sources_and_metadata_default_to_empty(self, mm):
        mm.add_chat_turn(task_id="task-3", role="user", content="no extras", conversation_id="conv-3")

        row = self._rows(mm)[0]
        assert json.loads(row["sources_json"]) == []
        assert json.loads(row["metadata_json"]) == {}

    def test_fts_sync_on_insert(self, mm):
        mm.add_chat_turn(task_id="task-4", role="user", content="a searchable phrase", conversation_id="conv-4")

        conn = sqlite3.connect(str(mm._db_path))
        conn.row_factory = sqlite3.Row
        match = conn.execute(
            "SELECT rowid FROM chat_turns_fts WHERE chat_turns_fts MATCH ?",
            ("searchable",),
        ).fetchone()
        conn.close()

        assert match is not None
        assert match["rowid"] == self._rows(mm)[0]["id"]

    def test_does_not_touch_conversation_log(self, mm):
        mm.add_chat_turn(task_id="task-5", role="user", content="x", conversation_id="conv-5")

        conn = sqlite3.connect(str(mm._db_path))
        count = conn.execute("SELECT COUNT(*) FROM conversation_log").fetchone()[0]
        conn.close()

        assert count == 0

    def test_multiple_turns_ordered_by_insertion(self, mm):
        mm.add_chat_turn(task_id="task-6", role="user", content="first", conversation_id="conv-6")
        mm.add_chat_turn(task_id="task-6", role="assistant", content="second", conversation_id="conv-6")

        rows = self._rows(mm)
        assert [r["role"] for r in rows] == ["user", "assistant"]
        assert [r["content"] for r in rows] == ["first", "second"]


# ---------------------------------------------------------------------------
# TestMarkDiffApplied — review-then-apply wiki diff UI's persisted state
# transition (scope-review-then-apply-diff-ui.md)
# ---------------------------------------------------------------------------

class TestMarkDiffApplied:

    @pytest.fixture()
    def mm(self, tmp_path) -> MemoryManager:
        return MemoryManager(db_path=tmp_path / "mark_diff_applied.db")

    def _metadata(self, mm: MemoryManager, task_id: str) -> dict:
        conn = sqlite3.connect(str(mm._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT metadata_json FROM chat_turns WHERE task_id = ? AND role = 'assistant'",
            (task_id,),
        ).fetchone()
        conn.close()
        return json.loads(row["metadata_json"])

    def test_marks_matching_entry_applied_and_returns_true(self, mm):
        mm.add_chat_turn(
            task_id="task-1", role="assistant", content="Proposed a diff.",
            conversation_id="conv-1",
            metadata={"pending_diffs": [{"page_name": "some-page", "diff": "@@ ... @@", "status": "pending"}]},
        )

        result = mm.mark_diff_applied("task-1", "some-page")

        assert result is True
        assert self._metadata(mm, "task-1")["pending_diffs"] == [
            {"page_name": "some-page", "diff": "@@ ... @@", "status": "applied"}
        ]

    def test_no_matching_task_id_returns_false(self, mm):
        result = mm.mark_diff_applied("nonexistent-task", "some-page")
        assert result is False

    def test_row_without_pending_diffs_returns_false(self, mm):
        mm.add_chat_turn(
            task_id="task-2", role="assistant", content="A plain answer.",
            conversation_id="conv-2", metadata={"agent": "conversational_agent"},
        )
        result = mm.mark_diff_applied("task-2", "some-page")
        assert result is False

    def test_no_matching_page_name_returns_false_and_leaves_metadata_unchanged(self, mm):
        mm.add_chat_turn(
            task_id="task-3", role="assistant", content="Proposed a diff.",
            conversation_id="conv-3",
            metadata={"pending_diffs": [{"page_name": "other-page", "diff": "d", "status": "pending"}]},
        )

        result = mm.mark_diff_applied("task-3", "some-page")

        assert result is False
        assert self._metadata(mm, "task-3")["pending_diffs"][0]["status"] == "pending"

    def test_multiple_pending_diffs_only_the_matching_page_flips_to_applied(self, mm):
        """
        docs/architecture/17-wiki-agent-diff-target.md's multi-diff open
        item: mark_diff_applied() already matches by page_name within the
        list (not "the diff" singular), so a turn proposing 2+ diffs
        should already apply/track them independently — this is the
        missing direct coverage proving that (episode-browsing-ui-plan.md
        Phase 3), not a code change.
        """
        mm.add_chat_turn(
            task_id="task-multi", role="assistant", content="Proposed two diffs.",
            conversation_id="conv-1",
            metadata={"pending_diffs": [
                {"page_name": "page-a", "diff": "@@ a @@", "status": "pending"},
                {"page_name": "page-b", "diff": "@@ b @@", "status": "pending"},
            ]},
        )

        result = mm.mark_diff_applied("task-multi", "page-a")

        assert result is True
        assert self._metadata(mm, "task-multi")["pending_diffs"] == [
            {"page_name": "page-a", "diff": "@@ a @@", "status": "applied"},
            {"page_name": "page-b", "diff": "@@ b @@", "status": "pending"},
        ]

    def test_only_matches_assistant_role_not_user(self, mm):
        # A user row incidentally sharing the same task_id must not match —
        # pending_diffs only ever lives on the assistant row.
        mm.add_chat_turn(task_id="task-4", role="user", content="update the page", conversation_id="conv-4")
        mm.add_chat_turn(
            task_id="task-4", role="assistant", content="Proposed a diff.",
            conversation_id="conv-4",
            metadata={"pending_diffs": [{"page_name": "some-page", "diff": "d", "status": "pending"}]},
        )

        result = mm.mark_diff_applied("task-4", "some-page")
        assert result is True


# ---------------------------------------------------------------------------
# TestRetentionSettings — global retention preset read/write
# (memory_manager.MemoryManager.get/set_retention_preset)
# ---------------------------------------------------------------------------

class TestRetentionSettings:

    @pytest.fixture()
    def mm(self, tmp_path) -> MemoryManager:
        return MemoryManager(db_path=tmp_path / "retention_settings.db")

    def test_get_returns_none_on_fresh_db(self, mm):
        assert mm.get_retention_preset() is None

    @pytest.mark.parametrize("preset", ["7d", "30d", "90d", "forever"])
    def test_set_then_get_round_trips(self, mm, preset):
        mm.set_retention_preset(preset)
        assert mm.get_retention_preset() == preset

    def test_set_twice_overwrites_not_duplicates(self, mm):
        mm.set_retention_preset("7d")
        mm.set_retention_preset("90d")
        assert mm.get_retention_preset() == "90d"

        conn = sqlite3.connect(str(mm._db_path))
        count = conn.execute("SELECT COUNT(*) FROM retention_settings").fetchone()[0]
        conn.close()
        assert count == 1

    def test_set_invalid_preset_raises_value_error(self, mm):
        with pytest.raises(ValueError, match="60d"):
            mm.set_retention_preset("60d")

        # No row should have been written on rejection.
        assert mm.get_retention_preset() is None


# ---------------------------------------------------------------------------
# TestGetChatTurns — read/list path (memory_manager.MemoryManager.get_chat_turns)
# ---------------------------------------------------------------------------

class TestGetChatTurns:

    @pytest.fixture()
    def mm(self, tmp_path) -> MemoryManager:
        return MemoryManager(db_path=tmp_path / "get_chat_turns.db")

    def test_empty_table_returns_empty_list_and_zero_total(self, mm):
        rows, total = mm.get_chat_turns()
        assert rows == []
        assert total == 0

    def test_unfiltered_pagination_and_total_count(self, mm):
        for i in range(5):
            mm.add_chat_turn(task_id="t", role="user", content=f"turn {i}", conversation_id="conv-1")

        page1, total1 = mm.get_chat_turns(limit=2, offset=0)
        page2, total2 = mm.get_chat_turns(limit=2, offset=2)
        page3, total3 = mm.get_chat_turns(limit=2, offset=4)

        assert total1 == total2 == total3 == 5
        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1

    def test_unfiltered_ordering_is_created_at_desc(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="first", conversation_id="conv-1")
        mm.add_chat_turn(task_id="t", role="assistant", content="second", conversation_id="conv-1")
        mm.add_chat_turn(task_id="t", role="user", content="third", conversation_id="conv-1")

        rows, total = mm.get_chat_turns(limit=10)
        assert total == 3
        assert [r["content"] for r in rows] == ["third", "second", "first"]

    def test_row_shape_includes_expected_keys_and_json_decoded_fields(self, mm):
        mm.add_chat_turn(
            task_id         = "t1",
            role            = "assistant",
            content         = "the answer",
            sources         = [{"name": "doc-a"}],
            metadata        = {"agent": "wiki"},
            conversation_id = "conv-1",
        )
        rows, _ = mm.get_chat_turns()
        row = rows[0]

        assert set(row.keys()) == {
            "id", "task_id", "role", "content", "sources",
            "status_message", "metadata", "conversation_id",
            "conversation_title", "created_at",
        }
        assert row["sources"]  == [{"name": "doc-a"}]
        assert row["metadata"] == {"agent": "wiki"}
        assert row["status_message"] is None

    def test_fts_search_returns_only_matching_rows(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="tell me about zebras", conversation_id="conv-1")
        mm.add_chat_turn(task_id="t", role="assistant", content="zebras are striped equines", conversation_id="conv-1")
        mm.add_chat_turn(task_id="t", role="user", content="what is the capital of France", conversation_id="conv-1")

        rows, total = mm.get_chat_turns(query="zebras")

        assert total == 2
        assert len(rows) == 2
        assert all("zebra" in r["content"].lower() for r in rows)

    def test_fts_search_ranks_best_match_first(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="a passing mention of pangolins", conversation_id="conv-1")
        mm.add_chat_turn(task_id="t", role="assistant", content="pangolins pangolins pangolins", conversation_id="conv-1")

        rows, total = mm.get_chat_turns(query="pangolins")

        assert total == 2
        assert rows[0]["content"] == "pangolins pangolins pangolins"

    def test_fts_search_no_match_returns_empty_not_error(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="hello world", conversation_id="conv-1")

        rows, total = mm.get_chat_turns(query="nonexistentterm")

        assert rows == []
        assert total == 0

    def test_fts_search_pagination(self, mm):
        for i in range(5):
            mm.add_chat_turn(task_id="t", role="user", content=f"widget number {i}", conversation_id="conv-1")

        page1, total1 = mm.get_chat_turns(query="widget", limit=2, offset=0)
        page2, total2 = mm.get_chat_turns(query="widget", limit=2, offset=4)

        assert total1 == total2 == 5
        assert len(page1) == 2
        assert len(page2) == 1

    def test_date_from_excludes_earlier_rows(self, mm):
        _insert_chat_turn_raw(mm._db_path, conversation_id="c", conversation_title=None, created_at=10.0, content="old")
        _insert_chat_turn_raw(mm._db_path, conversation_id="c", conversation_title=None, created_at=30.0, content="new")

        rows, total = mm.get_chat_turns(date_from=20.0)

        assert total == 1
        assert rows[0]["content"] == "new"

    def test_date_to_excludes_later_rows(self, mm):
        _insert_chat_turn_raw(mm._db_path, conversation_id="c", conversation_title=None, created_at=10.0, content="old")
        _insert_chat_turn_raw(mm._db_path, conversation_id="c", conversation_title=None, created_at=30.0, content="new")

        rows, total = mm.get_chat_turns(date_to=20.0)

        assert total == 1
        assert rows[0]["content"] == "old"

    def test_date_range_is_inclusive_on_both_ends(self, mm):
        _insert_chat_turn_raw(mm._db_path, conversation_id="c", conversation_title=None, created_at=10.0, content="a")
        _insert_chat_turn_raw(mm._db_path, conversation_id="c", conversation_title=None, created_at=20.0, content="b")
        _insert_chat_turn_raw(mm._db_path, conversation_id="c", conversation_title=None, created_at=30.0, content="c")

        rows, total = mm.get_chat_turns(date_from=10.0, date_to=20.0)

        assert total == 2
        assert {r["content"] for r in rows} == {"a", "b"}

    def test_date_range_combines_with_fts_query(self, mm):
        mm.add_chat_turn(task_id="t", role="user", content="pangolin facts", conversation_id="c")
        _insert_chat_turn_raw(mm._db_path, conversation_id="c", conversation_title=None, created_at=1.0, content="pangolin history")

        rows, total = mm.get_chat_turns(query="pangolin", date_from=time.time() - 5)

        assert total == 1
        assert rows[0]["content"] == "pangolin facts"

    def test_has_tool_result_filters_to_chart_pending_diffs_or_workflow_only(self, mm):
        mm.add_chat_turn(task_id="t1", role="assistant", content="a chart", conversation_id="c", metadata={"chart": {"png_path": "x"}})
        mm.add_chat_turn(task_id="t2", role="assistant", content="a diff", conversation_id="c", metadata={"pending_diffs": [{"page_name": "p", "diff": "d", "status": "pending"}]})
        mm.add_chat_turn(task_id="t3", role="assistant", content="a workflow", conversation_id="c", metadata={"workflow_id": "wf-1", "workflow_steps": []})
        mm.add_chat_turn(task_id="t4", role="assistant", content="plain answer", conversation_id="c", metadata={"agent": "conversational_agent"})

        rows, total = mm.get_chat_turns(has_tool_result=True)

        assert total == 3
        assert {r["content"] for r in rows} == {"a chart", "a diff", "a workflow"}

    def test_has_tool_result_false_returns_all_rows(self, mm):
        mm.add_chat_turn(task_id="t1", role="assistant", content="a chart", conversation_id="c", metadata={"chart": {}})
        mm.add_chat_turn(task_id="t2", role="assistant", content="plain answer", conversation_id="c", metadata={})

        rows, total = mm.get_chat_turns(has_tool_result=False)

        assert total == 2

    @pytest.mark.parametrize("special_query", [
        "what's up",
        "-",
        '"unterminated quote',
        "AND OR NOT",
        "()",
        "col:value",
    ])
    def test_fts_special_characters_do_not_raise(self, mm, special_query):
        mm.add_chat_turn(task_id="t", role="user", content="ordinary content", conversation_id="conv-1")

        rows, total = mm.get_chat_turns(query=special_query)

        assert isinstance(rows, list)
        assert isinstance(total, int)


# ---------------------------------------------------------------------------
# TestGetConversations — conversation summary path
# (memory_manager.MemoryManager.get_conversations)
# ---------------------------------------------------------------------------

class TestGetConversations:

    @pytest.fixture()
    def mm(self, tmp_path) -> MemoryManager:
        return MemoryManager(db_path=tmp_path / "get_conversations.db")

    def test_multiple_conversations_return_one_row_each(self, mm):
        mm.add_chat_turn(task_id="t1", role="user", content="hi",    conversation_id="conv-1")
        mm.add_chat_turn(task_id="t2", role="user", content="hello", conversation_id="conv-2")
        mm.add_chat_turn(task_id="t3", role="user", content="hey",   conversation_id="conv-3")

        conversations = mm.get_conversations()

        assert len(conversations) == 3
        assert {c["conversation_id"] for c in conversations} == {"conv-1", "conv-2", "conv-3"}

    def test_conversation_title_resolves_from_non_first_row(self, mm):
        # The title is on the *middle* row by created_at — neither the
        # earliest (MIN) nor the latest (MAX) row in the group — so this
        # only passes if conversation_title is resolved via "any row with
        # a non-null title" semantics, not by (accidentally or otherwise)
        # tracking whichever of MIN/MAX(created_at) a query happens to key
        # off of.
        _insert_chat_turn_raw(
            mm._db_path, conversation_id="conv-1", conversation_title=None,
            created_at=100.0, content="earliest, no title",
        )
        _insert_chat_turn_raw(
            mm._db_path, conversation_id="conv-1", conversation_title="Chosen Title",
            created_at=200.0, content="middle, has title",
        )
        _insert_chat_turn_raw(
            mm._db_path, conversation_id="conv-1", conversation_title=None,
            created_at=300.0, content="latest, no title",
        )

        conversations = mm.get_conversations()

        assert len(conversations) == 1
        convo = conversations[0]
        assert convo["conversation_id"]    == "conv-1"
        assert convo["conversation_title"] == "Chosen Title"
        assert convo["first_created_at"]   == 100.0
        assert convo["last_created_at"]    == 300.0

    def test_conversation_title_none_when_no_row_has_it(self, mm):
        mm.add_chat_turn(task_id="t1", role="user",      content="hi",    conversation_id="legacy")
        mm.add_chat_turn(task_id="t1", role="assistant", content="hello", conversation_id="legacy")

        conversations = mm.get_conversations()

        assert len(conversations) == 1
        assert conversations[0]["conversation_id"]    == "legacy"
        assert conversations[0]["conversation_title"] is None

    def test_first_and_last_created_at_reflect_min_max_per_conversation(self, mm):
        _insert_chat_turn_raw(mm._db_path, conversation_id="conv-1", conversation_title=None, created_at=10.0)
        _insert_chat_turn_raw(mm._db_path, conversation_id="conv-1", conversation_title=None, created_at=30.0)
        _insert_chat_turn_raw(mm._db_path, conversation_id="conv-1", conversation_title=None, created_at=20.0)

        conversations = mm.get_conversations()

        assert len(conversations) == 1
        assert conversations[0]["first_created_at"] == 10.0
        assert conversations[0]["last_created_at"]  == 30.0

    def test_ordered_by_last_created_at_desc(self, mm):
        _insert_chat_turn_raw(mm._db_path, conversation_id="conv-old", conversation_title=None, created_at=10.0)
        _insert_chat_turn_raw(mm._db_path, conversation_id="conv-new", conversation_title=None, created_at=50.0)
        _insert_chat_turn_raw(mm._db_path, conversation_id="conv-mid", conversation_title=None, created_at=30.0)

        conversations = mm.get_conversations()

        assert [c["conversation_id"] for c in conversations] == ["conv-new", "conv-mid", "conv-old"]
