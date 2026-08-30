"""
Tests for MemoryManager's GitHub Watch Feed methods (memory_manager.py) —
github_watch_cache, schema v11.

Follows the existing news_brief_cache test convention
(test_memory_manager_news_brief.py): a real MemoryManager against a
tmp_path SQLite file, no mocking of the DB layer itself.
"""

import sqlite3

from localist.memory_manager import MemoryManager, _SCHEMA_VERSION


class TestGithubWatchCache:
    def test_get_returns_none_when_unset(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "gh.db")
        assert mm.get_github_watch_cache() is None

    def test_set_then_get_roundtrips(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "gh.db")
        content = [{
            "key": "anthropics/claude-code", "label": "anthropics/claude-code",
            "repo_url": "https://github.com/anthropics/claude-code",
            "latest_release": {"tag_name": "v1.0.0"}, "error": None,
        }]
        mm.set_github_watch_cache(content)

        cache = mm.get_github_watch_cache()
        assert cache["content"] == content
        assert cache["generated_at"] > 0

    def test_set_overwrites_previous_cache(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "gh.db")
        mm.set_github_watch_cache([{"key": "o/old"}])
        mm.set_github_watch_cache([{"key": "o/new"}])

        cache = mm.get_github_watch_cache()
        assert cache["content"] == [{"key": "o/new"}]


class TestGithubWatchSchemaMigration:
    def test_v10_database_migrates_and_creates_table(self, tmp_path):
        path = tmp_path / "test.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (10);
            -- Minimal chat_turns stub (with its v9 'embedding' column already
            -- present) so _init_db's unconditional self-heal check finds
            -- the table it expects on any real v10 database.
            CREATE TABLE chat_turns (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id   TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding BLOB
            );
        """)
        conn.commit()
        conn.close()

        # Open with MemoryManager -> triggers _migrate(from_version=10) -> v11.
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        tables  = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert version == _SCHEMA_VERSION
        assert "github_watch_cache" in tables

    def test_fresh_db_has_github_watch_cache_table(self, tmp_path):
        path = tmp_path / "test.db"
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert "github_watch_cache" in tables
