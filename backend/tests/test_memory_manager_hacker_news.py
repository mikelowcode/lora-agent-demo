"""
Tests for MemoryManager's Hacker News Live Feed methods (memory_manager.py) —
hacker_news_cache, schema v12.

Follows the existing github_watch_cache test convention
(test_memory_manager_github_watch.py): a real MemoryManager against a
tmp_path SQLite file, no mocking of the DB layer itself.
"""

import sqlite3

from localist.memory_manager import MemoryManager, _SCHEMA_VERSION


class TestHackerNewsCache:
    def test_get_returns_none_when_unset(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "hn.db")
        assert mm.get_hacker_news_cache() is None

    def test_set_then_get_roundtrips(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "hn.db")
        content = [{
            "key": "111", "title": "Some story",
            "url": "https://example.com/a", "hn_url": "https://news.ycombinator.com/item?id=111",
            "score": 42, "by": "alice", "error": None,
        }]
        mm.set_hacker_news_cache(content)

        cache = mm.get_hacker_news_cache()
        assert cache["content"] == content
        assert cache["generated_at"] > 0

    def test_set_overwrites_previous_cache(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "hn.db")
        mm.set_hacker_news_cache([{"key": "1"}])
        mm.set_hacker_news_cache([{"key": "2"}])

        cache = mm.get_hacker_news_cache()
        assert cache["content"] == [{"key": "2"}]


class TestHackerNewsSchemaMigration:
    def test_v11_database_migrates_and_creates_table(self, tmp_path):
        path = tmp_path / "test.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (11);
            -- Minimal chat_turns stub (with its v9 'embedding' column already
            -- present) so _init_db's unconditional self-heal check finds
            -- the table it expects on any real v11 database.
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

        # Open with MemoryManager -> triggers _migrate(from_version=11) -> v12.
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        tables  = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert version == _SCHEMA_VERSION
        assert "hacker_news_cache" in tables

    def test_fresh_db_has_hacker_news_cache_table(self, tmp_path):
        path = tmp_path / "test.db"
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert "hacker_news_cache" in tables
