"""
Tests for MemoryManager's pinned GitHub repos methods (memory_manager.py) —
pinned_github_repos, schema v15.

Follows the existing github_watch_cache test convention
(test_memory_manager_github_watch.py): a real MemoryManager against a
tmp_path SQLite file, no mocking of the DB layer itself.
"""

import sqlite3

import pytest

from localist.memory_manager import MemoryManager, _SCHEMA_VERSION


class TestPinnedGithubRepos:
    def test_get_returns_empty_list_when_unset(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "gh.db")
        assert mm.get_pinned_github_repos() == []

    def test_set_then_get_roundtrips(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "gh.db")
        mm.set_pinned_github_repos(["ollama/ollama", "ml-explore/mlx"])

        assert mm.get_pinned_github_repos() == ["ollama/ollama", "ml-explore/mlx"]

    def test_set_overwrites_not_appends(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "gh.db")
        mm.set_pinned_github_repos(["ollama/ollama"])
        mm.set_pinned_github_repos(["ml-explore/mlx"])

        assert mm.get_pinned_github_repos() == ["ml-explore/mlx"]

    def test_set_raises_over_max_count(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "gh.db")
        too_many = [f"owner/repo{i}" for i in range(21)]

        with pytest.raises(ValueError):
            mm.set_pinned_github_repos(too_many)


class TestPinnedGithubReposSchemaMigration:
    def test_v14_database_migrates_and_creates_table(self, tmp_path):
        path = tmp_path / "test.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (14);
            -- Minimal chat_turns stub (with its v9 'embedding' column already
            -- present) so _init_db's unconditional self-heal check finds
            -- the table it expects on any real v14 database.
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

        # Open with MemoryManager -> triggers _migrate(from_version=14) -> v15.
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        tables  = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert version == _SCHEMA_VERSION
        assert "pinned_github_repos" in tables

    def test_fresh_db_has_pinned_github_repos_table(self, tmp_path):
        path = tmp_path / "test.db"
        MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert "pinned_github_repos" in tables
