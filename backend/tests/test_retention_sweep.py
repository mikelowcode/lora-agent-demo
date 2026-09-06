"""
Tests for MemoryManager.sweep_expired_memory() — the retention TTL
enforcement that runs once at backend startup (main.py's lifespan()).

Two independent presets as of schema v16 (docs/architecture/
20-episode-browsing-ui.md §20.12) — previously one shared
retention_settings.eviction_preset column governed both tables identically,
found live to be a real problem: a stable fact ("the user games on an Xbox
Series X") was silently retracted by the same 30-day window set for
chat-history cleanup, even though it was still true and had been used
since. Now:
  - chat_turns : hard-deleted once created_at is older than
    get_retention_preset()'s TTL. Unset default: keep everything (None).
  - episodes   : soft-retracted (status='active' -> 'retracted') once
    created_at is older than get_episode_retention_preset()'s OWN,
    independent TTL — never hard-deleted, matching the existing
    approve/reject/retract lifecycle. Unset default: "forever" (a
    concrete value, not merely "unset") — episodic memory doesn't inherit
    chat-history's TTL.

Either preset unset/"forever" must be a true no-op for its own table (not
"sweep with an infinite TTL"), independent of what the other preset is set
to.
"""

import sqlite3
import time

import pytest

from localist.memory_manager import MemoryManager


@pytest.fixture()
def mm(tmp_path) -> MemoryManager:
    return MemoryManager(db_path=tmp_path / "retention_sweep.db")


def _insert_chat_turn(mm: MemoryManager, *, created_at: float, content: str = "turn") -> int:
    conn = sqlite3.connect(str(mm._db_path))
    cursor = conn.execute(
        """
        INSERT INTO chat_turns (task_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        ("t", "user", content, created_at),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def _insert_episode(
    mm: MemoryManager, *, created_at: float, status: str = "active", subject: str = "fact",
) -> int:
    conn = sqlite3.connect(str(mm._db_path))
    cursor = conn.execute(
        """
        INSERT INTO episodes (episode_type, subject, content, source, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("project_fact", subject, "some content", "explicit", status, created_at),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


class TestSweepNoPreset:

    def test_no_preset_set_is_a_no_op(self, mm):
        # Neither preset ever set: chat_turns defaults to None (keep
        # everything), episodes defaults to "forever" — both no-ops.
        old = time.time() - 365 * 86400
        _insert_chat_turn(mm, created_at=old)
        _insert_episode(mm, created_at=old)

        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 0, "episodes_retracted": 0}
        assert mm.count_episodes(status="active") == 1
        conn = sqlite3.connect(str(mm._db_path))
        assert conn.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0] == 1
        conn.close()

    def test_forever_preset_is_a_no_op_for_both(self, mm):
        old = time.time() - 365 * 86400
        _insert_chat_turn(mm, created_at=old)
        _insert_episode(mm, created_at=old)
        mm.set_retention_preset("forever")
        mm.set_episode_retention_preset("forever")

        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 0, "episodes_retracted": 0}
        assert mm.count_episodes(status="active") == 1


class TestSweepEnforcement:

    def test_7d_preset_deletes_old_chat_turns_and_retracts_old_episodes(self, mm):
        # Both presets explicitly set to 7d — the "old shared behavior"
        # scenario, now reached via two independent calls instead of one.
        now = time.time()
        old = now - 8 * 86400
        recent = now - 1 * 86400

        old_turn_id    = _insert_chat_turn(mm, created_at=old, content="old")
        recent_turn_id = _insert_chat_turn(mm, created_at=recent, content="recent")
        old_ep_id      = _insert_episode(mm, created_at=old, subject="old fact")
        recent_ep_id   = _insert_episode(mm, created_at=recent, subject="recent fact")

        mm.set_retention_preset("7d")
        mm.set_episode_retention_preset("7d")
        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 1, "episodes_retracted": 1}

        conn = sqlite3.connect(str(mm._db_path))
        remaining_turn_ids = {
            r[0] for r in conn.execute("SELECT id FROM chat_turns").fetchall()
        }
        conn.close()
        assert remaining_turn_ids == {recent_turn_id}

        assert mm.count_episodes(status="active") == 1
        active_ids = {row["id"] for row in mm.list_episodes(status="active")}
        assert active_ids == {recent_ep_id}

        retracted_ids = {row["id"] for row in mm.list_episodes(status="retracted")}
        assert retracted_ids == {old_ep_id}

    def test_already_retracted_episode_is_not_recounted(self, mm):
        old = time.time() - 8 * 86400
        _insert_episode(mm, created_at=old, status="retracted", subject="already gone")

        mm.set_episode_retention_preset("7d")
        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 0, "episodes_retracted": 0}

    def test_sweep_is_idempotent(self, mm):
        old = time.time() - 8 * 86400
        _insert_chat_turn(mm, created_at=old)
        _insert_episode(mm, created_at=old)
        mm.set_retention_preset("7d")
        mm.set_episode_retention_preset("7d")

        first  = mm.sweep_expired_memory()
        second = mm.sweep_expired_memory()

        assert first  == {"chat_turns_deleted": 1, "episodes_retracted": 1}
        assert second == {"chat_turns_deleted": 0, "episodes_retracted": 0}


class TestPresetsAreIndependent:
    """
    The core behavior this decoupling exists for: setting one preset must
    never affect the other table, in either direction — the exact live
    failure (chat_turns' 30d window silently retracting a still-true
    episode) this schema change was built to prevent.
    """

    def test_chat_preset_alone_does_not_touch_episodes(self, mm):
        old = time.time() - 8 * 86400
        _insert_chat_turn(mm, created_at=old)
        old_ep_id = _insert_episode(mm, created_at=old, subject="stable fact")

        mm.set_retention_preset("7d")  # episode preset left at its "forever" default
        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 1, "episodes_retracted": 0}
        assert mm.count_episodes(status="active") == 1
        active_ids = {row["id"] for row in mm.list_episodes(status="active")}
        assert active_ids == {old_ep_id}

    def test_episode_preset_alone_does_not_touch_chat_turns(self, mm):
        old = time.time() - 8 * 86400
        old_turn_id = _insert_chat_turn(mm, created_at=old)
        _insert_episode(mm, created_at=old, subject="ephemeral note")

        mm.set_episode_retention_preset("7d")  # chat preset left unset ("forever" behavior)
        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 0, "episodes_retracted": 1}
        conn = sqlite3.connect(str(mm._db_path))
        remaining_turn_ids = {
            r[0] for r in conn.execute("SELECT id FROM chat_turns").fetchall()
        }
        conn.close()
        assert remaining_turn_ids == {old_turn_id}

    def test_different_values_for_each_are_respected_independently(self, mm):
        # chat_turns: 7d window (the old turn is 8d old -> swept).
        # episodes: 90d window (the old episode is only 8d old -> kept).
        old = time.time() - 8 * 86400
        _insert_chat_turn(mm, created_at=old)
        old_ep_id = _insert_episode(mm, created_at=old, subject="still fresh under 90d")

        mm.set_retention_preset("7d")
        mm.set_episode_retention_preset("90d")
        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 1, "episodes_retracted": 0}
        assert mm.count_episodes(status="active") == 1
        assert {row["id"] for row in mm.list_episodes(status="active")} == {old_ep_id}


class TestEpisodeRetentionPresetAccessors:
    """Direct unit tests for get/set_episode_retention_preset(), mirroring
    test_chat_turns_schema.py's coverage of the chat_turns equivalents."""

    def test_defaults_to_forever_when_never_set(self, mm):
        assert mm.get_episode_retention_preset() == "forever"

    def test_set_then_get_roundtrips(self, mm):
        mm.set_episode_retention_preset("30d")
        assert mm.get_episode_retention_preset() == "30d"

    def test_set_twice_overwrites(self, mm):
        mm.set_episode_retention_preset("7d")
        mm.set_episode_retention_preset("90d")
        assert mm.get_episode_retention_preset() == "90d"

    def test_invalid_preset_raises(self, mm):
        with pytest.raises(ValueError):
            mm.set_episode_retention_preset("60d")
        assert mm.get_episode_retention_preset() == "forever"

    def test_setting_episode_preset_does_not_affect_chat_preset(self, mm):
        mm.set_episode_retention_preset("30d")
        assert mm.get_retention_preset() is None

    def test_setting_chat_preset_does_not_affect_episode_preset(self, mm):
        mm.set_retention_preset("30d")
        assert mm.get_episode_retention_preset() == "forever"


class TestSchemaV16Migration:
    """
    Schema v16 (docs/architecture/20-episode-browsing-ui.md §20.12):
    retention_settings gains episode_eviction_preset. Mirrors
    test_memory_manager_pinned_github_repos.py's
    TestPinnedGithubReposSchemaMigration convention — a real v15 database
    (not a mock), migrated by opening it with MemoryManager.
    """

    def test_v15_database_with_existing_chat_preset_migrates_cleanly(self, tmp_path):
        # The exact live scenario this decoupling was built for: an
        # existing install that had already set eviction_preset="30d"
        # (governing both tables pre-v16) must, after migrating, keep that
        # value for chat_turns but get episodes newly decoupled to
        # "forever" — not silently inherit "30d" for episodes too.
        path = tmp_path / "test.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (15);
            CREATE TABLE chat_turns (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id   TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding BLOB
            );
            CREATE TABLE retention_settings (
                id              INTEGER PRIMARY KEY CHECK (id = 1),
                eviction_preset TEXT
            );
            INSERT INTO retention_settings (id, eviction_preset) VALUES (1, '30d');
        """)
        conn.commit()
        conn.close()

        mm = MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        row = conn.execute("SELECT * FROM retention_settings WHERE id = 1").fetchone()
        conn.close()

        from localist.memory_manager import _SCHEMA_VERSION
        assert version == _SCHEMA_VERSION
        assert row["eviction_preset"] == "30d"                    # untouched
        assert row["episode_eviction_preset"] == "forever"        # newly decoupled
        assert mm.get_retention_preset() == "30d"
        assert mm.get_episode_retention_preset() == "forever"

    def test_v15_database_with_no_row_at_all_migrates_cleanly(self, tmp_path):
        path = tmp_path / "test.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (15);
            CREATE TABLE chat_turns (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id   TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding BLOB
            );
            CREATE TABLE retention_settings (
                id              INTEGER PRIMARY KEY CHECK (id = 1),
                eviction_preset TEXT
            );
        """)
        conn.commit()
        conn.close()

        mm = MemoryManager(db_path=path)

        assert mm.get_retention_preset() is None
        assert mm.get_episode_retention_preset() == "forever"

    def test_fresh_db_has_both_columns(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "fresh.db")
        assert mm.get_retention_preset() is None
        assert mm.get_episode_retention_preset() == "forever"


class TestSchemaV17Migration:
    """
    Schema v17 (PLAN_semantic_gating_calibration.md): adds
    embedding_model_thresholds, the persisted-calibration table backing
    MemoryManager.get/set_calibrated_thresholds(). Mirrors
    TestSchemaV16Migration's convention exactly — a real v16 database (not a
    mock), migrated by opening it with MemoryManager.
    """

    def _v16_database(self, tmp_path):
        path = tmp_path / "test.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (16);
            CREATE TABLE chat_turns (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id   TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding BLOB
            );
            CREATE TABLE retention_settings (
                id                      INTEGER PRIMARY KEY CHECK (id = 1),
                eviction_preset         TEXT,
                episode_eviction_preset TEXT NOT NULL DEFAULT 'forever'
            );
        """)
        conn.commit()
        conn.close()
        return path

    def test_v16_database_migrates_to_v17_with_new_table(self, tmp_path):
        path = self._v16_database(tmp_path)

        mm = MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        cols = {row[1] for row in conn.execute("PRAGMA table_info(embedding_model_thresholds)").fetchall()}
        conn.close()

        from localist.memory_manager import _SCHEMA_VERSION
        assert version == _SCHEMA_VERSION == 17
        assert cols == {
            "model", "explicit_search_action", "lookup_request",
            "research_intent", "episodic_relevance", "calibrated_at",
        }
        # Empty on migration — no row exists until a calibration actually runs.
        assert mm.get_calibrated_thresholds("any-model") is None

    def test_v16_database_get_set_round_trip_after_migration(self, tmp_path):
        path = self._v16_database(tmp_path)
        mm = MemoryManager(db_path=path)

        mm.set_calibrated_thresholds("mystery-model:latest", {
            "lookup_request": 0.62, "research_intent": 0.58,
        })

        assert mm.get_calibrated_thresholds("mystery-model:latest") == {
            "lookup_request": 0.62, "research_intent": 0.58,
        }
        # Untouched retention behavior from v16 — this migration doesn't
        # regress the prior one.
        assert mm.get_episode_retention_preset() == "forever"

    def test_fresh_db_has_embedding_model_thresholds_table(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "fresh.db")
        assert mm.get_calibrated_thresholds("any-model") is None
        mm.set_calibrated_thresholds("any-model", {"lookup_request": 0.5})
        assert mm.get_calibrated_thresholds("any-model") == {"lookup_request": 0.5}
