"""
Tests for MemoryManager.sweep_expired_memory() — the global retention TTL
enforcement that runs once at backend startup (main.py's lifespan()).

Governs two tables under one preset (retention_settings.eviction_preset):
  - chat_turns : hard-deleted once created_at is older than the TTL.
  - episodes   : soft-retracted (status='active' -> 'retracted') once
    created_at is older than the TTL — never hard-deleted, matching the
    existing approve/reject/retract lifecycle.

No preset set, or preset == "forever", must be a true no-op (not "sweep
with an infinite TTL").
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
        old = time.time() - 365 * 86400
        _insert_chat_turn(mm, created_at=old)
        _insert_episode(mm, created_at=old)

        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 0, "episodes_retracted": 0}
        assert mm.count_episodes(status="active") == 1
        conn = sqlite3.connect(str(mm._db_path))
        assert conn.execute("SELECT COUNT(*) FROM chat_turns").fetchone()[0] == 1
        conn.close()

    def test_forever_preset_is_a_no_op(self, mm):
        old = time.time() - 365 * 86400
        _insert_chat_turn(mm, created_at=old)
        _insert_episode(mm, created_at=old)
        mm.set_retention_preset("forever")

        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 0, "episodes_retracted": 0}
        assert mm.count_episodes(status="active") == 1


class TestSweepEnforcement:

    def test_7d_preset_deletes_old_chat_turns_and_retracts_old_episodes(self, mm):
        now = time.time()
        old = now - 8 * 86400
        recent = now - 1 * 86400

        old_turn_id    = _insert_chat_turn(mm, created_at=old, content="old")
        recent_turn_id = _insert_chat_turn(mm, created_at=recent, content="recent")
        old_ep_id      = _insert_episode(mm, created_at=old, subject="old fact")
        recent_ep_id   = _insert_episode(mm, created_at=recent, subject="recent fact")

        mm.set_retention_preset("7d")
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

        mm.set_retention_preset("7d")
        result = mm.sweep_expired_memory()

        assert result == {"chat_turns_deleted": 0, "episodes_retracted": 0}

    def test_sweep_is_idempotent(self, mm):
        old = time.time() - 8 * 86400
        _insert_chat_turn(mm, created_at=old)
        _insert_episode(mm, created_at=old)
        mm.set_retention_preset("7d")

        first  = mm.sweep_expired_memory()
        second = mm.sweep_expired_memory()

        assert first  == {"chat_turns_deleted": 1, "episodes_retracted": 1}
        assert second == {"chat_turns_deleted": 0, "episodes_retracted": 0}
