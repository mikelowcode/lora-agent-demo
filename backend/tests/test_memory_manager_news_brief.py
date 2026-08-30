"""
Tests for MemoryManager's Daily News Brief methods (memory_manager.py,
docs/daily-news-brief-plan.md §4) — news_preferences and news_brief_cache,
schema v10.

Follows the existing chat_history_settings test convention: a real
MemoryManager against a tmp_path SQLite file, no mocking of the DB layer
itself.
"""

from localist.memory_manager import MemoryManager


class TestNewsPreferences:
    def test_get_returns_none_when_unset(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "news.db")
        assert mm.get_news_preferences() is None

    def test_set_then_get_roundtrips(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "news.db")
        mm.set_news_preferences("us", "Seattle", ["finance", "technology", "sports"])

        prefs = mm.get_news_preferences()
        assert prefs == {
            "home_country": "us",
            "local_query":  "Seattle",
            "topics":       ["finance", "technology", "sports"],
        }

    def test_set_upserts_not_duplicates(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "news.db")
        mm.set_news_preferences("us", "Seattle", ["finance", "technology", "sports"])
        mm.set_news_preferences("gb", None, ["science", "health", "crypto"])

        prefs = mm.get_news_preferences()
        assert prefs["home_country"] == "gb"
        assert prefs["local_query"] is None
        assert prefs["topics"] == ["science", "health", "crypto"]

    def test_set_rejects_wrong_topic_count(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "news.db")
        try:
            mm.set_news_preferences("us", None, ["finance", "technology"])
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "exactly 3" in str(exc)

    def test_set_rejects_too_many_topics(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "news.db")
        try:
            mm.set_news_preferences("us", None, ["a", "b", "c", "d"])
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "exactly 3" in str(exc)


class TestNewsBriefCache:
    def test_get_returns_none_when_unset(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "news.db")
        assert mm.get_news_brief_cache() is None

    def test_set_then_get_roundtrips(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "news.db")
        content = [{"key": "world", "label": "World", "articles": [], "error": None}]
        mm.set_news_brief_cache("2026-07-22", content, "conv-123")

        cache = mm.get_news_brief_cache()
        assert cache["brief_date"] == "2026-07-22"
        assert cache["content"] == content
        assert cache["conversation_id"] == "conv-123"
        assert cache["generated_at"] > 0

    def test_set_overwrites_previous_cache(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "news.db")
        mm.set_news_brief_cache("2026-07-21", [{"key": "world"}], "conv-old")
        mm.set_news_brief_cache("2026-07-22", [{"key": "national"}], "conv-new")

        cache = mm.get_news_brief_cache()
        assert cache["brief_date"] == "2026-07-22"
        assert cache["conversation_id"] == "conv-new"
        assert cache["content"] == [{"key": "national"}]
