"""
Tests for POST /wiki/apply-diff (main.py) — the review-then-apply wiki
diff UI's write endpoint (scope-review-then-apply-diff-ui.md).

Covers:
  - 200 + file written + chat_turn marked applied on success.
  - 404 when the target page doesn't exist.
  - 409 when the diff no longer applies cleanly (stale content).
  - mark_diff_applied() failure is swallowed — the disk write already
    succeeded, so a memory_manager hiccup must not fail the response.
  - 503 when wiki_agent/wiki_dir aren't configured.

The FastAPI lifespan (real runtime / embedding model / controller wiring)
is never triggered here — TestClient instantiates the app directly, and
_state.wiki_agent / _state.wiki_dir / _state.memory_manager are set up
per-test, mirroring test_main_task_chat_turns.py's pattern.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager
from localist.wiki_agent import WikiAgent


class _FakeRuntime:
    def infer(self, *a, **kw) -> str:
        return ""

    def embed(self, text: str) -> list[float]:
        return [0.0] * 768


@pytest.fixture()
def client(tmp_path: Path):
    prev_wiki_agent = main._state.wiki_agent
    prev_wiki_dir   = main._state.wiki_dir
    prev_memory     = main._state.memory_manager

    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()

    main._state.memory_manager = MemoryManager(db_path=tmp_path / "apply_diff_endpoint.db")
    main._state.wiki_agent     = WikiAgent(runtime=_FakeRuntime(), project_root=tmp_path)
    main._state.wiki_dir       = wiki_dir

    yield TestClient(main.app), wiki_dir

    main._state.wiki_agent     = prev_wiki_agent
    main._state.wiki_dir       = prev_wiki_dir
    main._state.memory_manager = prev_memory


_PAGE_CONTENT = "## Summary\n\nOld summary.\n"
_DIFF_TEXT = "@@ -3,1 +3,1 @@\n-Old summary.\n+New summary.\n"


class TestApplyDiffEndpoint:

    def test_success_writes_file_and_marks_chat_turn_applied(self, client):
        test_client, wiki_dir = client
        page_path = wiki_dir / "existing-page.md"
        page_path.write_text(_PAGE_CONTENT, encoding="utf-8")

        main._state.memory_manager.add_chat_turn(
            task_id="task-1", role="assistant", content="Proposed a diff.",
            conversation_id="conv-1",
            metadata={"pending_diffs": [{"page_name": "existing-page", "diff": _DIFF_TEXT, "status": "pending"}]},
        )

        resp = test_client.post(
            "/wiki/apply-diff",
            json={"task_id": "task-1", "page_name": "existing-page", "diff": _DIFF_TEXT},
        )

        assert resp.status_code == 200
        assert resp.json() == {"success": True, "page_name": "existing-page"}
        assert "New summary." in page_path.read_text(encoding="utf-8")

        turns, _ = main._state.memory_manager.get_chat_turns()
        assert turns[0]["metadata"]["pending_diffs"][0]["status"] == "applied"

    def test_applying_one_of_two_pending_diffs_leaves_the_other_untouched(self, client):
        """
        docs/architecture/17-wiki-agent-diff-target.md's multi-diff open
        item, exercised end to end through the real endpoint: a turn
        proposing diffs for two separate pages, applying one must write
        only that page and mark only that page_name's entry "applied" —
        the other stays "pending" and on disk unmodified. episode-
        browsing-ui-plan.md Phase 3.
        """
        test_client, wiki_dir = client
        page_a = wiki_dir / "page-a.md"
        page_b = wiki_dir / "page-b.md"
        page_a.write_text(_PAGE_CONTENT, encoding="utf-8")
        page_b.write_text(_PAGE_CONTENT, encoding="utf-8")

        main._state.memory_manager.add_chat_turn(
            task_id="task-multi", role="assistant", content="Proposed two diffs.",
            conversation_id="conv-1",
            metadata={"pending_diffs": [
                {"page_name": "page-a", "diff": _DIFF_TEXT, "status": "pending"},
                {"page_name": "page-b", "diff": _DIFF_TEXT, "status": "pending"},
            ]},
        )

        resp = test_client.post(
            "/wiki/apply-diff",
            json={"task_id": "task-multi", "page_name": "page-a", "diff": _DIFF_TEXT},
        )

        assert resp.status_code == 200
        assert "New summary." in page_a.read_text(encoding="utf-8")
        assert "New summary." not in page_b.read_text(encoding="utf-8")

        turns, _ = main._state.memory_manager.get_chat_turns()
        diffs = {d["page_name"]: d["status"] for d in turns[0]["metadata"]["pending_diffs"]}
        assert diffs == {"page-a": "applied", "page-b": "pending"}

    def test_missing_page_returns_404(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/wiki/apply-diff",
            json={"task_id": "task-2", "page_name": "nonexistent-page", "diff": _DIFF_TEXT},
        )
        assert resp.status_code == 404

    def test_stale_content_returns_409(self, client):
        test_client, wiki_dir = client
        (wiki_dir / "existing-page.md").write_text("Completely different content.\n", encoding="utf-8")

        resp = test_client.post(
            "/wiki/apply-diff",
            json={"task_id": "task-3", "page_name": "existing-page", "diff": _DIFF_TEXT},
        )
        assert resp.status_code == 409

    def test_mark_diff_applied_failure_does_not_break_response(self, client):
        test_client, wiki_dir = client
        (wiki_dir / "existing-page.md").write_text(_PAGE_CONTENT, encoding="utf-8")

        with patch.object(
            main._state.memory_manager, "mark_diff_applied", side_effect=RuntimeError("db down"),
        ):
            resp = test_client.post(
                "/wiki/apply-diff",
                json={"task_id": "task-4", "page_name": "existing-page", "diff": _DIFF_TEXT},
            )

        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_missing_wiki_agent_returns_503(self, client):
        test_client, _ = client
        main._state.wiki_agent = None

        resp = test_client.post(
            "/wiki/apply-diff",
            json={"task_id": "task-5", "page_name": "existing-page", "diff": _DIFF_TEXT},
        )
        assert resp.status_code == 503

    def test_request_validation_rejects_empty_fields(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/wiki/apply-diff",
            json={"task_id": "", "page_name": "existing-page", "diff": _DIFF_TEXT},
        )
        assert resp.status_code == 422
