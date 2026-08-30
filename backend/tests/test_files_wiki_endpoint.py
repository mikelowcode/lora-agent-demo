"""
Tests for GET /files/wiki (main.py) — the wiki-page listing endpoint reused
by both the Files sidebar and Feature A's wiki-pin picker.

Covers OKF alignment (§18): index.md/logs.md/MEMORY.md must never appear in
the response — they're structural/generated files, never a page a user
would pin as a diff target. Follows test_wiki_apply_diff_endpoint.py's
fixture convention: the real FastAPI lifespan is never triggered, and
_state.wiki_dir is set up per-test.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localist import main


@pytest.fixture()
def client(tmp_path: Path):
    prev_wiki_dir = main._state.wiki_dir

    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    main._state.wiki_dir = wiki_dir

    yield TestClient(main.app), wiki_dir

    main._state.wiki_dir = prev_wiki_dir


class TestFilesWikiEndpoint:

    def test_excludes_meta_wiki_filenames(self, client):
        test_client, wiki_dir = client
        (wiki_dir / "real-page.md").write_text("# Real Page\n", encoding="utf-8")
        (wiki_dir / "index.md").write_text("# Wiki Index\n", encoding="utf-8")
        (wiki_dir / "logs.md").write_text("# Wiki Changelog\n", encoding="utf-8")
        (wiki_dir / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")

        resp = test_client.get("/files/wiki")

        assert resp.status_code == 200
        names = [f["name"] for f in resp.json()["files"]]
        assert names == ["real-page"]

    def test_empty_wiki_dir_returns_empty_list(self, client):
        test_client, _ = client
        resp = test_client.get("/files/wiki")
        assert resp.status_code == 200
        assert resp.json()["files"] == []
