"""
Tests for POST /chat/pin-wiki-page (main.py) — pins an existing wiki page
into the ephemeral session file cache by stem (see
docs/architecture/11-session-file-attachments.md).

Covers:
  - 200 + session_files cache updated with source="wiki_pin" on success.
  - 404 when no {stem}.md exists on disk (validated against the real wiki
    directory, not the graph index, since the graph is only rebuilt on an
    explicit trigger and can lag behind real files).
  - 400 pass-through when session_files.add_file() rejects (e.g. oversized
    page), matching POST /chat/files's existing behavior.
  - 503 when wiki_dir isn't configured.

Follows test_wiki_apply_diff_endpoint.py's fixture convention: the real
FastAPI lifespan is never triggered, and _state.wiki_dir is set up per-test.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist import session_files


@pytest.fixture()
def client(tmp_path: Path):
    prev_wiki_dir = main._state.wiki_dir

    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    main._state.wiki_dir = wiki_dir

    session_files.clear()
    yield TestClient(main.app), wiki_dir

    main._state.wiki_dir = prev_wiki_dir
    session_files.clear()


class TestPinWikiPageEndpoint:

    def test_success_pins_page_with_wiki_pin_source(self, client):
        test_client, wiki_dir = client
        (wiki_dir / "how-localist-works.md").write_text(
            "# How Localist Works\n\ncontent\n", encoding="utf-8",
        )

        resp = test_client.post("/chat/pin-wiki-page", json={"stem": "how-localist-works"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "how-localist-works.md"
        assert body["source"] == "wiki_pin"
        assert body["token_estimate"] > 0

        files = session_files.get_files()
        assert len(files) == 1
        assert files[0].filename == "how-localist-works.md"
        assert files[0].source == "wiki_pin"
        assert "How Localist Works" in files[0].content

    def test_missing_page_returns_404(self, client):
        test_client, _ = client
        resp = test_client.post("/chat/pin-wiki-page", json={"stem": "nonexistent-page"})
        assert resp.status_code == 404

    def test_oversized_page_returns_400(self, client):
        test_client, wiki_dir = client
        oversized = "x" * ((session_files.MAX_FILE_TOKENS + 1) * session_files._CHARS_PER_TOKEN)
        (wiki_dir / "huge-page.md").write_text(oversized, encoding="utf-8")

        resp = test_client.post("/chat/pin-wiki-page", json={"stem": "huge-page"})

        assert resp.status_code == 400
        assert "too large to pin" in resp.json()["detail"]

    def test_missing_wiki_dir_returns_503(self, client):
        test_client, _ = client
        main._state.wiki_dir = None

        resp = test_client.post("/chat/pin-wiki-page", json={"stem": "anything"})
        assert resp.status_code == 503

    def test_request_validation_rejects_empty_stem(self, client):
        test_client, _ = client
        resp = test_client.post("/chat/pin-wiki-page", json={"stem": ""})
        assert resp.status_code == 422

    @pytest.mark.parametrize("stem", ["index", "logs", "MEMORY"])
    def test_meta_wiki_filename_rejected_even_if_present_on_disk(self, client, stem):
        """index.md/logs.md/MEMORY.md are structural, never pinnable — even
        if one exists on disk, pinning it must be rejected (defense in
        depth beyond the picker UI, which already excludes these from
        GET /files/wiki — see docs/architecture/18-okf-wiki-alignment.md)."""
        test_client, wiki_dir = client
        (wiki_dir / f"{stem}.md").write_text("structural file content", encoding="utf-8")

        resp = test_client.post("/chat/pin-wiki-page", json={"stem": stem})

        assert resp.status_code == 400
        assert "not a pinnable page" in resp.json()["detail"]
        assert session_files.get_files() == []
