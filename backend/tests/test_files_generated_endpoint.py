"""
Tests for POST /files/generated (main.py) — the direct-write counterpart to
the model-driven file_op write_file MCP tool, added so a user-triggered
"Save as" on a chat turn can land a file in generated_files/ without going
through the agent/MCP round trip.

Follows test_files_upload_endpoint.py's fixture convention: the real FastAPI
lifespan is never triggered, and _state.generated_dir is set up per-test —
plus file_ops' own module-global sandbox root is synced to it, mirroring
what main.py's startup path does via _set_generated_file_root().
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.mcp_server import file_ops


@pytest.fixture()
def client(tmp_path: Path):
    prev_generated_dir = main._state.generated_dir
    prev_file_ops_root = file_ops._project_root

    generated_dir = tmp_path / "generated_files"
    generated_dir.mkdir()
    main._state.generated_dir = generated_dir
    file_ops.set_project_root(generated_dir)

    yield TestClient(main.app), generated_dir

    main._state.generated_dir = prev_generated_dir
    file_ops._project_root = prev_file_ops_root


class TestFilesGeneratedEndpoint:

    def test_saves_txt_file(self, client):
        test_client, generated_dir = client
        resp = test_client.post(
            "/files/generated",
            json={"filename": "meeting-notes", "extension": "txt", "content": "hello"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "meeting-notes.txt"
        assert body["type"] == "generated"
        assert (generated_dir / "meeting-notes.txt").read_text(encoding="utf-8") == "hello"

    def test_saves_md_file(self, client):
        test_client, generated_dir = client
        resp = test_client.post(
            "/files/generated",
            json={"filename": "summary", "extension": "md", "content": "# Summary\n"},
        )
        assert resp.status_code == 200
        assert resp.json()["filename"] == "summary.md"
        assert (generated_dir / "summary.md").read_text(encoding="utf-8") == "# Summary\n"

    def test_auto_versions_on_name_collision(self, client):
        test_client, generated_dir = client
        (generated_dir / "note.txt").write_text("old", encoding="utf-8")

        resp = test_client.post(
            "/files/generated",
            json={"filename": "note", "extension": "txt", "content": "new"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "note_2.txt"
        assert (generated_dir / "note.txt").read_text(encoding="utf-8") == "old"
        assert (generated_dir / "note_2.txt").read_text(encoding="utf-8") == "new"

    def test_sanitizes_path_traversal_attempt(self, client):
        test_client, generated_dir = client
        resp = test_client.post(
            "/files/generated",
            json={"filename": "../../etc/passwd", "extension": "txt", "content": "x"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "/" not in body["filename"]
        assert Path(body["path"]).parent == generated_dir.resolve()

    def test_rejects_empty_content(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/files/generated",
            json={"filename": "empty", "extension": "txt", "content": "   "},
        )
        assert resp.status_code == 400
        assert "content" in resp.json()["detail"]

    def test_rejects_filename_with_no_valid_characters(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/files/generated",
            json={"filename": "!!!???", "extension": "txt", "content": "hello"},
        )
        assert resp.status_code == 400
        assert "filename" in resp.json()["detail"]

    def test_rejects_unsupported_extension(self, client):
        test_client, _ = client
        resp = test_client.post(
            "/files/generated",
            json={"filename": "note", "extension": "pdf", "content": "hello"},
        )
        assert resp.status_code == 422
