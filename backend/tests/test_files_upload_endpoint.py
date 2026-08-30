"""
Tests for POST /files/upload and GET /files/raw (main.py) — the wiki
raw-file upload/listing endpoints, widened alongside the local OCR service
(docs/architecture/22-local-ocr-service.md) to accept OCR-eligible
(image/PDF) files in addition to .md/.txt. No test previously existed for
either endpoint at all; this also backfills basic .md/.txt coverage.

Follows test_files_wiki_endpoint.py's fixture convention: the real FastAPI
lifespan is never triggered, and _state.raw_dir is set up per-test.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localist import main


@pytest.fixture()
def client(tmp_path: Path):
    prev_raw_dir = main._state.raw_dir
    prev_memory_manager = main._state.memory_manager

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    main._state.raw_dir = raw_dir
    main._state.memory_manager = None  # index_document call is skipped

    yield TestClient(main.app), raw_dir

    main._state.raw_dir = prev_raw_dir
    main._state.memory_manager = prev_memory_manager


class TestFilesUploadEndpoint:

    def test_uploads_md_file(self, client):
        test_client, raw_dir = client
        resp = test_client.post(
            "/files/upload",
            files={"file": ("note.md", b"# A note\n", "text/markdown")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "note.md"
        assert body["type"] == "raw"
        assert (raw_dir / "note.md").read_bytes() == b"# A note\n"

    def test_uploads_ocr_eligible_png_bytes_unchanged(self, client):
        test_client, raw_dir = client
        png_bytes = b"\x89PNG\r\n\x1a\n\xff\xd8\xff\x00not-a-real-png-but-bytes"
        resp = test_client.post(
            "/files/upload",
            files={"file": ("scan.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "scan.png"
        assert body["type"] == "raw"
        # Saved as-is — no OCR at upload time (WikiAgent extracts lazily).
        assert (raw_dir / "scan.png").read_bytes() == png_bytes

    def test_uploads_ocr_eligible_pdf(self, client):
        test_client, raw_dir = client
        pdf_bytes = b"%PDF-1.4\xff\xfe not a real pdf but bytes"
        resp = test_client.post(
            "/files/upload",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        assert (raw_dir / "doc.pdf").read_bytes() == pdf_bytes

    def test_rejects_unsupported_extension(self, client):
        test_client, raw_dir = client
        resp = test_client.post(
            "/files/upload",
            files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")},
        )
        assert resp.status_code == 422
        assert "Unsupported file type" in resp.json()["detail"]
        assert not (raw_dir / "archive.zip").exists()

    def test_overwrites_existing_file_with_same_name(self, client):
        test_client, raw_dir = client
        (raw_dir / "note.txt").write_text("old", encoding="utf-8")
        resp = test_client.post(
            "/files/upload",
            files={"file": ("note.txt", b"new", "text/plain")},
        )
        assert resp.status_code == 200
        assert (raw_dir / "note.txt").read_text(encoding="utf-8") == "new"


class TestFilesRawEndpoint:

    def test_lists_md_txt_and_ocr_eligible_files(self, client):
        test_client, raw_dir = client
        (raw_dir / "a.md").write_text("A", encoding="utf-8")
        (raw_dir / "b.txt").write_text("B", encoding="utf-8")
        (raw_dir / "c.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (raw_dir / "d.pdf").write_bytes(b"%PDF-1.4")
        (raw_dir / "e.heic").write_bytes(b"\x00\x00\x00 ftypheic")

        resp = test_client.get("/files/raw")

        assert resp.status_code == 200
        names = {f["filename"] for f in resp.json()["files"]}
        assert names == {"a.md", "b.txt", "c.png", "d.pdf", "e.heic"}

    def test_excludes_unsupported_extensions(self, client):
        test_client, raw_dir = client
        (raw_dir / "keep.md").write_text("keep", encoding="utf-8")
        (raw_dir / "skip.py").write_text("print(1)", encoding="utf-8")

        resp = test_client.get("/files/raw")

        assert resp.status_code == 200
        names = {f["filename"] for f in resp.json()["files"]}
        assert names == {"keep.md"}

    def test_empty_raw_dir_returns_empty_list(self, client):
        test_client, _ = client
        resp = test_client.get("/files/raw")
        assert resp.status_code == 200
        assert resp.json()["files"] == []
