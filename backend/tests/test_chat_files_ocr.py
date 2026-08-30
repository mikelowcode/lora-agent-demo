"""
Tests for POST /chat/files (main.py) — the ephemeral chat-attachment
upload endpoint. Covers both branches: the original UTF-8-decode text path
(previously untested directly anywhere in this suite — test_session_files.py
only covers the underlying cache module, never the endpoint itself) and the
new image/PDF OCR-routing path added in
docs/architecture/22-local-ocr-service.md (Phase 1-4).

MCPToolDispatcher.dispatch() is monkeypatched per-test rather than
exercising a real MCP round trip or real Vision/PyMuPDF — those are
covered by test_mcp_server.py (ocr.extract_text unit tests) and
test_mcp_tool_dispatcher.py (ocr_extract routing), plus live verification
against the real running stack during development (see
docs/architecture/22-local-ocr-service.md).

Follows test_chat_pin_wiki_page.py's fixture convention: the real FastAPI
lifespan is never triggered, and _state.runtime is set up per-test.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist import session_files
from localist.mcp_server import ocr
from localist.mcp_tool_dispatcher import MCPToolDispatcher
from localist.prompt_builder import ToolResult


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ocr, "_upload_root", tmp_path)

    prev_runtime = main._state.runtime
    main._state.runtime = MagicMock(name="fake_runtime")

    session_files.clear()
    yield TestClient(main.app)

    main._state.runtime = prev_runtime
    session_files.clear()


def _ok_dispatch(result_text: str):
    def fake_dispatch(self, tools_to_call, instruction, context=None):
        assert tools_to_call == ["ocr_extract"]
        return [ToolResult(tool_name="ocr_extract", parameters="", result=result_text, success=True)]
    return fake_dispatch


def _failing_dispatch(error_text: str):
    def fake_dispatch(self, tools_to_call, instruction, context=None):
        return [ToolResult(tool_name="ocr_extract", parameters="", result=error_text, success=False)]
    return fake_dispatch


class TestTextFileUploadPath:
    """The original, pre-OCR text-upload path — unaffected by this feature,
    backfilled here since no prior test hit the endpoint directly."""

    def test_text_file_uploads_successfully(self, client):
        resp = client.post("/chat/files", files={"file": ("notes.md", b"# Hello\nsome notes", "text/markdown")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "notes.md"
        assert body["token_estimate"] > 0

        files = session_files.get_files()
        assert len(files) == 1
        assert files[0].filename == "notes.md"
        assert files[0].content == "# Hello\nsome notes"
        assert files[0].source == "upload"

    def test_binary_non_utf8_file_returns_422(self, client):
        resp = client.post("/chat/files", files={"file": ("script.py", b"\xff\xfe\x00binary", "text/x-python")})
        assert resp.status_code == 422
        assert "could not be read as UTF-8" in resp.json()["detail"]
        assert session_files.get_files() == []

    def test_disallowed_extension_returns_400(self, client):
        resp = client.post("/chat/files", files={"file": ("archive.zip", b"not really a zip", "application/zip")})
        assert resp.status_code == 400
        assert "not supported" in resp.json()["detail"]


class TestOcrUploadPath:
    """Image/PDF uploads — routed through ocr_extract instead of the
    UTF-8-decode branch. See main.py's _OCR_MIME_BY_EXTENSION."""

    def test_png_upload_success_lands_in_session_files(self, client):
        with patch.object(MCPToolDispatcher, "dispatch", _ok_dispatch("Extracted screenshot text")):
            resp = client.post("/chat/files", files={"file": ("screenshot.png", b"\x89PNG fake bytes", "image/png")})

        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "screenshot.png"
        assert body["token_estimate"] > 0

        files = session_files.get_files()
        assert len(files) == 1
        assert files[0].filename == "screenshot.png"
        assert files[0].content == "Extracted screenshot text"
        assert files[0].source == "upload"

    def test_pdf_upload_success_lands_in_session_files(self, client):
        with patch.object(MCPToolDispatcher, "dispatch", _ok_dispatch("Extracted PDF text")):
            resp = client.post("/chat/files", files={"file": ("report.pdf", b"%PDF-fake", "application/pdf")})

        assert resp.status_code == 200
        files = session_files.get_files()
        assert files[0].filename == "report.pdf"
        assert files[0].content == "Extracted PDF text"

    def test_heic_upload_routes_through_ocr(self, client):
        calls = []

        def fake_dispatch(self, tools_to_call, instruction, context=None):
            calls.append(tools_to_call)
            return [ToolResult(tool_name="ocr_extract", parameters="", result="Extracted HEIC text", success=True)]

        with patch.object(MCPToolDispatcher, "dispatch", fake_dispatch):
            resp = client.post("/chat/files", files={"file": ("photo.heic", b"heic fake bytes", "image/heic")})

        assert resp.status_code == 200
        # Confirm it actually took the OCR branch (mime_type threaded through
        # correctly), not a lucky UTF-8 decode of the fake bytes.
        assert calls == [["ocr_extract"]]

    def test_ocr_failure_returns_422_with_tool_detail(self, client):
        with patch.object(MCPToolDispatcher, "dispatch", _failing_dispatch("ERROR: no readable text detected")):
            resp = client.post("/chat/files", files={"file": ("blank.png", b"\x89PNG fake bytes", "image/png")})

        assert resp.status_code == 422
        assert "no readable text detected" in resp.json()["detail"]
        assert session_files.get_files() == []

    def test_context_carries_correct_mime_type_per_extension(self, client):
        seen_context = {}

        def fake_dispatch(self, tools_to_call, instruction, context=None):
            seen_context.update(context or {})
            return [ToolResult(tool_name="ocr_extract", parameters="", result="text", success=True)]

        with patch.object(MCPToolDispatcher, "dispatch", fake_dispatch):
            client.post("/chat/files", files={"file": ("photo.jpg", b"fake jpg", "image/jpeg")})

        assert seen_context["ocr_mime_type"] == "image/jpeg"
        assert seen_context["ocr_file_path"]  # non-empty temp filename

    def test_temp_file_is_written_and_cleaned_up(self, client, tmp_path: Path):
        seen_paths = []

        def fake_dispatch(self, tools_to_call, instruction, context=None):
            tmp_name = context["ocr_file_path"]
            resolved = tmp_path / tmp_name
            seen_paths.append(resolved)
            assert resolved.exists(), "temp file must exist while the OCR call is in flight"
            return [ToolResult(tool_name="ocr_extract", parameters="", result="text", success=True)]

        with patch.object(MCPToolDispatcher, "dispatch", fake_dispatch):
            resp = client.post("/chat/files", files={"file": ("screenshot.png", b"\x89PNG fake bytes", "image/png")})

        assert resp.status_code == 200
        assert seen_paths and not seen_paths[0].exists(), "temp file must be deleted after the call"

    def test_temp_file_cleaned_up_even_on_ocr_failure(self, client, tmp_path: Path):
        seen_paths = []

        def fake_dispatch(self, tools_to_call, instruction, context=None):
            resolved = tmp_path / context["ocr_file_path"]
            seen_paths.append(resolved)
            return [ToolResult(tool_name="ocr_extract", parameters="", result="ERROR: failed", success=False)]

        with patch.object(MCPToolDispatcher, "dispatch", fake_dispatch):
            client.post("/chat/files", files={"file": ("blank.png", b"\x89PNG fake bytes", "image/png")})

        assert seen_paths and not seen_paths[0].exists()
