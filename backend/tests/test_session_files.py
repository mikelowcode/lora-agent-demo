"""
Tests for session_files.py — the ephemeral per-session file cache.

Covers the "source" parameter added for wiki-page pinning
(docs/architecture/11-session-file-attachments.md): add_file() defaults to
source="upload" (unchanged behavior for existing callers), accepts
source="wiki_pin" to mark a pinned wiki page, threads it through
get_files() as SessionFile.source, and gives the per-file-too-large
rejection distinct wording for a pin (the user didn't choose the page's
size, the corpus did).
"""

import pytest

from localist import session_files


@pytest.fixture(autouse=True)
def _clear_cache():
    """The module-level cache is process-lifetime global state — reset
    it before and after every test so tests never leak into each other."""
    session_files.clear()
    yield
    session_files.clear()


class TestAddFileSourceDefault:

    def test_default_source_is_upload(self):
        error = session_files.add_file("notes.md", "hello world")
        assert error is None

        files = session_files.get_files()
        assert len(files) == 1
        assert files[0].filename == "notes.md"
        assert files[0].content == "hello world"
        assert files[0].source == "upload"


class TestAddFileWikiPin:

    def test_wiki_pin_source_is_recorded(self):
        error = session_files.add_file(
            "localist-software-stack.md", "# Stack\n\ncontent", source="wiki_pin",
        )
        assert error is None

        files = session_files.get_files()
        assert len(files) == 1
        assert files[0].source == "wiki_pin"

    def test_wiki_pin_too_large_gets_distinct_message(self):
        oversized = "x" * ((session_files.MAX_FILE_TOKENS + 1) * session_files._CHARS_PER_TOKEN)
        error = session_files.add_file("huge-page.md", oversized, source="wiki_pin")

        assert error is not None
        assert "too large to pin" in error
        assert "huge-page.md" in error

    def test_upload_too_large_keeps_original_message(self):
        oversized = "x" * ((session_files.MAX_FILE_TOKENS + 1) * session_files._CHARS_PER_TOKEN)
        error = session_files.add_file("huge-upload.md", oversized)

        assert error is not None
        assert "too large to pin" not in error
        assert "is too large" in error

    def test_extension_and_budget_rules_apply_unchanged_to_pins(self):
        error = session_files.add_file("page.bin", "content", source="wiki_pin")
        assert error is not None
        assert "not supported" in error


class TestOcrExtensionsAllowed:
    """Extensions routed through the local ocr_extract MCP tool (Phase 1-4,
    docs/architecture/22-local-ocr-service.md). By the time add_file() sees
    these, main.py has already replaced the raw bytes with OCR'd text —
    this module has no OCR awareness at all, so a plain string is all
    these tests need to prove the allowlist accepts them."""

    def test_image_and_pdf_extensions_are_accepted(self):
        for filename in ("photo.png", "photo.jpg", "photo.jpeg", "photo.webp", "photo.heic", "scan.pdf"):
            error = session_files.add_file(filename, "extracted OCR text")
            assert error is None, f"{filename} should be accepted, got: {error}"

        files = {f.filename for f in session_files.get_files()}
        assert files == {"photo.png", "photo.jpg", "photo.jpeg", "photo.webp", "photo.heic", "scan.pdf"}

    def test_still_subject_to_normal_budget_rules(self):
        oversized = "x" * ((session_files.MAX_FILE_TOKENS + 1) * session_files._CHARS_PER_TOKEN)
        error = session_files.add_file("huge-scan.pdf", oversized)
        assert error is not None
        assert "is too large" in error


class TestGetFilesPreservesMixedSources:

    def test_upload_and_pin_coexist_with_correct_sources(self):
        session_files.add_file("uploaded.md", "a")
        session_files.add_file("pinned.md", "b", source="wiki_pin")

        files = {f.filename: f.source for f in session_files.get_files()}
        assert files == {"uploaded.md": "upload", "pinned.md": "wiki_pin"}

    def test_remove_file_works_regardless_of_source(self):
        session_files.add_file("pinned.md", "b", source="wiki_pin")
        assert session_files.remove_file("pinned.md") is True
        assert session_files.get_files() == []
