"""
OllamaVisionOCRProvider (mcp_server/ocr_ollama.py) — the cross-platform
image-OCR fallback route, step 7 of the OSS release build order.

requests.post is patched directly (patch.object(ocr_ollama.requests,
"post", ...)), same convention as test_ollama_runtime_client.py's
TestTimeoutOverride — no real network call, no real Ollama server needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localist.mcp_server import ocr_ollama
from localist.mcp_server.ocr_provider import OCRProvider


def _fake_response(status_code: int = 200, content: str = "", text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text or content
    response.json.return_value = {"message": {"content": content}, "done": True}
    return response


@pytest.fixture()
def upload_root(tmp_path: Path, monkeypatch):
    from localist.mcp_server import ocr
    monkeypatch.setattr(ocr, "_upload_root", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def vision_model_env(monkeypatch):
    monkeypatch.setenv("LOCALIST_OLLAMA_VISION_MODEL", "llava")
    monkeypatch.delenv("LOCALIST_OLLAMA_URL", raising=False)


class TestOcrOllamaProtocolConformance:
    def test_satisfies_ocr_provider(self):
        assert isinstance(ocr_ollama.OllamaVisionOCRProvider(), OCRProvider)


class TestOcrOllamaSuccess:
    def test_returns_transcribed_text_and_sends_expected_payload(self, upload_root):
        (upload_root / "photo.png").write_bytes(b"fake-png-bytes")
        response = _fake_response(content="Hello, world!")

        with patch.object(ocr_ollama.requests, "post", return_value=response) as mock_post:
            result = ocr_ollama.OllamaVisionOCRProvider().extract_text("photo.png", "image/png")

        assert result == "Hello, world!"
        assert mock_post.call_args.args[0] == "http://localhost:11434/api/chat"
        sent = json.loads(mock_post.call_args.kwargs["data"])
        assert sent["model"] == "llava"
        assert sent["stream"] is False
        assert sent["messages"][0]["role"] == "user"
        assert isinstance(sent["messages"][0]["images"], list)
        assert len(sent["messages"][0]["images"]) == 1

    def test_respects_custom_ollama_url(self, upload_root, monkeypatch):
        monkeypatch.setenv("LOCALIST_OLLAMA_URL", "http://localhost:12345")
        (upload_root / "photo.png").write_bytes(b"fake-png-bytes")
        response = _fake_response(content="text")

        with patch.object(ocr_ollama.requests, "post", return_value=response) as mock_post:
            ocr_ollama.OllamaVisionOCRProvider().extract_text("photo.png", "image/png")

        assert mock_post.call_args.args[0] == "http://localhost:12345/api/chat"


class TestOcrOllamaNoTextFound:
    def test_sentinel_response_raises(self, upload_root):
        (upload_root / "photo.png").write_bytes(b"fake-png-bytes")
        response = _fake_response(content="NO_TEXT_FOUND")

        with patch.object(ocr_ollama.requests, "post", return_value=response):
            with pytest.raises(ValueError, match="no readable text detected"):
                ocr_ollama.OllamaVisionOCRProvider().extract_text("photo.png", "image/png")

    def test_sentinel_response_case_insensitive(self, upload_root):
        (upload_root / "photo.png").write_bytes(b"fake-png-bytes")
        response = _fake_response(content="no_text_found.")

        with patch.object(ocr_ollama.requests, "post", return_value=response):
            with pytest.raises(ValueError, match="no readable text detected"):
                ocr_ollama.OllamaVisionOCRProvider().extract_text("photo.png", "image/png")

    def test_near_empty_response_raises(self, upload_root):
        (upload_root / "photo.png").write_bytes(b"fake-png-bytes")
        response = _fake_response(content="  ")

        with patch.object(ocr_ollama.requests, "post", return_value=response):
            with pytest.raises(ValueError, match="no readable text detected"):
                ocr_ollama.OllamaVisionOCRProvider().extract_text("photo.png", "image/png")


class TestOcrOllamaConfigAndInput:
    def test_missing_vision_model_env_raises_before_network_call(self, upload_root, monkeypatch):
        monkeypatch.delenv("LOCALIST_OLLAMA_VISION_MODEL", raising=False)
        (upload_root / "photo.png").write_bytes(b"fake-png-bytes")

        with patch.object(ocr_ollama.requests, "post") as mock_post:
            with pytest.raises(ValueError, match="LOCALIST_OLLAMA_VISION_MODEL"):
                ocr_ollama.OllamaVisionOCRProvider().extract_text("photo.png", "image/png")

        mock_post.assert_not_called()

    def test_pdf_mime_type_raises_before_network_call(self, upload_root):
        (upload_root / "doc.pdf").write_bytes(b"%PDF-fake")

        with patch.object(ocr_ollama.requests, "post") as mock_post:
            with pytest.raises(ValueError, match="requires macOS on Apple Silicon"):
                ocr_ollama.OllamaVisionOCRProvider().extract_text("doc.pdf", "application/pdf")

        mock_post.assert_not_called()

    def test_missing_file_raises(self, upload_root):
        with pytest.raises(ValueError, match="file not found"):
            ocr_ollama.OllamaVisionOCRProvider().extract_text("ghost.png", "image/png")

    def test_path_traversal_blocked(self, upload_root):
        with pytest.raises(ValueError, match="path traversal"):
            ocr_ollama.OllamaVisionOCRProvider().extract_text("../../etc/passwd", "image/png")


class TestOcrOllamaTransportErrors:
    def test_connection_error_raises_value_error(self, upload_root):
        (upload_root / "photo.png").write_bytes(b"fake-png-bytes")

        with patch.object(
            ocr_ollama.requests, "post",
            side_effect=ocr_ollama.requests.ConnectionError("refused"),
        ):
            with pytest.raises(ValueError, match="cannot reach Ollama"):
                ocr_ollama.OllamaVisionOCRProvider().extract_text("photo.png", "image/png")

    def test_timeout_raises_value_error(self, upload_root):
        (upload_root / "photo.png").write_bytes(b"fake-png-bytes")

        with patch.object(
            ocr_ollama.requests, "post",
            side_effect=ocr_ollama.requests.Timeout(),
        ):
            with pytest.raises(ValueError, match="did not respond within"):
                ocr_ollama.OllamaVisionOCRProvider().extract_text("photo.png", "image/png")

    def test_non_200_raises_value_error_with_body(self, upload_root):
        (upload_root / "photo.png").write_bytes(b"fake-png-bytes")
        response = _fake_response(status_code=500, text="model not found")

        with patch.object(ocr_ollama.requests, "post", return_value=response):
            with pytest.raises(ValueError, match="model not found"):
                ocr_ollama.OllamaVisionOCRProvider().extract_text("photo.png", "image/png")
