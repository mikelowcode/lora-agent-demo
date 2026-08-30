"""
Localist MCP Server — Ollama vision-model OCR route
========================================================
Cross-platform image text-extraction, implementing OCRProvider
(ocr_provider.py) via a local Ollama vision-capable chat model — the
fallback ocr_extract (mcp_server/main.py) reaches for on any platform
where Apple's Vision framework (ocr.py) isn't available.

Images only. PDFs are not handled here — extending PDF support would mean
un-gating PyMuPDF (currently bundled Apple-Silicon-only alongside Vision in
the [ocr] extra) for every platform, a real license-footprint decision
(PyMuPDF is AGPL-3.0, already flagged in THIRD_PARTY_LICENSES.md as pending
replacement) deliberately left for a future step, not a side effect of this
one. See docs/architecture/22-local-ocr-service.md §22.12.

Text extraction only, same boundary as ocr.py (see §22.10): the prompt
below instructs verbatim transcription only, never description or
captioning. This matters more here than for Vision's VNRecognizeTextRequest
— a real OCR API that returns literally nothing for a textless image — a
vision-language model will otherwise answer in prose ("I don't see any
text..."), which would silently misrepresent that prose as extracted
content. The prompt asks for an exact sentinel string on no legible text,
checked in addition to the same near-empty-length threshold ocr.py uses.

This process does not inherit backend/main.py's own load_dotenv() call
(see mcp_server/main.py's module docstring) — LOCALIST_OLLAMA_VISION_MODEL
and LOCALIST_OLLAMA_URL are read fresh on every call via os.environ, same
convention as ollama_web_search.py.

Sandboxing duplicates ocr.py's/file_ops.py's _sandbox_resolve() (same
checks, same error text) rather than importing a private helper across
modules — this codebase's established convention (see ocr.py's own
docstring). get_upload_root()/set_upload_root() are the one piece of
shared, non-duplicated infrastructure — both OCR providers must resolve
against the same upload_root.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

from .ocr import get_upload_root

_DEFAULT_BASE_URL: str = "http://localhost:11434"
_CHAT_PATH: str = "/api/chat"
_REQUEST_TIMEOUT: float = 60.0

_NO_TEXT_SENTINEL: str = "NO_TEXT_FOUND"

_TRANSCRIBE_PROMPT: str = (
    "Transcribe all readable text in this image exactly as it appears, "
    "verbatim. Output only the transcribed text — no description of the "
    "image, no commentary, no added formatting or markup. "
    f"If there is no legible text anywhere in the image, respond with "
    f"exactly this and nothing else: {_NO_TEXT_SENTINEL}"
)

# Below this many extracted characters, treat the result as a failed
# extraction — same threshold and rationale as ocr.py's _MIN_EXTRACTED_CHARS.
_MIN_EXTRACTED_CHARS: int = 3


def _sandbox_resolve(rel_path: str) -> Path:
    """
    Resolve rel_path against upload_root and enforce sandboxing.

    Same checks and error text convention as ocr.py's/file_ops.py's
    _sandbox_resolve.
    """
    upload_root = get_upload_root()
    try:
        resolved = (upload_root / rel_path).resolve()
    except Exception as exc:
        raise ValueError(f"ERROR: invalid path — {exc}") from exc

    if not str(resolved).startswith(str(upload_root)):
        raise ValueError(
            "ERROR: path traversal outside upload_root is not permitted"
        )
    return resolved


class OllamaVisionOCRProvider:
    """
    Cross-platform OCRProvider implementation via a local Ollama
    vision-capable chat model. See module docstring for scope (images
    only) and the sentinel-based no-text contract.
    """

    def extract_text(
        self,
        path:          str,
        mime_type:     str,
        max_pdf_pages: int | None = None,
    ) -> str:
        if not mime_type.startswith("image/"):
            raise ValueError(
                f"ERROR: OllamaVisionOCRProvider only supports image/* "
                f"uploads (got '{mime_type}') — PDF extraction requires "
                f"macOS on Apple Silicon."
            )

        model = os.environ.get("LOCALIST_OLLAMA_VISION_MODEL", "")
        if not model:
            raise ValueError(
                "ERROR: LOCALIST_OLLAMA_VISION_MODEL is not set — required "
                "for image OCR on this platform (no Apple Vision framework "
                "available). Set it to a vision-capable Ollama model you've "
                "pulled locally, e.g. 'llava' or 'qwen2.5vl'."
            )

        resolved = _sandbox_resolve(path)
        if not resolved.exists():
            raise ValueError(f"ERROR: file not found — {resolved}")

        base_url = os.environ.get("LOCALIST_OLLAMA_URL", _DEFAULT_BASE_URL).rstrip("/")
        endpoint = base_url + _CHAT_PATH
        image_b64 = base64.b64encode(resolved.read_bytes()).decode("ascii")

        payload = {
            "model": model,
            "messages": [
                {
                    "role":    "user",
                    "content": _TRANSCRIBE_PROMPT,
                    "images":  [image_b64],
                },
            ],
            "stream": False,
        }

        try:
            response = requests.post(
                endpoint,
                headers = {"Content-Type": "application/json"},
                data    = json.dumps(payload),
                timeout = _REQUEST_TIMEOUT,
            )
        except requests.ConnectionError as exc:
            raise ValueError(
                f"ERROR: cannot reach Ollama at {endpoint} for image OCR — "
                f"is the service running? Detail: {exc}"
            ) from exc
        except requests.Timeout:
            raise ValueError(
                f"ERROR: Ollama did not respond within {_REQUEST_TIMEOUT}s "
                f"(endpoint: {endpoint}) — image OCR request timed out."
            )

        if response.status_code != 200:
            raise ValueError(
                f"ERROR: Ollama returned HTTP {response.status_code} from "
                f"{endpoint} for image OCR: {response.text[:400]}"
            )

        try:
            text = response.json()["message"]["content"].strip()
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"ERROR: unexpected response shape from Ollama at "
                f"{endpoint} for image OCR: {exc}"
            ) from exc

        if text.strip(".").strip().upper() == _NO_TEXT_SENTINEL or len(text) < _MIN_EXTRACTED_CHARS:
            raise ValueError(
                f"ERROR: no readable text detected in '{resolved.name}' — "
                f"extraction produced no usable content."
            )

        return text
