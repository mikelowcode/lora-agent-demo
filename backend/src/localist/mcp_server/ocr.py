"""
Localist MCP Server — ocr_extract tool implementation
========================================================
Extracts text from images and PDFs entirely locally, independent of
whichever chat inference backend (oMLX/Ollama/Foundry) is active — see
docs/architecture/22-local-ocr-service.md.

Images (including HEIC — see _ocr_image_bytes) are OCR'd via Apple's Vision
framework (VNRecognizeTextRequest): Neural Engine-accelerated, no model
download, no extra process, not an inference engine in the Localist sense
at all. PDFs try a direct text-layer extraction first (PyMuPDF) and only
fall back to per-page rasterize-then-OCR when no real text layer exists
(scanned PDFs) — API surface for both live-verified against this venv
(Vision: VNImageRequestHandler.initWithData_options_ +
VNRecognizeTextRequest.initWithCompletionHandler_; PyMuPDF:
fitz.open()/page.get_text()/page.get_pixmap().tobytes("png")).

macOS/Apple Silicon only, gated the same way EmbeddingEngine is gated in
main.py's is_apple_silicon check. Like file_ops.py, every function here
raises on failure rather than swallowing errors into a result string — the
MCP protocol layer converts a raised exception into isError=True for the
client (see file_ops.py's module docstring for the underlying mechanism).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

_IMAGE_MIME_PREFIX: str = "image/"
_PDF_MIME: str = "application/pdf"

# Below this average chars/page, a PDF's embedded text layer is treated as
# absent (scanned PDF) rather than real — falls back to rasterize+OCR.
_MIN_CHARS_PER_TEXT_LAYER_PAGE: int = 20

# Below this many total extracted characters, treat the result as a failed
# extraction (blank scan, non-text photo) rather than forwarding near-
# nothing into a chat prompt.
_MIN_EXTRACTED_CHARS: int = 3

_DEFAULT_MAX_PDF_PAGES: int = 20

# Extensions routed through ocr_extract instead of a plain UTF-8-decode/text
# path — shared by main.py's /chat/files and /files/upload routing and by
# WikiAgent's raw_path ingest routing, so all three callers agree on exactly
# which extensions are OCR-eligible. Kept as an explicit map rather than
# mimetypes.guess_type() — .heic in particular isn't reliably registered in
# the stdlib mimetypes database across platforms. Must stay in sync with this
# module's own supported mime types (image/*, application/pdf) and
# ChatPanel.svelte's/Sidebar.svelte's OCR extension lists.
OCR_MIME_BY_EXTENSION: dict[str, str] = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".pdf":  "application/pdf",
}

_upload_root: Path | None = None


def get_upload_root() -> Path:
    """
    Resolve the sandbox root for uploaded images/PDFs awaiting OCR.

    Configurable via the LOCALIST_MCP_UPLOAD_ROOT environment variable.
    Defaults to the backend/ directory (parent of this package), matching
    file_ops.get_project_root()'s convention. OCR uploads are sandboxed
    under a fixed "chat_uploads" subdirectory of that root — distinct from
    file_op's "generated_files" since this holds ephemeral user-uploaded
    input, not agent-authored output — created on first resolution if it
    doesn't yet exist.
    """
    global _upload_root
    if _upload_root is None:
        env_root = os.environ.get("LOCALIST_MCP_UPLOAD_ROOT")
        base = (
            Path(env_root).resolve()
            if env_root
            # backend/src/localist/mcp_server/ocr.py -> backend/
            else Path(__file__).resolve().parent.parent.parent.parent
        )
        _upload_root = base / "chat_uploads"
        _upload_root.mkdir(parents=True, exist_ok=True)
    return _upload_root


def set_upload_root(path: Path | str) -> None:
    """Override the sandbox root. Used at startup and by tests."""
    global _upload_root
    _upload_root = Path(path).resolve()


def get_max_pdf_pages() -> int:
    """
    Default PDF page cap for the rasterize+OCR fallback path, read fresh on
    every call (same "read fresh, never cache" convention as
    mcp_tool_dispatcher.py's WEB_SEARCH_PROVIDER checks). Overridable via
    LOCALIST_OCR_MAX_PDF_PAGES; falls back to _DEFAULT_MAX_PDF_PAGES on an
    unset or non-integer value. A real text-layer PDF is read directly
    regardless of page count — this cap only bounds the OCR fallback path.
    """
    raw = os.environ.get("LOCALIST_OCR_MAX_PDF_PAGES", "")
    try:
        return int(raw) if raw else _DEFAULT_MAX_PDF_PAGES
    except ValueError:
        return _DEFAULT_MAX_PDF_PAGES


def _sandbox_resolve(rel_path: str) -> Path:
    """
    Resolve rel_path against upload_root and enforce sandboxing.

    Same checks and error text convention as file_ops._sandbox_resolve.
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


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def extract_text(path: str, mime_type: str, max_pdf_pages: int | None = None) -> str:
    """
    Extract text from an uploaded image or PDF, sandboxed under upload_root.

    Raises ValueError on: unsupported platform, unsupported mime_type,
    missing file, PDF page count over max_pdf_pages (rasterize+OCR path
    only), or a near-empty extraction result.
    """
    if not _is_apple_silicon():
        raise ValueError(
            "ERROR: OCR requires macOS on Apple Silicon (Vision framework) — "
            f"this platform is {platform.system()}/{platform.machine()}."
        )

    resolved = _sandbox_resolve(path)
    if not resolved.exists():
        raise ValueError(f"ERROR: file not found — {resolved}")

    if mime_type.startswith(_IMAGE_MIME_PREFIX):
        text = _ocr_image_bytes(resolved.read_bytes())
    elif mime_type == _PDF_MIME:
        pages_cap = max_pdf_pages if max_pdf_pages is not None else get_max_pdf_pages()
        text = _extract_pdf(resolved, pages_cap)
    else:
        raise ValueError(f"ERROR: unsupported mime_type '{mime_type}' for OCR")

    if len(text.strip()) < _MIN_EXTRACTED_CHARS:
        raise ValueError(
            f"ERROR: no readable text detected in '{resolved.name}' — "
            f"extraction produced no usable content."
        )
    return text


def _ocr_image_bytes(data: bytes) -> str:
    """
    Run Apple Vision's VNRecognizeTextRequest against raw image bytes.

    VNImageRequestHandler(imageData:) decodes via ImageIO internally, which
    already handles HEIC/HEIF (same codec Photos/Preview use) alongside
    PNG/JPEG/WEBP system-wide on macOS — no format-specific branching
    needed here, HEIC support falls out of using this initializer rather
    than assuming a specific pixel format.
    """
    try:
        import Vision
        from Foundation import NSData
    except ImportError as exc:
        raise ValueError(
            "ERROR: pyobjc-framework-Vision is not installed — "
            "see backend/requirements.txt."
        ) from exc

    ns_data = NSData.dataWithBytes_length_(data, len(data))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)

    results: list[str] = []
    errors: list[object] = []

    def _completion(request, error):
        if error is not None:
            errors.append(error)
            return
        for observation in request.results():
            candidates = observation.topCandidates_(1)
            if candidates:
                results.append(candidates[0].string())

    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(_completion)
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    ok, err = handler.performRequests_error_([request], None)
    if not ok or errors:
        detail = errors[0] if errors else err
        raise ValueError(f"ERROR: Vision OCR failed — {detail}")

    return "\n".join(results)


def _extract_pdf(resolved: Path, max_pdf_pages: int) -> str:
    """
    Text-layer extraction first (fast, exact for digitally-created PDFs);
    falls back to per-page rasterize+OCR only when the text layer is
    absent/sparse (scanned PDF signal — see _MIN_CHARS_PER_TEXT_LAYER_PAGE).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ValueError(
            "ERROR: PyMuPDF is not installed — see backend/requirements.txt."
        ) from exc

    doc = fitz.open(resolved)
    try:
        text_layer = "\n".join(page.get_text() for page in doc)
        if len(text_layer.strip()) >= _MIN_CHARS_PER_TEXT_LAYER_PAGE * doc.page_count:
            return text_layer

        if doc.page_count > max_pdf_pages:
            raise ValueError(
                f"ERROR: '{resolved.name}' has {doc.page_count} pages, "
                f"exceeding the {max_pdf_pages}-page OCR limit "
                f"(no usable text layer to read directly instead)."
            )

        ocr_parts: list[str] = []
        for page_index, page in enumerate(doc):
            pixmap = page.get_pixmap(dpi=200)
            page_text = _ocr_image_bytes(pixmap.tobytes("png")).strip()
            # Skip pages Vision found nothing on entirely, rather than
            # emitting a bare "--- page N ---" marker with no content under
            # it — a PDF where every page comes back empty must still trip
            # extract_text()'s final near-empty check, not be padded past it
            # by marker text alone.
            if page_text:
                ocr_parts.append(f"--- page {page_index + 1} ---\n{page_text}")
        return "\n\n".join(ocr_parts)
    finally:
        doc.close()
