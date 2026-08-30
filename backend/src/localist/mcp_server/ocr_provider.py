"""
Localist MCP Server — OCRProvider Protocol
==============================================
The interface any local text-extraction implementation for `ocr_extract`
must satisfy.

Layer placement
---------------
  mcp_server/main.py's ocr_extract tool  →  OCRProvider (this contract)
                                                   ↓
                                          VisionOCRProvider (ocr.py)  |  …

Architectural contract
----------------------
- This module defines the Protocol only. Zero platform-specific logic.
- `ocr.py`'s `VisionOCRProvider` implements this interface today (Apple
  Vision + PyMuPDF, macOS/Apple Silicon only).
- Text extraction only — see docs/architecture/22-local-ocr-service.md
  §22.10. A future implementation may target a different platform or
  technique (e.g. an Ollama vision model prompted to transcribe text
  verbatim), but must honor the same text-extraction-only contract; general
  image understanding (captioning, "what's in this photo") is explicitly
  out of scope for anything implementing this Protocol and would need its
  own interface and its own scoping pass.

Why a Protocol rather than an ABC?
-----------------------------------
Same rationale as `base_runtime_client.BaseRuntimeClient`: typing.Protocol
keeps this structurally typed, so an implementation satisfies the interface
without inheriting from it — `ocr.py`'s existing free-function
`extract_text()` stays exactly as-is, `VisionOCRProvider` just delegates to
it. @runtime_checkable enables isinstance() checks where useful (tests,
future provider-selection logic).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class OCRProvider(Protocol):
    """
    Structural interface for local text-extraction providers.

    Methods
    -------
    extract_text(path, mime_type, max_pdf_pages=None) → str
        Extract plain text from a local image or PDF file.

    Parameters
    ----------
    path:
        Path to the file, resolved against the implementation's own
        sandbox root (see e.g. ocr.py's get_upload_root()/_sandbox_resolve).
    mime_type:
        The file's MIME type — implementations decide which types they
        support (ocr.py: image/* and application/pdf).
    max_pdf_pages:
        Optional page cap for a PDF fallback path that needs to render
        pages (e.g. rasterize+OCR for a scanned PDF with no text layer).
        None means "use the implementation's own configured default."
        Implementations that never rasterize may ignore this parameter.

    Returns
    -------
    str
        The extracted plain text. Never a partial or placeholder string —
        a failed or near-empty extraction must raise instead of returning
        one.

    Raises
    ------
    ValueError
        On any failure: unsupported platform, unsupported mime_type,
        missing file, PDF page count over max_pdf_pages (fallback path
        only), or a near-empty extraction result. Every implementation
        must raise rather than return a degraded result — matches
        file_ops.py's raise-on-failure convention, converted to
        isError=True by the MCP protocol layer.
    """

    def extract_text(
        self,
        path:          str,
        mime_type:     str,
        max_pdf_pages: int | None = None,
    ) -> str:
        ...
