## 22. Local OCR Service (Apple Vision + PyMuPDF)

### 22.1 Overview

Chat uploads accept images (incl. HEIC) and PDFs, extracted to plain text at
upload time by a dedicated local OCR tool — entirely independent of whichever
chat inference backend (oMLX/Ollama/Foundry) is active. **This is text
extraction only** — a photo, screenshot, or scan with visible text works; a
photo with no text in it (a portrait, a movie poster with only stylized
logo art, a landscape) does not describe or caption, it correctly finds
nothing and rejects. See §22.10.

This closes §11.6's
Open Item 4 and the matching note in §3 (Slot SF), but *not* the way either
originally proposed: both assumed oMLX's native multimodal support (Gemma
4B's built-in OCR) would be routed into directly. That approach was scoped in
full at §21 and explicitly rejected in favor of this one — see §22.8 for why.

The core design decision: OCR happens once, at upload time, and the result is
plain text. By the time it reaches `session_files.py`, an OCR'd image is
indistinguishable from a `.md` upload — same cache, same token budget, same
`[SESSION FILES]` prompt slot (§3, §11.5). Nothing downstream of the upload
endpoint (`PromptBuilder`, `ConversationalAgent`, `BaseRuntimeClient`, any
runtime client) has any awareness that OCR happened at all. This is what
makes the feature "truly engine-agnostic": there is no dependency on which
chat backend is active, no per-turn re-sending of image bytes, and no new
multimodal wire format to maintain across three runtime clients.

### 22.2 `ocr_extract` MCP Tool (`mcp_server/ocr.py`)

A 14th tool on `localist-mcp` (§14), registered in `mcp_server/main.py`
alongside `file_op`/`fetch_url`/etc. `extract_text(path, mime_type,
max_pdf_pages=None) -> str`, sandboxed the same way `file_ops.py` sandboxes
`generated_files/` — a separate `chat_uploads/` subdirectory of the same
configurable root (`LOCALIST_MCP_UPLOAD_ROOT`, defaults to `backend/`),
because this holds ephemeral user-uploaded input, not agent-authored output.
Raises `ValueError` on any failure (unsupported platform, missing file,
unsupported mime type, page-cap exceeded, near-empty extraction result) —
same raise-on-failure contract as `file_ops.py`, converted to `isError=True`
by the MCP protocol layer.

**Images** (`image/*`, including `image/heic`): Apple's Vision framework via
PyObjC (`VNImageRequestHandler` + `VNRecognizeTextRequest`, accurate
recognition level, language correction on). Neural Engine-accelerated, no
model download, no extra process — not an inference engine in the Localist
sense at all. HEIC support requires no format-specific code: `Vision
.VNImageRequestHandler.alloc().initWithData_options_()` decodes via ImageIO
internally, which already handles HEIC/HEIF system-wide on macOS (the same
codec Photos/Preview use) alongside PNG/JPEG/WEBP — using that initializer on
raw bytes is the entire HEIC story.

**PDFs** (`application/pdf`): text-layer extraction first via PyMuPDF
(`fitz.open()` + `page.get_text()`) — fast, exact for digitally-created
PDFs, and skips OCR entirely when the average extracted-chars-per-page clears
`_MIN_CHARS_PER_TEXT_LAYER_PAGE` (20). Below that threshold (scanned-PDF
signal), each page is rasterized (`page.get_pixmap(dpi=200)`) and OCR'd
through the same Vision path, joined with `--- page N ---` markers — pages
Vision finds nothing on are dropped entirely rather than emitting a bare
marker with no content under it (§22.9). `max_pdf_pages` (default from
`LOCALIST_OCR_MAX_PDF_PAGES`, 20) bounds only the rasterize+OCR fallback
path — a real text layer is read directly regardless of page count.

Platform-gated the same way `EmbeddingEngine` is gated in `main.py`
(`platform.system() == "Darwin" and platform.machine() in ("arm64",
"aarch64")`) — on an unsupported platform, `extract_text()` raises rather
than silently degrading.

### 22.3 Upload Routing (`backend/main.py`)

`POST /chat/files` (`attach_chat_file`) branches on extension before the
pre-existing UTF-8-decode attempt: `_OCR_MIME_BY_EXTENSION` (`.png`/`.jpg`/
`.jpeg`/`.webp`/`.heic`/`.pdf`) routes to `_extract_text_via_ocr()`; every
other extension keeps the original text path unchanged. An explicit dict
rather than `mimetypes.guess_type()` — `.heic` isn't reliably registered in
the stdlib mimetypes database across platforms.

`_extract_text_via_ocr()` writes the upload to a temp file under
`mcp_server.ocr.get_upload_root()` — imported directly from `mcp_server.ocr`
(not re-derived) so `main.py` and the separate `localist-mcp` process always
agree on the same directory; both are launched from `backend/` as cwd by
`start_localist.sh`, so a plain relative filename resolves identically for
both. It then constructs `MCPToolDispatcher(runtime=_require_runtime())` and
calls `await asyncio.to_thread(dispatcher.dispatch, ["ocr_extract"], "",
{...})` — `dispatch()` is a synchronous entry point that internally calls
`asyncio.run()`, so it must be thread-offloaded from an `async def` FastAPI
endpoint, the same convention every other blocking call in `main.py` already
follows. The temp file is deleted in a `finally` block regardless of outcome.
On OCR failure, raises the same `HTTPException(422, ...)` shape the
UTF-8-decode-failure branch already used, so the frontend needed no new error
handling.

**`MCPToolDispatcher` routing** (`mcp_tool_dispatcher.py`): `ocr_extract` is
never planner-routed — Planner's `tools_to_call` never names it (unlike every
other tool in this dispatcher, it exists solely for `main.py`'s direct call
above). `_run_ocr_extract()` reads `context["ocr_file_path"]`/
`context["ocr_mime_type"]` directly (no instruction-parsing — always
context-driven), modeled on `_run_file_op()`'s existing convention of
resolving real arguments from `context` when present. No new dispatcher API
was needed; `dispatch()`'s existing `(tools_to_call, instruction, context)`
signature already supported this shape.

### 22.4 `session_files.py` Integration

`ALLOWED_EXTENSIONS` widened with the same six OCR extensions. No other
change — the module has zero OCR/image awareness by design; an OCR'd upload
is just a string by the time `add_file()` sees it, subject to the same
4,000-token per-file / 20,000-token total budget as any text upload (§11.2).

### 22.5 Frontend (`ChatPanel.svelte`)

`ALLOWED_EXTENSIONS`/the file input's `accept` attribute widened via a new
`OCR_EXTENSIONS` subset constant. No runtime-backend-aware gating — unlike
the rejected §21 approach, OCR works identically regardless of which chat
backend is active, so the attach affordance is never disabled based on
runtime state. `AttachedFile` gained an `extracting?: boolean` field:
`handleFileSelect()` pushes an immediate placeholder pill for OCR-routed
uploads (OCR has real wall-clock latency, unlike a text file's near-instant
UTF-8 decode), then replaces it with the real `{filename, token_estimate}`
result, or removes it cleanly on failure. The pill template branches on
`f.extracting` to show a pulsing dot (reusing the existing global
`.dot`/`.dot-pulse` classes, already used for the assistant "planning…"
status) and "Extracting text…" instead of the normal icon/tokens/remove
layout.

### 22.6 Configuration

New in `backend/.env.example`, alongside the existing MCP-server block:

- `LOCALIST_MCP_UPLOAD_ROOT` — sandbox root for `ocr_extract`'s temp
  uploads, same convention as `LOCALIST_MCP_PROJECT_ROOT`. Blank defaults to
  `backend/`, with uploads further sandboxed under `chat_uploads/`.
- `LOCALIST_OCR_MAX_PDF_PAGES` — page cap for the rasterize+OCR fallback
  path only. Default 20.

New in `backend/requirements.txt`: `pyobjc-framework-Vision`,
`pyobjc-framework-Quartz`, `PyMuPDF`.

### 22.7 Test Coverage

- `tests/test_mcp_server.py` — `TestOcrExtractImages`/`TestOcrExtractPdf`/
  `TestOcrGetMaxPdfPages` (direct unit tests of `ocr.extract_text()`) plus
  two `ocr_extract` cases in the existing in-process MCP-wiring class.
  `_ocr_image_bytes` (the actual PyObjC/Vision call) is mocked throughout —
  Vision can't run in CI the way `EmbeddingEngine.embed` is already mocked in
  this suite; PDF fixtures use real PyMuPDF rather than mocking `fitz`'s API
  surface.
- `tests/test_mcp_tool_dispatcher.py` — `TestOcrExtractRouting`: success,
  tool-level error, two distinct MCP-unreachable code paths
  (`_call_mcp_tool` raising vs. `_open_session` raising), missing-context
  rejection.
- `tests/test_session_files.py` — `TestOcrExtensionsAllowed`: widened
  allowlist accepted, normal budget rules still apply.
- New `tests/test_chat_files_ocr.py` — full `POST /chat/files` coverage,
  including backfilling the plain text-upload path (had no direct test
  anywhere before this feature) and temp-file lifecycle (written before the
  OCR call, deleted after, including on failure).

Full suite: 1406 → 1443 passed, 0 failed. Live-verified against the real
running stack (real Apple Vision OCR, real PyMuPDF, real MCP round trip, no
mocks) via direct `curl` upload during development — see §22.9's note on
environment constraints during that pass.

### 22.8 Rejected Alternative: oMLX-Native Multimodal Routing (§21)

§21 scoped wiring chat images through `OMLXRuntimeClient.infer_with_file()`
— the one existing multimodal call in the codebase, currently used only by
`WikiAgent`'s document-ingest path (§14, `wiki_agent.py:1272`). That approach
was fully scoped (oMLX-only, `PromptBuilder`-integrated, separate image
byte-cache) but never built. Reasons this local-OCR design was chosen
instead, in order of weight:

1. **Backend independence.** `infer_with_file()` only exists on
   `OMLXRuntimeClient`. Wiring chat images through it would mean the attach
   affordance had to be disabled whenever Ollama or Foundry was the active
   runtime (§16) — a real UX regression compared to every other chat
   capability, none of which care which backend is active.
2. **No new multimodal contract.** Routing images through the model would
   have required widening `PromptBuilder.build()`'s return shape and
   `BaseRuntimeClient`'s string-only `infer()`/`infer_stream()` signatures —
   real surface area across three runtime clients for a feature whose actual
   requirement (OCR text extraction) doesn't need model-level multimodal
   reasoning at all.
3. **No per-turn resend cost.** §21 flagged, as an unresolved open item,
   that re-sending base64 image bytes on every subsequent turn (matching how
   text session files behave) would be a real payload/latency cost working
   against `PromptBuilder`'s KV-cache-reuse design principle (§3). Once OCR
   output is plain text, this problem doesn't exist — it's cached and
   re-injected exactly like any text file, cheaply.

§21 is kept in place (not moved to `archive/`) as a documented record of the
alternative considered and why it lost, per its own updated status line.

**Explicitly not addressed by this build** (kept out of scope on purpose):
`WikiAgent`'s existing `infer_with_file()` PDF/image ingest path still
depends on oMLX specifically and still degrades to a plain-text prompt with
no real PDF/image handling on Ollama/Foundry. The new `ocr_extract` tool
could close that gap too — same tool, second caller — but that's a separate,
not-yet-scoped follow-up, not bundled into this feature.

**Update, 2026-08-03:** closed — see §24.

### 22.9 Bugs Caught During Build

Two real bugs were caught by testing, not by inspection — both are locked in
by permanent regression tests (§22.7):

1. **Falsy-zero page cap.** `extract_text(..., max_pdf_pages=0)` was
   originally resolved via `max_pdf_pages or get_max_pdf_pages()` — since
   `0` is falsy in Python, an explicit "reject everything" cap was silently
   replaced by the default. Fixed with an explicit `is not None` check.
   Caught via live verification (a real Apple Vision + PyMuPDF smoke test
   run during Phase 1 development, not a unit test written after the fact).
2. **Empty-page marker passing the near-empty check.** When every page of a
   scanned PDF produced no OCR text, `_extract_pdf()` still joined
   `"--- page 1 ---\n"` — non-trivial-looking text with zero real content —
   which cleared `extract_text()`'s final near-empty-result rejection
   undetected. Fixed by only emitting a page marker for pages that actually
   produced text; a fully-blank scan now correctly raises "no readable text
   detected." Caught while writing `test_near_empty_ocr_result_raises`
   (§22.7).

Frontend UI verification (the "Extracting text…" pill, widened file picker)
was confirmed visually by Michael directly against the running dev server —
no browser-automation tool was available in the build environment, and the
app was already running live under an active session at the time, so it was
deliberately not driven or restarted programmatically.

### 22.10 Known Limitation: Text Extraction Only, No General Image Understanding

This service does OCR — literal text recognition — not image captioning or
description. Live-reported by Michael (2026-08-01): uploading a photo/poster
with no visible text (a movie poster, `Runaways 2021.jpg`) fails with:

```
'Runaways 2021.jpg' could not be read — ERROR: no readable text detected in
'<temp-filename>.jpg' — extraction produced no usable content.
```

This is working as designed, not a bug. `_ocr_image_bytes()` (§22.2) uses
Vision's `VNRecognizeTextRequest` specifically — a text-recognition request
type that returns recognized text observations and nothing else. It has no
concept of "describe what's in this image"; a photo with genuinely no
legible text in it produces zero text observations, correctly triggering
`extract_text()`'s near-empty-result rejection (`_MIN_EXTRACTED_CHARS`, same
check that catches a blank scan). The tool cannot distinguish "this image
has no text" from "OCR failed" — both look identical from `VNRecognizeText
Request`'s output, so both get the same rejection message today.

**Not in scope for this service.** General image understanding (captioning,
object/scene description, "what's in this photo") is a fundamentally
different capability — it needs a real vision-language model, not a
text-recognition API. That's exactly the capability the rejected §21
alternative would have used (oMLX's Gemma 4B multimodal support), but §21
was rejected specifically for OCR-shaped uploads (§22.8) — a photo without
text isn't an OCR use case, and routing it through Gemma 4B would reopen the
backend-independence/multimodal-contract tradeoffs §22.8 rejected. If
image captioning becomes a real want, it needs its own scoping pass, not a
quiet extension of `ocr_extract`.

### 22.11 `OCRProvider` Protocol (OSS release, step 6)

`extract_text(path, mime_type, max_pdf_pages=None) -> str` (§22.2) was
already the single interface every caller used — `mcp_server/main.py`'s
`ocr_extract` tool, `wiki_agent.py`'s raw ingest, `main.py`'s chat-upload
route — but it wasn't formally typed as a swappable interface. New
`mcp_server/ocr_provider.py` defines `OCRProvider` as a
`@runtime_checkable typing.Protocol`, mirroring `base_runtime_client.py`'s
`BaseRuntimeClient` (same rationale: structural typing, no forced
inheritance, trivial to mock/isinstance-check). `ocr.py` gained one small
additive class, `VisionOCRProvider`, that implements the Protocol by
delegating to the existing module-level `extract_text()` — no change to
`_ocr_image_bytes`, `_extract_pdf`, sandboxing, or any call site; every
caller still uses the free function directly today.

This exists to give a future second implementation (e.g. an Ollama
vision-model route for non-Apple-Silicon platforms, scoped separately and
not yet built) a real interface to implement against. **§22.10's boundary
is unaffected and applies to any `OCRProvider` implementation, not just
this one** — the Protocol's docstring says so explicitly: text extraction
only, never general image understanding, regardless of which model or
platform is doing the extracting.
