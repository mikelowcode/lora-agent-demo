"""
Localist — WikiAgent
================
Self-contained wiki ingestion agent.  Owns all logic directly.

Previously this module wrapped ``agent_wiki_loop_streaming.py`` and
imported pure functions, Pydantic models, and the SYSTEM_PROMPT from it.
That standalone script has been deleted.  All logic now lives here.

Layer placement
---------------
  ControllerAgent  →  WikiAgent  →  FoundryRuntimeClient (inference)
                                 →  MemoryManager (optional — index updates)

Architectural contract
----------------------
- Pure Python module.  No FastAPI, no HTTP, no stdin, no sys.exit().
- Satisfies the AgentInterface Protocol defined in controller_agent.py.
- All model inference is requested through the injected RuntimeClient —
  never by calling the Foundry HTTP API directly.
- All file I/O is scoped to paths supplied by the Controller via SubTask.context.
  The agent does NOT resolve paths relative to its own __file__ location.
- Changes are written to the wiki directory only when
  SubTask.context["auto_apply"] is True (default: False).  When False,
  the proposed actions are returned in AgentResult.output for the Controller
  or a human review step to approve.
- Journal entries are written to SubTask.context["journal_path"] when
  provided; silently skipped otherwise.
- When a MemoryManager is supplied at construction time, every page
  successfully written to disk is immediately indexed via
  memory_manager.index_document().  This keeps the document index current
  without requiring a full corpus re-scan on the next ResearchAgent call.

SubTask.context schema
----------------------
Required keys (exactly one of the following two — raw_path wins if both
are present; see WikiAgent.run())
    raw_path : str | Path
        Absolute path to the raw file to ingest (.md or .txt).
    diff_target : str
        Stem of an existing wiki page (e.g. "localist-software-stack") to
        propose a minimal unified diff against. subtask.instruction is
        used as the free-text description of the desired change. No raw
        file is read on this path — see WikiAgent._run_diff_only().

Optional keys
    wiki_dir : str | Path
        Wiki pages directory.  Defaults to <project_root>/wiki.
    schema_path : str | Path
        SCHEMA.md path.  Defaults to <project_root>/SCHEMA.md.
    templates_dir : str | Path
        Templates directory.  Defaults to <project_root>/templates.
    journal_path : str | Path
        Path for the agent_journal.jsonl file.  Omit to skip journalling.
    auto_apply : bool
        Write approved changes to disk immediately.  Default False.
    max_tokens : int
        Max tokens for the model call.  Default 2048.
    temperature : float
        Sampling temperature.  Default 0.2.

AgentResult.output schema (on success)
---------------------------------------
    new_pages : list[dict]
        Each dict: {page_name, page_type, content}
    diffs : list[dict]
        Each dict: {page_name, diff}
    applied : bool
        True when changes were written to disk in this call.
    raw_filename : str | None
        Basename of the ingested raw file. None on the diff_target path
        (there is no raw file) — never repurposed to mean "target page".
    diff_target : str | None
        The resolved target page stem when this call took the diff-only
        path (WikiAgent._run_diff_only()). None on the raw_path ingest
        path.
    wiki_page_count : int
        Number of wiki pages in the index at call time.

    Additional keys present only when auto_apply=True
    written : list[str]
        Page names successfully written or patched.
    skipped : list[str]
        Page names skipped (page already existed for new-page actions).
    diff_errors : list[str]
        Page names where diff application failed.

Porting notes (changes from agent_wiki_loop_streaming.py)
----------------------------------------------------------
The following changes were made during consolidation. The first four are
deliberate architectural removals. The last is a bug fix.

1. sys.exit() removed — all error paths return FAILED AgentResult instead.
2. Interactive prompts removed — file selection and approval loop are gone.
   raw_path comes from SubTask.context; auto_apply replaces the approval prompt.
3. Direct requests.post() removed — all inference goes through RuntimeClient.
4. __file__-relative PROJECT_ROOT removed — project_root is a constructor
   argument with a sensible default; all paths come from SubTask.context.
5. apply_unified_diff fixed — the original used difflib.restore(lines, 2),
   which is designed for ndiff format, not unified diff format.  Since the
   SYSTEM_PROMPT instructs the model to emit unified diffs, restore() always
   returned an empty list.  Replaced with a correct hunk-level parser that
   handles the unified diff format the model actually produces.
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .build_graph import build_graph
from .mcp_server.ocr import OCR_MIME_BY_EXTENSION, get_upload_root as _ocr_get_upload_root
from .mcp_tool_dispatcher import MCPToolDispatcher
from .prompt_builder import PromptBuilder
from . import wiki_maintenance_log
from .wiki_doc import META_WIKI_FILENAMES, parse_wiki_doc

from .controller_agent import (
    AgentResult,
    SubTask,
    TaskStatus,
)

if TYPE_CHECKING:
    from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV-cache guard rails (match original constants)
# ---------------------------------------------------------------------------

_MAX_WIKI_CHARS   = 6_000
_RELEVANT_PAGES_N = 4
_SUMMARY_LINES    = 3
_MAX_SCHEMA_LINES = 120


# ---------------------------------------------------------------------------
# Pre-write snapshot safety net (§17.8)
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR_NAME = ".snapshots"
_SNAPSHOT_TTL_DEFAULT_SECONDS = 30 * 24 * 60 * 60  # 30 days
_SNAPSHOT_TTL_ENV_VAR = "LOCALIST_WIKI_SNAPSHOT_TTL_SECONDS"


def _snapshot_ttl_seconds() -> int:
    """
    Read LOCALIST_WIKI_SNAPSHOT_TTL_SECONDS from the environment.

    Read at call time (not cached) — same convention as Planner's
    _tool_fallback_mode() — so tests and ops tooling can exercise the
    prune-on-write and startup-sweep paths with a short TTL instead of
    requiring a real 30-day-old file. Falls back to the 30-day default on
    missing or invalid values.
    """
    raw = os.environ.get(_SNAPSHOT_TTL_ENV_VAR)
    if raw is None:
        return _SNAPSHOT_TTL_DEFAULT_SECONDS
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "[wiki_agent] Invalid %s=%r — falling back to default TTL of %d seconds.",
            _SNAPSHOT_TTL_ENV_VAR, raw, _SNAPSHOT_TTL_DEFAULT_SECONDS,
        )
        return _SNAPSHOT_TTL_DEFAULT_SECONDS


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a deterministic wiki agent. Your ONLY job is to output a single "
    "valid XML document. You MUST NOT output prose, comments, explanations, or "
    "Markdown fences. Your entire response must be the XML block and nothing else."
)

# System prompt for the diff-only path (WikiAgent._run_diff_only()) — no raw
# file is involved, and create_page is not a legal action here (the target
# page already exists), so the model is told that explicitly rather than
# relying on prompt-body instructions alone to prevent it from defaulting to
# create_page.
DIFF_SYSTEM_PROMPT = (
    "You are a deterministic wiki agent. Your ONLY job is to output a single "
    "valid XML document proposing a unified diff against ONE existing wiki "
    "page. Only <action name=\"apply_diff\"> is legal on this path — the "
    "target page already exists, so you MUST NOT use create_page. You MUST "
    "NOT output prose, comments, explanations, or Markdown fences. Your "
    "entire response must be the XML block and nothing else."
)

_PROMPT_BUILDER = PromptBuilder()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreatePage(BaseModel):
    """A request to create a new wiki page."""
    page_name: str
    page_type: Literal["SYSTEM", "CONCEPT", "RESEARCH_NOTE"]
    content:   str


class ApplyDiff(BaseModel):
    """A request to patch an existing wiki page with a unified diff."""
    page_name: str
    diff:      str


class Actions(BaseModel):
    """The complete set of wiki actions parsed from one model response."""
    new_pages: list[CreatePage] = Field(default_factory=list)
    diffs:     list[ApplyDiff]  = Field(default_factory=list)


class JournalEntry(BaseModel):
    """One append-only record in the agent journal file."""
    step:      str
    timestamp: datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))
    actions:   Actions | None = None
    approved:  bool    | None = None
    error:     str     | None = None


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_raw_content_via_ocr(runtime: Any, raw_path: Path, mime_type: str) -> str:
    """
    Extract raw_path's text via the ocr_extract MCP tool (Apple Vision +
    PyMuPDF — mcp_server/ocr.py), mirroring main.py's
    _extract_text_via_ocr() for chat uploads (docs/architecture/
    22-local-ocr-service.md). ocr_extract's sandboxing only resolves paths
    under its own upload root (mcp_server.ocr.get_upload_root()), so
    raw_path — which lives under the wiki's raw/ directory, a separate
    sandbox — is copied to a temp file there first and removed afterward
    regardless of outcome. Raises RuntimeError on any OCR failure.
    """
    upload_root = _ocr_get_upload_root()
    tmp_name = f"{uuid.uuid4().hex}{raw_path.suffix.lower()}"
    tmp_path = upload_root / tmp_name
    tmp_path.write_bytes(raw_path.read_bytes())

    try:
        dispatcher = MCPToolDispatcher(runtime=runtime)
        results = dispatcher.dispatch(
            ["ocr_extract"], "",
            {"ocr_file_path": tmp_name, "ocr_mime_type": mime_type},
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    result = results[0]
    if not result.success:
        raise RuntimeError(str(result.result))
    return result.result


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def sweep_expired_snapshots(wiki_dir: Path) -> list[Path]:
    """
    Remove every snapshot file under wiki_dir/.snapshots older than the
    effective TTL (see _snapshot_ttl_seconds()), regardless of page. Called
    once at startup (main.py's lifespan()) to catch snapshots belonging to
    pages that haven't been edited again since — WikiAgent._prune_page_snapshots()
    only prunes a page's own snapshots when that same page is next written.
    Returns the paths removed, for the caller to log.
    """
    snapshots_dir = wiki_dir / _SNAPSHOT_DIR_NAME
    if not snapshots_dir.exists():
        return []
    cutoff = time.time() - _snapshot_ttl_seconds()
    pruned: list[Path] = []
    for p in snapshots_dir.glob("*.md"):
        if p.is_file() and p.stat().st_mtime < cutoff:
            p.unlink()
            pruned.append(p)
    return pruned


def is_text_file(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except Exception:
        return False
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


# ---------------------------------------------------------------------------
# KV-cache-safe wiki context builder
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _one_line_summary(content: str, n: int = _SUMMARY_LINES) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return " | ".join(lines[:n])


def build_wiki_context(wiki_pages: dict[str, str], raw_content: str) -> str:
    """
    Build a prompt-safe wiki context block.

    Accepts a plain dict (stem → content) as before, so callers that load
    pages from the filesystem or from MemoryManager.get_all_documents() both
    work after normalising to this shape.
    """
    if not wiki_pages:
        return "(no existing wiki pages)"

    raw_tokens = _tokenize(raw_content)

    scored = sorted(
        (
            (len(raw_tokens & _tokenize(content)), name, content)
            for name, content in wiki_pages.items()
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    index_lines = ["## WIKI PAGE INDEX (all pages)\n"]
    for _, name, content in scored:
        index_lines.append(f"- {name}: {_one_line_summary(content)}")
    index_block = "\n".join(index_lines)

    full_blocks: list[str] = []
    budget = _MAX_WIKI_CHARS

    for _, name, content in scored[:_RELEVANT_PAGES_N]:
        chunk = f"---\n## WIKI PAGE (full): {name}\n\n{content}\n"
        if len(chunk) > budget:
            full_blocks.append(
                chunk[:budget] + "\n... [truncated to fit context budget]\n"
            )
            break
        full_blocks.append(chunk)
        budget -= len(chunk)
        if budget <= 0:
            break

    full_block = "\n".join(full_blocks)
    return f"{index_block}\n\n{full_block}".strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_EXAMPLE = """
# EXAMPLE OUTPUT (follow this structure exactly)
<actions>
  <action name="create_page">
    <page_name>example-research-note</page_name>
    <page_type>RESEARCH_NOTE</page_type>
    <content>
---
type: RESEARCH_NOTE
title: Example Research Note
description: One-sentence summary of what this page documents.
resource: example-raw-file.txt
tags: [example-tag-one, example-tag-two]
timestamp: YYYY-MM-DD
status: draft
source: example-raw-file.txt
---

# example-research-note

## Summary

Two to five sentences summarising the raw file. No speculation. Grounded only
in what the raw file explicitly states.

## Details

### Extracted Concepts

- Concept one extracted directly from the raw file.
- Concept two extracted directly from the raw file.

### Mapped Pages

- [[localist-software-stack]] — use the page_name from EXISTING WIKI PAGES verbatim, not a paraphrase.

### Proposed New Pages

- `new-page-name` (CONCEPT) — reason a new page is justified.

## Related Pages

- [[localist-software-stack]]

## Revision History

- YYYY-MM-DD — Initial research note created from example-raw-file.txt.
    </content>
  </action>
</actions>
"""


def build_user_prompt(
    schema_text:  str,
    templates:    dict[str, str],
    wiki_context: str,
    raw_filename: str,
    raw_content:  str,
) -> str:
    template_block = "".join(
        f"### TEMPLATE: {key}\n\n{content}\n"
        for key, content in templates.items()
    )

    schema_lines   = schema_text.splitlines()
    schema_snippet = "\n".join(schema_lines[:_MAX_SCHEMA_LINES])
    if len(schema_lines) > _MAX_SCHEMA_LINES:
        schema_snippet += "\n... [schema truncated for context budget]"

    today = date.today().isoformat()

    prompt = f"""\
# SCHEMA (rules you must follow)
{schema_snippet}

# PAGE TEMPLATES
{template_block}

# EXISTING WIKI PAGES
{wiki_context}

# RAW FILE TO INGEST
Filename: {raw_filename}

{textwrap.dedent(raw_content).strip()}

# YOUR TASK
1. Create exactly one RESEARCH_NOTE for the raw file. It MUST include all five
   sections in order: Summary · Details · Related Pages · Revision History,
   each as an H2 heading, plus front-matter at the top.
2. Front matter MUST also include: title (human-readable page title),
   description (one sentence summarizing the page), resource (the raw
   filename or URL this page was derived from), tags (2-5 kebab-case topic
   tags), and timestamp ({today}) — all real, specific values, never the
   literal placeholder text shown in the example below.
3. Details MUST contain three H3 subsections: Extracted Concepts · Mapped Pages
   · Proposed New Pages. Use the string "null" when a subsection is empty.
4. Optionally propose new CONCEPT or SYSTEM pages only if clearly justified.
5. For existing wiki pages, propose minimal unified diffs only where necessary.
6. Page names MUST be kebab-case (lowercase letters, digits, hyphens only).
7. Use {today} as the date in all Revision History entries.
8. Every [[...]] link target in "### Mapped Pages" and "## Related Pages"
   MUST exactly match an existing page name from EXISTING WIKI PAGES above,
   or a page_name you are proposing in this same response — not a
   paraphrase, not a page title, not a longer or shorter description of
   it. If you are not certain a page exists, propose it as a new page
   instead of linking to a guessed name.
{_EXAMPLE}
OUTPUT RULES (read before generating):
- Output ONLY the <actions> XML block. Nothing before it. Nothing after it.
- No prose, no code fences, no explanations outside the XML.
- Every <content> block MUST follow the example structure above exactly.\
"""
    return textwrap.dedent(prompt).strip()


def build_slim_prompt(
    schema_text:  str,
    templates:    dict[str, str],
    wiki_context: str,
    raw_filename: str,
) -> str:
    """
    Slim prompt for the oMLX native file ingestion path.

    Omits the RAW FILE TO INGEST section — the file content is delivered as
    a ``type="file"`` content block and processed by MarkItDown server-side.
    """
    template_block = "".join(
        f"### TEMPLATE: {key}\n\n{content}\n"
        for key, content in templates.items()
    )

    schema_lines   = schema_text.splitlines()
    schema_snippet = "\n".join(schema_lines[:_MAX_SCHEMA_LINES])
    if len(schema_lines) > _MAX_SCHEMA_LINES:
        schema_snippet += "\n... [schema truncated for context budget]"

    today = date.today().isoformat()

    prompt = f"""\
# SCHEMA (rules you must follow)
{schema_snippet}

# PAGE TEMPLATES
{template_block}

# EXISTING WIKI PAGES
{wiki_context}

# YOUR TASK
The file "{raw_filename}" has been provided directly for processing.
1. Create exactly one RESEARCH_NOTE for the raw file. It MUST include all five
   sections in order: Summary · Details · Related Pages · Revision History,
   each as an H2 heading, plus front-matter at the top.
2. Front matter MUST also include: title (human-readable page title),
   description (one sentence summarizing the page), resource (the raw
   filename or URL this page was derived from), tags (2-5 kebab-case topic
   tags), and timestamp ({today}) — all real, specific values, never the
   literal placeholder text shown in the example below.
3. Details MUST contain three H3 subsections: Extracted Concepts · Mapped Pages
   · Proposed New Pages. Use the string "null" when a subsection is empty.
4. Optionally propose new CONCEPT or SYSTEM pages only if clearly justified.
5. For existing wiki pages, propose minimal unified diffs only where necessary.
6. Page names MUST be kebab-case (lowercase letters, digits, hyphens only).
7. Use {today} as the date in all Revision History entries.
8. Every [[...]] link target in "### Mapped Pages" and "## Related Pages"
   MUST exactly match an existing page name from EXISTING WIKI PAGES above,
   or a page_name you are proposing in this same response — not a
   paraphrase, not a page title, not a longer or shorter description of
   it. If you are not certain a page exists, propose it as a new page
   instead of linking to a guessed name.
{_EXAMPLE}
OUTPUT RULES (read before generating):
- Output ONLY the <actions> XML block. Nothing before it. Nothing after it.
- No prose, no code fences, no explanations outside the XML.
- Every <content> block MUST follow the example structure above exactly.\
"""
    return textwrap.dedent(prompt).strip()


def build_diff_prompt(
    schema_text:  str,
    templates:    dict[str, str],
    page_name:    str,
    page_content: str,
    instruction:  str,
    tool_context: str | None = None,
) -> str:
    """
    Prompt for the diff-only path (WikiAgent._run_diff_only()).

    Unlike build_user_prompt()/build_slim_prompt(), this omits the wiki-wide
    context and the "RAW FILE TO INGEST" section entirely — the target page's
    full current content is the only content in scope, and the user's free-
    text instruction (not a raw file) describes the desired change. Only
    apply_diff is a legal action here; create_page is explicitly disallowed
    in both this prompt body and DIFF_SYSTEM_PROMPT.

    tool_context, when provided (a Priority 1c compound plan dispatched a
    tool alongside the pinned-page diff target — see
    Planner._priority1c_pinned_diff()), is spliced in as a "# FETCHED
    CONTEXT" section between the target page and the task instructions.
    Omitted entirely when None, reproducing prior output exactly.
    """
    template_block = "".join(
        f"### TEMPLATE: {key}\n\n{content}\n"
        for key, content in templates.items()
    )

    schema_lines   = schema_text.splitlines()
    schema_snippet = "\n".join(schema_lines[:_MAX_SCHEMA_LINES])
    if len(schema_lines) > _MAX_SCHEMA_LINES:
        schema_snippet += "\n... [schema truncated for context budget]"

    tool_section = f"\n# FETCHED CONTEXT\n\n{tool_context}\n" if tool_context else ""

    prompt = f"""\
# SCHEMA (rules you must follow)
{schema_snippet}

# PAGE TEMPLATES
{template_block}

# TARGET WIKI PAGE: {page_name}

{page_content}
{tool_section}
# YOUR TASK
The user wants the TARGET WIKI PAGE above updated. Their instruction:

{textwrap.dedent(instruction).strip()}

1. Propose exactly one minimal unified diff against the TARGET WIKI PAGE
   content shown above, using the <action name="apply_diff"> action. Do
   NOT propose a create_page action — this page already exists.
2. Removed/added/context lines must reproduce the TARGET WIKI PAGE content
   verbatim. The @@ line-number header is used only as a hint, not a
   guarantee, so it is more important that the actual line content is an
   exact, verbatim match than that the header numbers are precise.
3. If a line you are removing or adding begins with a markdown list bullet
   (e.g. "- " or "* "), your diff line MUST start with exactly one diff
   marker character (-, +, or a space for context) followed by the ENTIRE
   original line INCLUDING its bullet. Do not merge the bullet's dash with
   the diff marker into a single "-" — a removed bulleted line must read
   "--" then a space then the text (marker, then bullet, then space, then
   text), never a single "-" followed directly by the text.
4. If no change is actually needed, output an empty <actions></actions>
   block rather than fabricating a no-op diff.

# EXAMPLE OUTPUT (follow this structure exactly)
<actions>
  <action name="apply_diff">
    <page_name>{page_name}</page_name>
    <diff>
@@ -12,3 +12,4 @@
 unchanged context line
-old line to remove
+new line to add
+another new line
@@ -20,1 +21,1 @@
-- **Old Bulleted Item:** description text.
+- **New Bulleted Item:** description text.
    </diff>
  </action>
</actions>

OUTPUT RULES (read before generating):
- Output ONLY the <actions> XML block. Nothing before it. Nothing after it.
- No prose, no code fences, no explanations outside the XML.
- Only apply_diff actions are valid on this path; any create_page action
  will be discarded.\
"""
    return textwrap.dedent(prompt).strip()


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------

def _extract_actions_xml(text: str) -> str | None:
    """
    Extract the first complete <actions>…</actions> block.

    Uses str.find / str.rfind rather than regex so that nested angle brackets
    inside <content> blocks do not confuse the extractor.  Validation is
    deferred to parse_model_xml() via ET.fromstring() on the shielded block.
    """
    start = text.find("<actions")
    if start == -1:
        return None
    end = text.rfind("</actions>")
    if end == -1:
        return None
    return text[start : end + len("</actions>")].strip()


def _shield_content_blocks(xml_text: str) -> tuple[str, list[str]]:
    """
    Replace <content>...</content> AND <diff>...</diff> blocks with safe
    placeholders before XML parsing, so raw Markdown/prose/diff-syntax
    characters inside them — including bare "&", "<", ">" (e.g. a real page
    line like "Local Tools & Libraries" appearing as unchanged diff context;
    confirmed live, gemma4:31b-cloud, 2026-07-09 — see
    diagnostics/diag_wiki_agent_diff_only.py) — never reach ET.fromstring()
    unescaped. Both tag kinds share one placeholder list/counter since a
    single response can contain either or both.
    """
    blocks: list[str] = []

    def _make_replacer(tag: str):
        def _replacer(m: re.Match) -> str:
            blocks.append(m.group(1))
            return f"<{tag}>__CONTENT_{len(blocks) - 1}__</{tag}>"
        return _replacer

    shielded = re.sub(
        r"<content>(.*?)</content>",
        _make_replacer("content"),
        xml_text,
        flags=re.DOTALL,
    )
    shielded = re.sub(
        r"<diff>(.*?)</diff>",
        _make_replacer("diff"),
        shielded,
        flags=re.DOTALL,
    )
    return shielded, blocks


def parse_model_xml(raw_output: str) -> Actions:
    """
    Parse the model's XML response into an Actions object.

    Raises
    ------
    ValueError
        If no <actions>…</actions> block is found, or if it cannot be parsed.
    """
    xml_block = _extract_actions_xml(raw_output)
    if xml_block is None:
        raise ValueError("No valid <actions> XML block found in model output.")

    shielded, contents = _shield_content_blocks(xml_block)
    try:
        root = ET.fromstring(shielded)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse <actions> XML: {exc}") from exc

    if root.tag != "actions":
        raise ValueError(f"Root XML element must be <actions>, got <{root.tag}>.")

    new_pages: list[dict[str, str]] = []
    diffs:     list[dict[str, str]] = []

    for action in root.findall("action"):
        action_name = action.get("name", "")

        if action_name == "create_page":
            raw_content = action.findtext("content") or ""
            if raw_content.startswith("__CONTENT_") and raw_content.endswith("__"):
                idx = int(raw_content[10:-2])
                raw_content = contents[idx] if idx < len(contents) else ""
            entry = {
                "page_name": (action.findtext("page_name") or "").strip(),
                "page_type": (action.findtext("page_type") or "").strip(),
                "content":   raw_content.strip(),
            }
            if not entry["page_name"] or not entry["page_type"]:
                logger.warning(
                    "parse_model_xml: skipping create_page with missing page_name or page_type."
                )
                continue
            new_pages.append(entry)

        elif action_name == "apply_diff":
            raw_diff = action.findtext("diff") or ""
            if raw_diff.startswith("__CONTENT_") and raw_diff.endswith("__"):
                idx = int(raw_diff[10:-2])
                raw_diff = contents[idx] if idx < len(contents) else ""
            entry = {
                "page_name": (action.findtext("page_name") or "").strip(),
                "diff":      raw_diff.strip(),
            }
            if not entry["page_name"] or not entry["diff"]:
                logger.warning(
                    "parse_model_xml: skipping apply_diff with missing page_name or diff."
                )
                continue
            diffs.append(entry)

        else:
            logger.debug("parse_model_xml: ignoring unknown action name=%r.", action_name)

    logger.debug(
        "parse_model_xml: parsed %d create_page and %d apply_diff action(s).",
        len(new_pages), len(diffs),
    )
    return Actions.model_validate({"new_pages": new_pages, "diffs": diffs})


# ---------------------------------------------------------------------------
# Diff application
# ---------------------------------------------------------------------------

def apply_unified_diff(original: str, diff_text: str) -> str:
    """
    Apply a unified diff string to the original file content and return the
    patched result.

    BUG FIX vs original: agent_wiki_loop_streaming.py called
    ``difflib.restore(diff_lines, 2)``, which is designed for *ndiff* format,
    not unified diff format.  This implementation correctly parses @@ hunks.

    BUG FIX (2026-07-09, confirmed live via diagnostics/diag_wiki_agent_diff_
    only.py against gemma4:31b-cloud): hunks are now located by *content*
    (via _locate_hunk()), not by trusting the model-authored ``@@ -N,M``
    header as a literal position. Models are unreliable at counting exact
    1-indexed line numbers in a prompt-sized file, but reliably reproduce the
    actual text being replaced — the live failure had a hunk header claiming
    line 21 while the removed/context lines it emitted actually lived at a
    different offset entirely, so a position-trusting apply silently
    compared against unrelated content. orig_start is now used only as a
    disambiguation hint (see _locate_hunk()) when a hunk's content matches
    more than one position in the file.

    BUG FIX (2026-07-09, same live session, second distinct finding): a
    removed/added line that itself begins with a markdown bullet ("- ")
    collides with the unified-diff marker character, which is also "-"/"+".
    The model sometimes collapses the two into one dash (writing
    "-**Bold Text**..." instead of "-- **Bold Text**..."), silently dropping
    the bullet from the extracted before/after text. When the primary
    (as-authored) interpretation doesn't match anywhere, this function
    retries once assuming every removed/added line in the hunk is missing a
    "- " bullet prefix (see _extract_hunk_lines()) before giving up.

    BUG FIX (2026-07-09, same live session, third distinct finding —
    confirmed live via the review-then-apply UI's POST /wiki/apply-diff):
    the model's LAST emitted "+" line of a diff frequently has no trailing
    newline (it's the last line of the text block it generated), even when
    that line replaces a line that is nowhere near the end of the real
    file. Splicing such a line into `result_lines` mid-file then joins it
    directly onto the next line with no line break at all — live example:
    a one-line replacement mid-document glued itself onto the following
    "### Mapped Pages" heading with zero separation. Every line except the
    file's own last line is normalized to end with "\n" after all hunks are
    applied, regardless of which hunk's `after` lines were missing one.

    BUG FIX (2026-07-28, generalizing the second finding above): the bullet/
    marker collision can also happen in the other direction — an unchanged
    CONTEXT line (marker " ") whose own text happens to start with a
    markdown bullet ("- "/"* ") can have its leading " " marker dropped by
    the model, leaving a raw line that is shape-identical to a collapsed
    removed/added bulleted line. Previously this only had one recovery
    hypothesis to try (recover_bullet_marker) and, when the true cause was
    this context-side drop, fell through to a safe-but-blocked 409 (see
    docs/architecture/17-wiki-agent-diff-target.md §17.7 point 3 — confirmed
    live, deliberately left unfixed at the time). The two collisions are
    distinguishable by content: a removed/added line missing its bullet
    entirely has no leading space once its marker char is stripped (e.g.
    "-**Bold**" strips to "**Bold**"), whereas a context line that dropped
    its " " marker while keeping its own bullet dash intact still has a
    leading space after stripping one char (e.g. "- **Bold**" strips to
    " **Bold**", since the bullet format is dash+space+text). See
    recover_context_bullet_marker in _extract_hunk_lines().
    apply_unified_diff() now tries a small ordered list of interpretations
    per hunk (as-authored, bullet-recovered, context-recovered, and both
    recoveries combined) against _locate_hunk(), taking the first one that
    actually matches file content, rather than hard-coding exactly two
    tiers.

    Raises
    ------
    ValueError
        If the diff contains no @@ hunks, or if a hunk's content does not
        match anywhere in the file under any interpretation.
    """
    hunks = _parse_unified_hunks(diff_text)
    if not hunks:
        raise ValueError("Diff contains no recognisable @@ hunks.")

    result_lines = original.splitlines(keepends=True)

    for orig_start, hunk_lines in hunks:
        primary_exc: ValueError | None = None
        before: list[str] | None = None
        after:  list[str] | None = None
        idx = 0
        tried: set[tuple[str, ...]] = set()

        # Recomputed against the *current* state of result_lines on every
        # hunk — deliberately not offset-tracked across hunks, since each
        # hunk's position is now found by content, not carried arithmetic.
        for attempt, kwargs in enumerate(_RECOVERY_MODES):
            candidate_before, candidate_after = _extract_hunk_lines(hunk_lines, **kwargs)
            key = tuple(candidate_before)
            if key in tried:
                # This recovery hypothesis produced the same before-block as
                # one already tried (e.g. a hunk with no -/+ lines at all) —
                # skip the redundant lookup rather than re-raising or
                # re-searching.
                continue
            tried.add(key)
            try:
                idx = _locate_hunk(result_lines, candidate_before, orig_start)
            except ValueError as exc:
                if attempt == 0:
                    primary_exc = exc
                continue
            before, after = candidate_before, candidate_after
            if attempt:
                logger.debug(
                    "apply_unified_diff: recovered via %r at hunk originally "
                    "headed '@@ -%d'.", kwargs, orig_start,
                )
            break

        if before is None:
            # No interpretation matched — re-raise the original (as-authored)
            # failure, the most informative one, rather than the last one tried.
            raise primary_exc

        result_lines[idx : idx + len(before)] = after

    # Normalize: only the file's actual last line may lack a trailing "\n".
    # A spliced-in "after" line that came from the model's diff text may be
    # missing one even mid-file (see third bug-fix note above) — restore it
    # so the following line isn't silently glued onto it.
    for i in range(len(result_lines) - 1):
        if not result_lines[i].endswith("\n"):
            result_lines[i] += "\n"

    return "".join(result_lines)


# Ordered interpretations tried by apply_unified_diff() for each hunk, most
# to least literal. Each entry is the kwargs passed to _extract_hunk_lines().
# The last entry (both recoveries at once) covers a hunk unlucky enough to
# suffer both collision directions at once — rare, but each flag only acts
# on the lines its own signal actually matches (see _extract_hunk_lines()),
# so combining them is safe even when only one direction is actually present.
_RECOVERY_MODES: tuple[dict[str, bool], ...] = (
    {},
    {"recover_bullet_marker": True},
    {"recover_context_bullet_marker": True},
    {"recover_bullet_marker": True, "recover_context_bullet_marker": True},
)


def _extract_hunk_lines(
    hunk_lines:                    list[str],
    recover_bullet_marker:         bool = False,
    recover_context_bullet_marker: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Split one hunk's raw lines into (before, after) — the removed+context
    lines and the added+context lines, in file order.

    recover_bullet_marker=True re-derives before/after under the hypothesis
    that a removed/added bulleted line collapsed its markdown bullet's "- "
    into its own "-"/"+" diff marker (see apply_unified_diff()'s second
    2026-07-09 bug-fix note): the line dropped its "- " prefix entirely, so
    every removed/added line gets it prepended back. A dropped bullet always
    leaves no space directly after the marker char (e.g. "-**Bold**" strips
    to "**Bold**") — that's what recover_context_bullet_marker below uses to
    tell the two collisions apart.

    recover_context_bullet_marker=True re-derives before/after under the
    complementary hypothesis (2026-07-28, generalizing the above): a hunk's
    unchanged CONTEXT line that itself starts with a markdown bullet dropped
    its leading " " marker instead, reproducing the bullet's own "- "/"* "
    dash verbatim — which then reads exactly like a removed/added line whose
    marker char happens to equal the bullet's dash. The two are
    distinguishable by content: stripping one char off a genuinely collapsed
    removed/added bulleted line leaves no leading space ("**Bold**"), while
    stripping one char off a collapsed-context bulleted line leaves the
    bullet's own space intact (dash+space+text format ⇒ " **Bold**"). Only
    -/+ lines whose stripped content starts with a space are reclassified as
    unchanged context; every other -/+ line is handled by recover_bullet_
    marker (or left as-authored, if that's also False) — content that
    doesn't look like a bullet has no plausible context-collision reading.

    Context lines using the correctly-formatted " " marker are never touched
    by either recovery mode — both hypotheses only apply to lines that begin
    with "-" or "+".
    """
    before: list[str] = []
    after:  list[str] = []

    for line in hunk_lines:
        if line.startswith("-") or line.startswith("+"):
            content = line[1:]
            if recover_context_bullet_marker and content.startswith(" "):
                before.append(line)
                after.append(line)
                continue
            recovered = ("- " + content) if recover_bullet_marker else content
            if line.startswith("-"):
                before.append(recovered)
            else:
                after.append(recovered)
        else:
            ctx = line[1:] if line.startswith(" ") else line
            before.append(ctx)
            after.append(ctx)

    return before, after


def _locate_hunk(
    result_lines: list[str],
    before:       list[str],
    orig_start:   int,
) -> int:
    """
    Locate the position of `before` (context + removed lines, in that order)
    within `result_lines` by content, not by trusting the model-authored
    `@@ -orig_start` header as a literal position. `orig_start` is used only
    as a disambiguation hint when `before` matches at more than one position
    — most `before` blocks (a full sentence of real prose) are unique in a
    wiki page, so this is a rare path, but it needs a deterministic tiebreak
    rather than silently taking the first match in file order.

    A hunk with no context/removed lines (pure insertion — `before == []`)
    has no content to search for; falls back to the hinted position,
    clamped to the valid range, since there's nothing else to go on.

    Raises
    ------
    ValueError
        If `before` is non-empty and matches nowhere in `result_lines` — a
        genuine content mismatch (e.g. the page changed since the model
        read it), not a position problem.
    """
    if not before:
        return max(0, min(orig_start - 1, len(result_lines)))

    matches = [
        i for i in range(len(result_lines) - len(before) + 1)
        if result_lines[i : i + len(before)] == before
    ]

    if not matches:
        raise ValueError(
            "Diff hunk does not match file content — the following block "
            "was not found anywhere in the file.\n"
            f"Expected: {''.join(before)!r}"
        )

    if len(matches) == 1:
        return matches[0]

    # Ambiguous — before content appears at more than one position.
    # Deterministic tiebreak: prefer whichever match sits closest to the
    # model's stated (possibly wrong) line number, rather than silently
    # picking the first occurrence in file order.
    hinted = orig_start - 1
    return min(matches, key=lambda i: abs(i - hinted))


def _parse_unified_hunks(diff_text: str) -> list[tuple[int, list[str]]]:
    hunks:         list[tuple[int, list[str]]] = []
    current_start: int        = 0
    current_lines: list[str]  = []
    in_hunk = False

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("@@"):
            if in_hunk and current_lines:
                hunks.append((current_start, current_lines))
            m = re.match(r"^@@ -(\d+)", line)
            current_start = int(m.group(1)) if m else 1
            current_lines = []
            in_hunk = True
        elif in_hunk:
            current_lines.append(line)

    if in_hunk and current_lines:
        hunks.append((current_start, current_lines))

    return hunks


# ---------------------------------------------------------------------------
# Link validation
# ---------------------------------------------------------------------------

_LINK_SCAN_HEADERS = ["### Mapped Pages", "## Related Pages"]


def _validate_links(
    actions: Actions,
    wiki_pages: dict[str, str],
) -> dict[str, list[str]]:
    """
    Scan each new_pages[i].content for [[...]] occurrences within the
    Mapped Pages and Related Pages sections, and flag any whose target
    does not match an existing wiki page or a page proposed in this same
    response.

    Returns a dict keyed by page_name, each value a list of unresolved
    link targets found in that page's content. Pages with no unresolved
    links are omitted from the dict (empty dict overall if none found).

    Does not modify actions or any page content. Pure validation —
    flagging only.

    Normalization rule: link_text.lower().replace(" ", "-") — lowercase
    plus space-to-hyphen only. Resolves case-only mismatches (e.g.
    "Localist Master Project Outline" → "localist-master-project-outline")
    but intentionally does NOT resolve word-count mismatches (e.g.
    "Localist Software Stack Overview" ≠ "localist-software-stack").
    This rule must stay consistent with Phase B's graph builder (Prompt 5).
    """
    valid_targets = set(wiki_pages.keys()) | {p.page_name for p in actions.new_pages}
    unresolved: dict[str, list[str]] = {}

    for page in actions.new_pages:
        parsed = parse_wiki_doc(page.content)
        in_scope_links: list = []

        for header in _LINK_SCAN_HEADERS:
            idx = parsed.body.find(header)
            if idx == -1:
                continue
            line_end = parsed.body.find("\n", idx)
            if line_end == -1:
                continue  # header is last line, no content follows
            section_start = line_end + 1
            rest = parsed.body[section_start:]
            # Section ends at the next line beginning with '#' (any heading level)
            m = re.search(r"^#", rest, re.MULTILINE)
            section_end = section_start + m.start() if m else len(parsed.body)
            section_text = parsed.body[section_start:section_end]
            # Use parse_wiki_doc for link extraction — no second [[...]] regex
            in_scope_links.extend(parse_wiki_doc(section_text).links)

        page_unresolved = [
            link.link_text.lower().replace(" ", "-")
            for link in in_scope_links
            if link.link_text.lower().replace(" ", "-") not in valid_targets
        ]

        if page_unresolved:
            unresolved[page.page_name] = page_unresolved

    return unresolved


# ---------------------------------------------------------------------------
# Routing keywords used by can_handle()
# ---------------------------------------------------------------------------

_WIKI_KEYWORDS: frozenset[str] = frozenset({
    "ingest", "wiki", "research note", "research_note",
    "create page", "update page", "raw file", "document",
    "knowledge base", "summarise", "summarize", "extract concepts",
    "diff", "revise page", "modify page",
})


# ---------------------------------------------------------------------------
# WikiAgent
# ---------------------------------------------------------------------------

class WikiAgent:
    """
    Self-contained wiki ingestion agent satisfying the AgentInterface Protocol.

    Parameters
    ----------
    runtime :
        A RuntimeClient instance.  Used for all model inference calls.
    project_root :
        Fallback root for resolving wiki_dir, schema_path, and templates_dir
        when they are not supplied in SubTask.context.  Defaults to the
        directory containing this file.
    memory_manager :
        Optional SQLite-backed MemoryManager.  When supplied, every page
        written to disk is immediately passed to
        memory_manager.index_document() so the document index stays current
        without a full corpus re-scan.  When absent, disk writes are
        unchanged — the index is simply not updated automatically.
    """

    def __init__(
        self,
        runtime:        Any,
        project_root:   Path | None = None,
        memory_manager: "MemoryManager | None" = None,
    ) -> None:
        self._runtime        = runtime
        # backend/src/localist/wiki_agent.py -> backend/
        self._project_root   = project_root or Path(__file__).resolve().parent.parent.parent
        self._memory_manager = memory_manager

    # -----------------------------------------------------------------------
    # AgentInterface — name
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "wiki_agent"

    # -----------------------------------------------------------------------
    # AgentInterface — can_handle
    # -----------------------------------------------------------------------

    def can_handle(self, instruction: str) -> bool:
        lowered = instruction.lower()
        return any(kw in lowered for kw in _WIKI_KEYWORDS)

    # -----------------------------------------------------------------------
    # AgentInterface — run
    # -----------------------------------------------------------------------

    def run(self, subtask: SubTask) -> AgentResult:
        """
        Ingest one raw file, or propose a diff against one existing wiki
        page, and return proposed (or applied) wiki actions.

        Dispatch
        --------
        - "raw_path" in context             → ingest path (below).
        - "diff_target" in context, no
          "raw_path"                        → _run_diff_only().
        - both present                      → raw_path wins; diff_target
                                               is ignored (logged at debug).
        - neither present                   → fails, same as always.

        Ingest pipeline
        ----------------
        1. Resolve and validate paths from subtask.context.
        2. Load schema, templates, wiki index, and raw document.
        3. Build wiki_context (always — informs model regardless of path).
        4. Detect runtime capability and call model:
             - infer_with_file() if runtime supports it (oMLX 0.4.2+)
             - infer() with full string prompt otherwise (Foundry + others)
        5. Parse the model's XML response into Actions.
        6. Journal / apply / index / return — see _finalize().

        No stdin.  No sys.exit().  No interactive prompts.
        """
        ctx = subtask.context

        if "raw_path" not in ctx and ctx.get("diff_target") is not None:
            return self._run_diff_only(subtask)

        if "raw_path" in ctx and ctx.get("diff_target") is not None:
            logger.debug(
                "[%s] Both raw_path and diff_target present in context; "
                "raw_path wins — diff_target=%r ignored.",
                self.name, ctx["diff_target"],
            )

        # -- 1. Resolve paths ------------------------------------------------

        try:
            raw_path = self._resolve_raw_path(ctx)
        except (KeyError, ValueError) as exc:
            return self._fail(subtask, f"Context error: {exc}")

        wiki_dir      = Path(ctx.get("wiki_dir",      self._project_root / "wiki"))
        schema_path   = Path(ctx.get("schema_path",   self._project_root / "SCHEMA.md"))
        templates_dir = Path(ctx.get("templates_dir", self._project_root / "templates"))
        journal_path  = Path(ctx["journal_path"]) if "journal_path" in ctx else None
        auto_apply    = bool(ctx.get("auto_apply",  False))
        max_tokens    = int(ctx.get("max_tokens",   2048))
        temperature   = float(ctx.get("temperature", 0.2))

        # -- 2. Validate required paths --------------------------------------

        missing = [
            label for label, p in [
                ("SCHEMA.md",  schema_path),
                ("templates/", templates_dir),
                ("wiki/",      wiki_dir),
                ("raw_path",   raw_path),
            ]
            if not p.exists()
        ]
        if missing:
            return self._fail(subtask, f"Required paths not found: {missing}")

        logger.info(
            "[%s] Ingesting '%s' — auto_apply=%s",
            self.name, raw_path.name, auto_apply,
        )

        # -- 3. Load inputs --------------------------------------------------
        #
        # wiki_pages dict is built either from the MemoryManager index (fast,
        # no filesystem walk) or from disk (fallback when no manager is present).
        # raw_content is always loaded from disk — needed for build_wiki_context()
        # keyword scoring and for the string-prompt infer() path.
        #
        # OCR-eligible raw files (image/PDF — mcp_server.ocr.OCR_MIME_BY_
        # EXTENSION) are extracted to plain text via the same ocr_extract MCP
        # tool chat uploads use, rather than read as UTF-8 text — this is
        # what makes ingestion backend-agnostic instead of depending on
        # oMLX's infer_with_file()/MarkItDown path (§22.8 of the
        # architecture spec). Done ahead of the try/except below so an OCR
        # failure gets its own clear error message rather than "File load
        # error".

        ocr_mime_type = OCR_MIME_BY_EXTENSION.get(raw_path.suffix.lower())
        if ocr_mime_type is not None:
            try:
                raw_content = extract_raw_content_via_ocr(self._runtime, raw_path, ocr_mime_type)
            except Exception as exc:
                return self._fail(subtask, f"OCR extraction error: {exc}")

        try:
            schema_text = read_text_file(schema_path)
            templates   = self._load_templates(templates_dir)
            if ocr_mime_type is None:
                raw_content = read_text_file(raw_path)

            if self._memory_manager is not None:
                wiki_pages = self._load_wiki_pages_from_index(wiki_dir)
                logger.debug(
                    "[%s] wiki_pages loaded from MemoryManager index (%d pages).",
                    self.name, len(wiki_pages),
                )
            else:
                wiki_pages = self._load_wiki_pages(wiki_dir)
                logger.debug(
                    "[%s] wiki_pages loaded from filesystem (%d pages).",
                    self.name, len(wiki_pages),
                )
        except Exception as exc:
            return self._fail(subtask, f"File load error: {exc}")

        # -- 4. Build prompt -------------------------------------------------

        wiki_context = build_wiki_context(wiki_pages, raw_content)

        # -- 5. Call model via RuntimeClient ---------------------------------
        #
        # Never true for an OCR-extracted raw file, even on oMLX — raw_content
        # is already plain text at this point, so it takes the infer() string
        # path below exactly like a .md/.txt raw file (§22.8).

        use_file_upload = ocr_mime_type is None and hasattr(self._runtime, "infer_with_file")
        logger.debug(
            "[%s] inference path: %s",
            self.name,
            "infer_with_file (oMLX native)" if use_file_upload else "infer (string prompt)",
        )

        try:
            if use_file_upload:
                slim_prompt = build_slim_prompt(
                    schema_text  = schema_text,
                    templates    = templates,
                    wiki_context = wiki_context,
                    raw_filename = raw_path.name,
                )
                logger.debug(
                    "[%s] slim_prompt_chars=%d  file=%s",
                    self.name, len(slim_prompt), raw_path.name,
                )
                raw_output = self._runtime.infer_with_file(
                    file_path   = raw_path,
                    prompt      = slim_prompt,
                    system      = SYSTEM_PROMPT,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                )
            else:
                try:
                    raw_user_prompt = build_user_prompt(
                        schema_text  = schema_text,
                        templates    = templates,
                        wiki_context = wiki_context,
                        raw_filename = raw_path.name,
                        raw_content  = raw_content,
                    )
                    _sys, user_prompt = _PROMPT_BUILDER.build(
                        instruction      = raw_user_prompt,
                        current_datetime = datetime.now().astimezone(),
                    )
                    # _sys (PromptBuilder's generic identity prompt) is discarded —
                    # WikiAgent must use SYSTEM_PROMPT (XML-only directive) here.
                except Exception as exc:
                    return self._fail(subtask, f"Prompt build error: {exc}")
                logger.debug(
                    "[%s] user_prompt_chars=%d",
                    self.name, len(user_prompt),
                )
                raw_output = self._runtime.infer(
                    system      = SYSTEM_PROMPT,
                    prompt      = user_prompt,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                )
        except (RuntimeError, ValueError) as exc:
            return self._fail(subtask, f"Inference error: {exc}")

        # -- 6. Parse XML response -------------------------------------------

        try:
            actions = parse_model_xml(raw_output)
        except ValueError as exc:
            return self._fail(subtask, f"XML parse error: {exc}")

        return self._finalize(
            subtask      = subtask,
            actions      = actions,
            wiki_pages   = wiki_pages,
            journal_path = journal_path,
            auto_apply   = auto_apply,
            wiki_dir     = wiki_dir,
            raw_filename = raw_path.name,
            diff_target  = None,
        )

    # -----------------------------------------------------------------------
    # Diff-only path — apply a diff to an existing page, no raw file
    # -----------------------------------------------------------------------

    def _run_diff_only(self, subtask: SubTask) -> AgentResult:
        """
        Propose (and optionally apply) a diff against one existing wiki page,
        driven by subtask.instruction rather than a raw file. See the
        "diff_target" entry in the module docstring's SubTask.context schema.

        Pipeline mirrors run()'s ingest path minus everything that only makes
        sense for a raw file: no raw_content load, no build_wiki_context()
        (the target page's full content is already the entire context), and
        only apply_diff is a legal model action (create_page actions are
        discarded with a warning if the model emits one anyway).
        """
        ctx = subtask.context
        diff_target = ctx["diff_target"]

        wiki_dir      = Path(ctx.get("wiki_dir",      self._project_root / "wiki"))
        schema_path   = Path(ctx.get("schema_path",   self._project_root / "SCHEMA.md"))
        templates_dir = Path(ctx.get("templates_dir", self._project_root / "templates"))
        journal_path  = Path(ctx["journal_path"]) if "journal_path" in ctx else None
        auto_apply    = bool(ctx.get("auto_apply",  False))
        max_tokens    = int(ctx.get("max_tokens",   2048))
        temperature   = float(ctx.get("temperature", 0.2))
        tool_context  = ctx.get("tool_context")

        missing = [
            label for label, p in [
                ("SCHEMA.md",  schema_path),
                ("templates/", templates_dir),
                ("wiki/",      wiki_dir),
            ]
            if not p.exists()
        ]
        if missing:
            return self._fail(subtask, f"Required paths not found: {missing}")

        logger.info(
            "[%s] Diff-only run — target=%r auto_apply=%s",
            self.name, diff_target, auto_apply,
        )

        try:
            schema_text = read_text_file(schema_path)
            templates   = self._load_templates(templates_dir)

            if self._memory_manager is not None:
                wiki_pages = self._load_wiki_pages_from_index(wiki_dir)
            else:
                wiki_pages = self._load_wiki_pages(wiki_dir)
        except Exception as exc:
            return self._fail(subtask, f"File load error: {exc}")

        if diff_target not in wiki_pages:
            return self._fail(subtask, f"diff_target page not found: {diff_target}")

        diff_prompt = build_diff_prompt(
            schema_text  = schema_text,
            templates    = templates,
            page_name    = diff_target,
            page_content = wiki_pages[diff_target],
            instruction  = subtask.instruction,
            tool_context = tool_context,
        )
        logger.debug(
            "[%s] diff_prompt_chars=%d  target=%s",
            self.name, len(diff_prompt), diff_target,
        )

        try:
            raw_output = self._runtime.infer(
                system      = DIFF_SYSTEM_PROMPT,
                prompt      = diff_prompt,
                max_tokens  = max_tokens,
                temperature = temperature,
            )
        except (RuntimeError, ValueError) as exc:
            return self._fail(subtask, f"Inference error: {exc}")

        try:
            actions = parse_model_xml(raw_output)
        except ValueError as exc:
            return self._fail(subtask, f"XML parse error: {exc}")

        if actions.new_pages:
            logger.warning(
                "[%s] Diff-only run for target=%r received %d create_page "
                "action(s); discarding — only apply_diff is valid on this "
                "path.",
                self.name, diff_target, len(actions.new_pages),
            )
            actions = Actions(new_pages=[], diffs=actions.diffs)

        return self._finalize(
            subtask      = subtask,
            actions      = actions,
            wiki_pages   = wiki_pages,
            journal_path = journal_path,
            auto_apply   = auto_apply,
            wiki_dir     = wiki_dir,
            raw_filename = None,
            diff_target  = diff_target,
        )

    # -----------------------------------------------------------------------
    # Shared post-inference tail — journal / apply / index / build result
    # -----------------------------------------------------------------------

    def _finalize(
        self,
        subtask:      SubTask,
        actions:      Actions,
        wiki_pages:   dict[str, str],
        journal_path: Path | None,
        auto_apply:   bool,
        wiki_dir:     Path,
        raw_filename: str | None,
        diff_target:  str | None,
    ) -> AgentResult:
        """
        Shared tail for both run() (ingest) and _run_diff_only(): validate
        links, journal, optionally apply to disk + reindex + rebuild the
        graph, and build the AgentResult. Exactly one of raw_filename/
        diff_target is expected to be non-None (the other stays None to
        record which path produced this result — see the module docstring's
        AgentResult.output schema).
        """
        # -- Validate [[...]] link targets ------------------------------------

        unresolved_links = _validate_links(actions, wiki_pages)
        if unresolved_links:
            for page_name, targets in unresolved_links.items():
                for target in targets:
                    logger.warning(
                        "[%s] Unresolved link in '%s': [[%s]]",
                        self.name, page_name, target,
                    )

        # -- Optionally journal the result ------------------------------------

        if journal_path is not None:
            self._write_journal(actions, journal_path)

        # -- Optionally apply changes to disk ----------------------------------

        applied    = False
        skipped:   list[str] = []
        written:   list[str] = []
        diff_errs: list[str] = []

        if auto_apply:
            applied, written, skipped, diff_errs = self._apply_changes(
                actions, wiki_dir
            )
            self._reindex_and_rebuild_graph(wiki_dir, written)

            if written:
                created_names = {p.page_name for p in actions.new_pages}
                log_entries = [
                    ("Creation" if name in created_names else "Update", name)
                    for name in written
                ]
                self._append_logs_md(log_entries, wiki_dir)
                self._regenerate_index_md(self._load_wiki_pages(wiki_dir), wiki_dir)

        # -- Build and return AgentResult --------------------------------------

        output: dict[str, Any] = {
            "new_pages":        [p.model_dump() for p in actions.new_pages],
            "diffs":            [d.model_dump() for d in actions.diffs],
            "applied":          applied,
            "raw_filename":     raw_filename,
            "diff_target":      diff_target,
            "wiki_page_count":  len(wiki_pages),
            "unresolved_links": unresolved_links,
        }

        if auto_apply:
            output["written"]     = written
            output["skipped"]     = skipped
            output["diff_errors"] = diff_errs

        logger.info(
            "[%s] Complete — new_pages=%d  diffs=%d  applied=%s",
            self.name,
            len(actions.new_pages),
            len(actions.diffs),
            applied,
        )

        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = self.name,
            status     = TaskStatus.COMPLETE,
            output     = output,
        )

    def _reindex_and_rebuild_graph(
        self,
        wiki_dir:      Path,
        written_pages: list[str],
    ) -> None:
        """
        Shared post-write tail: reindex each written page in MemoryManager
        (if present) and rebuild the concept graph. Used by both
        _finalize()'s auto_apply path and apply_pending_diff() — the
        review-then-apply UI's single-diff write has the same disk-state
        implications as an ingest-time auto_apply write and must keep the
        index/graph equally current. No-op if there's no MemoryManager or
        nothing was written.
        """
        if self._memory_manager is None or not written_pages:
            return

        for page_name in written_pages:
            page_path = wiki_dir / f"{page_name}.md"
            if page_path.exists():
                try:
                    self._memory_manager.index_document(
                        path     = page_path,
                        doc_type = "wiki",
                        embed    = True,
                    )
                    logger.debug(
                        "[%s] Indexed '%s' in MemoryManager.", self.name, page_name
                    )
                except Exception as exc:
                    # Non-fatal — disk write succeeded; index can be
                    # rebuilt by calling index_directory() later.
                    logger.warning(
                        "[%s] MemoryManager.index_document failed for '%s': %s",
                        self.name, page_name, exc,
                    )

        try:
            graph_summary = build_graph(wiki_dir, self._memory_manager)
            logger.info(
                "[%s] Graph rebuilt after write — nodes=%d edges=%d resolved=%d unresolved=%d",
                self.name, graph_summary["nodes"], graph_summary["edges"],
                graph_summary["resolved"], graph_summary["unresolved"],
            )
        except Exception as exc:
            logger.warning("[%s] Graph rebuild failed after write (non-fatal): %s", self.name, exc)

    @staticmethod
    def _regenerate_index_md(wiki_pages: dict[str, str], wiki_dir: Path) -> None:
        """
        Regenerate wiki_dir/index.md from the current on-disk page set
        (OKF's per-directory summary file), grouped by front-matter `type`.

        Deterministic — never model-authored — so it can never drift from
        what's actually on disk or hallucinate a link. A page missing
        `type` is grouped under "UNSPECIFIED"; missing `title`/
        `description` fall back to the page's stem / no description line,
        so a page that hasn't been through the OKF backfill never breaks
        generation. wiki_pages is expected to already exclude
        META_WIKI_FILENAMES (see _load_wiki_pages()).
        """
        groups: dict[str, list[tuple[str, str, str | None]]] = {}
        for stem, content in sorted(wiki_pages.items()):
            frontmatter = parse_wiki_doc(content).frontmatter
            page_type   = str(frontmatter.get("type") or "UNSPECIFIED")
            title       = str(frontmatter.get("title") or stem)
            description = frontmatter.get("description")
            groups.setdefault(page_type, []).append((stem, title, description))

        lines = ["# Wiki Index", ""]
        for page_type in sorted(groups):
            lines.append(f"## {page_type}")
            lines.append("")
            for stem, title, description in groups[page_type]:
                if description:
                    lines.append(f"- [[{stem}]] — {title}: {description}")
                else:
                    lines.append(f"- [[{stem}]] — {title}")
            lines.append("")

        write_text_file(wiki_dir / "index.md", "\n".join(lines).rstrip() + "\n")

    @staticmethod
    def _append_logs_md(entries: list[tuple[str, str]], wiki_dir: Path) -> None:
        """
        Append a dated changelog section to wiki_dir/logs.md (OKF's
        changelog file) — one bullet per (kind, page_name) entry, kind one
        of "Creation"/"Update", linking [[page_name]]. Deterministic —
        computed from what _apply_changes()/apply_pending_diff() actually
        wrote, never model-authored. Creates the file with a header on
        first write; every call after that appends a new dated section
        rather than trying to merge into an existing same-day section, to
        keep this a pure, simple append with no re-parsing of logs.md's
        own structure.

        No-op if entries is empty (nothing was actually written).
        """
        if not entries:
            return

        today = date.today().isoformat()
        lines = [f"## {today}", ""]
        for kind, page_name in entries:
            lines.append(f"- {kind}: [[{page_name}]]")
        block = "\n".join(lines) + "\n"

        logs_path = wiki_dir / "logs.md"
        if not logs_path.exists():
            write_text_file(logs_path, "# Wiki Changelog\n\n" + block)
        else:
            with open(logs_path, "a", encoding="utf-8") as fh:
                fh.write("\n" + block)

    # -----------------------------------------------------------------------
    # Review-then-apply UI — apply a single previously-proposed diff
    # -----------------------------------------------------------------------

    def apply_pending_diff(
        self,
        page_name: str,
        diff:      str,
        wiki_dir:  Path,
    ) -> AgentResult:
        """
        Apply one previously-proposed diff directly to an existing wiki
        page — the review-then-apply UI's Apply action (see
        scope-review-then-apply-diff-ui.md). No fresh model call, no
        SubTask/Actions wrapper: the diff text round-trips back from the
        client exactly as WikiAgent originally proposed it.

        Content-based matching inside apply_unified_diff() (see its
        2026-07-09 bug fixes) is the staleness check: if the page changed
        on disk since the diff was proposed — someone hand-edited it, or a
        second diff landed first — the match legitimately fails and this
        returns a FAILED AgentResult rather than corrupting the page.
        output["error_kind"] distinguishes "not_found" (page doesn't exist)
        from "stale" (content mismatch) so callers (main.py's
        POST /wiki/apply-diff) can map to the right HTTP status without
        string-matching the error message.
        """
        subtask_id = f"apply-diff:{page_name}"
        page_path = wiki_dir / f"{page_name}.md"

        if not page_path.exists():
            return AgentResult(
                subtask_id = subtask_id,
                agent_name = self.name,
                status     = TaskStatus.FAILED,
                output     = {"error_kind": "not_found"},
                error      = f"Page not found: {page_name}",
            )

        self._snapshot_page(page_name, wiki_dir)

        original = read_text_file(page_path)
        try:
            updated = apply_unified_diff(original, diff)
        except ValueError as exc:
            return AgentResult(
                subtask_id = subtask_id,
                agent_name = self.name,
                status     = TaskStatus.FAILED,
                output     = {"error_kind": "stale"},
                error      = (
                    "Diff no longer applies cleanly — the page may have "
                    f"changed since it was proposed: {exc}"
                ),
            )

        write_text_file(page_path, updated)
        logger.info("[%s] apply_pending_diff: wrote '%s'.", self.name, page_name)

        self._reindex_and_rebuild_graph(wiki_dir, [page_name])
        self._append_logs_md([("Update", page_name)], wiki_dir)
        self._regenerate_index_md(self._load_wiki_pages(wiki_dir), wiki_dir)

        return AgentResult(
            subtask_id = subtask_id,
            agent_name = self.name,
            status     = TaskStatus.COMPLETE,
            output     = {"page_name": page_name, "written": True},
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_raw_path(ctx: dict[str, Any]) -> Path:
        if "raw_path" not in ctx:
            raise KeyError(
                "'raw_path' is required in SubTask.context. "
                "Provide the absolute path to the file to ingest."
            )
        path = Path(ctx["raw_path"])
        if not path.exists():
            raise ValueError(f"raw_path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"raw_path is not a file: {path}")
        suffix = path.suffix.lower()
        if suffix in OCR_MIME_BY_EXTENSION:
            # OCR-eligible (image/PDF) — extracted to text later in run();
            # not UTF-8 text on disk, so is_text_file() below doesn't apply.
            return path
        if suffix not in {".md", ".txt"}:
            raise ValueError(
                f"raw_path must be a .md, .txt, or OCR-eligible "
                f"(image/PDF) file, got: {path.suffix}"
            )
        if not is_text_file(path):
            raise ValueError(
                f"raw_path does not appear to be a UTF-8 text file: {path}"
            )
        return path

    @staticmethod
    def _load_templates(templates_dir: Path) -> dict[str, str]:
        templates: dict[str, str] = {}
        for name in ["system", "concept", "research-note"]:
            p = templates_dir / f"{name}.md"
            if p.exists():
                key = name.upper().replace("-", "_")
                templates[key] = read_text_file(p)
        return templates

    @staticmethod
    def _load_wiki_pages(wiki_dir: Path) -> dict[str, str]:
        """Load all .md files from the wiki directory (stem → content).

        Excludes META_WIKI_FILENAMES (index.md, logs.md, MEMORY.md) — these
        are structural/generated files, never content pages the model
        should see as "EXISTING WIKI PAGES" or resolve a diff_target
        against.
        """
        if not wiki_dir.exists():
            return {}
        return {
            p.stem: read_text_file(p)
            for p in sorted(wiki_dir.iterdir())
            if p.is_file() and p.suffix == ".md" and p.name not in META_WIKI_FILENAMES
        }

    def _load_wiki_pages_from_index(self, wiki_dir: Path) -> dict[str, str]:
        """
        Load wiki page content from the MemoryManager index.

        Falls back to the filesystem walk if the index returns no results
        for this wiki_dir (e.g. on first run before any pages are indexed).
        Filters by the absolute wiki_dir so pages from other projects don't
        bleed in when a single MemoryManager is shared.
        """
        assert self._memory_manager is not None  # guarded by caller

        docs = self._memory_manager.get_all_documents(doc_type="wiki")

        # Filter to pages that live under this wiki_dir, excluding
        # META_WIKI_FILENAMES (index.md, logs.md, MEMORY.md) — same
        # exclusion _load_wiki_pages() applies to the filesystem-walk path.
        wiki_dir_str = str(wiki_dir.resolve())
        filtered = {
            d.name: d.content
            for d in docs
            if str(d.path).startswith(wiki_dir_str)
            and Path(d.path).name not in META_WIKI_FILENAMES
        }

        if not filtered and wiki_dir.exists():
            logger.debug(
                "[%s] MemoryManager index empty for wiki_dir=%s — falling back to filesystem.",
                self.name, wiki_dir,
            )
            return self._load_wiki_pages(wiki_dir)

        return filtered

    @staticmethod
    def _snapshot_page(page_name: str, wiki_dir: Path) -> None:
        """
        Copy the current on-disk content of an existing page to
        wiki_dir/.snapshots/ before it gets overwritten. Non-fatal — a
        snapshot failure must never block the real write, matching the
        pattern used for journal writes (see _write_journal()).

        Naming: {page_name}.v{N}.{YYYYMMDDTHHMMSS}.md, UTC timestamp,
        N = count of existing snapshots for this page_name + 1.
        """
        try:
            content = read_text_file(wiki_dir / f"{page_name}.md")
            snapshots_dir = wiki_dir / _SNAPSHOT_DIR_NAME
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            n = len(list(snapshots_dir.glob(f"{page_name}.v*.md"))) + 1
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            snapshot_path = snapshots_dir / f"{page_name}.v{n}.{timestamp}.md"
            write_text_file(snapshot_path, content)
        except Exception as exc:
            logger.warning("[wiki_agent] Snapshot failed for '%s': %s", page_name, exc)
            return

        logger.info("[wiki_agent] Snapshotted '%s' -> %s", page_name, snapshot_path)

        try:
            WikiAgent._prune_page_snapshots(page_name, wiki_dir)
        except Exception as exc:
            logger.warning("[wiki_agent] Snapshot prune failed for '%s': %s", page_name, exc)

    @staticmethod
    def _prune_page_snapshots(page_name: str, wiki_dir: Path) -> None:
        """Remove this page's own snapshots older than the effective TTL
        (see _snapshot_ttl_seconds()). Runs after every successful
        _snapshot_page() call; complements the startup-wide
        sweep_expired_snapshots() for pages that get edited again before
        the TTL elapses. Writes the same wiki_maintenance_log audit entry
        as the startup sweep so both prune paths share one audit trail."""
        snapshots_dir = wiki_dir / _SNAPSHOT_DIR_NAME
        cutoff = time.time() - _snapshot_ttl_seconds()
        for p in snapshots_dir.glob(f"{page_name}.v*.md"):
            if p.stat().st_mtime < cutoff:
                p.unlink()
                wiki_maintenance_log.log_snapshot_pruned(p.name, str(p))

    @staticmethod
    def _apply_changes(
        actions:  Actions,
        wiki_dir: Path,
    ) -> tuple[bool, list[str], list[str], list[str]]:
        written:     list[str] = []
        skipped:     list[str] = []
        diff_errors: list[str] = []

        for entry in actions.new_pages:
            path = wiki_dir / f"{entry.page_name}.md"
            if path.exists():
                logger.warning("[wiki_agent] Skipping existing page: %s", entry.page_name)
                skipped.append(entry.page_name)
                continue
            write_text_file(path, entry.content)
            logger.info("[wiki_agent] Created: %s", path)
            written.append(entry.page_name)

        for entry in actions.diffs:
            path = wiki_dir / f"{entry.page_name}.md"
            if not path.exists():
                logger.warning("[wiki_agent] Diff target missing: %s", entry.page_name)
                diff_errors.append(entry.page_name)
                continue
            WikiAgent._snapshot_page(entry.page_name, wiki_dir)
            original = read_text_file(path)
            try:
                updated = apply_unified_diff(original, entry.diff)
                write_text_file(path, updated)
                logger.info("[wiki_agent] Patched: %s", path)
                written.append(entry.page_name)
            except Exception as exc:
                logger.error("[wiki_agent] Diff failed for %s: %s", entry.page_name, exc)
                diff_errors.append(entry.page_name)

        applied = bool(written)
        return applied, written, skipped, diff_errors

    @staticmethod
    def _write_journal(actions: Actions, journal_path: Path) -> None:
        try:
            entry = JournalEntry(step="wiki_agent_run", actions=actions)
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(journal_path, "a", encoding="utf-8") as fh:
                fh.write(entry.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("[wiki_agent] Journal write failed: %s", exc)

    @staticmethod
    def _fail(subtask: SubTask, reason: str) -> AgentResult:
        logger.error("[wiki_agent] subtask %s failed: %s", subtask.subtask_id, reason)
        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = "wiki_agent",
            status     = TaskStatus.FAILED,
            output     = {},
            error      = reason,
        )


# ---------------------------------------------------------------------------
# Protocol conformance check
# ---------------------------------------------------------------------------

def _assert_protocol_conformance() -> None:
    from .controller_agent import AgentInterface

    class _MockRuntime:
        def infer(self, *a, **kw) -> str:  return ""
        def embed(self, text: str) -> list: return []

    agent = WikiAgent(runtime=_MockRuntime())
    assert isinstance(agent, AgentInterface), (
        "WikiAgent does not satisfy the AgentInterface Protocol."
    )
    logger.debug("Protocol conformance check passed.")


# ---------------------------------------------------------------------------
# Wiring example — for reference, not executed in production
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    import uuid
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, stream=sys.stdout)

    from .foundry_runtime_client import FoundryRuntimeClient
    from .controller_agent import ControllerAgent

    runtime    = FoundryRuntimeClient()
    wiki_agent = WikiAgent(runtime=runtime)
    controller = ControllerAgent(runtime=runtime, agents=[wiki_agent])

    result = controller.handle_task({
        "task_id":     str(uuid.uuid4()),
        "instruction": "Ingest the raw file and create a wiki research note.",
        "context": {
            "raw_path":    "/absolute/path/to/your/raw/file.md",
            "wiki_dir":    "/absolute/path/to/your/wiki",
            "schema_path": "/absolute/path/to/your/SCHEMA.md",
            "auto_apply":  False,
        },
    })

    print(json.dumps(result, indent=2))