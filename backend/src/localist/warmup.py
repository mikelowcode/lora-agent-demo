"""
§3.7c Lever 3 — Startup cache warm-up.

Parses warmup_fixture.md and issues a single best-effort runtime.infer()
call at startup to promote block 0 of the Localist-shaped prompt prefix
to a cache-resident tier. PromptBuilder.build() assembles an identical
slot layout to production traffic so the prefill matches real requests.

This module has no FastAPI or pydantic dependencies and is safe to import
in test environments that lack those packages.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .prompt_builder import PromptBuilder, ToolResult, Turn

if TYPE_CHECKING:
    from .base_runtime_client import BaseRuntimeClient

logger = logging.getLogger(__name__)

_WARMUP_BUILDER = PromptBuilder()

# Matches metadata lines of the form "key: value" where key is a lowercase
# identifier (no spaces). Prevents content lines that happen to contain ": "
# from being misread as metadata.
_META_RE = re.compile(r"^([a-z][a-z_]*): (.*)$")


def parse_warmup_fixture(path: Path) -> tuple[list[ToolResult], list[Turn]]:
    """
    Parse warmup_fixture.md into (tool_results, turns).

    File format — two top-level sections separated by '## TOOL RESULTS' and
    '## WORKING MEMORY' headers. Within each section, entries are separated
    by lines containing only '---'. The first lines of each entry are
    'key: value' metadata (lowercase key, no spaces); a blank line separates
    metadata from body text.

    Raises ValueError when required headers are missing.
    Raises OSError when the file cannot be read.
    Callers should catch both; run_cache_warmup() does so.
    """
    text = path.read_text("utf-8")

    TOOL_HDR = "## TOOL RESULTS"
    WM_HDR   = "## WORKING MEMORY"
    if TOOL_HDR not in text or WM_HDR not in text:
        raise ValueError(
            f"warmup_fixture.md missing required section header(s): "
            f"'{TOOL_HDR}' and/or '{WM_HDR}'"
        )

    _, after_tool    = text.split(TOOL_HDR, 1)
    tool_section, wm_section = after_tool.split(WM_HDR, 1)

    def _parse_entry(raw: str) -> tuple[dict[str, str], str]:
        lines = raw.strip().split("\n")
        meta: dict[str, str] = {}
        i = 0
        while i < len(lines):
            line = lines[i]
            m = _META_RE.match(line)
            if m:
                meta[m.group(1)] = m.group(2).strip()
                i += 1
            elif line.strip() == "":
                i += 1
                break
            else:
                break
        body = "\n".join(lines[i:]).strip()
        return meta, body

    tool_results: list[ToolResult] = []
    for raw in re.split(r"\n---\n", tool_section):
        if not raw.strip():
            continue
        meta, body = _parse_entry(raw)
        if "tool" not in meta or not body:
            continue
        tool_results.append(ToolResult(
            tool_name  = meta["tool"],
            parameters = meta.get("params", ""),
            result     = body,
        ))

    turns: list[Turn] = []
    for raw in re.split(r"\n---\n", wm_section):
        if not raw.strip():
            continue
        meta, body = _parse_entry(raw)
        if "role" not in meta or not body:
            continue
        turns.append(Turn(role=meta["role"], content=body))

    return tool_results, turns


def run_cache_warmup(
    controller:    Any,
    runtime:       "BaseRuntimeClient",
    templates_dir: Path,
) -> None:
    """
    One-shot startup cache warm-up (§3.7c Lever 3).

    Issues a single runtime.infer() call using a prompt built from
    warmup_fixture.md via PromptBuilder.build(). The KV-prefill promotes
    block 0 to a cache-resident tier (disk or hot). Any failure is logged
    as a warning; startup continues regardless. Never raises.
    """
    fixture_path = templates_dir / "warmup_fixture.md"

    # Step 1 — load and parse the fixture.
    try:
        tool_results, working_memory = parse_warmup_fixture(fixture_path)
    except Exception as exc:
        logger.warning(
            "Cache warm-up skipped: fixture load/parse failed (%s: %s).",
            type(exc).__name__, exc,
        )
        return

    if not tool_results and not working_memory:
        logger.warning(
            "Cache warm-up skipped: fixture parsed to empty content — "
            "check warmup_fixture.md."
        )
        return

    # Step 2 — build the prompt. Persona is fetched via the controller's
    # existing lazy-load path; falls back to None on any failure. Same for
    # assistant_name — resolved via the controller's own helper so the warm
    # cache primes the identity string that will actually be used on the
    # first real request (may be stale for one warmup cycle after a live
    # name change; acceptable, see PUT /settings/assistant-name).
    try:
        persona = controller._load_persona()
        assistant_name = controller._current_assistant_name()
        system_prompt, user_prompt = _WARMUP_BUILDER.build(
            instruction      = "Summarise the key themes from the tool results above.",
            current_datetime = datetime.now().astimezone(),
            persona          = persona,
            assistant_name   = assistant_name,
            tool_results     = tool_results or None,
            working_memory   = working_memory or None,
        )
    except Exception as exc:
        logger.warning(
            "Cache warm-up skipped: prompt assembly failed (%s: %s).",
            type(exc).__name__, exc,
        )
        return

    # Step 3 — one inference call; completion content is irrelevant.
    try:
        t0 = time.perf_counter()
        runtime.infer(
            prompt     = user_prompt,
            system     = system_prompt,
            max_tokens = 16,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Cache warm-up complete — block 0 promoted to cache-resident tier "
            "(%.0f ms).", elapsed_ms,
        )
    except Exception as exc:
        logger.warning(
            "Cache warm-up call failed (%s: %s) — startup continues normally.",
            type(exc).__name__, exc,
        )
