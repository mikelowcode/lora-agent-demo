"""
wiki_doc — Parse wiki markdown files into frontmatter, body, and link references.

Pure parsing module: no I/O beyond load_wiki_doc(), no SQLite, no embeddings,
no knowledge of any other project module.
"""

from __future__ import annotations

import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WikiLink:
    link_text: str   # literal bracket content as written, e.g. "Localist Master Project Outline"
    target_path: str # same as link_text today; Phase B will normalize this independently
                     # without changing link_text, so it lives as a distinct field to make
                     # that a one-line addition rather than a schema change.


@dataclass(frozen=True)
class ParsedWikiDoc:
    frontmatter: dict[str, Any]
    body: str
    links: list[WikiLink]


_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Structural/generated files that live in wiki_dir alongside real content
# pages but must never be treated as one: not indexed into the RAG corpus,
# not a graph node / resolve_graph_target() candidate stem, not shown to
# the model as "EXISTING WIKI PAGES" context, not offered as a pinnable
# page in the UI. index.md/logs.md are themselves generated from this
# exclusion set (see WikiAgent._regenerate_index_md()/_append_logs_md());
# MEMORY.md is EpisodicMemoryWriter's rendered output (memory_manager.py).
META_WIKI_FILENAMES: frozenset[str] = frozenset({"index.md", "logs.md", "MEMORY.md"})


def parse_wiki_doc(content: str) -> ParsedWikiDoc:
    """Parse already-loaded markdown text. Pure function, no I/O."""
    lines = content.splitlines(keepends=True)

    frontmatter: dict[str, Any] = {}
    body = content

    # Detect the opening fence at line 0 (standard) or line 1 (one leading
    # blank line — common in LLM-generated files).  At most one blank line
    # is tolerated; unbounded stripping would mask deeper structural problems.
    if lines and lines[0].rstrip("\r\n") == "---":
        fence_idx = 0
    elif (
        len(lines) >= 2
        and lines[0].rstrip("\r\n") == ""
        and lines[1].rstrip("\r\n") == "---"
    ):
        fence_idx = 1
    else:
        fence_idx = None

    if fence_idx is not None:
        # Scan for closing fence
        close_idx = None
        for i, line in enumerate(lines[fence_idx + 1 :], start=fence_idx + 1):
            if line.rstrip("\r\n") == "---":
                close_idx = i
                break

        if close_idx is not None:
            yaml_block = "".join(lines[fence_idx + 1 : close_idx])
            frontmatter = yaml.safe_load(yaml_block) or {}

            # Body is everything after the closing fence line
            remainder = "".join(lines[close_idx + 1 :])
            # Strip at most one leading blank line
            if remainder.startswith("\n"):
                remainder = remainder[1:]
            elif remainder.startswith("\r\n"):
                remainder = remainder[2:]
            body = remainder
        # else: malformed (no closing fence) — frontmatter stays {}, body stays content

    links = [
        WikiLink(link_text=m.group(1).strip(), target_path=m.group(1).strip())
        for m in _LINK_RE.finditer(body)
    ]

    return ParsedWikiDoc(frontmatter=frontmatter, body=body, links=links)


def load_wiki_doc(path: Path) -> ParsedWikiDoc:
    """Read `path` from disk and parse it. Thin wrapper around parse_wiki_doc()."""
    return parse_wiki_doc(path.read_text(encoding="utf-8"))
