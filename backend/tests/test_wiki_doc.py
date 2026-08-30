"""
Tests for wiki_doc — parse_wiki_doc() and load_wiki_doc().

All nine cases from the Phase A/B Graph Retrieval Layer spec.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from localist.wiki_doc import WikiLink, ParsedWikiDoc, META_WIKI_FILENAMES, parse_wiki_doc, load_wiki_doc


# ---------------------------------------------------------------------------
# Fixtures — verbatim real file content (byte-for-byte from corpus)
# ---------------------------------------------------------------------------

LOCALIST_BUILD_ORDER = """\
---
title: Localist Build Order
type: research-note
query: Analyze Localist Build Order.md
created: 2026-06-16
updated: 2026-06-16
---

## Summary

This document outlines the nine-phase development roadmap for the Localist project. The phases detail the sequential implementation of core components, starting with the runtime and inference layer (Phase 1), through the Agent Core and Persistent Memory integration (Phases 2 & 3), and culminating in the full FastAPI backend and SvelteKit frontend (Phases 5 & 6).

## Details

### Extracted Concepts

- **Multi-Layer Architecture:** The project is structured into distinct, interdependent layers: Inference (oMLX/Foundry), Agent Core (Controller/Wiki/Research), Persistent Memory (SQLite/Indexing), Backend (FastAPI), and Frontend (SvelteKit).
- **Development Milestones:** Specific phases detail critical implementations, such as implementing the Agent Controller (Phase 2), integrating Persistent Memory (Phase 3), and achieving full frontend/backend connectivity (Phases 5 & 6).
- **Key Technologies:** The build relies on oMLX for inference, MLX-LM for embeddings, SQLite for memory management, FastAPI for the backend, and SvelteKit for the frontend.

### Mapped Pages

- [[Localist Master Project Outline]] — Provides the high-level goals and vision for the Localist project.
- [[Localist Software Stack Overview]] — Details the specific technologies and models used during the build phases (e.g., gemma-4-e4b-it-4bit, mlx-community/embeddinggemma-300m-4bit).

### Proposed New Pages

- `Localist Architecture Diagram` (CONCEPT) — To formally map the relationships between the various components developed across the nine phases.

## Related Pages

- [[Localist Master Project Outline]]
- [[Localist Software Stack Overview]]

## Revision History

- 2026-06-16 — Initial research note created from Localist Build Order.md.
"""

LOCALIST_MASTER_PROJECT_OUTLINE = """\
---
title: Localist Master Project Outline
type: research-note
query: Analyze Localist Master Project Outline.md
created: 2026-06-16
updated: 2026-06-16
---

## Summary

The Localist project is a Local First Agent Framework designed to run entirely on a local machine. It utilizes a multi-agent reasoning pipeline powered by oMLX for inference, MLX-LM for embeddings, and SQLite for persistent memory management. The system integrates a FastAPI backend and a SvelteKit frontend to provide a complete, local solution.

## Details

### Extracted Concepts

- **Localist:** The core framework providing local, self-contained AI agent capabilities.
- **Multi-Agent Pipeline:** A system involving specialized agents (Controller, Wiki, Research, Conversational, Memory) each with a distinct role in the workflow.
- **oMLX:** Serves the large language model via an OpenAI-compatible HTTP API, handling local inference.
- **MLX-LM:** Provides standalone embedding generation using models like `gemma-300m-4bit`.
- **Persistent Memory:** Managed by `MemoryManager` using SQLite to store the document index, conversation logs, and retrieval cache.
- **Workflow:** The system supports ingestion (WikiAgent), research retrieval (ResearchAgent), and conversational Q&A (ConversationalAgent).

### Mapped Pages

- null

### Proposed New Pages

- `Localist Design Philosophy` (CONCEPT) — To formally define the underlying principles of the Localist architecture.
- `Localist Software Stack` (CONCEPT) — To detail the specific technologies used (oMLX, MLX-LM, FastAPI, SQLite, SvelteKit).
- `Localist Build Order` (SYSTEM) — To document the sequence required to set up and run the project components.

## Related Pages

- [[Localist Design Philosophy]]
- [[Localist Software Stack]]
- [[Localist Build Order]]
- [[Localist Wiki Evolution Ideas]]

## Revision History

- 2026-06-16 — Initial research note created from Localist Master Project Outline.md.
"""

LOCALIST_SOFTWARE_STACK = """\
---
title: Localist Software Stack Overview
type: research-note
query: Analyze Localist Software Stack.md
created: 2026-06-16
updated: 2026-06-16
---

## Summary

This document details the complete software stack required for the Localist project, categorizing components into Core Software (Required), Local Tools & Libraries (Required), Local sLMs, Cloud Models (Optional), and Optional Enhancements. The stack is designed to run locally, utilizing MLX and oMLX for inference and FastAPI/SvelteKit for the application layer.

## Details

### Extracted Concepts

- **Core Software Stack:** Includes oMLX (Local inference server), MLX (Apple Silicon ML framework), MLX-LM (LLM/embedding inference), Python 3.13, FastAPI (Backend HTTP server), SvelteKit (Frontend UI), and VS Code.
- **Local Tools & Libraries:** Essential dependencies like SQLite (Persistent memory/index), pydantic (Validation), uvicorn (ASGI server), requests (HTTP client), and python-multipart (File upload support).
- **sLMs:** The active chat/reasoning model is `gemma-4-e4b-it-4bit` (via oMLX). The embedding model is `mlx-community/embeddinggemma-300m-4bit`, producing 768-dim vectors.
- **Cloud Models:** Azure AI Foundry is listed as an optional backend, accessible via FoundryRuntimeClient.
- **Cost Structure:** All listed software components are noted as Free, with Azure AI Foundry having a Free tier/Paid option.

### Mapped Pages

- null

### Proposed New Pages

- `Localist Software Stack` (CONCEPT) — To formally define the specific technologies and their roles within the Localist architecture.
- `Localist Build Order` (SYSTEM) — To document the sequence required to set up and run the various components of the stack.

## Related Pages

- [[Localist Master Project Outline]]

## Revision History

- 2026-06-16 — Initial research note created from Localist Software Stack.md.
"""

LORA_PERSONA = (
    "You are LORA, a local‑first thinking partner.\n"
    "You speak clearly, directly, and in a natural conversational tone.\n"
    "You use tools when they are needed and follow tool instructions precisely.\n"
    "When you state facts, you cite where they came from."
)


# ---------------------------------------------------------------------------
# Test 1 — No frontmatter
# ---------------------------------------------------------------------------

def test_no_frontmatter_plain_body():
    content = "## Hello\n\nSome text with [[A Link]] inside.\n"
    result = parse_wiki_doc(content)
    assert result.frontmatter == {}
    assert result.body == content
    assert result.links == [WikiLink(link_text="A Link", target_path="A Link")]


# ---------------------------------------------------------------------------
# Test 2 — Valid frontmatter, real corpus fixtures
# ---------------------------------------------------------------------------

def test_build_order_frontmatter_and_body():
    result = parse_wiki_doc(LOCALIST_BUILD_ORDER)
    fm = result.frontmatter
    assert fm["title"] == "Localist Build Order"
    assert fm["type"] == "research-note"
    assert fm["query"] == "Analyze Localist Build Order.md"
    assert fm["created"] == datetime.date(2026, 6, 16)
    assert fm["updated"] == datetime.date(2026, 6, 16)
    assert not result.body.startswith("---")
    assert "title:" not in result.body


def test_master_outline_frontmatter_and_body():
    result = parse_wiki_doc(LOCALIST_MASTER_PROJECT_OUTLINE)
    fm = result.frontmatter
    assert fm["title"] == "Localist Master Project Outline"
    assert fm["type"] == "research-note"
    assert fm["query"] == "Analyze Localist Master Project Outline.md"
    assert fm["created"] == datetime.date(2026, 6, 16)
    assert fm["updated"] == datetime.date(2026, 6, 16)
    assert not result.body.startswith("---")
    assert "title:" not in result.body


def test_software_stack_frontmatter_and_body():
    result = parse_wiki_doc(LOCALIST_SOFTWARE_STACK)
    fm = result.frontmatter
    assert fm["title"] == "Localist Software Stack Overview"
    assert fm["type"] == "research-note"
    assert fm["query"] == "Analyze Localist Software Stack.md"
    assert fm["created"] == datetime.date(2026, 6, 16)
    assert fm["updated"] == datetime.date(2026, 6, 16)
    assert not result.body.startswith("---")
    assert "title:" not in result.body


# ---------------------------------------------------------------------------
# Test 3 — lora-persona.md real-world case (no frontmatter, passthrough)
# ---------------------------------------------------------------------------

def test_lora_persona_no_frontmatter_passthrough():
    result = parse_wiki_doc(LORA_PERSONA)
    assert result.frontmatter == {}
    assert result.body == LORA_PERSONA


# ---------------------------------------------------------------------------
# Test 4 — Malformed frontmatter (opening fence, no closing fence)
# ---------------------------------------------------------------------------

def test_malformed_frontmatter_no_close():
    content = "---\ntitle: Broken File\nsome body text\n[[A Link]]\n"
    result = parse_wiki_doc(content)
    assert result.frontmatter == {}
    assert result.body == content
    # Links are still parsed from body (the whole content in fail-safe mode)
    assert any(link.link_text == "A Link" for link in result.links)


# ---------------------------------------------------------------------------
# Test 5 — Casing/wording inconsistency preserved exactly (no normalization)
# ---------------------------------------------------------------------------

def test_build_order_preserves_title_case_link():
    result = parse_wiki_doc(LOCALIST_BUILD_ORDER)
    link_texts = [link.link_text for link in result.links]
    # Exact string as written in localist-build-order.md
    assert "Localist Master Project Outline" in link_texts


def test_master_outline_preserves_different_stack_link():
    result = parse_wiki_doc(LOCALIST_MASTER_PROJECT_OUTLINE)
    link_texts = [link.link_text for link in result.links]
    # "Localist Software Stack" — different from "Localist Software Stack Overview"
    # in localist-build-order.md; both preserved unmodified (normalization is Phase B)
    assert "Localist Software Stack" in link_texts
    assert "Localist Software Stack Overview" not in link_texts


# ---------------------------------------------------------------------------
# Test 6 — Unresolved/aspirational links parsed identically to resolvable ones
# ---------------------------------------------------------------------------

def test_aspirational_links_parsed_as_ordinary_wiki_links():
    result = parse_wiki_doc(LOCALIST_MASTER_PROJECT_OUTLINE)
    link_texts = [link.link_text for link in result.links]
    # These pages do not exist in the corpus — must parse identically to any other link
    assert "Localist Design Philosophy" in link_texts
    assert "Localist Wiki Evolution Ideas" in link_texts
    # Confirm they are plain WikiLink instances, not some special "unresolved" type
    for link in result.links:
        assert isinstance(link, WikiLink)
        assert link.target_path == link.link_text


# ---------------------------------------------------------------------------
# Test 7 — Document order preserved, duplicates not deduplicated
# ---------------------------------------------------------------------------

def test_build_order_link_count_and_order():
    result = parse_wiki_doc(LOCALIST_BUILD_ORDER)
    # [[Localist Master Project Outline]] x2, [[Localist Software Stack Overview]] x2
    assert len(result.links) == 4
    assert result.links[0].link_text == "Localist Master Project Outline"
    assert result.links[1].link_text == "Localist Software Stack Overview"
    assert result.links[2].link_text == "Localist Master Project Outline"
    assert result.links[3].link_text == "Localist Software Stack Overview"


# ---------------------------------------------------------------------------
# Test 8 — load_wiki_doc() file I/O wrapper
# ---------------------------------------------------------------------------

def test_load_wiki_doc_matches_parse_wiki_doc(tmp_path: Path):
    wiki_file = tmp_path / "localist-build-order.md"
    wiki_file.write_text(LOCALIST_BUILD_ORDER, encoding="utf-8")
    from_disk = load_wiki_doc(wiki_file)
    from_str = parse_wiki_doc(LOCALIST_BUILD_ORDER)
    assert from_disk == from_str


# ---------------------------------------------------------------------------
# Test 9 — Frontmatter date values are datetime.date, not strings
# ---------------------------------------------------------------------------

def test_frontmatter_dates_are_date_objects():
    result = parse_wiki_doc(LOCALIST_BUILD_ORDER)
    assert isinstance(result.frontmatter["created"], datetime.date)
    assert isinstance(result.frontmatter["updated"], datetime.date)
    # Sanity-check value
    assert result.frontmatter["created"] == datetime.date(2026, 6, 16)


# ---------------------------------------------------------------------------
# Tests 10–13 — Leading-blank-line regression (Open Item 6 fix)
# ---------------------------------------------------------------------------

# Verbatim format as produced by Gemma and written to disk:
# line 0 is a blank line; the real opening --- fence is on line 1.
_LEADING_BLANK_DOC = (
    "\n"
    "---\n"
    "title: Localist Build Order\n"
    "type: research-note\n"
    "query: Analyze Localist Build Order.md\n"
    "created: 2026-06-16\n"
    "updated: 2026-06-16\n"
    "---\n"
    "\n"
    "## Summary\n"
    "\n"
    "Body text here.\n"
)


def test_leading_blank_line_before_fence_parses_frontmatter():
    """One leading blank line before --- must not defeat frontmatter detection."""
    result = parse_wiki_doc(_LEADING_BLANK_DOC)
    assert result.frontmatter["title"] == "Localist Build Order"
    assert result.frontmatter["type"] == "research-note"
    assert result.frontmatter["created"] == datetime.date(2026, 6, 16)


def test_leading_blank_line_before_fence_body_clean():
    """Body returned when fence is at line 1 must not include the YAML block or leading blank."""
    result = parse_wiki_doc(_LEADING_BLANK_DOC)
    assert not result.body.startswith("\n---")
    assert "title:" not in result.body
    assert "## Summary" in result.body


def test_leading_blank_no_closing_fence_fallback_unchanged():
    """Leading blank + opening fence but no closing fence → body = content, frontmatter = {}."""
    content = "\n---\ntitle: Broken\nsome body text without a closing fence\n"
    result = parse_wiki_doc(content)
    assert result.frontmatter == {}
    assert result.body == content  # silent fallback, body is the whole content unchanged


def test_standard_fence_at_line_zero_unaffected_by_fix():
    """Standard well-formed doc with fence on line 0 must behave identically after the fix."""
    result = parse_wiki_doc(LOCALIST_BUILD_ORDER)
    assert result.frontmatter["title"] == "Localist Build Order"
    assert not result.body.startswith("---")
    assert "title:" not in result.body


# ---------------------------------------------------------------------------
# Test 14 — META_WIKI_FILENAMES (OKF alignment — structural/generated files
# excluded from indexing, graph nodes, and model-visible wiki context)
# ---------------------------------------------------------------------------

def test_meta_wiki_filenames_contains_expected_names():
    assert META_WIKI_FILENAMES == frozenset({"index.md", "logs.md", "MEMORY.md"})
