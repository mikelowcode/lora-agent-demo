"""
Tests for build_graph.py — MemoryManager graph methods and builder pipeline.

Five test classes:
  1. TestUpsertGraphNode      — insert then update; created_at preserved.
  2. TestUpsertGraphEdge      — insert then update; resolution can change.
  3. TestBuildGraphCorpus     — end-to-end against real three-document corpus.
  4. TestBuildGraphIdempotent — rerun produces no duplicates.
  5. TestBuildGraphResolution — unresolved link resolves after target page added.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from localist.memory_manager import MemoryManager
from localist.build_graph import build_graph


# ---------------------------------------------------------------------------
# Real corpus fixtures — exact content from wiki/ files (verbatim)
# ---------------------------------------------------------------------------

_BUILD_ORDER = """\

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

_MASTER_OUTLINE = """\

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

_SOFTWARE_STACK = """\

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_mm(tmp_path: Path) -> MemoryManager:
    return MemoryManager(db_path=tmp_path / "test.db")


def _make_wiki(base: Path, files: dict[str, str]) -> Path:
    """Write {filename: content} to base/wiki/ and return the wiki dir."""
    wiki = base / "wiki"
    wiki.mkdir(exist_ok=True)
    for name, content in files.items():
        (wiki / name).write_text(content, encoding="utf-8")
    return wiki


def _raw_query(mm: MemoryManager, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    db_path = mm._db_path  # access via the attribute set in __init__
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# 1. upsert_graph_node — insert then update
# ---------------------------------------------------------------------------

class TestUpsertGraphNode:

    def test_insert_sets_created_and_updated_equal(self, tmp_path):
        mm = _fresh_mm(tmp_path)
        node_id = mm.upsert_graph_node(
            doc_path=tmp_path / "page-a.md",
            node_type="research-note",
            title="Page A",
        )
        assert node_id > 0
        rows = _raw_query(mm, "SELECT * FROM graph_nodes WHERE id = ?", (node_id,))
        assert len(rows) == 1
        row = rows[0]
        # On first insert, created_at == updated_at (same timestamp call)
        assert row["created_at"] == pytest.approx(row["updated_at"], abs=1e-3)
        assert row["node_type"] == "research-note"
        assert row["title"] == "Page A"

    def test_update_advances_updated_at_preserves_created_at(self, tmp_path):
        mm = _fresh_mm(tmp_path)
        doc = tmp_path / "page-b.md"
        node_id = mm.upsert_graph_node(doc_path=doc, node_type="old-type", title="Old Title")

        rows_before = _raw_query(mm, "SELECT * FROM graph_nodes WHERE id = ?", (node_id,))
        created_at_before = rows_before[0]["created_at"]
        updated_at_before = rows_before[0]["updated_at"]

        time.sleep(0.02)  # ensure timestamp advances

        id2 = mm.upsert_graph_node(doc_path=doc, node_type="new-type", title="New Title")

        # Same node, same id
        assert id2 == node_id

        rows_after = _raw_query(mm, "SELECT * FROM graph_nodes WHERE id = ?", (node_id,))
        row = rows_after[0]

        # created_at must be unchanged
        assert row["created_at"] == pytest.approx(created_at_before, abs=1e-6)
        # updated_at must have advanced
        assert row["updated_at"] > updated_at_before
        # values overwritten
        assert row["node_type"] == "new-type"
        assert row["title"] == "New Title"
        # Only one row total
        all_rows = _raw_query(mm, "SELECT id FROM graph_nodes")
        assert len(all_rows) == 1


# ---------------------------------------------------------------------------
# 2. upsert_graph_edge — insert then update (resolution can change on rebuild)
# ---------------------------------------------------------------------------

class TestUpsertGraphEdge:

    def _insert_node(self, mm: MemoryManager, path: Path) -> int:
        return mm.upsert_graph_node(doc_path=path, node_type=None, title=None)

    def test_insert_then_resolve_on_update(self, tmp_path):
        mm = _fresh_mm(tmp_path)

        source_path = tmp_path / "source.md"
        target_path = tmp_path / "target.md"
        source_id   = self._insert_node(mm, source_path)
        target_id   = self._insert_node(mm, target_path)

        # First call — edge doesn't resolve yet (target didn't exist at parse time)
        mm.upsert_graph_edge(
            source_node_id  = source_id,
            source_doc_path = source_path,
            target_path     = "target",
            target_node_id  = None,
            target_resolved = False,
            link_text       = "Target",
        )

        rows = _raw_query(mm,
            "SELECT * FROM graph_edges WHERE source_doc_path = ? AND target_path = ?",
            (str(source_path.resolve()), "target"),
        )
        assert len(rows) == 1
        assert rows[0]["target_resolved"] == 0
        assert rows[0]["target_node_id"] is None

        # Second call — same natural key, now resolved
        mm.upsert_graph_edge(
            source_node_id  = source_id,
            source_doc_path = source_path,
            target_path     = "target",
            target_node_id  = target_id,
            target_resolved = True,
            link_text       = "Target",
        )

        rows_after = _raw_query(mm,
            "SELECT * FROM graph_edges WHERE source_doc_path = ? AND target_path = ?",
            (str(source_path.resolve()), "target"),
        )
        # Exactly one row — not a duplicate
        assert len(rows_after) == 1
        assert rows_after[0]["target_resolved"] == 1
        assert rows_after[0]["target_node_id"] == target_id


# ---------------------------------------------------------------------------
# 3. End-to-end against the real three-document corpus
# ---------------------------------------------------------------------------

class TestBuildGraphCorpus:
    """
    Edge count derivation:
      localist-build-order.md:
        [[Localist Master Project Outline]]   × 2 (Mapped + Related) → 1 edge RESOLVED
        [[Localist Software Stack Overview]]  × 2 (Mapped + Related) → 1 edge UNRESOLVED
            (normalized: localist-software-stack-overview; stem is localist-software-stack)
        Subtotal: 2 edges (1 resolved, 1 unresolved)
      localist-master-project-outline.md:
        [[Localist Design Philosophy]]   × 1 → 1 edge UNRESOLVED
        [[Localist Software Stack]]      × 1 → 1 edge RESOLVED (stem matches)
        [[Localist Build Order]]         × 1 → 1 edge RESOLVED
        [[Localist Wiki Evolution Ideas]]× 1 → 1 edge UNRESOLVED
        Subtotal: 4 edges (2 resolved, 2 unresolved)
      localist-software-stack.md:
        [[Localist Master Project Outline]] × 1 → 1 edge RESOLVED
        Subtotal: 1 edge (1 resolved, 0 unresolved)
      TOTAL: 7 edges — 4 resolved, 3 unresolved
    """

    _CORPUS = {
        "localist-build-order.md":              _BUILD_ORDER,
        "localist-master-project-outline.md":   _MASTER_OUTLINE,
        "localist-software-stack.md":           _SOFTWARE_STACK,
    }
    _EXPECTED_EDGES      = 7
    _EXPECTED_RESOLVED   = 4
    _EXPECTED_UNRESOLVED = 3

    @pytest.fixture()
    def corpus_result(self, tmp_path):
        mm       = _fresh_mm(tmp_path)
        wiki_dir = _make_wiki(tmp_path, self._CORPUS)
        return build_graph(wiki_dir, mm), mm

    def test_node_count(self, corpus_result):
        summary, _ = corpus_result
        assert summary["nodes"] == 3

    def test_edge_counts(self, corpus_result):
        summary, _ = corpus_result
        assert summary["edges"]      == self._EXPECTED_EDGES
        assert summary["resolved"]   == self._EXPECTED_RESOLVED
        assert summary["unresolved"] == self._EXPECTED_UNRESOLVED

    def test_specific_resolved_edges(self, corpus_result):
        _, mm = corpus_result
        rows = _raw_query(mm,
            "SELECT target_path, target_resolved FROM graph_edges WHERE target_resolved = 1"
        )
        resolved_targets = {r["target_path"] for r in rows}
        assert "localist-master-project-outline" in resolved_targets  # build-order → outline
        assert "localist-software-stack"         in resolved_targets  # outline → software-stack
        assert "localist-build-order"            in resolved_targets  # outline → build-order
        # software-stack → outline (same target, already asserted above)

    def test_specific_unresolved_edges(self, corpus_result):
        _, mm = corpus_result
        rows = _raw_query(mm,
            "SELECT target_path FROM graph_edges WHERE target_resolved = 0"
        )
        unresolved_targets = {r["target_path"] for r in rows}
        # Word-count mismatch: "Localist Software Stack Overview" → stem would need to be
        # "localist-software-stack-overview" but the file stem is "localist-software-stack"
        assert "localist-software-stack-overview" in unresolved_targets
        # Genuinely nonexistent pages
        assert "localist-design-philosophy"       in unresolved_targets
        assert "localist-wiki-evolution-ideas"    in unresolved_targets

    def test_build_order_duplicates_collapsed(self, corpus_result):
        """localist-build-order links to localist-master-project-outline twice
        (Mapped Pages and Related Pages) — must produce exactly one edge row."""
        _, mm = corpus_result
        rows = _raw_query(mm,
            """SELECT COUNT(*) as cnt FROM graph_edges
               WHERE target_path = 'localist-master-project-outline'
                 AND source_doc_path LIKE '%localist-build-order%'"""
        )
        assert rows[0]["cnt"] == 1


# ---------------------------------------------------------------------------
# 4. Idempotency — rerun produces no duplicates
# ---------------------------------------------------------------------------

class TestBuildGraphIdempotent:

    def test_rerun_node_and_edge_counts_stable(self, tmp_path):
        mm = _fresh_mm(tmp_path)
        wiki_dir = _make_wiki(tmp_path, {
            "localist-build-order.md":            _BUILD_ORDER,
            "localist-master-project-outline.md": _MASTER_OUTLINE,
            "localist-software-stack.md":         _SOFTWARE_STACK,
        })

        summary1 = build_graph(wiki_dir, mm)
        summary2 = build_graph(wiki_dir, mm)

        assert summary2["nodes"]      == summary1["nodes"]
        assert summary2["edges"]      == summary1["edges"]
        assert summary2["resolved"]   == summary1["resolved"]
        assert summary2["unresolved"] == summary1["unresolved"]

        node_count = _raw_query(mm, "SELECT COUNT(*) as cnt FROM graph_nodes")[0]["cnt"]
        edge_count = _raw_query(mm, "SELECT COUNT(*) as cnt FROM graph_edges")[0]["cnt"]
        assert node_count == summary1["nodes"]
        assert edge_count == summary1["edges"]


# ---------------------------------------------------------------------------
# 5. Unresolved link resolves after target page is added
# ---------------------------------------------------------------------------

class TestBuildGraphResolution:

    def test_link_resolves_on_rebuild_after_target_added(self, tmp_path):
        """
        Run 1: source.md links to [[Missing Page]] which doesn't exist yet.
        Run 2: missing-page.md is added; the edge should show target_resolved=1.
        The edge count remains 1 (not doubled) after the second run.
        """
        source_content = """\
## Summary

A test page.

## Related Pages

- [[Missing Page]]

## Revision History

- 2026-06-19 — Created.
"""
        mm       = _fresh_mm(tmp_path)
        wiki_dir = _make_wiki(tmp_path, {"source.md": source_content})

        summary1 = build_graph(wiki_dir, mm)

        assert summary1["nodes"]      == 1
        assert summary1["edges"]      == 1
        assert summary1["unresolved"] == 1

        rows1 = _raw_query(mm,
            "SELECT * FROM graph_edges WHERE target_path = 'missing-page'"
        )
        assert len(rows1) == 1
        assert rows1[0]["target_resolved"] == 0
        assert rows1[0]["target_node_id"] is None

        # Add the missing page
        missing_content = """\
## Summary

The previously-missing page.

## Related Pages

## Revision History

- 2026-06-19 — Created.
"""
        (wiki_dir / "missing-page.md").write_text(missing_content, encoding="utf-8")

        summary2 = build_graph(wiki_dir, mm)

        assert summary2["nodes"] == 2
        assert summary2["edges"] == 1          # still 1 edge, not doubled
        assert summary2["resolved"] == 1
        assert summary2["unresolved"] == 0

        rows2 = _raw_query(mm,
            "SELECT * FROM graph_edges WHERE target_path = 'missing-page'"
        )
        assert len(rows2) == 1                  # one edge row, not two
        assert rows2[0]["target_resolved"] == 1
        assert rows2[0]["target_node_id"] is not None


# ---------------------------------------------------------------------------
# META_WIKI_FILENAMES exclusion (OKF alignment) — index.md/logs.md/MEMORY.md
# must never become graph nodes or resolve_graph_target() candidate stems.
# ---------------------------------------------------------------------------

class TestBuildGraphExcludesMetaFiles:

    def test_meta_filenames_excluded_from_graph_nodes(self, tmp_path):
        mm = _fresh_mm(tmp_path)
        wiki_dir = _make_wiki(tmp_path, {
            "localist-build-order.md": _BUILD_ORDER,
            "index.md":                "# Wiki Index\n\n## RESEARCH_NOTE\n\n- [[localist-build-order]]\n",
            "logs.md":                 "# Wiki Changelog\n\n## 2026-07-21\n\n- Creation: [[localist-build-order]]\n",
            "MEMORY.md":               "# Memory\n\n- some episodic fact\n",
        })

        summary = build_graph(wiki_dir, mm)

        assert summary["nodes"] == 1  # only localist-build-order
        assert mm.list_graph_node_stems() == ["localist-build-order"]
