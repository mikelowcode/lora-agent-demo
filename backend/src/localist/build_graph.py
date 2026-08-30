"""
build_graph.py — Offline link-graph builder for the Localist wiki.

Walk wiki/ non-recursively, parse every [[wiki-link]] reference in each
page, and populate the graph_nodes / graph_edges tables in the
MemoryManager database.

Run manually from the backend/ directory:
    python build_graph.py

Design notes
------------
- doc_path convention: absolute, resolved paths — matches the existing
  document_index.path convention in MemoryManager (Path.resolve() → str).

- Normalization rule, identical to wiki_agent._validate_links():
      wiki_agent:   link_text.lower().replace(" ", "-")
      build_graph:  link_text.lower().replace(" ", "-")
  These must stay in sync. See _normalize() below.

- Two-pass algorithm so file processing order doesn't affect resolution:
    Pass 1 — upsert graph_nodes for every .md file.
    Pass 2 — clear all graph_edges (whole-corpus clear), then for each
              file resolve each [[link]] against the complete node-stem
              set and upsert one edge per unique (source, target_path)
              pair.

- Same-page-same-target duplicate [[...]] links (e.g. a link that
  appears in both Mapped Pages and Related Pages) collapse to ONE edge
  row per unique (source_doc_path, target_path) pair. This is deliberate:
  graph_edges counts relationships, not link-mention occurrences.

- clear_graph_edges() is used (not clear_graph_for_doc()) because this
  is a full offline rebuild, not an incremental per-document update.
  clear_graph_for_doc() is available for future per-document hooks.
"""
from __future__ import annotations

from pathlib import Path

# backend/src/localist/build_graph.py -> backend/
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

from .memory_manager import MemoryManager
from .wiki_doc import META_WIKI_FILENAMES, load_wiki_doc

_WIKI_DIR = _BACKEND_DIR / "wiki"


def _normalize(link_text: str) -> str:
    # Identical to wiki_agent._validate_links(): link_text.lower().replace(" ", "-")
    return link_text.lower().replace(" ", "-")


def build_graph(wiki_dir: Path, mm: MemoryManager) -> dict[str, int]:
    """
    Populate graph_nodes and graph_edges from all .md files in wiki_dir.

    Parameters
    ----------
    wiki_dir : directory to walk (non-recursive, .md files only, excluding
               META_WIKI_FILENAMES — index.md/logs.md/MEMORY.md never
               become graph nodes or resolve_graph_target() candidates)
    mm       : MemoryManager connected to the target database

    Returns
    -------
    dict with keys: nodes, edges, resolved, unresolved
    """
    wiki_files = sorted(
        p for p in wiki_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".md" and p.name not in META_WIKI_FILENAMES
    )

    # ---- Pass 1: upsert all graph_nodes ------------------------------------
    # Done before edge resolution so the full stem set is complete regardless
    # of file processing order.
    parsed: dict[str, tuple] = {}   # str(abs_path) → (frontmatter, links)
    node_ids: dict[str, int] = {}   # str(abs_path) → node_id

    for path in wiki_files:
        doc      = load_wiki_doc(path)
        abs_path = str(path.resolve())
        node_id  = mm.upsert_graph_node(
            doc_path=abs_path,
            node_type=doc.frontmatter.get("type"),
            title=doc.frontmatter.get("title"),
        )
        parsed[abs_path]   = (doc.frontmatter, doc.links)
        node_ids[abs_path] = node_id

    # Stem → node_id lookup for resolution.
    # Stems match wiki_agent.py's wiki_pages keys (p.stem), so normalization
    # comparisons are apples-to-apples between the two layers.
    stem_to_node_id: dict[str, int] = {
        Path(abs_path).stem: nid
        for abs_path, nid in node_ids.items()
    }

    # ---- Clear all graph_edges before the edge pass -----------------------
    mm.clear_graph_edges()

    # ---- Pass 2: resolve links and upsert graph_edges ---------------------
    total_edges    = 0
    resolved_count = 0

    for abs_path, (_fm, links) in parsed.items():
        source_node_id = node_ids[abs_path]
        seen_targets: set[str] = set()

        for link in links:
            norm = _normalize(link.link_text)
            if norm in seen_targets:
                # Collapse duplicate same-page-same-target links to one edge.
                continue
            seen_targets.add(norm)

            target_nid = stem_to_node_id.get(norm)
            resolved   = target_nid is not None

            mm.upsert_graph_edge(
                source_node_id  = source_node_id,
                source_doc_path = abs_path,
                target_path     = norm,
                target_node_id  = target_nid,
                target_resolved = resolved,
                link_text       = link.link_text,
            )
            total_edges += 1
            if resolved:
                resolved_count += 1

    return {
        "nodes":      len(node_ids),
        "edges":      total_edges,
        "resolved":   resolved_count,
        "unresolved": total_edges - resolved_count,
    }


if __name__ == "__main__":
    # MemoryManager() bare default writes to lora_memory.db; the live backend
    # uses localist_memory.db (main.py:254) — must not silently diverge.
    mm      = MemoryManager(db_path=_BACKEND_DIR / "localist_memory.db")
    summary = build_graph(_WIKI_DIR, mm)
    print("Graph build complete.")
    print(f"  Nodes:      {summary['nodes']}")
    print(f"  Edges:      {summary['edges']}")
    print(f"  Resolved:   {summary['resolved']}")
    print(f"  Unresolved: {summary['unresolved']}")
