"""
backfill_okf_frontmatter.py — One-time script to bring existing wiki pages'
front matter up to the OKF-aligned convention (see SCHEMA.md).

Adds any of title/description/resource/tags/timestamp that are missing from
a page's front matter, deriving values heuristically from the page's own
body (no inference call, no model involved) — this is a one-time pass over
*existing* pages only; new pages get these fields from the model directly,
per _EXAMPLE's expanded template in wiki_agent.py. Existing fields
(type/status/created/updated/query/anything else) are never modified or
removed, only appended after the new OKF fields.

Run manually:
    python backfill_okf_frontmatter.py
Not wired into any startup/reconcile path — mirrors build_graph.py's
offline __main__ pattern (same project convention).
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

from .paths import get_backend_root

_BACKEND_DIR = get_backend_root()

from .wiki_doc import META_WIKI_FILENAMES, parse_wiki_doc

_WIKI_DIR = _BACKEND_DIR / "wiki"

_CONCEPT_BULLET_RE = re.compile(r"^- \*\*([^:*]+):\*\*", re.MULTILINE)
_RESOURCE_FILENAME_RE = re.compile(r"([\w\- ]+\.\w+)")


class _NoAliasDumper(yaml.SafeDumper):
    """Disable anchor/alias generation — a repeated value (e.g. `timestamp`
    derived from `updated`) must render as a plain scalar twice, not
    `&id001`/`*id001`. Front matter is meant to be hand-readable YAML."""

    def ignore_aliases(self, data):
        return True


def _derive_title(stem: str) -> str:
    return stem.replace("-", " ").title()


def _derive_description(body: str) -> str | None:
    """First sentence of the ## Summary section, or None if absent."""
    m = re.search(r"##\s*Summary\s*\n+(.*?)(?:\n##|\Z)", body, re.DOTALL)
    if not m:
        return None
    summary = " ".join(m.group(1).split())
    if not summary:
        return None
    sentence_end = summary.find(". ")
    return summary[: sentence_end + 1] if sentence_end != -1 else summary


def _derive_tags(body: str) -> list[str] | None:
    """Kebab-cased bullet labels under ### Extracted Concepts, max 5."""
    m = re.search(r"###\s*Extracted Concepts\s*\n+(.*?)(?:\n###|\n##|\Z)", body, re.DOTALL)
    if not m:
        return None
    labels = _CONCEPT_BULLET_RE.findall(m.group(1))
    tags = [
        re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
        for label in labels
    ]
    tags = [t for t in tags if t]
    return tags[:5] or None


def _derive_resource(frontmatter: dict) -> str | None:
    """
    Pull a referenced filename out of the existing `query` field, if any.

    This project's `query` values follow an "Analyze <filename>" convention
    (e.g. "Analyze Localist Software Stack.md") — strip that leading verb
    so `resource` holds just the filename, matching OKF's "canonical URI of
    the underlying asset" intent rather than the full ingestion prompt.
    """
    query = frontmatter.get("query")
    if not query:
        return None
    query = re.sub(r"^analyze\s+", "", str(query), flags=re.IGNORECASE)
    m = _RESOURCE_FILENAME_RE.search(query)
    return m.group(1).strip() if m else query.strip()


def backfill_page(path: Path) -> list[str]:
    """
    Rewrite `path`'s front matter with any missing OKF fields added.
    Returns the list of field names that were actually added (empty if the
    page already had everything).
    """
    doc = parse_wiki_doc(path.read_text(encoding="utf-8"))
    fm = doc.frontmatter
    stem = path.stem

    added: list[str] = []
    new_fm: dict = {}

    def _set(key: str, value) -> None:
        if fm.get(key):
            new_fm[key] = fm[key]
        elif value is not None:
            new_fm[key] = value
            added.append(key)

    updated = fm.get("updated")
    timestamp_default = updated.isoformat() if hasattr(updated, "isoformat") else updated

    new_fm["type"] = fm.get("type", "RESEARCH_NOTE")
    _set("title", _derive_title(stem))
    _set("description", _derive_description(doc.body))
    _set("resource", _derive_resource(fm))
    _set("tags", _derive_tags(doc.body))
    _set("timestamp", timestamp_default or date.today().isoformat())

    # Preserve every other original field (status/created/updated/query/...)
    # in its original order, appended after the OKF fields above.
    for key, value in fm.items():
        if key not in new_fm:
            new_fm[key] = value

    if not added:
        return []

    yaml_block = yaml.dump(
        new_fm, Dumper=_NoAliasDumper, sort_keys=False, default_flow_style=False,
    ).strip()
    new_content = f"---\n{yaml_block}\n---\n{doc.body}"
    path.write_text(new_content, encoding="utf-8")
    return added


def main() -> None:
    if not _WIKI_DIR.exists():
        print(f"wiki dir not found: {_WIKI_DIR}")
        return

    pages = sorted(
        p for p in _WIKI_DIR.iterdir()
        if p.is_file() and p.suffix == ".md" and p.name not in META_WIKI_FILENAMES
    )

    updated_count = 0
    for path in pages:
        added = backfill_page(path)
        if added:
            updated_count += 1
            print(f"{path.name}: added {', '.join(added)}")
        else:
            print(f"{path.name}: already OKF-complete, skipped")

    print(f"\n{updated_count} of {len(pages)} pages updated.")


if __name__ == "__main__":
    main()
