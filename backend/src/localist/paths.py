"""
Localist — backend root resolution
======================================
Single source of truth for "where is backend/ (or its packaged
equivalent)" — replaces 13 previously-independent `Path(__file__).resolve
().parent.parent[.parent]` computations scattered across main.py,
mcp_server/main.py, mcp_server/file_ops.py, mcp_server/ocr.py,
controller_agent.py, memory_manager.py, wiki_agent.py,
wiki_maintenance_log.py, build_graph.py, backfill_embeddings.py,
backfill_episode_embeddings.py, and backfill_okf_frontmatter.py — the exact
scattered-duplication pattern that already caused a stray-file-write bug
during the src/ layout restructure (six of those files had their depth
silently wrong afterward). Every caller now gets the same answer
regardless of its own nesting depth, instead of each file counting
`.parent`s itself.

Two distinct roots, not one — get_backend_root() and get_resource_root()
are the same answer in source-tree mode (both backend/), but diverge once
frozen:

  get_backend_root() — user-writable data: wiki/, raw/, generated_files/,
  the SQLite db, logs. When frozen, this must NOT be inside the (typically
  read-only, code-signed) app bundle.

  get_resource_root() — read-only, ship-with-the-app resources:
  templates/ and SCHEMA.md, which WikiAgent requires to exist (validated,
  not just best-effort — see wiki_agent.py's `missing = [...]` check).
  These are part of the app's own logic, never user-modified, and when
  frozen they live inside the PyInstaller bundle (sys._MEIPASS), bundled
  via each .spec file's `datas`, not in Application Support (which starts
  genuinely empty for a new user — there'd be nothing there to find).

Resolution order (both functions):
  1. Their own env var (LOCALIST_BACKEND_ROOT / LOCALIST_RESOURCE_ROOT),
     if set — explicit override. Real escape hatch for testing (avoids
     writing to a real ~/Library/Application Support/ during development)
     and for advanced users who want a custom location either way.
  2. Frozen (PyInstaller sets sys.frozen=True) — get_backend_root() uses
     ~/Library/Application Support/Localist (created if missing);
     get_resource_root() uses sys._MEIPASS, PyInstaller's own bundle
     extraction directory.
  3. Otherwise — both computed from this file's own fixed location
     (backend/src/localist/paths.py -> backend/), the same answer every
     one of the 13 replaced call sites used to compute independently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_SUPPORT_DIR_NAME = "Localist"


def _source_tree_root() -> Path:
    # backend/src/localist/paths.py -> backend/
    return Path(__file__).resolve().parent.parent.parent


def get_backend_root() -> Path:
    env_override = os.environ.get("LOCALIST_BACKEND_ROOT")
    if env_override:
        return Path(env_override).resolve()

    if getattr(sys, "frozen", False):
        root = Path.home() / "Library" / "Application Support" / _APP_SUPPORT_DIR_NAME
        root.mkdir(parents=True, exist_ok=True)
        return root

    return _source_tree_root()


def get_resource_root() -> Path:
    env_override = os.environ.get("LOCALIST_RESOURCE_ROOT")
    if env_override:
        return Path(env_override).resolve()

    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]

    return _source_tree_root()
