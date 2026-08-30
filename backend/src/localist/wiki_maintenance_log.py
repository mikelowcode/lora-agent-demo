"""
Wiki maintenance audit log.
============================
Append-only, timestamped record of destructive wiki maintenance actions
(currently: orphan removals from document_index, performed by
MemoryManager.reconcile_wiki()).

Distinct in purpose from:
  - application logging (ephemeral, console-only, via the standard logger)
  - sessions-log.md (hand/Claude-Code-narrated dev journal, not a runtime
    artifact)

This is a plain data file — no logging framework, no rotation, no size cap.
If the file ever needs rotation/retention at larger project scale, that is
a forward-looking concern, not handled here.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# backend/src/localist/wiki_maintenance_log.py -> backend/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "logs" / "wiki_maintenance.log"


def _append_line(line: str, entry_kind: str) -> None:
    """Shared append/flush logic for all entry kinds — write failures here
    must never break the caller's flow; any exception is caught, logged as
    a warning, and swallowed."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        f = open(_LOG_PATH, mode="a", encoding="utf-8")
        try:
            f.write(line)
            f.flush()
        finally:
            f.close()
    except Exception as exc:
        logger.warning("wiki_maintenance_log: failed to write %s entry — %s", entry_kind, exc)


def log_orphan_removed(name: str, path: str) -> None:
    """
    Append one line recording an orphaned document_index row removal.

    Line format (tab-separated):
        <ISO 8601 UTC timestamp>\torphan_removed\tname=<name>\tpath=<path>
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _append_line(f"{timestamp}\torphan_removed\tname={name}\tpath={path}\n", "orphan-removed")


def log_snapshot_pruned(name: str, path: str) -> None:
    """
    Append one line recording an expired (>30 days) wiki page snapshot
    removal from wiki/.snapshots/, performed by the startup TTL sweep in
    main.py's lifespan().

    Line format (tab-separated):
        <ISO 8601 UTC timestamp>\tsnapshot_pruned\tname=<name>\tpath=<path>
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _append_line(f"{timestamp}\tsnapshot_pruned\tname={name}\tpath={path}\n", "snapshot-pruned")
