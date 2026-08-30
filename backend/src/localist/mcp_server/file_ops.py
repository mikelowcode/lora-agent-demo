"""
Localist MCP Server — file_op tool implementations
====================================================
Ports the file_op logic from ToolDispatcher._file_read / _file_write /
_file_append (tool_dispatcher.py) verbatim: same sandboxing check (resolved
path must be a descendant of project_root), same _MAX_FILE_READ_CHARS
truncation behaviour, same truncation suffix.

Unlike ToolDispatcher, which swallows errors into an "ERROR: ..." string
inside a always-success-shaped ToolResult, these functions raise on failure
so the MCP protocol layer can surface isError=True to the client — see
mcp/server/lowlevel/server.py's call_tool() decorator, which converts any
raised exception into CallToolResult(isError=True, content=[TextContent(text=str(exc))]).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..paths import get_backend_root

# Maximum characters read from a file (mirrors tool_dispatcher._MAX_FILE_READ_CHARS —
# kept duplicated rather than imported so this service has no dependency on the
# backend's agent stack).
_MAX_FILE_READ_CHARS: int = 4000

_project_root: Path | None = None


def get_project_root() -> Path:
    """
    Resolve the sandbox root.

    Configurable via the LOCALIST_MCP_PROJECT_ROOT environment variable.
    Defaults to the backend/ directory (parent of this package), matching
    where main.py and start_localist.sh run the sibling services from.

    file_op reads/writes/appends are sandboxed under a fixed
    "generated_files" subdirectory of that root, not the root itself —
    created on first resolution if it doesn't yet exist.
    """
    global _project_root
    if _project_root is None:
        env_root = os.environ.get("LOCALIST_MCP_PROJECT_ROOT")
        base = Path(env_root).resolve() if env_root else get_backend_root()
        _project_root = base / "generated_files"
        _project_root.mkdir(parents=True, exist_ok=True)
    return _project_root


def set_project_root(path: Path | str) -> None:
    """Override the sandbox root. Used at startup and by tests."""
    global _project_root
    _project_root = Path(path).resolve()


def _sandbox_resolve(rel_path: str) -> Path:
    """
    Resolve rel_path against project_root and enforce sandboxing.

    Raises ValueError if the path is invalid or escapes project_root —
    same checks and same error text as ToolDispatcher._run_file_op.
    """
    project_root = get_project_root()
    try:
        resolved = (project_root / rel_path).resolve()
    except Exception as exc:
        raise ValueError(f"ERROR: invalid path — {exc}") from exc

    if not str(resolved).startswith(str(project_root)):
        raise ValueError(
            "ERROR: path traversal outside project_root is not permitted"
        )
    return resolved


def read_file(path: str) -> str:
    """Read a UTF-8 text file relative to project_root, sandboxed."""
    resolved = _sandbox_resolve(path)

    if not resolved.exists():
        raise ValueError(f"ERROR: file not found — {resolved}")

    text = resolved.read_text(encoding="utf-8", errors="replace")
    truncated = False
    if len(text) > _MAX_FILE_READ_CHARS:
        text = text[:_MAX_FILE_READ_CHARS]
        truncated = True
    suffix = "\n… [truncated]" if truncated else ""
    return text + suffix


def write_file(path: str, content: str) -> str:
    """
    Write content to a UTF-8 text file relative to project_root, sandboxed.

    Refuses to write when content is empty or whitespace-only (see
    Open Item 1, LOCALIST-Architecture.md §14.7) — the derived-content
    fallback upstream can resolve to "" when it finds nothing to write,
    which used to succeed silently as a 0-byte file.

    Never overwrites an existing file. If the resolved target already
    exists, tries stem_2{suffix} through stem_10{suffix} in the same
    directory (splitting the filename on its first dot, so a compound
    extension like ".tar.gz" stays intact — e.g. "archive.tar.gz" versions
    as "archive_2.tar.gz", not "archive.tar_2.gz") and writes to the first
    one that doesn't exist. Raises if all 10 versions are already taken.
    """
    resolved = _sandbox_resolve(path)

    if not content.strip():
        raise ValueError("ERROR: no content to write — refusing empty file write")

    if resolved.exists():
        original_name = resolved.name
        if "." in original_name:
            stem, _, ext = original_name.partition(".")
            suffix = "." + ext
        else:
            stem, suffix = original_name, ""

        for n in range(2, 11):
            candidate = resolved.with_name(f"{stem}_{n}{suffix}")
            if not candidate.exists():
                resolved = candidate
                break
        else:
            raise ValueError(
                f"ERROR: version cap reached — 10 versions of "
                f"'{original_name}' already exist"
            )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"ERROR: file_op failed — {exc}") from exc
    return f"OK: wrote {len(content)} characters to {resolved.name}"


_MAX_SIDECAR_TURNS: int = 50


def append_file(path: str, content: str, turn_id: str | None = None) -> str:
    """
    Append content to a UTF-8 text file relative to project_root, sandboxed.

    If turn_id is a non-empty string, dedup against a sidecar file
    (".{name}.append_turns", alongside the target, holding up to the last
    _MAX_SIDECAR_TURNS turn_ids already applied) so the same turn_id can't
    append twice — e.g. on a caller retry. turn_id=None or "" preserves
    the original always-append behavior with no sidecar involved.
    """
    resolved = _sandbox_resolve(path)

    prior_turns: list[str] = []
    sidecar = None
    if turn_id:
        sidecar = resolved.parent / f".{resolved.name}.append_turns"
        if sidecar.exists():
            try:
                prior_turns = [
                    line for line in sidecar.read_text(encoding="utf-8").splitlines()
                    if line
                ]
            except Exception:
                # Sidecar unreadable/corrupt — treat as no prior turn_ids
                # recorded rather than failing the append.
                prior_turns = []

        if turn_id in prior_turns:
            return (
                f"OK: skipped duplicate append for turn_id={turn_id} "
                f"(already applied)"
            )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a", encoding="utf-8") as f:
            f.write(content)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"ERROR: file_op failed — {exc}") from exc

    if turn_id:
        prior_turns.append(turn_id)
        prior_turns = prior_turns[-_MAX_SIDECAR_TURNS:]
        sidecar.write_text("\n".join(prior_turns) + "\n", encoding="utf-8")

    return f"OK: appended {len(content)} characters to {resolved.name}"
