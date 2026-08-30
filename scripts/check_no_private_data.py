#!/usr/bin/env python3
"""
pre-commit hook: block staged files that hold Localist runtime/personal
data or secrets.

.gitignore is the primary mechanism keeping these paths out of the repo;
this is defense-in-depth against `git add -f` and against a new
personal-data path landing without a matching .gitignore entry. Receives
staged filenames (relative to repo root) as argv, per pre-commit's
`pass_filenames: true`.
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

# Directories whose entire contents are runtime/personal data.
_DENIED_DIR_PREFIXES = (
    "backend/wiki/",
    "backend/raw/",
    "backend/generated_files/",
)

# Exact paths that are never meant to be committed.
_DENIED_EXACT_PATHS = {
    "sessions-log.md",
    "docs/architecture/23-auditlog-core-scope.md",
}

# Basename suffixes/patterns for secrets and local databases.
_DENIED_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".pem", ".key")

# Files that would otherwise match a denied pattern but are meant to ship.
_ALLOWED_EXACT_BASENAMES = {".env.example"}


def is_denied(path_str: str) -> str | None:
    """Return a human-readable reason if path_str is denied, else None."""
    path = PurePosixPath(path_str)
    basename = path.name

    if basename in _ALLOWED_EXACT_BASENAMES:
        return None

    if path_str in _DENIED_EXACT_PATHS:
        return "matches a path that must never be committed"

    for prefix in _DENIED_DIR_PREFIXES:
        if path_str == prefix.rstrip("/") or path_str.startswith(prefix):
            return f"lives under {prefix!r}, a runtime/personal-data directory"

    if basename == ".env" or basename.startswith(".env."):
        return "looks like an env file (secrets/config)"

    if basename.endswith(_DENIED_SUFFIXES):
        return "looks like a local database or secret key file"

    return None


def main(argv: list[str]) -> int:
    violations = [(f, reason) for f in argv if (reason := is_denied(f))]

    if not violations:
        return 0

    print("check_no_private_data: refusing to commit the following:\n")
    for f, reason in violations:
        print(f"  {f}  —  {reason}")
    print(
        "\nThese paths hold Localist runtime state or secrets and must "
        "never be committed. If this is a false positive, adjust "
        "scripts/check_no_private_data.py."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
