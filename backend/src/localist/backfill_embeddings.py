"""
Localist — Embedding Backfill Utility
===================================
Populates the ``embedding`` column for documents already in ``localist_memory.db``
that were indexed before EmbeddingEngine was available (rows where
``embedding IS NULL``).

Usage
-----
Run once from inside the ``backend/`` directory after wiring EmbeddingEngine:

    python backfill_embeddings.py

Options (all optional):
    --db        Path to SQLite database.  Default: ./localist_memory.db
    --model     MLX-LM model path.       Default: mlx-community/embeddinggemma-300m-4bit
    --dry-run   Print what would be updated without writing to the database.
    --force     Re-embed ALL documents, not just NULL rows.

Exit codes:
    0  All rows processed successfully.
    1  EmbeddingEngine failed to load — no changes made.
    2  Partial failure — some rows were not embedded (errors logged).

Architecture note
-----------------
This script imports EmbeddingEngine and MemoryManager directly.  It does NOT
start the FastAPI server.  Run it while the server is stopped (or while
SQLite WAL mode allows concurrent reads — it will wait for any active write
lock to release before committing).
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import struct
import sys
import time
from pathlib import Path

from .paths import get_backend_root

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("backfill")

# ---------------------------------------------------------------------------
# Embedding constants and pack helper.
#
# Mirrors memory_manager._pack_embedding() exactly:
#   struct.pack(f">{len(vector)}f", *vector)  — big-endian float32
#   768 floats × 4 bytes = 3072 bytes per document.
#
# IMPORTANT: memory_manager.index_document() truncates content to [:500]
# before calling embed_fn (line 632 of memory_manager.py).  We replicate
# that truncation here so backfill-generated vectors are identical to what
# a live index_document() call would produce, ensuring cache coherence.
# ---------------------------------------------------------------------------

_EMBED_DIM      = 768
_EMBED_TRUNCATE = 500    # must match index_document content[:500]
_EXPECTED_BYTES = _EMBED_DIM * 4   # 3072


def _pack(vec: list[float]) -> bytes:
    """Mirrors memory_manager._pack_embedding()."""
    if len(vec) != _EMBED_DIM:
        raise ValueError(f"Expected {_EMBED_DIM}-dim vector, got {len(vec)}")
    return struct.pack(f">{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for NULL rows in localist_memory.db"
    )
    parser.add_argument(
        "--db",
        default = str(get_backend_root() / "localist_memory.db"),
        help    = "Path to SQLite database (default: ./localist_memory.db)",
    )
    parser.add_argument(
        "--model",
        default = "mlx-community/embeddinggemma-300m-4bit",
        help    = "MLX-LM model path for embeddings",
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Print what would be updated without writing",
    )
    parser.add_argument(
        "--force",
        action  = "store_true",
        help    = "Re-embed ALL documents, not just NULL rows",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return 1

    # -- Load embedding engine -----------------------------------------------
    logger.info("Loading EmbeddingEngine (model=%s) …", args.model)
    try:
        from .embedding_engine import EmbeddingEngine
    except ImportError as exc:
        logger.error("Cannot import EmbeddingEngine: %s", exc)
        return 1

    engine = EmbeddingEngine(model_path=args.model)
    if not engine.available:
        logger.error(
            "EmbeddingEngine failed to load.  "
            "Ensure mlx-lm is installed: pip install mlx-lm"
        )
        return 1

    # -- Query candidate rows ------------------------------------------------
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")

    if args.force:
        query = "SELECT id, path, content FROM document_index"
        logger.info("--force: re-embedding ALL documents in document_index.")
    else:
        query = "SELECT id, path, content FROM document_index WHERE embedding IS NULL"
        logger.info("Querying for documents with embedding IS NULL …")

    rows = con.execute(query).fetchall()

    if not rows:
        logger.info("No documents require backfill.  Nothing to do.")
        con.close()
        return 0

    logger.info("Found %d document(s) to embed.", len(rows))

    # -- Embed and update ----------------------------------------------------
    errors   = 0
    embedded = 0
    t_start  = time.perf_counter()

    for row_id, path, content in rows:
        if not content:
            logger.warning("  [SKIP] id=%d  path=%s  (empty content)", row_id, path)
            continue

        # Truncate to 500 chars — mirrors index_document() content[:500] call
        # so backfill vectors are identical to what live indexing produces.
        embed_input = content[:_EMBED_TRUNCATE]
        logger.info(
            "  Embedding id=%-4d  path=%s  (%d chars, truncated to %d)",
            row_id, path, len(content), len(embed_input),
        )

        if args.dry_run:
            logger.info("  [DRY-RUN] would embed and write %d bytes", _EXPECTED_BYTES)
            embedded += 1
            continue

        try:
            vec  = engine.embed(embed_input)
            blob = _pack(vec)
        except Exception as exc:
            logger.error("  [ERROR] id=%d  embed failed: %s", row_id, exc)
            errors += 1
            continue

        try:
            con.execute(
                "UPDATE document_index SET embedding = ? WHERE id = ?",
                (blob, row_id),
            )
            con.commit()
            embedded += 1
            logger.info("  [OK]    id=%-4d  dim=%d  bytes=%d", row_id, len(vec), len(blob))
        except sqlite3.Error as exc:
            logger.error("  [ERROR] id=%d  DB write failed: %s", row_id, exc)
            errors += 1

    elapsed = time.perf_counter() - t_start

    # Invalidate retrieval cache once after all embedding updates so cached
    # scores computed before embeddings existed don't survive the backfill.
    if not args.dry_run and embedded > 0:
        con.execute("UPDATE retrieval_cache SET valid = 0")
        con.commit()
        logger.info("Invalidated retrieval cache after backfill.")

    con.close()

    # -- Summary -------------------------------------------------------------
    logger.info(
        "Backfill complete in %.1fs — embedded=%d  skipped/errors=%d  dry_run=%s",
        elapsed, embedded, errors, args.dry_run,
    )

    if errors:
        logger.warning(
            "%d document(s) were not embedded due to errors (see above).", errors
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
