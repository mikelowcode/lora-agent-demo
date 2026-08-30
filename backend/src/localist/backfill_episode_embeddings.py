"""
Localist — Episodic Memory Embedding Backfill
==========================================
One-off script: embeds every active `episodes` row whose `embedding` column
is currently NULL (i.e. everything written before EpisodicMemoryWriter
started embedding on insert()), and regenerates MEMORY.md from the current
DB state.

Why this exists
----------------
EpisodicMemoryWriter.insert() only embeds *new* rows going forward. Rows
already in lora_memory.db/localist_memory.db (episodes) — including the
oMLX V0.5.0 `project_fact` example — have embedding = NULL and would only
ever be found by EpisodicMemoryReader.by_similarity()'s keyword fallback.
Run this once after pulling the embedding-support change to make existing
memories reachable by real cosine similarity too.

Usage
-----
    cd backend
    source .venv/bin/activate
    python backfill_episode_embeddings.py [--dry-run]

Uses the same Settings() resolution as main.py's lifespan (same env vars,
same defaults), so it operates on the same DB and wiki dir the running
server uses. Safe to re-run — rows that already have an embedding are
skipped.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

from .paths import get_backend_root

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("backfill_episode_embeddings")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report how many rows would be embedded without writing anything.",
    )
    args = parser.parse_args()

    from .main import Settings  # Settings lives in main.py, not a separate config module
    from .embedding_engine import EmbeddingEngine
    from .memory_manager import EpisodicMemoryWriter, _pack_embedding

    settings = Settings()
    project_root = get_backend_root()
    memory_db = Path(settings.memory_db) if settings.memory_db else project_root / "localist_memory.db"
    wiki_dir = Path(settings.wiki_dir) if settings.wiki_dir else project_root / "wiki"
    memory_md_path = wiki_dir / "MEMORY.md"

    if not memory_db.exists():
        logger.error("Memory DB not found at %s — nothing to backfill.", memory_db)
        return 1

    engine = EmbeddingEngine()
    if not engine.available:
        logger.error(
            "EmbeddingEngine failed to load (see warnings above) — cannot "
            "backfill embeddings. Install mlx-embeddings and retry."
        )
        return 1

    conn = sqlite3.connect(str(memory_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, subject, content FROM episodes "
        "WHERE status = 'active' AND embedding IS NULL"
    ).fetchall()

    logger.info("Found %d active episode(s) missing an embedding.", len(rows))
    if not rows:
        conn.close()
    else:
        if args.dry_run:
            for row in rows:
                logger.info("  [dry-run] id=%d subject=%r", row["id"], row["subject"][:60])
            conn.close()
            logger.info("Dry run — no changes written.")
            return 0

        embedded = 0
        failed = 0
        for row in rows:
            try:
                vector = engine.embed(f"{row['subject']}. {row['content']}")
                blob = _pack_embedding(vector)
                conn.execute(
                    "UPDATE episodes SET embedding = ? WHERE id = ?",
                    (blob, row["id"]),
                )
                embedded += 1
            except Exception as exc:
                failed += 1
                logger.warning("  id=%d failed to embed: %s", row["id"], exc)

        conn.commit()
        conn.close()
        logger.info("Backfill complete — embedded %d, failed %d.", embedded, failed)

    # Regenerate MEMORY.md from current state regardless of whether any
    # rows needed embedding — covers first-run (no MEMORY.md yet) too.
    writer = EpisodicMemoryWriter(db_path=memory_db, memory_md_path=memory_md_path)
    writer.regenerate_memory_md()
    logger.info("MEMORY.md regenerated at %s.", memory_md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
