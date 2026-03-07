"""Migration: add collapse detection columns, model_id, and translation_attempts table.

Adds new columns to translations table, creates translation_attempts table,
backfills model_id for existing rows, and retroactively detects collapsed translations.

Usage:
    uv run python scripts/migrate_collapse_detection.py
"""

import sqlite3
from pathlib import Path

from detect_collapse import is_collapsed, collapse_score

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"

# The model used for all existing translations
KNOWN_MODEL_PATH = "tinker://a6453cc0-d0d8-5168-996a-c9b9ee3b8582:train:0/sampler_weights/final"
KNOWN_MODEL_ID = "a6453cc0-d0d8-5168-996a-c9b9ee3b8582"


def migrate():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    # 1. Add new columns to translations (idempotent)
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(translations)").fetchall()
    }

    new_cols = {
        "model_id": "TEXT",
        "is_collapsed": "BOOLEAN DEFAULT 0",
        "collapse_score": "REAL",
        "attempt_number": "INTEGER DEFAULT 1",
    }

    for col, col_type in new_cols.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE translations ADD COLUMN {col} {col_type}")
            print(f"  Added column translations.{col}")

    # 2. Create translation_attempts table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS translation_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            title_tvl TEXT,
            body_tvl TEXT,
            og_description_tvl TEXT,
            model_path TEXT NOT NULL,
            model_id TEXT,
            temperature REAL NOT NULL,
            is_collapsed BOOLEAN DEFAULT 0,
            collapse_score REAL,
            paragraph_count INTEGER,
            failed_paragraphs INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_attempts_article ON translation_attempts(article_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_translations_collapsed ON translations(is_collapsed)"
    )
    print("  Created translation_attempts table + indexes")

    # 3. Backfill model_id for existing translations
    updated = conn.execute(
        "UPDATE translations SET model_id = ? WHERE model_id IS NULL",
        (KNOWN_MODEL_ID,),
    ).rowcount
    print(f"  Backfilled model_id for {updated} translations")

    # 4. Retroactively detect collapsed translations
    rows = conn.execute(
        "SELECT id, article_id, title_tvl, body_tvl FROM translations"
    ).fetchall()

    collapsed_count = 0
    for row in rows:
        body = row["body_tvl"] or ""
        title = row["title_tvl"] or ""

        body_collapsed = is_collapsed(body)
        title_collapsed = is_collapsed(title)
        any_collapsed = body_collapsed or title_collapsed
        score = max(collapse_score(body), collapse_score(title))

        if any_collapsed:
            collapsed_count += 1

        conn.execute(
            "UPDATE translations SET is_collapsed = ?, collapse_score = ? WHERE id = ?",
            (1 if any_collapsed else 0, round(score, 4), row["id"]),
        )

    conn.commit()
    print(f"  Scanned {len(rows)} translations: {collapsed_count} flagged as collapsed")

    # 5. Show summary
    total = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
    collapsed = conn.execute("SELECT COUNT(*) FROM translations WHERE is_collapsed = 1").fetchone()[0]
    clean = total - collapsed
    print(f"\nSummary: {clean} clean, {collapsed} collapsed out of {total} total translations")

    conn.close()


if __name__ == "__main__":
    migrate()
