"""Initialize the football news SQLite database.

Usage:
    uv run python scripts/init_football_db.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  last_fetched_at TEXT,
  article_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS articles (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES sources(id),
  url TEXT NOT NULL UNIQUE,
  title_en TEXT NOT NULL,
  body_en TEXT NOT NULL,
  author TEXT,
  published_at TEXT NOT NULL,
  category TEXT,
  tags TEXT,
  image_url TEXT,
  image_alt TEXT,
  image_width INTEGER,
  image_height INTEGER,
  og_description_en TEXT,
  fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
  word_count INTEGER
);

CREATE TABLE IF NOT EXISTS translations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id TEXT NOT NULL REFERENCES articles(id),
  title_tvl TEXT,
  og_description_tvl TEXT,
  body_tvl TEXT,
  model_path TEXT NOT NULL,
  translated_at TEXT NOT NULL DEFAULT (datetime('now')),
  paragraph_count INTEGER,
  failed_paragraphs INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fetch_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL,
  fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
  articles_found INTEGER,
  articles_new INTEGER,
  errors TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category, published_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_translations_article ON translations(article_id);
"""

SEED_SOURCES = [
    ("goal", "Goal.com"),
    ("fifa", "FIFA.com"),
    ("sky", "Sky Sports"),
]


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)
    for source_id, name in SEED_SOURCES:
        conn.execute(
            "INSERT OR IGNORE INTO sources (id, name) VALUES (?, ?)",
            (source_id, name),
        )
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")
    print(f"  Sources: {', '.join(n for _, n in SEED_SOURCES)}")


if __name__ == "__main__":
    init_db()
