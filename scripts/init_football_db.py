"""Initialize the football news SQLite database.

Usage:
    uv run python scripts/init_football_db.py
"""

import sqlite3
from pathlib import Path

from db_conn import get_db, is_d1_mode

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
  model_id TEXT,
  translated_at TEXT NOT NULL DEFAULT (datetime('now')),
  paragraph_count INTEGER,
  failed_paragraphs INTEGER DEFAULT 0,
  is_collapsed BOOLEAN DEFAULT 0,
  collapse_score REAL,
  attempt_number INTEGER DEFAULT 1
);

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
);

CREATE TABLE IF NOT EXISTS fetch_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL,
  fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
  articles_found INTEGER,
  articles_new INTEGER,
  errors TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id TEXT NOT NULL REFERENCES articles(id),
  paragraph_idx INTEGER,
  feedback_type TEXT NOT NULL,
  island TEXT,
  session_id TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS implicit_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id TEXT NOT NULL,
  signal_type TEXT NOT NULL,
  paragraph_index INTEGER,
  session_id TEXT,
  island TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS article_feedback_forms (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id TEXT NOT NULL REFERENCES articles(id),
  session_id TEXT,
  island TEXT,
  helpful_score INTEGER NOT NULL,
  mode_preference TEXT NOT NULL,
  correction_paragraph_idx INTEGER,
  correction_text TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category, published_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_translations_article ON translations(article_id);
CREATE INDEX IF NOT EXISTS idx_feedback_article ON feedback(article_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_article ON implicit_signals(article_id);
CREATE INDEX IF NOT EXISTS idx_signals_created ON implicit_signals(created_at);
CREATE INDEX IF NOT EXISTS idx_article_feedback_forms_article ON article_feedback_forms(article_id);
CREATE INDEX IF NOT EXISTS idx_article_feedback_forms_created ON article_feedback_forms(created_at);
CREATE INDEX IF NOT EXISTS idx_article_feedback_forms_session ON article_feedback_forms(session_id, article_id);
CREATE INDEX IF NOT EXISTS idx_attempts_article ON translation_attempts(article_id);
CREATE INDEX IF NOT EXISTS idx_translations_collapsed ON translations(is_collapsed);
"""

SEED_SOURCES = [
    ("goal", "Goal.com"),
    ("fifa", "FIFA.com"),
    ("sky", "Sky Sports"),
]


def init_db():
    if is_d1_mode():
        conn = get_db()
        # D1: execute each statement individually (no executescript)
        for stmt in SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        for source_id, name in SEED_SOURCES:
            conn.execute(
                "INSERT OR IGNORE INTO sources (id, name) VALUES (?, ?)",
                (source_id, name),
            )
        conn.close()
        print("Database initialized on Cloudflare D1")
    else:
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
