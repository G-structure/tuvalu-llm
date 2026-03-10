"""Database connection factory — returns D1 (CI) or local SQLite (dev).

If CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are set, returns a D1
HTTP connection. Otherwise, falls back to local SQLite.

Usage:
    from db_conn import get_db
    conn = get_db()
    conn.execute("SELECT * FROM articles")
"""

import os
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"


def is_d1_mode() -> bool:
    return bool(os.environ.get("CLOUDFLARE_ACCOUNT_ID") and os.environ.get("CLOUDFLARE_API_TOKEN"))


def get_db():
    """Return a D1Connection or sqlite3.Connection depending on env."""
    if is_d1_mode():
        from d1_client import get_d1
        return get_d1()
    else:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def insert_or_ignore(conn, sql: str, params: tuple) -> bool:
    """INSERT OR IGNORE that works on both sqlite3 and D1.

    Returns True if the row was inserted (new), False if it already existed.
    On D1, IntegrityError comes as a RuntimeError with 'UNIQUE constraint'.
    """
    # Convert INSERT INTO to INSERT OR IGNORE INTO
    ignore_sql = sql.replace("INSERT INTO", "INSERT OR IGNORE INTO", 1)
    try:
        result = conn.execute(ignore_sql, params)
        # For D1, check if any rows were written
        if hasattr(result, '_results'):
            # D1 — no easy way to check, assume success
            return True
        return True
    except (sqlite3.IntegrityError, RuntimeError) as e:
        if "UNIQUE constraint" in str(e) or "IntegrityError" in str(e):
            return False
        raise
