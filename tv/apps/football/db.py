"""Football app DB helpers shared by export/repository code."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tv.common.config import get_repo_root

DEFAULT_DB_PATH = get_repo_root() / "data" / "football" / "football.db"


def row_to_dict(row: Any) -> dict[str, Any]:
    """Convert sqlite3.Row / D1Row / dict-like rows to a plain dict."""
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    raise TypeError(f"Unsupported row type: {type(row)!r}")


def fetch_all(conn: Any, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Execute a query and normalize the result rows to dicts."""
    cursor = conn.execute(sql, params)
    return [row_to_dict(row) for row in cursor.fetchall()]


def fetch_one(conn: Any, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    """Execute a query and normalize a single row to a dict."""
    cursor = conn.execute(sql, params)
    row = cursor.fetchone()
    return row_to_dict(row) if row is not None else None


def table_exists(conn: Any, table_name: str) -> bool:
    """Return True when *table_name* exists in the current SQLite/D1 DB."""
    row = fetch_one(
        conn,
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return row is not None


def first_existing_table(conn: Any, table_names: list[str]) -> str | None:
    """Return the first existing table from *table_names*, else None."""
    for name in table_names:
        if table_exists(conn, name):
            return name
    return None


_PARAGRAPH_RE = re.compile(r"<p[^>]*>([\s\S]*?)</p>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def split_paragraphs(body: str | None) -> list[str]:
    """Split article text into display-aligned paragraphs."""
    if not body:
        return []
    if "<p" in body.lower():
        matches = _PARAGRAPH_RE.findall(body)
        if matches:
            return [
                _TAG_RE.sub("", match).strip()
                for match in matches
                if _TAG_RE.sub("", match).strip()
            ]
    return [paragraph.strip() for paragraph in re.split(r"\n\n+", body) if paragraph.strip()]


def default_db_path() -> Path:
    """Return the canonical local football DB path."""
    return DEFAULT_DB_PATH
