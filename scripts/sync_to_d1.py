"""Sync local football.db to Cloudflare D1.

Exports articles and translations from local SQLite and pushes them to D1
via wrangler CLI. Designed to run after the scraper/translator pipeline
in GitHub Actions.

Usage:
    python scripts/sync_to_d1.py
    python scripts/sync_to_d1.py --dry-run
"""

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"
D1_DATABASE = "talafutipolo"


def escape_sql(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value).replace("'", "''")
    return f"'{s}'"


def build_upsert_sql(table: str, rows: list[dict], conflict_col: str = "id") -> list[str]:
    """Build INSERT OR REPLACE statements for a list of rows."""
    if not rows:
        return []
    statements = []
    for row in rows:
        cols = ", ".join(row.keys())
        vals = ", ".join(escape_sql(v) for v in row.values())
        statements.append(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({vals});")
    return statements


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _optional_rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    if not table_exists(conn, table):
        return []
    return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]


def get_local_data(conn: sqlite3.Connection) -> dict:
    """Read all data from local DB that needs syncing."""
    conn.row_factory = sqlite3.Row

    sources = [dict(r) for r in conn.execute("SELECT * FROM sources").fetchall()]
    articles = [dict(r) for r in conn.execute("SELECT * FROM articles").fetchall()]
    translations = [dict(r) for r in conn.execute("SELECT * FROM translations").fetchall()]
    feedback = _optional_rows(conn, "feedback")
    implicit_signals = _optional_rows(conn, "implicit_signals")
    article_feedback_forms = _optional_rows(conn, "article_feedback_forms")

    return {
        "sources": sources,
        "articles": articles,
        "translations": translations,
        "feedback": feedback,
        "implicit_signals": implicit_signals,
        "article_feedback_forms": article_feedback_forms,
    }


def build_sync_sql(data: dict) -> str:
    """Build full SQL for syncing to D1."""
    lines = ["PRAGMA foreign_keys = OFF;"]

    # Sources first (FK parent)
    lines.extend(build_upsert_sql("sources", data["sources"]))

    # Articles (FK parent for translations)
    lines.extend(build_upsert_sql("articles", data["articles"]))

    # Translations
    for row in data["translations"]:
        # Use article_id as conflict key (unique index)
        cols = ", ".join(row.keys())
        vals = ", ".join(escape_sql(v) for v in row.values())
        lines.append(f"INSERT OR REPLACE INTO translations ({cols}) VALUES ({vals});")

    lines.extend(build_upsert_sql("feedback", data["feedback"]))
    lines.extend(build_upsert_sql("implicit_signals", data["implicit_signals"]))
    lines.extend(build_upsert_sql("article_feedback_forms", data["article_feedback_forms"]))

    lines.append("PRAGMA foreign_keys = ON;")
    return "\n".join(lines)


def execute_d1(sql_file: Path, dry_run: bool = False) -> bool:
    """Execute SQL file against D1 via wrangler."""
    if dry_run:
        print(f"[DRY RUN] Would execute {sql_file} ({sql_file.stat().st_size} bytes)")
        return True

    result = subprocess.run(
        [
            "npx", "wrangler", "d1", "execute", D1_DATABASE,
            "--remote", f"--file={sql_file}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"D1 sync failed:\n{result.stderr}", file=sys.stderr)
        return False

    print(result.stdout)
    return True


def main():
    parser = argparse.ArgumentParser(description="Sync football.db to Cloudflare D1")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    data = get_local_data(conn)
    conn.close()

    print(
        f"Local DB: {len(data['sources'])} sources, "
        f"{len(data['articles'])} articles, "
        f"{len(data['translations'])} translations, "
        f"{len(data['feedback'])} paragraph feedback, "
        f"{len(data['implicit_signals'])} implicit signals, "
        f"{len(data['article_feedback_forms'])} coaching forms"
    )

    sql = build_sync_sql(data)
    statement_count = sql.count(";")
    print(f"Generated {statement_count} SQL statements")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write(sql)
        sql_path = Path(f.name)

    print(f"SQL file: {sql_path} ({sql_path.stat().st_size} bytes)")

    if args.dry_run:
        # Print first 20 lines as preview
        for line in sql.split("\n")[:20]:
            print(f"  {line}")
        if statement_count > 20:
            print(f"  ... ({statement_count - 20} more statements)")
        return

    success = execute_d1(sql_path)
    sql_path.unlink(missing_ok=True)

    if not success:
        sys.exit(1)

    print("D1 sync complete!")


if __name__ == "__main__":
    main()
