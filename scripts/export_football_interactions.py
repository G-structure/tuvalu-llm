#!/usr/bin/env python3
"""Export football interactions into normalized JSONL artifacts.

Usage:
    uv run python scripts/export_football_interactions.py
    uv run python scripts/export_football_interactions.py --output-dir out/football_interactions
    CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... \
        uv run python scripts/export_football_interactions.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tv.apps.football.db import default_db_path
from tv.apps.football.export import export_interactions
from tv.apps.football.repository import FootballInteractionRepository


def _get_connection(db_path: Path):
    """Use the existing env-based DB selection when available."""
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from db_conn import get_db  # type: ignore

    conn = get_db()

    # Local sqlite mode should honor an explicit db_path override.
    if hasattr(conn, "execute") and db_path != default_db_path():
        import sqlite3

        conn.close()
        sqlite_conn = sqlite3.connect(str(db_path))
        sqlite_conn.row_factory = sqlite3.Row
        return sqlite_conn
    return conn


def main() -> None:
    parser = argparse.ArgumentParser(description="Export football interactions to normalized JSONL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "football" / "exports" / "interactions",
        help="Directory where manifest.json and JSONL artifacts will be written.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=default_db_path(),
        help="Local SQLite football DB path. Ignored when D1 env vars are set unless it differs from the default path.",
    )
    parser.add_argument(
        "--skip-implicit",
        action="store_true",
        help="Exclude implicit engagement signals from the export.",
    )
    args = parser.parse_args()

    conn = _get_connection(args.db_path)
    try:
        repo = FootballInteractionRepository(conn)
        manifest = export_interactions(
            repo,
            args.output_dir,
            include_implicit=not args.skip_implicit,
        )
    finally:
        if hasattr(conn, "close"):
            conn.close()

    counts = manifest["counts"]
    print(f"Wrote football interaction export to {args.output_dir}")
    print(
        "  explicit_feedback={explicit_feedback} corrections={corrections} "
        "implicit_signals={implicit_signals} football_polls={football_polls}".format(**counts)
    )


if __name__ == "__main__":
    main()
