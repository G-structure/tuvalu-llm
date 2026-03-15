"""Tests for football interaction export artifacts."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tv.apps.football.export import export_interactions
from tv.apps.football.repository import FootballInteractionRepository


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE articles (
          id TEXT PRIMARY KEY,
          source_id TEXT NOT NULL,
          url TEXT NOT NULL,
          title_en TEXT NOT NULL,
          body_en TEXT NOT NULL,
          published_at TEXT NOT NULL,
          category TEXT
        );
        CREATE TABLE translations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          article_id TEXT NOT NULL,
          title_tvl TEXT,
          body_tvl TEXT
        );
        CREATE TABLE feedback (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          article_id TEXT NOT NULL,
          paragraph_idx INTEGER,
          feedback_type TEXT NOT NULL,
          island TEXT,
          session_id TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE implicit_signals (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          article_id TEXT NOT NULL,
          signal_type TEXT NOT NULL,
          paragraph_index INTEGER,
          session_id TEXT,
          island TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE article_feedback_forms (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          article_id TEXT NOT NULL,
          session_id TEXT,
          island TEXT,
          helpful_score INTEGER NOT NULL,
          mode_preference TEXT NOT NULL,
          correction_paragraph_idx INTEGER,
          correction_text TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE football_polls (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          article_id TEXT,
          question TEXT,
          prompt_tvl TEXT,
          options_json TEXT,
          opens_at TEXT,
          closes_at TEXT
        );
        CREATE TABLE football_poll_votes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          poll_id INTEGER NOT NULL,
          article_id TEXT,
          vote TEXT NOT NULL,
          island TEXT,
          session_id TEXT,
          created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO articles (id, source_id, url, title_en, body_en, published_at, category)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "goal-1",
            "goal",
            "https://example.com/goal-1",
            "Arsenal edge Liverpool in thriller",
            "First paragraph.\n\nSecond paragraph.",
            "2026-03-15T00:00:00Z",
            "premier-league",
        ),
    )
    conn.execute(
        """
        INSERT INTO translations (article_id, title_tvl, body_tvl)
        VALUES (?, ?, ?)
        """,
        (
            "goal-1",
            "Ko manumalo a Arsenal i se tafaoga faigata",
            "Palakalafa muamua.\n\nPalakalafa lua.",
        ),
    )
    conn.execute(
        """
        INSERT INTO feedback (article_id, paragraph_idx, feedback_type, island, session_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("goal-1", 1, "thumbs_up", "Funafuti", "sess-1", "2026-03-15T01:00:00Z"),
    )
    conn.execute(
        """
        INSERT INTO implicit_signals (article_id, signal_type, paragraph_index, session_id, island, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("goal-1", "reveal", 0, "sess-1", "Funafuti", "2026-03-15T01:05:00Z"),
    )
    conn.execute(
        """
        INSERT INTO article_feedback_forms
          (article_id, session_id, island, helpful_score, mode_preference, correction_paragraph_idx, correction_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "goal-1",
            "sess-2",
            "Nui",
            1,
            "tv+en",
            0,
            "Palakalafa muamua e mafai o sili atu i te faigofie.",
            "2026-03-15T02:00:00Z",
        ),
    )
    conn.execute(
        """
        INSERT INTO football_polls (article_id, question, prompt_tvl, options_json, opens_at, closes_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "goal-1",
            "Who wins the rematch?",
            "Ko oi ka manumalo i te toe tafaoga?",
            json.dumps(["Arsenal", "Liverpool", "Draw"]),
            "2026-03-15T00:00:00Z",
            "2026-03-16T00:00:00Z",
        ),
    )
    conn.execute(
        """
        INSERT INTO football_poll_votes (poll_id, article_id, vote, island, session_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (1, "goal-1", "Arsenal", "Nanumea", "sess-3", "2026-03-15T03:00:00Z"),
    )
    return conn


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_export_writes_all_artifacts(tmp_path: Path):
    conn = _make_db()
    repo = FootballInteractionRepository(conn)

    manifest = export_interactions(repo, tmp_path)

    assert manifest["counts"] == {
        "explicit_feedback": 2,
        "corrections": 1,
        "implicit_signals": 1,
        "football_polls": 1,
    }
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "explicit_feedback.jsonl").exists()
    assert (tmp_path / "corrections.jsonl").exists()
    assert (tmp_path / "implicit_signals.jsonl").exists()
    assert (tmp_path / "football_polls.jsonl").exists()


def test_export_records_include_article_context(tmp_path: Path):
    conn = _make_db()
    repo = FootballInteractionRepository(conn)
    export_interactions(repo, tmp_path)

    explicit = _read_jsonl(tmp_path / "explicit_feedback.jsonl")
    paragraph_vote = explicit[0]
    assert paragraph_vote["event_type"] == "paragraph_feedback"
    assert paragraph_vote["article"]["title_en"] == "Arsenal edge Liverpool in thriller"
    assert paragraph_vote["context"]["paragraph_en"] == "Second paragraph."
    assert paragraph_vote["context"]["paragraph_tvl"] == "Palakalafa lua."

    article_feedback = explicit[1]
    assert article_feedback["event_type"] == "article_feedback"
    assert article_feedback["label"]["preferred_mode"] == "tv+en"

    corrections = _read_jsonl(tmp_path / "corrections.jsonl")
    assert corrections[0]["training_signal_type"] == "freeform_correction"
    assert corrections[0]["label"]["suggested_tvl"].startswith("Palakalafa muamua")

    polls = _read_jsonl(tmp_path / "football_polls.jsonl")
    assert polls[0]["context"]["options"] == ["Arsenal", "Liverpool", "Draw"]
    assert polls[0]["label"]["vote"] == "Arsenal"


def test_export_can_skip_implicit_signals(tmp_path: Path):
    conn = _make_db()
    repo = FootballInteractionRepository(conn)
    manifest = export_interactions(repo, tmp_path, include_implicit=False)

    implicit = _read_jsonl(tmp_path / "implicit_signals.jsonl")
    assert implicit == []
    assert manifest["counts"]["implicit_signals"] == 0
