"""Repository helpers for football interaction export."""

from __future__ import annotations

from typing import Any

from .db import fetch_all, fetch_one, first_existing_table, split_paragraphs

ARTICLE_FEEDBACK_TABLES = [
    "article_feedback_forms",
    "article_feedback_submissions",
    "article_feedback",
    "translation_coach_submissions",
]

POLL_TABLE_CANDIDATES = [
    ("football_poll_votes", "football_polls", "poll_id"),
    ("poll_votes", "poll_questions", "poll_id"),
    ("match_prediction_votes", "match_prediction_prompts", "prompt_id"),
]


class FootballInteractionRepository:
    """Read football interaction rows and enrich them with article context."""

    def __init__(self, conn: Any):
        self.conn = conn
        self._article_cache: dict[str, dict[str, Any]] = {}

    def get_article_context(self, article_id: str) -> dict[str, Any]:
        """Return article metadata plus paragraph arrays for *article_id*."""
        if article_id in self._article_cache:
            return self._article_cache[article_id]

        row = fetch_one(
            self.conn,
            """
            SELECT
              a.id,
              a.source_id,
              a.url,
              a.title_en,
              a.body_en,
              a.category,
              a.published_at,
              t.title_tvl,
              t.body_tvl
            FROM articles a
            LEFT JOIN translations t ON t.article_id = a.id
            WHERE a.id = ?
            """,
            (article_id,),
        )
        if row is None:
            context = {
                "id": article_id,
                "source_id": None,
                "url": None,
                "title_en": None,
                "title_tvl": None,
                "category": None,
                "published_at": None,
                "paragraphs_en": [],
                "paragraphs_tvl": [],
            }
        else:
            context = {
                "id": row["id"],
                "source_id": row.get("source_id"),
                "url": row.get("url"),
                "title_en": row.get("title_en"),
                "title_tvl": row.get("title_tvl"),
                "category": row.get("category"),
                "published_at": row.get("published_at"),
                "paragraphs_en": split_paragraphs(row.get("body_en")),
                "paragraphs_tvl": split_paragraphs(row.get("body_tvl")),
            }
        self._article_cache[article_id] = context
        return context

    def get_paragraph_feedback_rows(self) -> list[dict[str, Any]]:
        """Return existing explicit paragraph feedback rows."""
        return fetch_all(
            self.conn,
            """
            SELECT id, article_id, paragraph_idx, feedback_type, island, session_id, created_at
            FROM feedback
            ORDER BY id
            """,
        )

    def get_implicit_signal_rows(self) -> list[dict[str, Any]]:
        """Return implicit engagement signal rows."""
        return fetch_all(
            self.conn,
            """
            SELECT id, article_id, signal_type, paragraph_index, island, session_id, created_at
            FROM implicit_signals
            ORDER BY id
            """,
        )

    def get_article_feedback_rows(self) -> tuple[str | None, list[dict[str, Any]]]:
        """Return richer article-level feedback rows when an interaction table exists."""
        table_name = first_existing_table(self.conn, ARTICLE_FEEDBACK_TABLES)
        if not table_name:
            return None, []

        return table_name, fetch_all(
            self.conn,
            f"""
            SELECT *
            FROM {table_name}
            ORDER BY id
            """,
        )

    def get_poll_vote_rows(self) -> tuple[str | None, list[dict[str, Any]]]:
        """Return poll/prediction votes joined with prompt metadata when present."""
        for vote_table, prompt_table, prompt_fk in POLL_TABLE_CANDIDATES:
            if not first_existing_table(self.conn, [vote_table]):
                continue
            if first_existing_table(self.conn, [prompt_table]):
                rows = fetch_all(
                    self.conn,
                    f"""
                    SELECT
                      v.*,
                      p.question,
                      p.prompt_tvl,
                      p.options_json,
                      p.article_id AS prompt_article_id,
                      p.opens_at,
                      p.closes_at
                    FROM {vote_table} v
                    LEFT JOIN {prompt_table} p ON p.id = v.{prompt_fk}
                    ORDER BY v.id
                    """,
                )
            else:
                rows = fetch_all(
                    self.conn,
                    f"""
                    SELECT *
                    FROM {vote_table}
                    ORDER BY id
                    """,
                )
            return vote_table, rows
        return None, []
