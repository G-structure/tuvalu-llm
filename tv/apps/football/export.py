"""Export football interaction rows into normalized JSONL artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tv.common.io import write_json, write_jsonl

from .models import ExportedInteraction
from .repository import FootballInteractionRepository


def _article_payload(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": article.get("id"),
        "source_id": article.get("source_id"),
        "url": article.get("url"),
        "title_en": article.get("title_en"),
        "title_tvl": article.get("title_tvl"),
        "category": article.get("category"),
        "published_at": article.get("published_at"),
    }


def _paragraph_context(article: dict[str, Any], paragraph_idx: int | None) -> dict[str, Any]:
    if paragraph_idx is None:
        return {}
    en_paragraphs = article.get("paragraphs_en", [])
    tvl_paragraphs = article.get("paragraphs_tvl", [])
    return {
        "paragraph_en": en_paragraphs[paragraph_idx] if paragraph_idx < len(en_paragraphs) else None,
        "paragraph_tvl": tvl_paragraphs[paragraph_idx] if paragraph_idx < len(tvl_paragraphs) else None,
    }


def _record_to_dict(record: ExportedInteraction) -> dict[str, Any]:
    return record.to_dict()


def export_interactions(
    repository: FootballInteractionRepository,
    out_dir: Path,
    *,
    include_implicit: bool = True,
) -> dict[str, Any]:
    """Write normalized football interaction artifacts under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    explicit_feedback: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    implicit_signals: list[dict[str, Any]] = []
    poll_votes: list[dict[str, Any]] = []

    for row in repository.get_paragraph_feedback_rows():
        article = repository.get_article_context(row["article_id"])
        paragraph_idx = row.get("paragraph_idx")
        record = ExportedInteraction(
            id=f"feedback:{row['id']}",
            event_type="paragraph_feedback",
            training_signal_type="binary_preference",
            source_table="feedback",
            article_id=row.get("article_id"),
            paragraph_idx=paragraph_idx,
            created_at=row.get("created_at"),
            article=_article_payload(article),
            context=_paragraph_context(article, paragraph_idx),
            label={
                "feedback_type": row.get("feedback_type"),
                "preference_score": 1 if row.get("feedback_type") == "thumbs_up" else 0,
            },
            user={
                "session_id": row.get("session_id"),
                "island": row.get("island"),
            },
            metadata={},
        )
        explicit_feedback.append(_record_to_dict(record))

    table_name, rich_rows = repository.get_article_feedback_rows()
    for row in rich_rows:
        article_id = row.get("article_id")
        article = repository.get_article_context(article_id) if article_id else {}
        paragraph_idx = row.get("correction_paragraph_idx")
        if paragraph_idx is None:
            paragraph_idx = row.get("paragraph_idx")
        preferred_mode = row.get("mode_preference") or row.get("preferred_mode") or row.get("view_mode")
        helpful_score = row.get("helpful_score")
        if helpful_score is None:
            helpful_score = row.get("helpful")
        if isinstance(helpful_score, str):
            try:
                helpful_score = int(helpful_score)
            except ValueError:
                helpful_score = 1 if helpful_score.lower() in {"true", "yes", "helpful"} else 0
        helpful = None if helpful_score is None else int(helpful_score) > 0
        correction_text = row.get("correction_text") or row.get("suggested_tvl") or row.get("better_tvl")

        record = ExportedInteraction(
            id=f"{table_name}:{row['id']}",
            event_type="article_feedback",
            training_signal_type="article_preference",
            source_table=table_name or "article_feedback",
            article_id=article_id,
            paragraph_idx=paragraph_idx,
            created_at=row.get("created_at") or row.get("submitted_at"),
            article=_article_payload(article),
            context=_paragraph_context(article, paragraph_idx),
            label={
                "helpful": helpful,
                "helpful_score": helpful_score,
                "preferred_mode": preferred_mode,
                "correction_text": correction_text,
            },
            user={
                "session_id": row.get("session_id"),
                "island": row.get("island"),
            },
            metadata={
                "notes": row.get("notes"),
            },
        )
        explicit_feedback.append(_record_to_dict(record))

        if correction_text:
            correction = ExportedInteraction(
                id=f"correction:{table_name}:{row['id']}",
                event_type="correction_suggestion",
                training_signal_type="freeform_correction",
                source_table=table_name or "article_feedback",
                article_id=article_id,
                paragraph_idx=paragraph_idx,
                created_at=row.get("created_at") or row.get("submitted_at"),
                article=_article_payload(article),
                context=_paragraph_context(article, paragraph_idx),
                label={
                    "suggested_tvl": correction_text,
                    "preferred_mode": preferred_mode,
                    "helpful_score": helpful_score,
                },
                user={
                    "session_id": row.get("session_id"),
                    "island": row.get("island"),
                },
                metadata={},
            )
            corrections.append(_record_to_dict(correction))

    if include_implicit:
        for row in repository.get_implicit_signal_rows():
            article_id = row.get("article_id")
            article = repository.get_article_context(article_id) if article_id else {}
            paragraph_idx = row.get("paragraph_index")
            record = ExportedInteraction(
                id=f"implicit:{row['id']}",
                event_type="implicit_signal",
                training_signal_type="engagement_signal",
                source_table="implicit_signals",
                article_id=article_id,
                paragraph_idx=paragraph_idx,
                created_at=row.get("created_at"),
                article=_article_payload(article),
                context=_paragraph_context(article, paragraph_idx),
                label={
                    "signal_type": row.get("signal_type"),
                },
                user={
                    "session_id": row.get("session_id"),
                    "island": row.get("island"),
                },
                metadata={},
            )
            implicit_signals.append(_record_to_dict(record))

    poll_table_name, vote_rows = repository.get_poll_vote_rows()
    for row in vote_rows:
        prompt_article_id = row.get("prompt_article_id") or row.get("article_id")
        article = repository.get_article_context(prompt_article_id) if prompt_article_id else {}
        options_json = row.get("options_json")
        try:
            options = json.loads(options_json) if options_json else None
        except json.JSONDecodeError:
            options = options_json
        vote = row.get("vote") or row.get("choice") or row.get("selected_option")
        record = ExportedInteraction(
            id=f"{poll_table_name}:{row['id']}",
            event_type="football_poll_vote",
            training_signal_type="community_preference",
            source_table=poll_table_name or "poll_votes",
            article_id=prompt_article_id,
            paragraph_idx=None,
            created_at=row.get("created_at") or row.get("submitted_at"),
            article=_article_payload(article),
            context={
                "question": row.get("question") or row.get("prompt_tvl"),
                "options": options,
            },
            label={
                "vote": vote,
            },
            user={
                "session_id": row.get("session_id"),
                "island": row.get("island"),
            },
            metadata={
                "prompt_id": row.get("poll_id") or row.get("prompt_id"),
                "opens_at": row.get("opens_at"),
                "closes_at": row.get("closes_at"),
            },
        )
        poll_votes.append(_record_to_dict(record))

    write_jsonl(out_dir / "explicit_feedback.jsonl", explicit_feedback)
    write_jsonl(out_dir / "corrections.jsonl", corrections)
    write_jsonl(out_dir / "implicit_signals.jsonl", implicit_signals)
    write_jsonl(out_dir / "football_polls.jsonl", poll_votes)

    manifest = {
        "out_dir": str(out_dir),
        "counts": {
            "explicit_feedback": len(explicit_feedback),
            "corrections": len(corrections),
            "implicit_signals": len(implicit_signals),
            "football_polls": len(poll_votes),
        },
        "files": {
            "explicit_feedback": "explicit_feedback.jsonl",
            "corrections": "corrections.jsonl",
            "implicit_signals": "implicit_signals.jsonl",
            "football_polls": "football_polls.jsonl",
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
