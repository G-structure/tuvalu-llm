"""Normalized football interaction export models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ExportedInteraction:
    """Single normalized interaction record ready for JSONL export."""

    id: str
    event_type: str
    training_signal_type: str
    source_table: str
    article_id: str | None
    paragraph_idx: int | None
    created_at: str | None
    article: dict[str, Any]
    context: dict[str, Any]
    label: dict[str, Any]
    user: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)
