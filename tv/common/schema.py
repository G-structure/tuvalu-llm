"""Normalized example schema for the training pipeline.

Every dataset (parallel MT, synthetic, capability) gets normalized to this
schema before being used in any training stage.
"""

from __future__ import annotations

from typing import Any, Literal

TASK_FAMILIES = ("chat", "tool", "math", "code", "qa", "summarization", "translation")

TaskFamily = Literal["chat", "tool", "math", "code", "qa", "summarization", "translation"]


def make_example(
    *,
    id: str,
    task_family: TaskFamily,
    messages: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
    translate_mask: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a normalized training example.

    Args:
        id: Unique example identifier.
        task_family: One of TASK_FAMILIES.
        messages: Chat-format messages list (role + content).
        metadata: Arbitrary metadata (source dataset, direction, etc.).
        translate_mask: Optional per-message annotation of which spans to
            translate vs preserve. Used by the selective translation engine.
    """
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"Unknown task_family: {task_family!r}")
    ex: dict[str, Any] = {
        "id": id,
        "task_family": task_family,
        "messages": messages,
        "metadata": metadata or {},
    }
    if translate_mask is not None:
        ex["translate_mask"] = translate_mask
    return ex


def validate_example(ex: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []
    if not ex.get("id"):
        errors.append("missing id")
    if ex.get("task_family") not in TASK_FAMILIES:
        errors.append(f"invalid task_family: {ex.get('task_family')!r}")
    messages = ex.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        errors.append("messages must be a non-empty list")
    else:
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"messages[{i}] not a dict")
            elif "role" not in msg or "content" not in msg:
                errors.append(f"messages[{i}] missing role or content")
    return errors
