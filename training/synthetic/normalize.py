"""Normalization utilities for converting raw dataset rows into common schema."""

from __future__ import annotations

from typing import Any

from training.common.schema import TASK_FAMILIES, TaskFamily, make_example


def normalize_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Coerce raw messages into [{role, content}, ...] form.

    Handles common variations: "text" vs "content", missing roles, etc.
    """
    out: list[dict[str, str]] = []
    for msg in raw_messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content") or msg.get("text") or msg.get("value") or ""
        out.append({"role": role, "content": str(content)})
    return out


def infer_task_family(
    messages: list[dict[str, str]],
    metadata: dict[str, Any],
) -> TaskFamily:
    """Heuristic task family inference from message content and metadata.

    Falls back to the family stored in metadata["task_family"] if present.
    """
    if "task_family" in metadata and metadata["task_family"] in TASK_FAMILIES:
        return metadata["task_family"]

    full_text = " ".join(m.get("content", "") for m in messages).lower()

    if any(kw in full_text for kw in ("def ", "function ", "```python", "```java", "```js")):
        return "code"
    if any(kw in full_text for kw in ("solve", "calculate", "equation", "answer is")):
        return "math"
    if "tool_call" in full_text or "function_call" in full_text:
        return "tool"
    if any(kw in full_text for kw in ("summarize", "summary", "highlights")):
        return "summarization"
    if any(kw in full_text for kw in ("translate", "translation")):
        return "translation"
    return "chat"


def generate_translate_mask(
    messages: list[dict[str, str]],
    task_family: TaskFamily,
) -> list[dict[str, Any]]:
    """Generate per-message translate_mask annotations.

    Rules:
    - system messages: translate=False (prompts are structural)
    - user messages: translate=True (natural language)
    - assistant messages: depends on task family
      - code/tool: translate="selective" (may contain code/JSON)
      - others: translate=True
    - tool messages: translate=False (preserve exactly)
    """
    mask: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            mask.append({"translate": False})
        elif role == "tool":
            mask.append({"translate": False})
        elif role == "user":
            mask.append({"translate": True})
        elif role == "assistant":
            if task_family in ("code", "tool"):
                mask.append({"translate": "selective"})
            else:
                mask.append({"translate": True})
        else:
            mask.append({"translate": True})
    return mask


def strip_metadata_for_training(example: dict[str, Any]) -> dict[str, Any]:
    """Remove internal-only metadata fields before writing training JSONL.

    Keeps: id, task_family, messages.
    Removes: metadata, translate_mask (used only during pipeline processing).
    """
    return {
        "id": example["id"],
        "task_family": example["task_family"],
        "messages": example["messages"],
    }
