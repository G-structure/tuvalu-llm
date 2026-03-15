"""Selective translation engine for Stage B synthetic data.

Translates human-language spans while preserving machine-parseable content
(code, JSON, URLs, placeholders, etc.) using a mask-translate-unmask pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Placeholder masking
# ---------------------------------------------------------------------------

# Ordered by priority (earlier patterns take precedence via non-overlapping spans).
_SPAN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Fenced code blocks (``` with optional language tag)
    ("code_fence", re.compile(r"```[\s\S]*?```", re.DOTALL)),
    # Inline code
    ("inline_code", re.compile(r"`[^`\n]+`")),
    # LaTeX display math ($$...$$) — before single $
    ("latex_display", re.compile(r"\$\$[\s\S]+?\$\$")),
    # LaTeX inline math ($...$) — only if content looks math-like
    ("latex_inline", re.compile(r"\$(?=[^\s$])(?:[^$\\]|\\.)+\$")),
    # LaTeX bracket math \[...\]
    ("latex_bracket", re.compile(r"\\\[[\s\S]+?\\\]")),
    # XML/HTML blocks: <tag ...>...</tag>
    ("xml_block", re.compile(
        r"<(?P<tag>[a-zA-Z_][\w.-]*)"   # opening tag name
        r"(?:\s[^>]*)?"                  # optional attributes
        r">"                             # close opening tag
        r"[\s\S]*?"                      # content (non-greedy)
        r"</(?P=tag)>"                   # matching closing tag
    )),
    # Self-closing XML/HTML tags
    ("xml_self", re.compile(r"<[a-zA-Z_][\w.-]*(?:\s[^>]*)?\s*/>")),
    # JSON objects — balanced braces heuristic (top level only)
    ("json_obj", re.compile(
        r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    )),
    # JSON arrays — balanced brackets heuristic
    ("json_arr", re.compile(
        r'\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]'
    )),
    # URLs
    ("url", re.compile(r"(?:https?|ftp|file)://[^\s<>\"')}\]]+", re.ASCII)),
    # File paths (absolute, relative, home)
    ("filepath", re.compile(
        r"(?:~|\.\.?)?/[A-Za-z0-9_./-]+(?:\.[A-Za-z0-9]+)?"
    )),
    # Template/format placeholders: {{var}}, {name}, %(name)s, %s, %d, ${var}
    ("placeholder", re.compile(
        r"\{\{[\w.]+\}\}"            # {{var}}
        r"|\{[\w.]+\}"              # {name}
        r"|%\([\w.]+\)[sd]"         # %(name)s
        r"|%[sd]"                   # %s, %d
        r"|\$\{[\w.]+\}"           # ${var}
        r"|<[\w.]+>"               # <id> — angle-bracket placeholders
    )),
    # Shell command lines (start with $ or >)
    ("shell_cmd", re.compile(r"^[\$>]\s+.+$", re.MULTILINE)),
    # Function/method calls: identifier( or identifier.identifier(
    ("func_call", re.compile(
        r"\b[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)*\s*\("
    )),
    # snake_case or camelCase identifiers (standalone, at least 2 segments)
    ("identifier", re.compile(
        r"\b(?:[a-z]+_[a-z_]+[a-z]"     # snake_case
        r"|[a-z]+[A-Z][a-zA-Z]+)\b"     # camelCase
    )),
    # Numbers with units
    ("number_unit", re.compile(
        r"\b\d+(?:\.\d+)?(?:px|em|rem|pt|%|ms|s|kb|mb|gb|tb)\b", re.IGNORECASE
    )),
    # Standalone monetary amounts
    ("money", re.compile(r"\$\d+(?:\.\d+)?")),
]


def mask_protected_spans(text: str) -> tuple[str, dict[str, str]]:
    """Replace protected (machine-parseable) spans with unique placeholders.

    Returns (masked_text, placeholder_map) where placeholder_map maps
    __PH_NNN__ -> original span text.
    """
    # Collect all non-overlapping spans ordered by start position.
    occupied: list[tuple[int, int]] = []
    raw_spans: list[tuple[int, int, str]] = []

    for _name, pattern in _SPAN_PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            # Skip if overlaps with any already-claimed span.
            if any(s < end and start < e for s, e in occupied):
                continue
            occupied.append((start, end))
            raw_spans.append((start, end, m.group()))

    # Sort by position for deterministic replacement.
    raw_spans.sort(key=lambda t: t[0])

    placeholder_map: dict[str, str] = {}
    parts: list[str] = []
    prev = 0

    for idx, (start, end, original) in enumerate(raw_spans):
        ph = f"__PH_{idx:03d}__"
        placeholder_map[ph] = original
        parts.append(text[prev:start])
        parts.append(ph)
        prev = end

    parts.append(text[prev:])
    masked = "".join(parts)
    return masked, placeholder_map


def unmask_protected_spans(text: str, placeholder_map: dict[str, str]) -> str:
    """Restore all __PH_NNN__ placeholders back to original spans.

    Handles minor formatting changes that translation might introduce
    (e.g., extra spaces around placeholders).
    """
    for ph, original in placeholder_map.items():
        # Try exact match first, then a fuzzy match that allows
        # surrounding whitespace changes.
        if ph in text:
            text = text.replace(ph, original)
        else:
            # Translation may have mangled the placeholder slightly:
            # e.g., added spaces inside underscores.  Try a regex.
            escaped = re.escape(ph)
            # Allow optional spaces around the inner marker.
            fuzzy = escaped.replace(r"\_\_", r"_\s*_")
            text = re.sub(fuzzy, original, text)
    return text


# ---------------------------------------------------------------------------
# Message content classification
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>|\"function\":\s*\{", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```")
_TOOL_SCHEMA_RE = re.compile(
    r'"type"\s*:\s*"function"'
    r'|"parameters"\s*:\s*\{'
    r'|"tools"\s*:\s*\[',
)


def classify_message_content(
    content: str,
    role: str,
    task_family: str,
) -> str:
    """Classify a message as 'translate', 'preserve', or 'selective'.

    Rules follow the spec:
    - role=tool -> always preserve
    - role=system -> translate (unless tool schema definition)
    - role=user -> translate (selective masking applied later)
    - role=assistant with tool calls -> preserve
    - role=assistant with code blocks -> selective
    - role=assistant with only natural language -> translate
    """
    if role == "tool":
        return "preserve"

    if role == "system":
        if _TOOL_SCHEMA_RE.search(content):
            return "preserve"
        return "translate"

    if role == "user":
        # Users may embed code snippets; masking handles those.
        if _CODE_BLOCK_RE.search(content):
            return "selective"
        return "translate"

    if role == "assistant":
        if _TOOL_CALL_RE.search(content):
            return "preserve"
        if _CODE_BLOCK_RE.search(content):
            return "selective"
        return "translate"

    # Unknown role: preserve to be safe.
    return "preserve"


# ---------------------------------------------------------------------------
# Message-level selective translation
# ---------------------------------------------------------------------------

TranslateFn = Callable[[str], str]


def selective_translate_message(
    message: dict[str, Any],
    translate_fn: TranslateFn,
    task_family: str,
) -> dict[str, Any]:
    """Translate a single message dict, preserving machine-parseable content.

    Args:
        message: Dict with at least 'role' and 'content'.
        translate_fn: (text: str) -> str  synchronous translation function.
        task_family: One of the TASK_FAMILIES.

    Returns:
        A new message dict with translated content where appropriate.
    """
    result = dict(message)
    content = message.get("content", "")
    role = message.get("role", "")

    if not content or not isinstance(content, str):
        return result

    action = classify_message_content(content, role, task_family)

    if action == "preserve":
        return result

    if action == "translate":
        result["content"] = translate_fn(content)
        return result

    # action == "selective"
    masked, ph_map = mask_protected_spans(content)
    translated = translate_fn(masked)
    result["content"] = unmask_protected_spans(translated, ph_map)
    return result


def selective_translate_example(
    example: dict[str, Any],
    translate_fn: TranslateFn,
    tool_mode: str = "safe",
) -> dict[str, Any]:
    """Translate a full normalized example, preserving structure.

    Args:
        example: A normalized example (see tv.common.schema).
        translate_fn: Synchronous translation function.
        tool_mode: 'safe' (default) preserves tool-call JSON in chat format;
                   'native' is experimental and currently treated as 'safe'.

    Returns:
        A new example dict with translated messages.
    """
    task_family = example.get("task_family", "chat")
    translate_mask = example.get("translate_mask")

    translated_messages: list[dict[str, Any]] = []
    preservation_metadata: dict[str, Any] = {}

    for i, msg in enumerate(example.get("messages", [])):
        # Use explicit mask if available, otherwise fall back to heuristic
        if translate_mask and i < len(translate_mask):
            mask_entry = translate_mask[i]
            mask_action = mask_entry.get("translate", True)

            if mask_action is False:
                translated_messages.append(dict(msg))
                continue
            elif mask_action is True:
                result_msg = dict(msg)
                content = msg.get("content", "")
                if content and isinstance(content, str):
                    result_msg["content"] = translate_fn(content)
                translated_messages.append(result_msg)
                continue
            elif mask_action == "selective":
                result_msg = dict(msg)
                content = msg.get("content", "")
                if content and isinstance(content, str):
                    masked, ph_map = mask_protected_spans(content)
                    translated = translate_fn(masked)
                    result_msg["content"] = unmask_protected_spans(translated, ph_map)
                    if ph_map:
                        preservation_metadata[f"msg_{i}_placeholders"] = len(ph_map)
                        ph_types = list(set(
                            k.split("_")[0] for k in list(ph_map.values())[:5] if "_" in k
                        ))
                        preservation_metadata[f"msg_{i}_placeholder_types"] = (
                            ph_types if len(ph_map) <= 20 else ["many"]
                        )
                translated_messages.append(result_msg)
                continue

        # Fallback to heuristic classification
        translated_messages.append(
            selective_translate_message(msg, translate_fn, task_family)
        )

    result = dict(example)
    result["messages"] = translated_messages

    # Add provenance metadata.
    meta = dict(result.get("metadata", {}))
    meta["selectively_translated"] = True
    meta["tool_mode"] = tool_mode
    if preservation_metadata:
        meta["preservation"] = preservation_metadata
    result["metadata"] = meta

    return result
