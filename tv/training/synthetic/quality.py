"""Validation pipeline for selectively translated examples.

Checks structural integrity, placeholder restoration, code preservation,
JSON validity, and length sanity. Collects rejected samples with reasons.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tv.common.io import append_jsonl

# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

_PH_LEAK_RE = re.compile(r"__PH_\d{3}__")
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)


def check_placeholder_leaks(text: str) -> list[str]:
    """Find any unreplaced __PH_NNN__ markers in the text."""
    return _PH_LEAK_RE.findall(text)


def validate_code_preservation(original: str, translated: str) -> bool:
    """Check that code blocks in translated text are byte-identical to source."""
    orig_blocks = _CODE_FENCE_RE.findall(original)
    trans_blocks = _CODE_FENCE_RE.findall(translated)
    if len(orig_blocks) != len(trans_blocks):
        return False
    return all(a == b for a, b in zip(orig_blocks, trans_blocks))


def validate_json_preservation(original_json: str, translated_json: str) -> bool:
    """Check that JSON structure is preserved (same keys, valid JSON)."""
    try:
        orig = json.loads(original_json)
        trans = json.loads(translated_json)
    except (json.JSONDecodeError, TypeError):
        return False
    return _same_structure(orig, trans)


def _same_structure(a: Any, b: Any) -> bool:
    """Recursively check that two JSON values have the same structure (keys/types)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_same_structure(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_same_structure(x, y) for x, y in zip(a, b))
    return True


# ---------------------------------------------------------------------------
# Full example validation
# ---------------------------------------------------------------------------

# Length ratio bounds: translated text should be within these bounds of source.
_MIN_LENGTH_RATIO = 0.3
_MAX_LENGTH_RATIO = 3.0


def validate_translation(
    original: dict[str, Any],
    translated: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate a translated example against its original.

    Returns (accepted, list_of_reasons_if_rejected).
    """
    reasons: list[str] = []

    orig_msgs = original.get("messages", [])
    trans_msgs = translated.get("messages", [])

    # Structural integrity: same number of messages, same roles.
    if len(orig_msgs) != len(trans_msgs):
        reasons.append(
            f"message_count_mismatch: {len(orig_msgs)} vs {len(trans_msgs)}"
        )
        # Can't do per-message checks if counts differ.
        return False, reasons

    for i, (om, tm) in enumerate(zip(orig_msgs, trans_msgs)):
        if om.get("role") != tm.get("role"):
            reasons.append(
                f"role_mismatch[{i}]: {om.get('role')} vs {tm.get('role')}"
            )

    # Per-message checks.
    for i, (om, tm) in enumerate(zip(orig_msgs, trans_msgs)):
        oc = om.get("content", "")
        tc = tm.get("content", "")

        if not isinstance(oc, str) or not isinstance(tc, str):
            continue

        # Placeholder leak check.
        leaks = check_placeholder_leaks(tc)
        if leaks:
            reasons.append(f"placeholder_leak[{i}]: {leaks}")

        # Code block preservation.
        if not validate_code_preservation(oc, tc):
            reasons.append(f"code_block_mismatch[{i}]")

        # JSON validity: any JSON in translated output should still parse.
        json_re = re.compile(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}')
        for m in json_re.finditer(tc):
            try:
                json.loads(m.group())
            except json.JSONDecodeError:
                reasons.append(f"invalid_json[{i}]: {m.group()[:60]}...")

        # Length sanity (skip if original is very short).
        if len(oc) > 10:
            ratio = len(tc) / len(oc)
            if ratio < _MIN_LENGTH_RATIO or ratio > _MAX_LENGTH_RATIO:
                reasons.append(
                    f"length_ratio[{i}]: {ratio:.2f} "
                    f"({len(oc)} -> {len(tc)})"
                )

    accepted = len(reasons) == 0
    return accepted, reasons


# ---------------------------------------------------------------------------
# Rejection log
# ---------------------------------------------------------------------------


class RejectionLog:
    """Collects rejected samples with reasons for post-mortem analysis."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def add(
        self,
        example_id: str,
        reasons: list[str],
        original: dict[str, Any] | None = None,
        translated: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "id": example_id,
            "reasons": reasons,
        }
        if original is not None:
            entry["original"] = original
        if translated is not None:
            entry["translated"] = translated
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        """Return rejection reason counts."""
        counter: Counter[str] = Counter()
        for entry in self._entries:
            for reason in entry["reasons"]:
                # Normalize: strip index/detail suffix for grouping.
                key = re.sub(r"\[.*", "", reason.split(":")[0])
                counter[key] += 1
        return dict(counter.most_common())

    def write_to_jsonl(self, path: str | Path) -> None:
        """Write all rejection entries to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for entry in self._entries:
            append_jsonl(path, entry)

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)
