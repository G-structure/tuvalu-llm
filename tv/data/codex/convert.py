"""Convert ``DistillJobRunner`` ``traces.jsonl`` rows into normalized
tuvalu training examples (``tv.common.schema.make_example``).

The DistillJobRunner emits one JSONL row per task containing the full
prompt + completion text + Turn item stream + verifier outputs. Tuvalu's
Stage B mix builder consumes the normalized ``make_example`` shape. The
conversion is straightforward — single (user, assistant) pair — but
needs the task_family field which is family-pinned per harvest batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from tv.common.schema import TASK_FAMILIES, make_example


# Map codex/tuvalu task_family names to the tv.common.schema literal set.
# qa_grounded -> qa, native_chat -> chat, hard_translation -> translation.
_FAMILY_MAP: dict[str, str] = {
    "qa_grounded": "qa",
    "native_chat": "chat",
    "hard_translation": "translation",
    "tool_call_trajectory": "tool",
    "rejection_candidate": "chat",
}


def load_traces_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream rows out of a ``traces.jsonl`` file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def codex_trace_to_tvl_example(
    row: dict[str, Any],
    *,
    release: str,
    task_family: str,
    keep_rejected: bool = False,
) -> dict[str, Any] | None:
    """Map one DistillJobRunner trace row into a normalized tuvalu example.

    Returns ``None`` if the row should be skipped (rejected and
    ``keep_rejected=False``, or empty completion).
    """
    if not keep_rejected and not row.get("accepted", False):
        return None

    completion = (row.get("completion") or "").strip()
    if not completion:
        return None
    prompt = row.get("prompt") or ""

    canonical_family = _FAMILY_MAP.get(task_family, task_family)
    if canonical_family not in TASK_FAMILIES:
        canonical_family = "chat"  # safe fallback

    rid = row.get("rollout_id") or row.get("task_id") or "unknown"
    ex_id = f"codex_{release}_{rid}"

    return make_example(
        id=ex_id,
        task_family=canonical_family,  # type: ignore[arg-type]
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        metadata={
            "provenance": "codex_subscription",
            "release": release,
            "codex_task_family": task_family,
            "teacher_model": row.get("teacher_model"),
            "teacher_endpoint": row.get("teacher_endpoint"),
            "rollout_id": row.get("rollout_id"),
            "source_task_id": row.get("task_id"),
            "reward": row.get("reward"),
            "verifier_ok": row.get("verifier_ok"),
            "verifier_reason": row.get("verifier_reason"),
            "hard_gate_passed": row.get("hard_gate_passed"),
            "accepted": row.get("accepted"),
            "gold_answer": row.get("gold_answer"),
            "token_usage": row.get("token_usage"),
            "min_reward_threshold": row.get("min_reward_threshold"),
            "require_hard_gate": row.get("require_hard_gate"),
            "written_at": row.get("written_at"),
        },
    )


def write_normalized_examples(
    *,
    traces_jsonl: Path | str,
    output_path: Path | str,
    release: str,
    task_family: str,
    keep_rejected: bool = False,
) -> tuple[int, int]:
    """Read traces.jsonl, write normalized examples to output_path.

    Returns ``(n_written, n_skipped)``.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    with out_p.open("w", encoding="utf-8") as fp:
        for row in load_traces_jsonl(traces_jsonl):
            ex = codex_trace_to_tvl_example(
                row,
                release=release,
                task_family=task_family,
                keep_rejected=keep_rejected,
            )
            if ex is None:
                n_skipped += 1
                continue
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written, n_skipped


__all__ = [
    "codex_trace_to_tvl_example",
    "load_traces_jsonl",
    "write_normalized_examples",
]
