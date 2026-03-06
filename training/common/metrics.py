"""Metric computation: chrF++, BLEU, exact match, structured logging."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for exact-match comparison."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def compute_translation_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute chrF++, BLEU, exact match over prediction records.

    Each record must have 'prediction' and 'reference' keys.
    Requires sacrebleu.
    """
    import sacrebleu  # type: ignore

    if not predictions:
        return {"count": 0}

    refs = [[r["reference"] for r in predictions]]
    hyps = [r["prediction"] for r in predictions]
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    bleu = sacrebleu.metrics.BLEU(effective_order=True)
    exact = sum(
        normalize_whitespace(r["prediction"]) == normalize_whitespace(r["reference"])
        for r in predictions
    )
    return {
        "count": len(predictions),
        "chrf_pp": chrf.corpus_score(hyps, refs).score,
        "bleu": bleu.corpus_score(hyps, refs).score,
        "exact_match": exact / len(predictions),
    }


def compute_grouped_metrics(
    predictions: list[dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, Any]]:
    """Compute translation metrics grouped by a metadata key."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        key = str(row.get(group_key, "unknown"))
        groups[key].append(row)
    return {key: compute_translation_metrics(rows) for key, rows in groups.items()}


def compute_preservation_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute preservation metrics for Stage B eval.

    Each record should have 'prediction' text and optional metadata about
    expected structure (json_valid, code_blocks, placeholders).
    """
    import json as json_module

    total = len(predictions)
    if total == 0:
        return {"count": 0}

    json_parse_ok = 0
    code_exact_ok = 0
    placeholder_leak = 0
    n_json = 0
    n_code = 0
    n_placeholder = 0

    for row in predictions:
        pred = row.get("prediction", "")
        meta = row.get("metadata", {})

        if meta.get("expected_json"):
            n_json += 1
            try:
                json_module.loads(pred)
                json_parse_ok += 1
            except (json_module.JSONDecodeError, TypeError):
                pass

        if meta.get("expected_code_blocks"):
            n_code += 1
            expected = meta["expected_code_blocks"]
            if all(block in pred for block in expected):
                code_exact_ok += 1

        if meta.get("placeholders"):
            n_placeholder += 1
            leaked = sum(
                1 for ph in meta["placeholders"]
                if ph.startswith("__PH_") and ph in pred
            )
            if leaked > 0:
                placeholder_leak += 1

    return {
        "count": total,
        "json_parse_rate": json_parse_ok / n_json if n_json else None,
        "code_exact_match_rate": code_exact_ok / n_code if n_code else None,
        "placeholder_leak_rate": placeholder_leak / n_placeholder if n_placeholder else None,
        "n_json_tested": n_json,
        "n_code_tested": n_code,
        "n_placeholder_tested": n_placeholder,
    }
