"""Decontamination: ensure codex-generated rows don't overlap any eval split.

Two checks per row:
1. Exact-text overlap of the answer against any eval-split document.
2. Held-out-n-gram check against the eval splits' n-gram fingerprints.

Both are the same primitives ``tv/corpus/cleanup`` uses on Stage A
synthetic data; we just point them at the eval-split files instead of
the train splits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿĀ-žŻ-ž']+", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _ngrams(text: str, n: int) -> set[str]:
    toks = _WORD_RE.findall(text.lower())
    return {" ".join(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def build_eval_fingerprint(
    eval_jsonl_paths: Iterable[Path | str],
    *,
    text_fields: tuple[str, ...] = ("tvl", "en", "answer", "completion"),
    n: int = 8,
) -> tuple[set[str], set[str]]:
    """Return ``(exact_text_set, ngram_set)`` derived from eval JSONL splits."""
    exact: set[str] = set()
    ngrams: set[str] = set()
    for p in eval_jsonl_paths:
        path = Path(p)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for f in text_fields:
                    t = row.get(f)
                    if not isinstance(t, str) or len(t) < 8:
                        continue
                    norm = _normalize(t)
                    if norm:
                        exact.add(norm)
                        ngrams.update(_ngrams(t, n))
    return exact, ngrams


def check_row_against_eval(
    row: dict,
    *,
    eval_exact: set[str],
    eval_ngrams: set[str],
    field: str = "completion",
    n: int = 8,
    ngram_overlap_max: float = 0.40,
) -> tuple[bool, str]:
    """Return ``(is_clean, reason)``.

    A row is contaminated if either:
    - normalized answer text exactly matches an eval entry, OR
    - more than ``ngram_overlap_max`` fraction of the answer's n-grams
      appear in the eval n-gram set.
    """
    text = row.get(field) or ""
    if not text:
        return True, "empty_text"
    norm = _normalize(text)
    if norm in eval_exact:
        return False, f"exact_match_eval:{field}"
    if eval_ngrams:
        gs = _ngrams(text, n)
        if gs:
            overlap = len(gs & eval_ngrams) / len(gs)
            if overlap > ngram_overlap_max:
                return False, f"ngram_overlap_{overlap:.2f}_above_{ngram_overlap_max}"
    return True, "ok"


def filter_traces(
    *,
    traces_jsonl: Path | str,
    eval_jsonl_paths: Iterable[Path | str],
    output_clean: Path | str,
    output_contaminated: Path | str,
    field: str = "completion",
    n: int = 8,
    ngram_overlap_max: float = 0.40,
) -> tuple[int, int]:
    """Stream a traces.jsonl, split into clean + contaminated sinks.

    Returns ``(n_clean, n_contaminated)``.
    """
    eval_exact, eval_ngrams = build_eval_fingerprint(
        eval_jsonl_paths, n=n,
    )
    out_clean = Path(output_clean)
    out_cont = Path(output_contaminated)
    out_clean.parent.mkdir(parents=True, exist_ok=True)
    out_cont.parent.mkdir(parents=True, exist_ok=True)
    n_clean = 0
    n_cont = 0
    with (
        Path(traces_jsonl).open("r", encoding="utf-8") as fp_in,
        out_clean.open("w", encoding="utf-8") as fp_clean,
        out_cont.open("w", encoding="utf-8") as fp_cont,
    ):
        for line in fp_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ok, reason = check_row_against_eval(
                row,
                eval_exact=eval_exact,
                eval_ngrams=eval_ngrams,
                field=field,
                n=n,
                ngram_overlap_max=ngram_overlap_max,
            )
            if ok:
                fp_clean.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_clean += 1
            else:
                row = dict(row)
                row["decontam_reason"] = reason
                fp_cont.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_cont += 1
    return n_clean, n_cont


__all__ = [
    "build_eval_fingerprint",
    "check_row_against_eval",
    "filter_traces",
]
