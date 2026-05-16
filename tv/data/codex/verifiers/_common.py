"""Shared helpers for the codex distill verifiers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# TVL stopwords drawn from tv/training/stage_c/pipeline.py's TVL_HINTS.
# Cheap langid: count how many of these appear in the answer. If the
# ratio of TVL stopword hits per word exceeds the threshold, call it
# Tuvaluan.
_TVL_STOPWORDS = {
    "te", "kae", "ko", "ki", "mai", "atu", "faka", "tuvalu", "malo",
    "fenua", "tala", "tenei", "tena", "konei", "konea", "fakatoka",
    "fakamau", "fakaaoga", "fakailoa", "faiga", "tagata", "fafine",
    "alofa", "tatou", "matou", "lautou", "laua", "ia", "au", "koe",
    "outou", "ai", "ne", "kae", "kona", "ona", "ne", "nā", "se", "sē",
    "mo", "ma", "i", "io", "ia", "lakau", "tau", "lavea", "lavea", "vae",
    "a", "o", "e", "lava", "loa", "ke", "mafai", "fakamoemoe", "mea",
    "iloa", "fakaaogā", "fakaali", "fakatoka", "fakatūlaga",
}

# English stopword set for the opposite check.
_EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "if", "then", "than", "so", "not", "no", "yes",
}


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿĀ-žŻ-ž'’-]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    # Drop single-letter tokens: 'i', 'a', 'e' are both Tuvaluan function
    # words AND English stopwords, so counting them confuses both
    # langid scorers. They're noise either way.
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= 2]


def langid_score_tvl(text: str) -> float:
    """Return TVL-likelihood score in [0, 1]. Uses stopword-ratio heuristic.

    A heavyweight langid replacement lives in ``tv.common.langid`` (if it
    exists) but we don't have it in v0; this stopword check is the same
    primitive tuvalu's stage_c uses and is good enough for the codex
    rejection filter.
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    tvl_hits = sum(1 for w in words if w in _TVL_STOPWORDS)
    en_hits = sum(1 for w in words if w in _EN_STOPWORDS)
    # If we see many English stopwords, that's a strong negative signal.
    if en_hits > 2 * tvl_hits + 3:
        return 0.0
    return min(1.0, tvl_hits / max(8, len(words)) * 4.0)


def langid_score_en(text: str) -> float:
    words = _tokenize(text)
    if not words:
        return 0.0
    en_hits = sum(1 for w in words if w in _EN_STOPWORDS)
    return min(1.0, en_hits / max(8, len(words)) * 4.0)


def read_last_assistant_text(workspace_dir: Path | str) -> str:
    """Read the assistant's final message from ``.codex/last_turn.json``
    (which DistillJobRunner writes for every turn)."""
    lt = Path(workspace_dir) / ".codex" / "last_turn.json"
    if not lt.exists():
        return ""
    try:
        payload = json.loads(lt.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for item in reversed(payload.get("items") or []):
        if item.get("type") in ("agentMessage", "assistantMessage"):
            txt = item.get("text") or item.get("content") or ""
            if isinstance(txt, str) and txt:
                return txt
    return ""


def protected_terms_recall(completion: str, terms: list[str]) -> tuple[float, list[str]]:
    """Fraction of protected terms that appear verbatim in completion.

    Returns ``(recall_score, missing_terms)``.
    """
    if not terms:
        return 1.0, []
    missing = [t for t in terms if t and t not in (completion or "")]
    recall = 1.0 - len(missing) / len(terms)
    return recall, missing


def chrf_plus_plus(hypothesis: str, reference: str) -> float:
    """Compute chrF++ (sentence-level) using sacrebleu if available;
    fall back to a 6-gram character F-score approximation otherwise."""
    try:
        from sacrebleu.metrics import CHRF  # type: ignore
        metric = CHRF(word_order=2)
        return metric.sentence_score(hypothesis or "", [reference or ""]).score / 100.0
    except Exception:
        return _approx_chrf6(hypothesis or "", reference or "")


def _approx_chrf6(hyp: str, ref: str) -> float:
    """Approximate chrF using char 6-gram F1. Slow but correct in shape."""
    if not hyp or not ref:
        return 0.0
    def grams(s: str, n: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in range(len(s) - n + 1):
            g = s[i : i + n]
            out[g] = out.get(g, 0) + 1
        return out
    hg = grams(hyp, 6)
    rg = grams(ref, 6)
    inter = sum(min(hg.get(k, 0), rg.get(k, 0)) for k in set(hg) | set(rg))
    p = inter / max(sum(hg.values()), 1)
    r = inter / max(sum(rg.values()), 1)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


__all__ = [
    "chrf_plus_plus",
    "langid_score_en",
    "langid_score_tvl",
    "protected_terms_recall",
    "read_last_assistant_text",
]
