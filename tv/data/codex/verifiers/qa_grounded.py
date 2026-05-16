"""WorkspaceVerifier for ``qa_grounded`` — TVL retrieval-backed QA.

This v0 verifier uses string-level checks only (no GPT-5.5 judge yet —
the judge build-out is Phase 5). It accepts iff the answer is:
- in Tuvaluan (stopword langid),
- non-empty, length-reasonable,
- protected terms preserved,
- contains at least one of the retrieved span ids' text snippets OR a
  shared rare n-gram with at least one retrieved span (cheap proxy for
  the source-support entailment the GPT-5.5 judge will measure later).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ._common import (
    langid_score_en,
    langid_score_tvl,
    protected_terms_recall,
    read_last_assistant_text,
)


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿĀ-žŻ-ž'’-]+", re.UNICODE)


def _rare_ngram_overlap(answer: str, span_texts: list[str], n: int = 4) -> float:
    """Fraction of n-grams in the answer that appear in any retrieved span."""
    def ngrams(s: str) -> set[str]:
        toks = _WORD_RE.findall(s.lower())
        return {" ".join(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}
    ans = ngrams(answer)
    if not ans:
        return 0.0
    spans = set()
    for st in span_texts:
        spans.update(ngrams(st))
    if not spans:
        return 0.0
    return len(ans & spans) / max(len(ans), 1)


class GroundedQAVerifier:
    def __init__(
        self,
        *,
        langid_min: float = 0.20,
        en_max: float = 0.35,
        protected_recall_min: float = 0.80,
        ngram_overlap_min: float = 0.05,
        len_min: int = 5,
        len_max: int = 600,
    ) -> None:
        self.langid_min = langid_min
        self.en_max = en_max
        self.protected_recall_min = protected_recall_min
        self.ngram_overlap_min = ngram_overlap_min
        self.len_min = len_min
        self.len_max = len_max

    async def __call__(self, *, workspace_dir, task):
        from codex_env.protocols import VerifierResult
        from tv.data.codex.task_builder import task_metadata_decode

        meta = task_metadata_decode(task.metadata or {})
        spans = list(meta.get("retrieved_span_texts") or [])
        protected = list(meta.get("protected_terms") or [])
        completion = read_last_assistant_text(workspace_dir).strip()

        len_ok = self.len_min <= len(completion) <= self.len_max
        tvl = langid_score_tvl(completion)
        en = langid_score_en(completion)
        langid_ok = tvl >= self.langid_min and en <= self.en_max
        pt_recall, missing = protected_terms_recall(completion, protected)
        pt_ok = pt_recall >= self.protected_recall_min
        overlap = _rare_ngram_overlap(completion, spans)
        overlap_ok = overlap >= self.ngram_overlap_min

        hard_gate = langid_ok and pt_ok
        reward = 1.0 if (hard_gate and len_ok and overlap_ok) else 0.0
        reason = (
            f"len={len(completion)}({len_ok}) tvl={tvl:.2f} en={en:.2f} "
            f"pt={pt_recall:.2f}({pt_ok}) ngram={overlap:.2f}({overlap_ok})"
        )
        return VerifierResult(
            reward=reward,
            ok=reward >= 1.0,
            reason=reason,
            hard_gate_passed=hard_gate,
            public_metrics={"answer_seen": completion[:200]},
            hidden_metrics={
                "len": float(len(completion)),
                "tvl_score": tvl,
                "en_score": en,
                "protected_recall": pt_recall,
                "ngram_overlap": overlap,
            },
            info={
                "completion_head": completion[:240],
                "missing_protected": missing,
            },
        )


__all__ = ["GroundedQAVerifier"]
