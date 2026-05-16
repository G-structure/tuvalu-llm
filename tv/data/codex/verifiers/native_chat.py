"""WorkspaceVerifier for the ``native_chat`` task family.

The codex teacher is asked to produce a TVL question/answer pair grounded
in a Tuvaluan passage. Accept iff:

- the answer contains a recognizable ``FESILI:`` and ``TALI:`` pair
- both halves are clearly Tuvaluan (stopword-ratio langid)
- protected terms (auto-extracted by ``task_builder``) survive
- the answer length is reasonable for a chat row (10..400 chars)

Reward is 1.0 on full accept, 0.0 on hard fail, in between for partial.
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


_FESILI_RE = re.compile(r"FESILI\s*:\s*(?P<q>.+?)(?:\n|$)", re.DOTALL | re.IGNORECASE)
_TALI_RE = re.compile(r"TALI\s*:\s*(?P<a>.+?)(?:\Z|\nFESILI\s*:)", re.DOTALL | re.IGNORECASE)


class NativeChatVerifier:
    def __init__(
        self,
        *,
        langid_min: float = 0.20,
        en_max: float = 0.50,
        protected_recall_min: float = 0.0,  # off by default for chat
        len_min: int = 10,
        len_max: int = 400,
    ) -> None:
        # Notes on defaults:
        # - en_max=0.50: a fluent TVL Q/A pair will pick up some English
        #   stopwords through borrowed proper names (Ieova, Iesu, Mata,
        #   etc.); cap at 0.50 instead of 0.35.
        # - protected_recall_min=0.0: chat answers paraphrase, they don't
        #   echo the source verbatim. Auto-extracted quoted phrases miss
        #   too often. Entity preservation belongs on hard_translation,
        #   not chat.
        self.langid_min = langid_min
        self.en_max = en_max
        self.protected_recall_min = protected_recall_min
        self.len_min = len_min
        self.len_max = len_max

    async def __call__(self, *, workspace_dir, task):
        from codex_env.protocols import VerifierResult
        from tv.data.codex.task_builder import task_metadata_decode

        meta = task_metadata_decode(task.metadata or {})
        protected = list(meta.get("protected_terms") or [])
        completion = read_last_assistant_text(workspace_dir)

        m_q = _FESILI_RE.search(completion)
        m_a = _TALI_RE.search(completion)
        format_ok = bool(m_q and m_a)
        question = m_q.group("q").strip() if m_q else ""
        answer = m_a.group("a").strip() if m_a else completion.strip()

        len_ok = self.len_min <= len(answer) <= self.len_max
        tvl_q = langid_score_tvl(question) if question else 0.0
        tvl_a = langid_score_tvl(answer)
        en_a = langid_score_en(answer)
        langid_ok = tvl_a >= self.langid_min and en_a <= self.en_max
        pt_recall, missing = protected_terms_recall(answer, protected)
        pt_ok = pt_recall >= self.protected_recall_min

        hard_gate = format_ok and langid_ok and pt_ok
        reward = 1.0 if (hard_gate and len_ok) else 0.0
        reason = (
            f"format={format_ok} len={len(answer)}({len_ok}) "
            f"tvl_a={tvl_a:.2f} en_a={en_a:.2f} pt_recall={pt_recall:.2f}"
        )
        return VerifierResult(
            reward=reward,
            ok=reward >= 1.0,
            reason=reason,
            hard_gate_passed=hard_gate,
            public_metrics={"answer_seen": answer[:200]},
            hidden_metrics={
                "format_ok": float(format_ok),
                "len": float(len(answer)),
                "tvl_q": tvl_q,
                "tvl_a": tvl_a,
                "en_a": en_a,
                "protected_recall": pt_recall,
            },
            info={
                "completion_head": completion[:240],
                "question": question[:200],
                "answer": answer[:240],
                "missing_protected": missing,
            },
        )


__all__ = ["NativeChatVerifier"]
