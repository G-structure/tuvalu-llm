"""WorkspaceVerifier for ``hard_translation`` — codex re-translation
of Stage A's challenging examples.

Reward = 1.0 iff:
- chrF++(hypothesis, gold) >= chrf_min
- protected-term recall >= entity_recall_min
- langid matches the target direction

Otherwise 0.0; hard gate is langid + entity recall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._common import (
    chrf_plus_plus,
    langid_score_en,
    langid_score_tvl,
    protected_terms_recall,
    read_last_assistant_text,
)


class HardTranslationVerifier:
    def __init__(
        self,
        *,
        chrf_min: float = 0.35,
        entity_recall_min: float = 0.80,
        target_langid_min: float = 0.20,
    ) -> None:
        self.chrf_min = chrf_min
        self.entity_recall_min = entity_recall_min
        self.target_langid_min = target_langid_min

    async def __call__(self, *, workspace_dir, task):
        from codex_env.protocols import VerifierResult
        from tv.data.codex.task_builder import task_metadata_decode

        meta = task_metadata_decode(task.metadata or {})
        gold = str(meta.get("gold_translation") or "")
        direction = str(meta.get("direction") or "tvl2en")
        protected = list(meta.get("protected_terms") or [])

        completion = read_last_assistant_text(workspace_dir).strip()
        chrf = chrf_plus_plus(completion, gold)
        pt_recall, missing = protected_terms_recall(completion, protected)

        if direction == "tvl2en":
            ans_langid = langid_score_en(completion)
        else:
            ans_langid = langid_score_tvl(completion)
        langid_ok = ans_langid >= self.target_langid_min

        chrf_ok = chrf >= self.chrf_min
        pt_ok = pt_recall >= self.entity_recall_min

        hard_gate = langid_ok and pt_ok
        reward = 1.0 if (chrf_ok and hard_gate) else 0.0
        reason = (
            f"direction={direction} chrf={chrf:.3f}({chrf_ok}) "
            f"langid={ans_langid:.2f}({langid_ok}) "
            f"pt_recall={pt_recall:.2f}({pt_ok})"
        )
        return VerifierResult(
            reward=reward,
            ok=reward >= 1.0,
            reason=reason,
            hard_gate_passed=hard_gate,
            public_metrics={"answer_seen": completion[:200]},
            hidden_metrics={
                "chrf": chrf,
                "answer_langid": ans_langid,
                "protected_recall": pt_recall,
            },
            info={
                "completion_head": completion[:240],
                "gold": gold[:240],
                "direction": direction,
                "missing_protected": missing,
            },
        )


__all__ = ["HardTranslationVerifier"]
