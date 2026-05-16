"""Codex-subscription distillation adapter for the tuvalu data pipeline.

Wraps the harness substrate that lives in the sibling ``rl-agent-work``
workspace (``codex_orchestrate.jobs.distill.DistillJobRunner``,
``codex_control``, ``codex_env``) and presents a tuvalu-shaped API: feed
in tuvalu source rows (TVL/EN pairs from ``data/external/tv2en-cleaned/``,
Stage C source spans, etc.), get back accepted training examples in
``tv.common.schema.make_example`` shape ready to plug into the Stage B
mix builder.

The runtime architecture is in ``docs/FULL_PIPELINE_AUDIT_AND_RL_JUDGE_PLAN.md``
under "Phase 4.5".
"""

from .convert import codex_trace_to_tvl_example, load_traces_jsonl
from .harvest import run_codex_harvest

__all__ = [
    "codex_trace_to_tvl_example",
    "load_traces_jsonl",
    "run_codex_harvest",
]
