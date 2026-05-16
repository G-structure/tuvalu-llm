"""Codex distill harvest runner.

Wraps the sibling ``codex_orchestrate.jobs.distill.DistillJobRunner``
behind a tuvalu-shaped entrypoint: feed in a source corpus + a task
family, get back ``traces.jsonl`` populated by the codex subscription
under ``~/.codex/auth.json``.

Run via the CLI:

    uv run python scripts/run_codex_harvest.py --task-family native_chat --n 50

or programmatically:

    from tv.data.codex import run_codex_harvest
    outcome = await run_codex_harvest(
        task_family="native_chat",
        source_rows=load_cleaned_subset(n=50),
        teacher_model="gpt-5.3-codex",
        out_dir=Path("runs/codex_harvest_..."),
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from ._codex_path import ensure_codex_packages_on_path

# Side-effect: add codex_orchestrate / codex_env to sys.path.
ensure_codex_packages_on_path()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source loaders — pull a sized subset of the cleaned HF dataset.
# ---------------------------------------------------------------------------

def load_cleaned_subset(
    *,
    cleaned_jsonl: Path | str = "data/external/tv2en-cleaned/cleaned.jsonl",
    n: int = 50,
    min_tvl_len: int = 40,
    max_tvl_len: int = 600,
    seed: int = 0,
    skip: int = 0,
    domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Stream cleaned.jsonl, sample ``n`` rows within length bounds.

    Reservoir-sample so the result distribution doesn't depend on file
    order. ``skip`` skips the first K rows of the reservoir output and
    is the cheap way to get a "held-out" subset for eval.
    """
    path = Path(cleaned_jsonl)
    if not path.exists():
        raise FileNotFoundError(
            f"cleaned dataset not at {path} — run "
            "`uv run python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download('FriezaForce/tv2en-cleaned', repo_type='dataset', "
            "filename='cleaned.jsonl', local_dir='data/external/tv2en-cleaned')\"`"
        )
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    domain_set = set(domains) if domains else None
    cnt = 0
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tvl = row.get("tvl") or ""
            en = row.get("en") or ""
            if not isinstance(tvl, str) or not isinstance(en, str):
                continue
            if not (min_tvl_len <= len(tvl) <= max_tvl_len):
                continue
            if domain_set is not None and row.get("domain") not in domain_set:
                continue
            cnt += 1
            target = n + skip
            if len(reservoir) < target:
                reservoir.append(row)
            else:
                i = rng.randint(0, cnt - 1)
                if i < target:
                    reservoir[i] = row
    rng.shuffle(reservoir)
    return reservoir[skip : skip + n]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

async def run_codex_harvest(
    *,
    task_family: str,
    source_rows: list[dict[str, Any]],
    out_dir: Path | str,
    teacher_model: str = "gpt-5.3-codex",
    teacher_endpoint: str = "https://api.openai.com",
    min_reward_threshold: float = 1.0,
    require_hard_gate: bool = True,
    budget_seconds: float = 180.0,
    max_dollars: float | None = None,
    translation_direction: str | None = None,  # for hard_translation only
) -> dict[str, Any]:
    """Run a codex distill harvest for ``task_family`` over ``source_rows``.

    Materializes a per-row codex_env.Task dir under ``out_dir/tasks/``,
    builds the right per-family WorkspaceVerifier, dispatches via
    ``codex_orchestrate.run_job(DistillJobSpec(...))`` with
    ``passthrough_to_default=True`` (codex uses ~/.codex/auth.json
    natively; no proxy in the model path), and persists per-task rows
    to ``out_dir/traces/traces.jsonl``.

    Returns a summary dict with the outcome's accepted/rejected counts
    plus paths to artifacts.
    """
    from codex_env import load_task_dir
    from codex_orchestrate.jobs.spec import DistillJobSpec, JobKind, RunContext
    from codex_orchestrate.lifecycle.run_job import run_job
    from codex_orchestrate.pools.verifier_runner import VerifierRunner

    from .task_builder import (
        build_hard_translation_task,
        build_native_chat_task,
        build_qa_grounded_task,
    )
    from .verifiers import (
        GroundedQAVerifier,
        HardTranslationVerifier,
        NativeChatVerifier,
    )

    out_root = Path(out_dir)
    bench_root = out_root / "tasks"
    if bench_root.exists():
        shutil.rmtree(bench_root)
    bench_root.mkdir(parents=True)

    # Build task dirs + select verifier based on task_family.
    if task_family == "native_chat":
        verifier = NativeChatVerifier()
        for i, src in enumerate(source_rows):
            tid = f"nc-{i:04d}"
            build_native_chat_task(
                bench_root=bench_root,
                task_id=tid,
                tvl_text=src.get("tvl") or "",
                en_text=src.get("en") or "",
                source_doc_id=str(src.get("doc_id") or ""),
            )
    elif task_family == "hard_translation":
        verifier = HardTranslationVerifier()
        direction = translation_direction or "tvl2en"
        for i, src in enumerate(source_rows):
            tid = f"ht-{i:04d}"
            if direction == "tvl2en":
                src_text, gold = src.get("tvl") or "", src.get("en") or ""
            else:
                src_text, gold = src.get("en") or "", src.get("tvl") or ""
            build_hard_translation_task(
                bench_root=bench_root,
                task_id=tid,
                src_text=src_text,
                gold_text=gold,
                direction=direction,
                source_doc_id=str(src.get("doc_id") or ""),
            )
    elif task_family == "qa_grounded":
        verifier = GroundedQAVerifier()
        for i, src in enumerate(source_rows):
            tid = f"qa-{i:04d}"
            # v0 shape: ask codex a generic "summarize key fact" question
            # against the source paragraph. A richer retrieval-corpus +
            # question-generation step lives in Phase 5 (judge build-out).
            tvl = src.get("tvl") or ""
            question = "Ne a te mea sili ne fakaasi mai i te tugapelu tenei?"
            build_qa_grounded_task(
                bench_root=bench_root,
                task_id=tid,
                question=question,
                spans=[(f"src-{i:04d}", tvl)],
                source_doc_id=str(src.get("doc_id") or ""),
            )
    else:
        raise ValueError(
            f"unknown task_family={task_family!r}. "
            "Supported: native_chat, hard_translation, qa_grounded"
        )

    # Load tasks back as codex_env.Task objects so the runner reads
    # them with the same loader the production path uses.
    tasks = [
        load_task_dir(p) for p in sorted(bench_root.iterdir()) if p.is_dir()
    ]
    log.info(
        "codex harvest: family=%s tasks=%d teacher=%s",
        task_family, len(tasks), teacher_model,
    )

    spec = DistillJobSpec(
        kind=JobKind.DISTILL,
        task=tasks[0],
        agent_policy_endpoint=teacher_endpoint,
        seed=0,
        budget_turns=1,
        budget_seconds=budget_seconds,
        teacher_model=teacher_model,
        teacher_endpoint=teacher_endpoint,
        teacher_temperature=0.7,
        min_reward_threshold=min_reward_threshold,
        require_hard_gate=require_hard_gate,
        max_dollars=max_dollars,
        output_dir=out_root / "traces",
    )
    ctx = RunContext(
        proxy=None,
        verifier_runner=VerifierRunner(),
        workdir=out_root / "_snapshots",
        extra={"tasks": tasks, "verifier": verifier},
    )

    t0 = time.perf_counter()
    outcome = await run_job(spec, ctx=ctx)
    elapsed = time.perf_counter() - t0

    return {
        "task_family": task_family,
        "teacher_model": teacher_model,
        "n_tasks": len(tasks),
        "accepted": outcome.accepted_count,
        "rejected": outcome.rejected_count,
        "dollars_spent": outcome.dollars_spent,
        "budget_hit": outcome.info.get("budget_hit", False),
        "wall_s": elapsed,
        "traces_jsonl": str(out_root / "traces" / "traces.jsonl"),
        "out_root": str(out_root),
        "traces": outcome.info.get("traces", []),
    }


__all__ = [
    "load_cleaned_subset",
    "run_codex_harvest",
]
