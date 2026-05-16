"""CLI driver for the codex distill harvest.

Usage::

    uv run python scripts/run_codex_harvest.py \\
        --task-family native_chat \\
        --n 50 \\
        --teacher-model gpt-5.3-codex \\
        --out runs/codex_harvest_$(date +%s)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add the tuvalu repo's tv/ to sys.path when run via uv run.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tv.data.codex.harvest import load_cleaned_subset, run_codex_harvest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run codex distill harvest.")
    parser.add_argument(
        "--task-family",
        choices=["native_chat", "hard_translation", "qa_grounded"],
        required=True,
    )
    parser.add_argument("--n", type=int, default=50, help="number of tasks to harvest")
    parser.add_argument("--skip", type=int, default=0, help="reservoir skip for held-out subsets")
    parser.add_argument("--teacher-model", default="gpt-5.3-codex")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--no-hard-gate", action="store_true")
    parser.add_argument(
        "--direction",
        choices=["tvl2en", "en2tvl"],
        default="tvl2en",
        help="hard_translation only",
    )
    parser.add_argument("--budget-seconds", type=float, default=180.0)
    parser.add_argument("--max-dollars", type=float, default=None)
    parser.add_argument(
        "--cleaned-jsonl",
        default="data/external/tv2en-cleaned/cleaned.jsonl",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help="restrict to one or more domains (book, article, etc). Repeatable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"loading source corpus subset n={args.n} from {args.cleaned_jsonl}", flush=True)
    source_rows = load_cleaned_subset(
        cleaned_jsonl=args.cleaned_jsonl,
        n=args.n,
        skip=args.skip,
        seed=args.seed,
        domains=args.domain,
    )
    print(f"loaded {len(source_rows)} source rows", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(
        run_codex_harvest(
            task_family=args.task_family,
            source_rows=source_rows,
            out_dir=args.out,
            teacher_model=args.teacher_model,
            min_reward_threshold=args.min_reward,
            require_hard_gate=not args.no_hard_gate,
            budget_seconds=args.budget_seconds,
            max_dollars=args.max_dollars,
            translation_direction=args.direction,
        )
    )

    # Print summary minus the per-trace verbose list (kept in traces.jsonl).
    out_min = {k: v for k, v in summary.items() if k != "traces"}
    print(json.dumps(out_min, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
