"""Convert a codex harvest into Stage B synthetic_tvl-format JSONL.

Takes the raw ``traces.jsonl`` written by ``DistillJobRunner``, runs
decontamination against the eval splits, normalizes the accepted rows
into ``tv.common.schema.make_example`` shape, and splits into
train/val/test under ``data/finetune/stage_b_synthetic_tvl/codex_<tag>/``.

Usage::

    uv run python scripts/stage_codex_harvest.py \\
        --traces runs/codex_harvest_nc500/traces/traces.jsonl \\
        --tag nc500 \\
        --task-family native_chat \\
        --val-frac 0.05 --test-frac 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tv.data.codex.convert import codex_trace_to_tvl_example, load_traces_jsonl  # noqa: E402
from tv.data.codex.decontam import filter_traces  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage codex harvest output.")
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--tag", required=True, help="suffix for the output directory")
    parser.add_argument(
        "--task-family",
        required=True,
        choices=["native_chat", "hard_translation", "qa_grounded"],
    )
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-jsonl",
        action="append",
        default=None,
        help="JSONL file(s) for decontamination. Repeatable. "
        "Skip if no eval splits are staged.",
    )
    parser.add_argument(
        "--output-root",
        default="data/finetune/stage_b_synthetic_tvl",
        type=Path,
    )
    parser.add_argument(
        "--keep-rejected",
        action="store_true",
        help="emit rejected rows into a parallel rejected.jsonl",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("stage_codex_harvest")

    out_root = args.output_root / f"codex_{args.tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Step 1: decontamination (if eval splits provided)
    if args.eval_jsonl:
        log.info("running decontamination against %d eval files", len(args.eval_jsonl))
        clean_path = out_root / "_decontam_clean.jsonl"
        cont_path = out_root / "_decontam_contaminated.jsonl"
        n_clean, n_cont = filter_traces(
            traces_jsonl=args.traces,
            eval_jsonl_paths=args.eval_jsonl,
            output_clean=clean_path,
            output_contaminated=cont_path,
        )
        log.info("decontam: %d clean, %d contaminated", n_clean, n_cont)
        traces_for_convert = clean_path
    else:
        log.info("skipping decontamination (no --eval-jsonl provided)")
        traces_for_convert = args.traces

    # Step 2: collect accepted (+ optionally rejected) normalized examples
    accepted: list[dict] = []
    rejected: list[dict] = []
    n_skipped_empty = 0
    for row in load_traces_jsonl(traces_for_convert):
        ex = codex_trace_to_tvl_example(
            row,
            release=args.tag,
            task_family=args.task_family,
            keep_rejected=args.keep_rejected,
        )
        if ex is None:
            n_skipped_empty += 1
            continue
        if row.get("accepted"):
            accepted.append(ex)
        else:
            rejected.append(ex)

    log.info(
        "normalized: %d accepted, %d rejected, %d skipped (empty)",
        len(accepted), len(rejected), n_skipped_empty,
    )

    # Step 3: train/val/test split on accepted rows (deterministic)
    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    n = len(accepted)
    n_val = max(1, int(n * args.val_frac))
    n_test = max(1, int(n * args.test_frac))
    val = accepted[:n_val]
    test = accepted[n_val : n_val + n_test]
    train = accepted[n_val + n_test :]

    # Write all four sinks
    for split, rows in (("train", train), ("validation", val), ("test", test)):
        path = out_root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fp:
            for ex in rows:
                fp.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("wrote %s: %d rows -> %s", split, len(rows), path)

    # accepted.jsonl is what Stage B mix builder consumes
    accepted_path = out_root / "accepted.jsonl"
    with accepted_path.open("w", encoding="utf-8") as fp:
        for ex in train + val + test:  # full accepted set
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("wrote accepted.jsonl: %d rows -> %s", len(train) + len(val) + len(test), accepted_path)

    if args.keep_rejected and rejected:
        rejected_path = out_root / "rejected.jsonl"
        with rejected_path.open("w", encoding="utf-8") as fp:
            for ex in rejected:
                fp.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info("wrote rejected.jsonl: %d rows -> %s", len(rejected), rejected_path)

    summary = {
        "out_root": str(out_root),
        "task_family": args.task_family,
        "tag": args.tag,
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_skipped_empty": n_skipped_empty,
        "n_train": len(train),
        "n_validation": len(val),
        "n_test": len(test),
        "decontam_eval_files": list(args.eval_jsonl) if args.eval_jsonl else [],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
