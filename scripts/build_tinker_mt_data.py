#!/usr/bin/env python3
"""DEPRECATED: Use scripts/build_stage_a_mt_data.py instead.

This wrapper delegates to tv.training.stage_a_mt.build_data for backwards
compatibility. It will be removed in a future version.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    warnings.warn(
        "scripts/build_tinker_mt_data.py is deprecated. "
        "Use scripts/build_stage_a_mt_data.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "WARNING: This script is deprecated. "
        "Use scripts/build_stage_a_mt_data.py instead.",
        file=sys.stderr,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/finetune/tinker_mt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--ratio-min", type=float, default=None)
    parser.add_argument("--ratio-max", type=float, default=None)
    parser.add_argument("--allow-low-confidence-articles", action="store_true", default=None)
    parser.add_argument("--bible-max-train-share", type=float, default=None)
    parser.add_argument("--non-bible-val-frac", type=float, default=None)
    parser.add_argument("--non-bible-test-frac", type=float, default=None)
    args = parser.parse_args()

    config: dict = {}
    cli_map = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "min_confidence": args.min_confidence,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "ratio_min": args.ratio_min,
        "ratio_max": args.ratio_max,
        "allow_low_confidence_articles": args.allow_low_confidence_articles,
        "bible_max_train_share": args.bible_max_train_share,
        "non_bible_val_frac": args.non_bible_val_frac,
        "non_bible_test_frac": args.non_bible_test_frac,
    }
    for key, value in cli_map.items():
        if value is not None:
            config[key] = value

    from tv.training.stage_a_mt.build_data import main as build_main

    build_main(config)


if __name__ == "__main__":
    main()
