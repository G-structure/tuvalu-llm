#!/usr/bin/env python3
"""CLI wrapper: Build Stage A MT dataset from aligned JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage A MT chat datasets from aligned JSONL."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file. CLI args override config values.",
    )
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
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
    return parser.parse_args()


def flatten_build_config(raw_config: dict) -> dict:
    """Flatten nested config into flat keys that build_data.main() expects."""
    config: dict = {}
    data_sec = raw_config.get("data", {})
    training_sec = raw_config.get("training", {})

    # All data.* keys map directly
    for key, value in data_sec.items():
        config[key] = value

    # seed comes from training section
    if "seed" in training_sec:
        config["seed"] = training_sec["seed"]

    return config


def main() -> None:
    args = parse_args()

    raw_config: dict = {}
    if args.config:
        with args.config.open() as f:
            raw_config = json.load(f)

    # Flatten nested config structure into flat keys build_data.main() expects
    config = flatten_build_config(raw_config)

    # CLI args override everything
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

    from training.stage_a_mt.build_data import main as build_main

    build_main(config)


if __name__ == "__main__":
    main()
