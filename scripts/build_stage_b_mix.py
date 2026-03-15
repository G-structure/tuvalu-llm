#!/usr/bin/env python3
"""CLI: Build Stage B mixed training dataset.

Usage:
    uv run python scripts/build_stage_b_mix.py
    uv run python scripts/build_stage_b_mix.py --config configs/stage_b_mix_default.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tv.common.config import load_config


def _translate_config(raw: dict) -> dict:
    """Map config file keys to build_mix.main() parameter names."""
    cfg: dict = {}

    sources = raw.get("sources", {})
    if "english_dir" in sources:
        cfg["english_dir"] = sources["english_dir"]
    if "synthetic_tvl_dir" in sources:
        cfg["synthetic_tvl_dir"] = sources["synthetic_tvl_dir"]
    if "crosslingual_dir" in sources:
        cfg["crosslingual_dir"] = sources["crosslingual_dir"]
    if "real_tvl_chat_dir" in sources:
        cfg["real_tvl_chat_dir"] = sources["real_tvl_chat_dir"]
    if "anchor_file" in sources:
        cfg["anchor_path"] = sources["anchor_file"]

    if "mix_ratios" in raw:
        cfg["mix_ratios"] = raw["mix_ratios"]
    if "pilot_size" in raw:
        cfg["pilot_size"] = raw["pilot_size"]
    if "validation_frac" in raw:
        cfg["validation_fraction"] = raw["validation_frac"]
    if "test_frac" in raw:
        cfg["test_fraction"] = raw["test_frac"]
    if "seed" in raw:
        cfg["seed"] = raw["seed"]
    if "output_dir" in raw:
        cfg["output_dir"] = raw["output_dir"]
    if "task_families" in raw:
        cfg["include_task_families"] = raw["task_families"]

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage B mixed dataset")
    parser.add_argument("--config", type=str, default=None, help="Config JSON file")
    args = parser.parse_args()

    config = {}
    if args.config:
        raw = load_config(args.config)
        config = _translate_config(raw)

    from tv.training.stage_b_agent.build_mix import main as build_main

    build_main(config)


if __name__ == "__main__":
    main()
