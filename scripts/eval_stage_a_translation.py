#!/usr/bin/env python3
"""CLI wrapper: Evaluate Stage A translation adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Tuvaluan<->English translation adapter."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file. CLI args override config values.",
    )
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def flatten_eval_config(raw_config: dict) -> dict:
    """Flatten nested config into flat keys that eval.main() expects."""
    config: dict = {}
    data_sec = raw_config.get("data", {})
    model_sec = raw_config.get("model", {})
    eval_sec = raw_config.get("eval", {})

    # data path: output_dir / test_file
    data_output = data_sec.get("output_dir", "data/finetune/stage_a_mt")
    test_file = eval_sec.get("test_file", "test.jsonl")
    config["data"] = str(Path(data_output) / test_file)

    # model
    if model_sec.get("name"):
        config["model_name"] = model_sec["name"]

    # eval params
    for key in ("max_tokens", "temperature", "out_dir"):
        if key in eval_sec:
            config[key] = eval_sec[key]

    return config


def main() -> None:
    args = parse_args()

    raw_config: dict = {}
    if args.config:
        with args.config.open() as f:
            raw_config = json.load(f)

    # Flatten nested config structure into flat keys eval.main() expects
    config = flatten_eval_config(raw_config)

    # CLI args override everything
    cli_map = {
        "data": args.data,
        "model_path": args.model_path,
        "base_model": args.base_model,
        "model_name": args.model_name,
        "out_dir": args.out_dir,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "limit": args.limit,
    }
    for key, value in cli_map.items():
        if value is not None:
            config[key] = value

    from training.stage_a_mt.eval import main as eval_main

    eval_main(config)


if __name__ == "__main__":
    main()
