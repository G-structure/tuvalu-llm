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


def main() -> None:
    args = parse_args()

    config: dict = {}
    if args.config:
        with args.config.open() as f:
            config = json.load(f)

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
