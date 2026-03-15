#!/usr/bin/env python3
"""Prepare Stage A / Stage B artifacts for local MLX-LM training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tv.training.local_mlx import prepare_local_mlx_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local MLX-LM training run for Stage A or Stage B.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Stage config JSON file.")
    parser.add_argument(
        "--mlx-model",
        type=str,
        default=None,
        help="Local MLX model path or HF repo. Overrides config local_mlx.model.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="out/local_mlx",
        help="Root directory where the prepared run will be written.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run directory name.")
    parser.add_argument("--pilot", action="store_true", help="Use pilot Stage B train file when available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        raw_config = json.load(f)
    summary = prepare_local_mlx_run(
        raw_config,
        pilot=args.pilot,
        mlx_model=args.mlx_model,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
