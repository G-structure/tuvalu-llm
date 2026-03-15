#!/usr/bin/env python3
"""Generate synthetic Tuvaluan data by selectively translating English capability datasets.

Uses the trained Stage A translation model to translate human-language spans
while preserving machine-readable structure (code, JSON, tool calls, etc.).

Example:
    uv run python scripts/generate_stage_b_synthetic_tvl.py \
        --config configs/synthetic_stage_b_core.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tv.common.cli import load_optional_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "synthetic_stage_b_core.json",
        help="Path to synthetic generation config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without generating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_optional_config(args.config)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    from tv.training.synthetic.generate import main as run_generation

    summary = run_generation(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
