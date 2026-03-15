#!/usr/bin/env python3
"""DEPRECATED: Use scripts/eval_stage_a_translation.py instead.

This wrapper delegates to tv.training.stage_a_mt.eval for backwards
compatibility. It will be removed in a future version.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.warn(
    "scripts/eval_tinker_mt.py is deprecated. "
    "Use scripts/eval_stage_a_translation.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="DEPRECATED: Use eval_stage_a_translation.py")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config: dict = {}
    for key in ("data", "model_path", "base_model", "model_name",
                "out_dir", "base_url", "max_tokens", "temperature", "limit"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            config[key] = val

    from tv.training.stage_a_mt.eval import main as eval_main

    eval_main(config)


if __name__ == "__main__":
    main()
