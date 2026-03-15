#!/usr/bin/env python3
"""DEPRECATED: Use scripts/train_stage_a_translation.py instead.

This wrapper delegates to tv.training.stage_a_mt.train for backwards
compatibility. It will be removed in a future version.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.warn(
    "scripts/train_tinker_mt.py is deprecated. "
    "Use scripts/train_stage_a_translation.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Parse legacy CLI args and forward to Stage A trainer.
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="DEPRECATED: Use train_stage_a_translation.py")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--train-on-what", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--ttl-seconds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--do-final-val-loss", action="store_true", default=None)
    args = parser.parse_args()

    config: dict = {}
    for key in ("data", "val_data", "model_name", "log_path", "base_url",
                "batch_size", "learning_rate", "epochs", "max_length",
                "lora_rank", "train_on_what", "save_every", "ttl_seconds",
                "seed", "do_final_val_loss"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            config[key] = val

    from tv.training.stage_a_mt.train import main as train_main

    train_main(config)


if __name__ == "__main__":
    main()
