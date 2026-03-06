#!/usr/bin/env python3
"""CLI wrapper: Train Stage A translation adapter via Tinker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Tuvaluan<->English LoRA translation adapter."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file. CLI args override config values.",
    )
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
    return parser.parse_args()


def flatten_train_config(raw_config: dict) -> dict:
    """Flatten nested config into flat keys that train.main() expects."""
    config: dict = {}
    data_sec = raw_config.get("data", {})
    training_sec = raw_config.get("training", {})
    model_sec = raw_config.get("model", {})
    logs_sec = raw_config.get("logs", {})

    # Data paths
    data_output = data_sec.get("output_dir", "data/finetune/stage_a_mt")
    train_file = data_sec.get("train_file", "train_balanced.jsonl")
    config["data"] = str(Path(data_output) / train_file)
    config["val_data"] = str(Path(data_output) / "validation.jsonl")

    # Model
    if model_sec.get("name"):
        config["model_name"] = model_sec["name"]

    # Training hyperparams
    for key in ("lora_rank", "max_length", "batch_size", "learning_rate",
                "epochs", "save_every", "seed", "train_on_what", "ttl_seconds"):
        if key in training_sec:
            config[key] = training_sec[key]

    # Logs
    if logs_sec.get("base_dir"):
        config["log_path"] = logs_sec["base_dir"]

    return config


def main() -> None:
    args = parse_args()

    raw_config: dict = {}
    if args.config:
        with args.config.open() as f:
            raw_config = json.load(f)

    # Flatten nested config structure into the flat keys train.main() expects.
    config = flatten_train_config(raw_config)

    # CLI args override everything
    cli_map = {
        "data": args.data,
        "val_data": args.val_data,
        "model_name": args.model_name,
        "log_path": args.log_path,
        "base_url": args.base_url,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "train_on_what": args.train_on_what,
        "save_every": args.save_every,
        "ttl_seconds": args.ttl_seconds,
        "seed": args.seed,
        "do_final_val_loss": args.do_final_val_loss,
    }
    for key, value in cli_map.items():
        if value is not None:
            config[key] = value

    from training.stage_a_mt.train import main as train_main

    train_main(config)


if __name__ == "__main__":
    main()
