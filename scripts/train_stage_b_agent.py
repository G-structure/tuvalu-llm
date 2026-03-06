#!/usr/bin/env python3
"""CLI: Train Stage B bilingual capability adapter.

Stage B starts from a base/chat model, NOT from Stage A weights.

Usage:
    uv run python scripts/train_stage_b_agent.py --config configs/stage_b_agent_oss120b.json
    uv run python scripts/train_stage_b_agent.py --config configs/stage_b_agent_qwen30b.json
    uv run python scripts/train_stage_b_agent.py --config configs/stage_b_agent_oss120b.json --pilot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.common.config import load_config


def _translate_config(raw: dict, pilot: bool = False) -> dict:
    """Map config file keys to train.main() parameter names."""
    cfg: dict = {}

    model = raw.get("model", {})
    if "name" in model:
        cfg["model_name"] = model["name"]

    training = raw.get("training", {})
    for key in ("lora_rank", "max_length", "batch_size", "learning_rate",
                "epochs", "save_every", "seed", "ablation_mode",
                "train_on_what", "ttl_seconds"):
        if key in training:
            cfg[key] = training[key]
    if "included_task_families" in training:
        cfg["include_task_families"] = training["included_task_families"]
    if "tool_mode" in training:
        cfg["tool_mode"] = training["tool_mode"]

    data = raw.get("data", {})
    if pilot and "train_pilot_file" in data:
        cfg["train_data"] = data["train_pilot_file"]
    elif "train_file" in data:
        cfg["train_data"] = data["train_file"]
    if "validation_file" in data:
        cfg["validation_data"] = data["validation_file"]

    logs = raw.get("logs", {})
    if "base_dir" in logs:
        cfg["output_dir"] = logs["base_dir"]

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage B agent adapter")
    parser.add_argument("--config", type=str, required=True, help="Config JSON file")
    parser.add_argument("--pilot", action="store_true", help="Use pilot dataset for shakeout")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    raw = load_config(args.config)
    config = _translate_config(raw, pilot=args.pilot)

    if args.run_name:
        config["run_name"] = args.run_name
    if args.resume:
        config["resume_from"] = args.resume

    from training.stage_b_agent.train import main as train_main

    train_main(config)


if __name__ == "__main__":
    main()
