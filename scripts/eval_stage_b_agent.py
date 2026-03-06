#!/usr/bin/env python3
"""CLI: Evaluate Stage B bilingual capability adapter.

Usage:
    uv run python scripts/eval_stage_b_agent.py --config configs/stage_b_agent_oss120b.json
    uv run python scripts/eval_stage_b_agent.py --config configs/stage_b_agent_qwen30b.json
    uv run python scripts/eval_stage_b_agent.py --config configs/stage_b_agent_oss120b.json --model-path /path/to/adapter
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.common.config import load_config


def _translate_config(raw: dict) -> dict:
    """Map config file keys to eval.main() parameter names."""
    cfg: dict = {}

    model = raw.get("model", {})
    if "name" in model:
        cfg["model_name"] = model["name"]

    eval_section = raw.get("eval", {})
    if "mt_test_file" in eval_section:
        cfg["mt_test_data"] = eval_section["mt_test_file"]
    if "capability_test_file" in eval_section:
        cfg["capability_test_data"] = eval_section["capability_test_file"]
    if "max_tokens" in eval_section:
        cfg["max_tokens"] = eval_section["max_tokens"]
    if "temperature" in eval_section:
        cfg["temperature"] = eval_section["temperature"]
    if "out_dir" in eval_section:
        cfg["output_dir"] = eval_section["out_dir"]

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage B agent adapter")
    parser.add_argument("--config", type=str, required=True, help="Config JSON file")
    parser.add_argument("--model-path", type=str, default=None, help="Path to Stage B adapter")
    parser.add_argument("--stage-a-path", type=str, default=None, help="Path to Stage A adapter for regression")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    raw = load_config(args.config)
    config = _translate_config(raw)

    if args.model_path:
        config["model_path"] = args.model_path
    if args.stage_a_path:
        config["stage_a_model_path"] = args.stage_a_path
    if args.run_name:
        config["run_name"] = args.run_name

    from training.stage_b_agent.eval import main as eval_main

    eval_main(config)


if __name__ == "__main__":
    main()
