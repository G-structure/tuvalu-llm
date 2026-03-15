#!/usr/bin/env python3
"""CLI: Export Stage A model path for synthetic generation."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Export Stage A translation adapter info")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        with args.config.open() as f:
            raw = json.load(f)
        if "logs" in raw and "base_dir" in raw["logs"]:
            config["log_path"] = raw["logs"]["base_dir"]
        if "model" in raw and "name" in raw["model"]:
            config["base_model"] = raw["model"]["name"]
    if args.log_path:
        config["log_path"] = args.log_path
    if args.base_model:
        config["base_model"] = args.base_model

    from tv.training.stage_a_mt.export import main as export_main

    result = export_main(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
