#!/usr/bin/env python3
"""Find the most recent Tinker sampler weights from the longest training run.

Scans all Stage B log directories, finds the one with the most training steps,
and returns the latest sampler_path checkpoint.

Usage:
    uv run python scripts/get_latest_model.py
    uv run python scripts/get_latest_model.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs" / "tinker"


def find_longest_run() -> tuple[Path, list[dict]]:
    """Find the training run with the most checkpoints/steps."""
    best_dir: Path | None = None
    best_checkpoints: list[dict] = []
    best_max_step = -1

    for log_dir in sorted(LOGS_DIR.iterdir()):
        ckpt_file = log_dir / "checkpoints.jsonl"
        if not ckpt_file.exists():
            continue

        checkpoints = []
        for line in ckpt_file.read_text().strip().split("\n"):
            if line.strip():
                checkpoints.append(json.loads(line))

        if not checkpoints:
            continue

        max_step = max(c.get("step", 0) for c in checkpoints)
        if max_step > best_max_step:
            best_max_step = max_step
            best_dir = log_dir
            best_checkpoints = checkpoints

    if best_dir is None:
        raise SystemExit("No training runs with checkpoints found")

    return best_dir, best_checkpoints


def get_latest_sampler(checkpoints: list[dict]) -> dict | None:
    """Get the latest checkpoint with a sampler_path."""
    sampler_ckpts = [c for c in checkpoints if "sampler_path" in c]
    if not sampler_ckpts:
        return None
    return sampler_ckpts[-1]


def get_latest_state(checkpoints: list[dict]) -> dict | None:
    """Get the latest checkpoint with a state_path."""
    state_ckpts = [c for c in checkpoints if "state_path" in c]
    if not state_ckpts:
        return None
    return state_ckpts[-1]


def main():
    parser = argparse.ArgumentParser(description="Find latest Tinker model")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    run_dir, checkpoints = find_longest_run()
    latest_sampler = get_latest_sampler(checkpoints)
    latest_state = get_latest_state(checkpoints)

    result = {
        "run_dir": str(run_dir),
        "total_checkpoints": len(checkpoints),
        "latest_sampler": latest_sampler,
        "latest_state": latest_state,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Run: {run_dir.name}")
        print(f"Checkpoints: {len(checkpoints)}")
        if latest_sampler:
            print(f"Latest sampler: {latest_sampler['sampler_path']}")
            print(f"  (step {latest_sampler.get('name', '?')})")
        else:
            print("No sampler weights available yet")
        if latest_state:
            print(f"Latest state: {latest_state['state_path']}")
            print(f"  (step {latest_state.get('step', '?')})")

    # Print just the sampler path to stdout for piping
    if latest_sampler and not args.json:
        print(f"\nMODEL_PATH={latest_sampler['sampler_path']}")


if __name__ == "__main__":
    main()
