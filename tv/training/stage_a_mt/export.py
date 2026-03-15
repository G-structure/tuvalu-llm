"""Stage A: Export the final checkpoint path for use by synthetic generation.

Reads the final checkpoint from a training log directory and returns
the model_path string that can be used with Tinker sampling clients.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tv.common.checkpoints import get_last_checkpoint
from tv.common.config import get_repo_root, resolve_path
from tv.common.io import write_json

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "log_path": "logs/stage_a_mt",
    "base_model": "Qwen/Qwen3-30B-A3B-Base",
}


def get_model_path(config: dict[str, Any] | None = None) -> str:
    """Get the final model path from a Stage A training run.

    Args:
        config: Configuration dict with at least 'log_path'.

    Returns:
        The Tinker model_path string for the final checkpoint.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    repo_root = get_repo_root()
    log_path = resolve_path(cfg["log_path"], repo_root)

    checkpoint_info = get_last_checkpoint(str(log_path))
    if not checkpoint_info:
        raise FileNotFoundError(f"No checkpoint found in {log_path}")

    model_path = checkpoint_info.get("weights_path") or checkpoint_info.get("state_path")
    if not model_path:
        raise FileNotFoundError(f"Checkpoint info has no usable path: {checkpoint_info}")

    logger.info("Stage A model path: %s", model_path)
    return model_path


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Export Stage A artifact info.

    Args:
        config: Configuration dict.

    Returns:
        Export info dict.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    model_path = get_model_path(cfg)

    repo_root = get_repo_root()
    log_path = resolve_path(cfg["log_path"], repo_root)

    export_info = {
        "model_path": model_path,
        "base_model": cfg["base_model"],
        "log_path": str(log_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(log_path / "export_info.json", export_info)

    print(f"Stage A export: {model_path}")
    print(f"  Base model: {cfg['base_model']}")
    print(f"  Export info written to: {log_path / 'export_info.json'}")

    return export_info
