"""Run manifest generation for reproducibility."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_repo_root
from .io import write_json


def get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=get_repo_root(),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def get_git_dirty() -> bool:
    """Check if the working tree is dirty."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=get_repo_root(),
        )
        return bool(result.stdout.strip())
    except FileNotFoundError:
        return True


def create_manifest(
    *,
    stage: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a run manifest dict."""
    manifest: dict[str, Any] = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": get_git_hash(),
        "git_dirty": get_git_dirty(),
        "config": config,
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write manifest to a JSON file."""
    write_json(path, manifest)
