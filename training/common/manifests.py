"""Run manifest generation for reproducibility."""

from __future__ import annotations

import hashlib
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


def get_git_diff() -> str:
    """Return the git diff of tracked files against HEAD."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            cwd=get_repo_root(),
        )
        return result.stdout if result.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def hash_file(path: Path) -> str:
    """Return SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_data_files(*paths: Path | str) -> dict[str, str]:
    """Return {filename: sha256} for each existing path."""
    result: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        if p.exists():
            result[p.name] = hash_file(p)
    return result


def create_manifest(
    *,
    stage: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
    data_files: list[Path | str] | None = None,
) -> dict[str, Any]:
    """Create a run manifest dict with full reproducibility info."""
    git_diff = get_git_diff()
    manifest: dict[str, Any] = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": get_git_hash(),
        "git_dirty": get_git_dirty(),
        "git_diff_length": len(git_diff),
        "config": config,
    }
    if data_files:
        manifest["data_hashes"] = hash_data_files(*data_files)
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write manifest to a JSON file."""
    write_json(path, manifest)


def save_git_diff(log_dir: Path) -> Path:
    """Save the current git diff to log_dir/git_diff.patch."""
    diff = get_git_diff()
    diff_path = log_dir / "git_diff.patch"
    diff_path.write_text(diff)
    return diff_path
