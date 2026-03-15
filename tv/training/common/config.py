"""Config loading and validation for staged training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return json.load(f)


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge overrides into base config. Returns new dict."""
    result = dict(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path_str: str, base_dir: Path | None = None) -> Path:
    """Resolve a path string relative to base_dir (or repo root)."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
    return (base_dir / p).resolve()


def get_repo_root() -> Path:
    """Return the repository root (parent of training/)."""
    return Path(__file__).resolve().parent.parent.parent


def get_stage_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    """Extract stage-specific section from a config, merged with top-level defaults."""
    top_level = {k: v for k, v in config.items() if not isinstance(v, dict)}
    stage_section = config.get(stage, {})
    return merge_config(top_level, stage_section)
