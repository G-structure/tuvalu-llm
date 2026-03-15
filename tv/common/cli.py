"""Shared CLI/config helpers for thin script wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config


def load_optional_config(path: str | Path | None) -> dict[str, Any]:
    """Load a JSON config file or return an empty config when absent."""
    if path is None:
        return {}
    return load_config(path)


def merge_cli_overrides(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of base with non-None CLI overrides applied."""
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
