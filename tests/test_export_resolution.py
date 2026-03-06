"""Test Stage A model path resolution in synthetic generation."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.io import write_json


def _make_stage_a_config(tmp_path, model_path=None, log_path=None):
    """Build a minimal stage_a_model config dict."""
    cfg = {"base_model": "test/model"}
    if model_path is not None:
        cfg["model_path"] = model_path
    if log_path is not None:
        cfg["log_path"] = log_path
    return cfg


def _resolve_model_path(stage_a_config, repo_root):
    """Extract the resolution logic matching generate.py."""
    from training.common.config import resolve_path
    from training.common.io import read_json

    model_path = stage_a_config.get("model_path")
    if not model_path and stage_a_config.get("log_path"):
        log_path = resolve_path(stage_a_config["log_path"], repo_root)
        export_info_path = log_path / "export_info.json"
        if export_info_path.exists():
            export_info = read_json(export_info_path)
            model_path = export_info.get("model_path")
        else:
            from training.stage_a_mt.export import get_model_path

            try:
                model_path = get_model_path({"log_path": stage_a_config["log_path"]})
            except FileNotFoundError:
                pass
    return model_path


def test_resolution_from_explicit_model_path(tmp_path):
    """When model_path is set explicitly, use it directly."""
    cfg = _make_stage_a_config(tmp_path, model_path="/trained/model/v1")
    result = _resolve_model_path(cfg, tmp_path)
    assert result == "/trained/model/v1"


def test_resolution_from_export_info(tmp_path):
    """When export_info.json exists in log_path, read model_path from it."""
    log_dir = tmp_path / "logs" / "stage_a"
    log_dir.mkdir(parents=True)
    write_json(log_dir / "export_info.json", {"model_path": "/exported/model/v2"})

    cfg = _make_stage_a_config(tmp_path, log_path=str(log_dir))
    result = _resolve_model_path(cfg, tmp_path)
    assert result == "/exported/model/v2"


def test_resolution_from_checkpoint_fallback(tmp_path):
    """When no export_info.json, fall back to get_model_path."""
    log_dir = tmp_path / "logs" / "stage_a"
    log_dir.mkdir(parents=True)
    # No export_info.json

    cfg = _make_stage_a_config(tmp_path, log_path=str(log_dir))

    with patch(
        "training.stage_a_mt.export.get_model_path",
        return_value="/checkpoint/model/v3",
    ):
        result = _resolve_model_path(cfg, tmp_path)
    assert result == "/checkpoint/model/v3"


def test_resolution_fallback_returns_none(tmp_path):
    """When nothing can resolve, model_path stays None."""
    log_dir = tmp_path / "logs" / "stage_a"
    log_dir.mkdir(parents=True)

    cfg = _make_stage_a_config(tmp_path, log_path=str(log_dir))

    with patch(
        "training.stage_a_mt.export.get_model_path",
        side_effect=FileNotFoundError("no checkpoint"),
    ):
        result = _resolve_model_path(cfg, tmp_path)
    assert result is None
