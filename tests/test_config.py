"""Test config loading and validation."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.config import load_config, merge_config, get_repo_root


def test_load_config():
    """Test loading a JSON config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"key": "value", "nested": {"a": 1}}, f)
        f.flush()
        config = load_config(f.name)
    assert config["key"] == "value"
    assert config["nested"]["a"] == 1


def test_load_config_not_found():
    """Test that missing config raises FileNotFoundError."""
    try:
        load_config("/nonexistent/path/config.json")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass


def test_merge_config():
    """Test deep merging of configs."""
    base = {"a": 1, "nested": {"x": 10, "y": 20}, "b": 2}
    overrides = {"a": 99, "nested": {"x": 99}}
    result = merge_config(base, overrides)
    assert result["a"] == 99
    assert result["nested"]["x"] == 99
    assert result["nested"]["y"] == 20
    assert result["b"] == 2


def test_merge_config_new_keys():
    """Test merging adds new keys."""
    base = {"a": 1}
    overrides = {"b": 2, "c": {"d": 3}}
    result = merge_config(base, overrides)
    assert result == {"a": 1, "b": 2, "c": {"d": 3}}


def test_get_repo_root():
    """Test repo root detection."""
    root = get_repo_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_all_stage_configs_valid():
    """Test that all stage config files parse as valid JSON."""
    configs_dir = get_repo_root() / "configs"
    for config_path in configs_dir.glob("*.json"):
        config = load_config(config_path)
        assert isinstance(config, dict), f"{config_path.name} not a dict"
