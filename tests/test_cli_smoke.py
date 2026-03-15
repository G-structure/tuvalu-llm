"""Smoke tests for CLI entrypoints (--help should work)."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "scripts/clean_pipeline.py",
    "scripts/build_splits.py",
    "scripts/render_training_data.py",
    "scripts/build_stage_a_mt_data.py",
    "scripts/train_stage_a_translation.py",
    "scripts/eval_stage_a_translation.py",
    "scripts/export_stage_a_translation.py",
    "scripts/prepare_local_mlx_training.py",
    "scripts/build_stage_b_sources.py",
    "scripts/generate_stage_b_synthetic_tvl.py",
    "scripts/build_stage_b_mix.py",
    "scripts/train_stage_b_agent.py",
    "scripts/eval_stage_b_agent.py",
]


def _run_help(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )


def test_build_stage_a_help():
    result = _run_help("scripts/build_stage_a_mt_data.py")
    assert result.returncode == 0, result.stderr


def test_clean_pipeline_help():
    result = _run_help("scripts/clean_pipeline.py")
    assert result.returncode == 0, result.stderr


def test_build_splits_help():
    result = _run_help("scripts/build_splits.py")
    assert result.returncode == 0, result.stderr


def test_render_training_data_help():
    result = _run_help("scripts/render_training_data.py")
    assert result.returncode == 0, result.stderr


def test_train_stage_a_help():
    result = _run_help("scripts/train_stage_a_translation.py")
    assert result.returncode == 0, result.stderr


def test_eval_stage_a_help():
    result = _run_help("scripts/eval_stage_a_translation.py")
    assert result.returncode == 0, result.stderr


def test_build_sources_help():
    result = _run_help("scripts/build_stage_b_sources.py")
    assert result.returncode == 0, result.stderr


def test_generate_synthetic_help():
    result = _run_help("scripts/generate_stage_b_synthetic_tvl.py")
    assert result.returncode == 0, result.stderr


def test_build_mix_help():
    result = _run_help("scripts/build_stage_b_mix.py")
    assert result.returncode == 0, result.stderr


def test_train_stage_b_help():
    result = _run_help("scripts/train_stage_b_agent.py")
    assert result.returncode == 0, result.stderr


def test_eval_stage_b_help():
    result = _run_help("scripts/eval_stage_b_agent.py")
    assert result.returncode == 0, result.stderr


def test_export_stage_a_help():
    result = _run_help("scripts/export_stage_a_translation.py")
    assert result.returncode == 0, result.stderr


def test_prepare_local_mlx_help():
    result = _run_help("scripts/prepare_local_mlx_training.py")
    assert result.returncode == 0, result.stderr


def test_deprecated_build_help():
    """Old deprecated script should still work."""
    result = _run_help("scripts/build_tinker_mt_data.py")
    assert result.returncode == 0, result.stderr


def test_all_scripts_exist():
    """All expected scripts exist."""
    for script in SCRIPTS:
        assert (REPO_ROOT / script).exists(), f"Missing: {script}"
