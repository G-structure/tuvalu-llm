"""Tests for local MLX training prep."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.local_mlx.prepare import prepare_local_mlx_run, stage_spec_from_config


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_stage_spec_stage_a():
    raw = {
        "stage": "stage_a_translation",
        "model": {"name": "Qwen/Qwen3-30B-A3B-Base"},
        "data": {"output_dir": "data/finetune/stage_a_mt", "train_file": "train_balanced.jsonl"},
        "eval": {"test_file": "test.jsonl"},
    }
    spec = stage_spec_from_config(raw)
    assert spec["stage"] == "stage_a_translation"
    assert spec["train_path"].endswith("data/finetune/stage_a_mt/train_balanced.jsonl")


def test_stage_spec_stage_b():
    raw = {
        "stage": "stage_b_agent",
        "model": {"name": "openai/gpt-oss-120b"},
        "data": {
            "train_file": "data/finetune/stage_b_mix/train.jsonl",
            "validation_file": "data/finetune/stage_b_mix/validation.jsonl",
        },
        "eval": {"capability_test_file": "data/finetune/stage_b_mix/test.jsonl"},
        "training": {"ablation_mode": "mixed"},
    }
    spec = stage_spec_from_config(raw)
    assert spec["stage"] == "stage_b_agent"
    assert spec["model_name"] == "openai/gpt-oss-120b"


def test_prepare_local_mlx_stage_a_completions(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "tmp_local_mlx_stage_a"
    rows = [
        {
            "id": "ex1",
            "messages": [
                {"role": "system", "content": "You are a translator."},
                {"role": "user", "content": "Translate this."},
                {"role": "assistant", "content": "Fakamatala."},
            ],
            "metadata": {"direction": "en_to_tvl"},
        }
    ]
    _write_jsonl(data_dir / "train.jsonl", rows)
    _write_jsonl(data_dir / "validation.jsonl", rows)
    _write_jsonl(data_dir / "test.jsonl", rows)

    raw = {
        "stage": "stage_a_translation",
        "model": {"name": "Qwen/Qwen3-30B-A3B-Base"},
        "data": {"output_dir": str(data_dir), "train_file": "train.jsonl"},
        "eval": {"test_file": "test.jsonl"},
    }
    summary = prepare_local_mlx_run(
        raw,
        output_root=tmp_path,
        run_name="stage_a_case",
        mlx_model="mlx-community/Qwen3-30B-A3B-Base-4bit",
    )
    train_export = Path(summary["data_dir"]) / "train.jsonl"
    exported = [json.loads(line) for line in train_export.read_text(encoding="utf-8").splitlines()]
    assert "prompt" in exported[0]
    assert "completion" in exported[0]
    assert exported[0]["prompt"].endswith("Assistant: ")


def test_prepare_local_mlx_stage_b_chat_filter(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "tmp_local_mlx_stage_b"
    train_rows = [
        {
            "id": "chat1",
            "task_family": "chat",
            "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Talofa"}],
            "metadata": {"stage_b_source": "english"},
        },
        {
            "id": "tool1",
            "task_family": "tool",
            "messages": [{"role": "user", "content": "Tool?"}, {"role": "assistant", "content": "No"}],
            "metadata": {"stage_b_source": "english"},
        },
        {
            "id": "anchor1",
            "task_family": "translation",
            "messages": [{"role": "user", "content": "Translate"}, {"role": "assistant", "content": "Fakaliliu"}],
            "metadata": {"stage_b_source": "anchor"},
        },
    ]
    _write_jsonl(data_dir / "train.jsonl", train_rows)
    _write_jsonl(data_dir / "validation.jsonl", train_rows)
    _write_jsonl(data_dir / "test.jsonl", train_rows)

    raw = {
        "stage": "stage_b_agent",
        "model": {"name": "openai/gpt-oss-120b"},
        "data": {
            "train_file": str(data_dir / "train.jsonl"),
            "validation_file": str(data_dir / "validation.jsonl"),
        },
        "eval": {"capability_test_file": str(data_dir / "test.jsonl")},
        "training": {
            "ablation_mode": "english_only",
            "included_task_families": ["chat"],
        },
    }
    summary = prepare_local_mlx_run(
        raw,
        output_root=tmp_path,
        run_name="stage_b_case",
        mlx_model="mlx-community/gpt-oss-120b-4bit",
    )
    train_export = Path(summary["data_dir"]) / "train.jsonl"
    exported = [json.loads(line) for line in train_export.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 1
    assert exported[0]["messages"][0]["role"] == "user"
    assert Path(summary["run_script_path"]).read_text(encoding="utf-8").strip().endswith("--mask-prompt")
