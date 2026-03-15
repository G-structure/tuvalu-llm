"""Prepare Stage A / Stage B datasets and configs for local MLX-LM training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tv.common.config import get_repo_root, resolve_path
from tv.common.io import read_jsonl, setup_run_dir, write_json, write_jsonl


DEFAULT_OUTPUT_ROOT = "out/local_mlx"


@dataclass(frozen=True)
class LocalPreset:
    """Conservative MLX-LM preset for a model/stage pair."""

    dataset_format: str
    mask_prompt: bool
    model: str | None
    batch_size: int
    grad_accumulation_steps: int
    num_layers: int
    max_seq_length: int
    grad_checkpoint: bool
    iters: int
    learning_rate: float
    val_batches: int
    steps_per_report: int
    steps_per_eval: int
    save_every: int
    adapter_subdir: str
    lora_keys: list[str]
    lora_rank: int
    lora_scale: float
    lora_dropout: float


PRESETS: dict[tuple[str, str], LocalPreset] = {
    (
        "stage_a_translation",
        "Qwen/Qwen3-30B-A3B-Base",
    ): LocalPreset(
        dataset_format="completions",
        mask_prompt=True,
        model=None,
        batch_size=1,
        grad_accumulation_steps=8,
        num_layers=8,
        max_seq_length=1536,
        grad_checkpoint=False,
        iters=2000,
        learning_rate=1e-5,
        val_batches=25,
        steps_per_report=25,
        steps_per_eval=500,
        save_every=250,
        adapter_subdir="adapters_stage_a_qwen30b",
        lora_keys=["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj"],
        lora_rank=8,
        lora_scale=16.0,
        lora_dropout=0.0,
    ),
    (
        "stage_b_agent",
        "Qwen/Qwen3-30B-A3B",
    ): LocalPreset(
        dataset_format="chat",
        mask_prompt=True,
        model=None,
        batch_size=1,
        grad_accumulation_steps=8,
        num_layers=8,
        max_seq_length=1536,
        grad_checkpoint=False,
        iters=2000,
        learning_rate=8e-6,
        val_batches=25,
        steps_per_report=25,
        steps_per_eval=500,
        save_every=250,
        adapter_subdir="adapters_stage_b_qwen30b",
        lora_keys=["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj"],
        lora_rank=8,
        lora_scale=16.0,
        lora_dropout=0.0,
    ),
    (
        "stage_b_agent",
        "openai/gpt-oss-120b",
    ): LocalPreset(
        dataset_format="chat",
        mask_prompt=True,
        model=None,
        batch_size=1,
        grad_accumulation_steps=8,
        num_layers=4,
        max_seq_length=1024,
        grad_checkpoint=False,
        iters=2000,
        learning_rate=5e-6,
        val_batches=25,
        steps_per_report=50,
        steps_per_eval=1000,
        save_every=500,
        adapter_subdir="adapters_stage_b_gpt_oss_120b",
        lora_keys=["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj"],
        lora_rank=8,
        lora_scale=16.0,
        lora_dropout=0.0,
    ),
}


def _stable_hash(value: str) -> int:
    import hashlib

    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _coerce_content(content: Any) -> str:
    """Best-effort conversion of structured message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "thinking":
                parts.append(str(item.get("thinking", "")))
            elif "content" in item:
                parts.append(str(item.get("content", "")))
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _render_prompt_completion(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Convert a chat example into prompt/completion form for base models."""
    if len(messages) < 2:
        raise ValueError("Need at least two messages to build a completion example")
    prompt_lines: list[str] = []
    for message in messages[:-1]:
        role = str(message.get("role", "user")).strip().capitalize()
        prompt_lines.append(f"{role}: {_coerce_content(message.get('content', ''))}".rstrip())
        prompt_lines.append("")
    prompt_lines.append("Assistant:")
    return {
        "prompt": "\n".join(prompt_lines).rstrip() + " ",
        "completion": _coerce_content(messages[-1].get("content", "")),
    }


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "role": message.get("role", "user"),
        "content": _coerce_content(message.get("content", "")),
    }
    for key in ("tool_calls", "tool_call_id", "name"):
        if key in message:
            out[key] = message[key]
    return out


def _export_rows(rows: list[dict[str, Any]], dataset_format: str) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages") or []
        if dataset_format == "completions":
            exported.append(_render_prompt_completion(messages))
            continue
        record: dict[str, Any] = {"messages": [_normalize_message(m) for m in messages]}
        if "tools" in row:
            record["tools"] = row["tools"]
        exported.append(record)
    return exported


def _filter_stage_b_rows(
    rows: list[dict[str, Any]],
    *,
    ablation_mode: str,
    include_task_families: list[str] | None,
    exclude_task_families: list[str] | None,
) -> list[dict[str, Any]]:
    if ablation_mode == "english_only":
        rows = [
            row
            for row in rows
            if row.get("metadata", {}).get("stage_b_source") == "english"
        ]
    elif ablation_mode == "tvl_only":
        rows = [
            row
            for row in rows
            if row.get("metadata", {}).get("stage_b_source") in ("synthetic_tvl", "anchor")
        ]
    elif ablation_mode != "mixed":
        raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

    if include_task_families is not None:
        rows = [row for row in rows if row.get("task_family") in include_task_families]
    if exclude_task_families is not None:
        rows = [row for row in rows if row.get("task_family") not in exclude_task_families]
    return rows


def _stage_a_spec(raw_config: dict[str, Any], *, pilot: bool) -> dict[str, Any]:
    data = raw_config.get("data", {})
    training = raw_config.get("training", {})
    logs = raw_config.get("logs", {})
    output_dir = data.get("output_dir", "data/finetune/stage_a_mt")
    train_file = data.get("train_file", "train_balanced.jsonl")
    if pilot and data.get("pilot_train_file"):
        train_file = data["pilot_train_file"]
    return {
        "stage": "stage_a_translation",
        "model_name": raw_config.get("model", {}).get("name", "Qwen/Qwen3-30B-A3B-Base"),
        "train_path": str(Path(output_dir) / train_file),
        "valid_path": str(Path(output_dir) / "validation.jsonl"),
        "test_path": str(Path(output_dir) / raw_config.get("eval", {}).get("test_file", "test.jsonl")),
        "epochs": int(training.get("epochs", 2)),
        "log_base_dir": logs.get("base_dir", "logs/local_mlx/stage_a"),
        "local_mlx": raw_config.get("local_mlx", {}),
    }


def _stage_b_spec(raw_config: dict[str, Any], *, pilot: bool) -> dict[str, Any]:
    data = raw_config.get("data", {})
    training = raw_config.get("training", {})
    logs = raw_config.get("logs", {})
    train_file = data.get("train_pilot_file") if pilot else data.get("train_file")
    if not train_file:
        train_file = "data/finetune/stage_b_mix/train.jsonl"
    return {
        "stage": "stage_b_agent",
        "model_name": raw_config.get("model", {}).get("name", "openai/gpt-oss-120b"),
        "train_path": train_file,
        "valid_path": data.get("validation_file", "data/finetune/stage_b_mix/validation.jsonl"),
        "test_path": raw_config.get("eval", {}).get("capability_test_file", "data/finetune/stage_b_mix/test.jsonl"),
        "epochs": int(training.get("epochs", 2)),
        "ablation_mode": training.get("ablation_mode", "mixed"),
        "include_task_families": training.get("included_task_families"),
        "exclude_task_families": training.get("excluded_task_families"),
        "log_base_dir": logs.get("base_dir", "logs/local_mlx/stage_b"),
        "local_mlx": raw_config.get("local_mlx", {}),
    }


def stage_spec_from_config(raw_config: dict[str, Any], *, pilot: bool = False) -> dict[str, Any]:
    stage = raw_config.get("stage")
    if stage == "stage_a_translation":
        return _stage_a_spec(raw_config, pilot=pilot)
    if stage == "stage_b_agent":
        return _stage_b_spec(raw_config, pilot=pilot)
    raise ValueError(f"Unsupported stage for local MLX prep: {stage!r}")


def _preset_for(stage: str, model_name: str) -> LocalPreset:
    key = (stage, model_name)
    if key in PRESETS:
        return PRESETS[key]
    if stage == "stage_b_agent":
        return LocalPreset(
            dataset_format="chat",
            mask_prompt=True,
            model=None,
            batch_size=1,
            grad_accumulation_steps=4,
            num_layers=4,
            max_seq_length=1024,
            grad_checkpoint=False,
            iters=1000,
            learning_rate=1e-5,
            val_batches=25,
            steps_per_report=50,
            steps_per_eval=500,
            save_every=250,
            adapter_subdir="adapters_stage_b",
            lora_keys=["self_attn.q_proj", "self_attn.v_proj"],
            lora_rank=8,
            lora_scale=16.0,
            lora_dropout=0.0,
        )
    return LocalPreset(
        dataset_format="completions",
        mask_prompt=True,
        model=None,
        batch_size=1,
        grad_accumulation_steps=4,
        num_layers=4,
        max_seq_length=1024,
        grad_checkpoint=False,
        iters=1000,
        learning_rate=1e-5,
        val_batches=25,
        steps_per_report=50,
        steps_per_eval=500,
        save_every=250,
        adapter_subdir="adapters_stage_a",
        lora_keys=["self_attn.q_proj", "self_attn.v_proj"],
        lora_rank=8,
        lora_scale=16.0,
        lora_dropout=0.0,
    )


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_dump(value: Any, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    return [f"{pad}{_yaml_scalar(value)}"]


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_yaml_dump(payload)) + "\n", encoding="utf-8")


def _build_mlx_config(
    *,
    stage_spec: dict[str, Any],
    preset: LocalPreset,
    data_dir: Path,
    run_dir: Path,
    mlx_model: str | None,
) -> dict[str, Any]:
    local_overrides = stage_spec.get("local_mlx", {})
    adapter_path = str(run_dir / preset.adapter_subdir)
    config: dict[str, Any] = {
        "model": mlx_model or local_overrides.get("model") or stage_spec["model_name"],
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "data": str(data_dir),
        "seed": int(local_overrides.get("seed", 0)),
        "num_layers": int(local_overrides.get("num_layers", preset.num_layers)),
        "batch_size": int(local_overrides.get("batch_size", preset.batch_size)),
        "iters": int(
            local_overrides.get(
                "iters",
                stage_spec["epochs"] * preset.iters,
            )
        ),
        "val_batches": int(local_overrides.get("val_batches", preset.val_batches)),
        "learning_rate": float(local_overrides.get("learning_rate", preset.learning_rate)),
        "steps_per_report": int(local_overrides.get("steps_per_report", preset.steps_per_report)),
        "steps_per_eval": int(local_overrides.get("steps_per_eval", preset.steps_per_eval)),
        "grad_accumulation_steps": int(
            local_overrides.get("grad_accumulation_steps", preset.grad_accumulation_steps)
        ),
        "resume_adapter_file": local_overrides.get("resume_adapter_file"),
        "adapter_path": adapter_path,
        "save_every": int(local_overrides.get("save_every", preset.save_every)),
        "test": False,
        "test_batches": int(local_overrides.get("test_batches", 100)),
        "max_seq_length": int(local_overrides.get("max_seq_length", preset.max_seq_length)),
        "grad_checkpoint": bool(
            local_overrides.get("grad_checkpoint", preset.grad_checkpoint)
        ),
        "lora_parameters": {
            "keys": local_overrides.get("lora_keys", preset.lora_keys),
            "rank": int(local_overrides.get("lora_rank", preset.lora_rank)),
            "scale": float(local_overrides.get("lora_scale", preset.lora_scale)),
            "dropout": float(local_overrides.get("lora_dropout", preset.lora_dropout)),
        },
    }
    if stage_spec["stage"] == "stage_a_translation" and preset.dataset_format == "completions":
        if "prompt_feature" in local_overrides:
            config["prompt_feature"] = local_overrides["prompt_feature"]
        if "completion_feature" in local_overrides:
            config["completion_feature"] = local_overrides["completion_feature"]
    return config


def _write_run_script(path: Path, *, config_path: Path, mask_prompt: bool) -> None:
    cmd = f'python -m mlx_lm lora --config "{config_path}"'
    if mask_prompt:
        cmd += " --mask-prompt"
    script = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + cmd + "\n"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _prepare_dataset(stage_spec: dict[str, Any], dataset_format: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    repo_root = get_repo_root()
    train_rows = _load_rows(resolve_path(stage_spec["train_path"], repo_root))
    valid_rows = _load_rows(resolve_path(stage_spec["valid_path"], repo_root))
    test_rows = _load_rows(resolve_path(stage_spec["test_path"], repo_root))

    if stage_spec["stage"] == "stage_b_agent":
        filter_kwargs = {
            "ablation_mode": stage_spec.get("ablation_mode", "mixed"),
            "include_task_families": stage_spec.get("include_task_families"),
            "exclude_task_families": stage_spec.get("exclude_task_families"),
        }
        train_rows = _filter_stage_b_rows(train_rows, **filter_kwargs)
        valid_rows = _filter_stage_b_rows(valid_rows, **filter_kwargs)
        test_rows = _filter_stage_b_rows(test_rows, **filter_kwargs)

    train_rows = sorted(train_rows, key=lambda row: _stable_hash(str(row.get("id", ""))))
    valid_rows = sorted(valid_rows, key=lambda row: _stable_hash(str(row.get("id", ""))))
    test_rows = sorted(test_rows, key=lambda row: _stable_hash(str(row.get("id", ""))))

    return (
        _export_rows(train_rows, dataset_format),
        _export_rows(valid_rows, dataset_format),
        _export_rows(test_rows, dataset_format),
    )


def prepare_local_mlx_run(
    raw_config: dict[str, Any],
    *,
    pilot: bool = False,
    mlx_model: str | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Prepare data + config artifacts for a local MLX-LM run."""
    stage_spec = stage_spec_from_config(raw_config, pilot=pilot)
    preset = _preset_for(stage_spec["stage"], stage_spec["model_name"])
    repo_root = get_repo_root()
    run_root = resolve_path(str(output_root), repo_root)
    run_dir = setup_run_dir(run_root / stage_spec["stage"], run_name)
    data_dir = run_dir / "data"
    train_rows, valid_rows, test_rows = _prepare_dataset(stage_spec, preset.dataset_format)

    write_jsonl(data_dir / "train.jsonl", train_rows)
    if valid_rows:
        write_jsonl(data_dir / "valid.jsonl", valid_rows)
    if test_rows:
        write_jsonl(data_dir / "test.jsonl", test_rows)

    mlx_config = _build_mlx_config(
        stage_spec=stage_spec,
        preset=preset,
        data_dir=data_dir,
        run_dir=run_dir,
        mlx_model=mlx_model,
    )
    config_path = run_dir / "mlx_lora_config.yaml"
    _write_yaml(config_path, mlx_config)

    run_script_path = run_dir / "run_local.sh"
    _write_run_script(run_script_path, config_path=config_path, mask_prompt=preset.mask_prompt)

    summary = {
        "stage": stage_spec["stage"],
        "model_name": stage_spec["model_name"],
        "mlx_model": mlx_config["model"],
        "dataset_format": preset.dataset_format,
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "train_examples": len(train_rows),
        "valid_examples": len(valid_rows),
        "test_examples": len(test_rows),
        "mlx_config_path": str(config_path),
        "run_script_path": str(run_script_path),
    }
    write_json(run_dir / "summary.json", summary)
    return summary
