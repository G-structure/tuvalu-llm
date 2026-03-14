# Local MLX Training

This repo now has a local MLX-LM preparation path for both:

- Stage A translation training
- Stage B bilingual capability training

The local path does not replace Tinker. It prepares a separate run directory with:

- MLX-ready dataset files
- an `mlx_lora_config.yaml`
- a `run_local.sh` launcher

## Why this path exists

The existing training scripts are built around Tinker. That is fine for remote training, but it is the wrong abstraction if the goal is to train locally on Apple Silicon as fast as possible.

For local runs, the fastest practical path is:

1. reuse the repo's existing dataset builders and split logic
2. export Stage A / Stage B data into MLX-LM-friendly formats
3. train directly with `mlx-lm`

## Stage behavior

### Stage A

Stage A starts from a base model by default (`Qwen/Qwen3-30B-A3B-Base`), so the local exporter writes prompt/completion records:

- `prompt`
- `completion`

### Stage B

Stage B targets chat models (`Qwen/Qwen3-30B-A3B`, `openai/gpt-oss-120b`), so the local exporter preserves chat-style:

- `messages`

The Stage B local path also respects:

- `ablation_mode`
- `included_task_families`
- `excluded_task_families`

## Commands

Prepare a Stage A local run:

```bash
uv run python scripts/prepare_local_mlx_training.py \
  --config configs/stage_a_translation_qwen30b_base.json \
  --mlx-model mlx-community/Qwen3-30B-A3B-Base-4bit
```

Prepare a Stage B local Qwen run:

```bash
uv run python scripts/prepare_local_mlx_training.py \
  --config configs/stage_b_agent_qwen30b.json \
  --mlx-model mlx-community/Qwen3-30B-A3B-4bit
```

Prepare a Stage B local `gpt-oss-120b` run:

```bash
uv run python scripts/prepare_local_mlx_training.py \
  --config configs/stage_b_agent_oss120b.json \
  --mlx-model mlx-community/gpt-oss-120b-4bit
```

For pilot Stage B data:

```bash
uv run python scripts/prepare_local_mlx_training.py \
  --config configs/stage_b_agent_oss120b.json \
  --mlx-model mlx-community/gpt-oss-120b-4bit \
  --pilot
```

The script prints a JSON summary with the generated run directory. Inside that directory:

- `data/train.jsonl`
- `data/valid.jsonl`
- `data/test.jsonl`
- `mlx_lora_config.yaml`
- `run_local.sh`

## Running the prepared job

The generated launcher runs:

```bash
python -m mlx_lm lora --config /path/to/mlx_lora_config.yaml --mask-prompt
```

Run the generated script directly:

```bash
bash out/local_mlx/.../run_local.sh
```

## Notes

- The generated hyperparameters are conservative speed-first presets, not final-quality settings.
- `gpt-oss-120b` is configured very defensively: batch 1, short sequences, few adapted layers.
- If the model does not fit without checkpointing, flip `grad_checkpoint` in the generated YAML.
- If you want persistent overrides, add a `local_mlx` section to the JSON config and rerun prep.
