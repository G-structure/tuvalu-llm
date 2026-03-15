# Local Training Plan

## Goal

Add a first-class local training path for this repo that covers both:

- Stage A translation training
- Stage B bilingual capability training

The local path should optimize for Apple Silicon throughput, use `mlx-lm` as the trainer, and reuse the existing dataset builders and config files instead of creating a second disconnected pipeline.

## Constraints

- Stage A currently trains a base model (`Qwen/Qwen3-30B-A3B-Base`) through Tinker. That means the local export format cannot assume a chat-template-capable instruct model.
- Stage B currently targets chat/instruct models (`Qwen/Qwen3-30B-A3B`, `openai/gpt-oss-120b`), so the local path should preserve chat messages where possible.
- `gpt-oss-120b` local training is only realistic as a tightly constrained QLoRA/LoRA run on a very large Apple Silicon machine. The repo should support it, but the defaults must be conservative.
- We should avoid making local MLX support depend on Tinker at runtime.

## Proposed Shape

### 1. Add a generic local MLX preparation path

Create a new local-training module plus a CLI that:

- reads an existing repo config file
- detects whether it is a Stage A or Stage B config
- resolves the correct train/validation/test files
- filters Stage B data using the existing ablation and task-family rules
- writes an MLX-ready dataset directory
- writes an `mlx-lm` config file
- emits the exact command to launch training locally

This should be one generic entrypoint, not separate duplicated Stage A and Stage B implementations.

### 2. Export different dataset formats by stage/model type

For Stage A:

- export prompt/completion format by default
- render prompts in the same simple role-based structure the current base-model pipeline expects
- keep assistant text in the completion side so prompt masking remains possible

For Stage B:

- export chat `messages` format for `mlx-lm`
- preserve the existing message structure from the mixed dataset
- avoid inventing a new internal schema

This split matters because Stage A uses a base model while Stage B uses chat models.

### 3. Add model-specific MLX presets

Encode conservative local presets for at least:

- `Qwen/Qwen3-30B-A3B-Base` for Stage A
- `Qwen/Qwen3-30B-A3B` for Stage B
- `openai/gpt-oss-120b` for Stage B

The presets should cover:

- micro-batch size
- gradient accumulation
- sequence length
- number of adapted layers
- checkpointing default
- LoRA rank and target modules
- logging / eval / save cadence

The defaults should be speed-first and memory-safe, not paper-maximal.

### 4. Keep config reuse, not config drift

Do not fork the training pipeline into “Tinker configs” and “local configs” unless necessary.

Instead:

- continue using the existing JSON configs
- optionally add a small `local_mlx` section for overrides
- let the new local CLI derive the rest from the existing config plus model presets

That keeps Stage A and Stage B aligned on the same source of truth for model choice, data paths, and train/val splits.

### 5. Document the operational path

Add documentation that explains:

- when to use local MLX vs Tinker
- which stages/models are supported locally
- why Stage A export differs from Stage B export
- how to point the script at a local MLX model directory
- recommended commands for Stage A Qwen, Stage B Qwen, and Stage B `gpt-oss-120b`
- hardware caveats for `gpt-oss-120b`

This should live in a dedicated local-training doc plus a short pointer from the main training pipeline doc.

### 6. Add light validation

Add tests for:

- config detection for Stage A vs Stage B
- Stage A prompt/completion export shape
- Stage B chat export shape
- Stage B ablation/task-family filtering in the local path
- CLI `--help` smoke coverage

The goal is to protect the translation/export layer, not to integration-test MLX itself.

## Implementation Order

1. Add `training/local_mlx/` helpers.
2. Add the generic local prep CLI under `scripts/`.
3. Wire Stage A and Stage B config translation into that CLI.
4. Add preset definitions for Qwen Stage A, Qwen Stage B, and `gpt-oss-120b`.
5. Add docs and usage examples.
6. Add tests.

## Non-Goals For The First Pass

- Full local evaluation parity with the Tinker eval scripts
- Automatic MLX model conversion or download
- Auto-detection of Mac unified memory and dynamic hyperparameter tuning
- Native distributed multi-Mac training
- Support for every model in the repo on day one

## Expected Outcome

After this change, we should be able to do the following without touching the remote Tinker path:

- prepare and launch Stage A training locally on MLX
- prepare and launch Stage B Qwen training locally on MLX
- prepare and launch Stage B `gpt-oss-120b` training locally on MLX with conservative defaults

The existing Tinker training flow should remain intact.
