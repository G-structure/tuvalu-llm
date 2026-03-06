#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SUBMODULE_PATH="vendor/tinker-cookbook"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required" >&2
  exit 1
fi

if [ ! -d .git ]; then
  echo "Run this from inside your git repo." >&2
  exit 1
fi

echo "Initializing/updating submodule..."
git submodule update --init --recursive "$SUBMODULE_PATH"

echo "Installing Python dependencies..."
uv pip install -e "$SUBMODULE_PATH"
uv pip install -e ".[training]"

echo
cat <<MSG
Bootstrap complete.

Staged training pipeline:
  1. Export your API key:  export TINKER_API_KEY=...

  Stage A (translation adapter):
  2. Build Stage A data:
       uv run python scripts/build_stage_a_mt_data.py --config configs/stage_a_translation_qwen30b_base.json
  3. Train Stage A:
       uv run python scripts/train_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json
  4. Evaluate Stage A:
       uv run python scripts/eval_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json

  Synthetic generation:
  5. Build English source pool:
       uv run python scripts/build_stage_b_sources.py --config configs/synthetic_stage_b_core.json
  6. Generate synthetic Tuvaluan:
       uv run python scripts/generate_stage_b_synthetic_tvl.py --config configs/synthetic_stage_b_core.json

  Stage B (bilingual agent):
  7. Build mixed training data:
       uv run python scripts/build_stage_b_mix.py --config configs/stage_b_mix_default.json
  8. Train Stage B:
       uv run python scripts/train_stage_b_agent.py --config configs/stage_b_agent_oss120b.json
  9. Evaluate Stage B:
       uv run python scripts/eval_stage_b_agent.py --config configs/stage_b_agent_oss120b.json

  See docs/TRAINING_PIPELINE.md for full documentation.
MSG
