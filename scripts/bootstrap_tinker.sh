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
uv pip install tinker datasets sacrebleu pyarrow pandas

echo
cat <<MSG
Bootstrap complete.

Next steps:
  1. Export your API key:  export TINKER_API_KEY=...
  2. Build MT data:
       uv run python scripts/build_tinker_mt_data.py
  3. Train a first adapter:
       uv run python scripts/train_tinker_mt.py --data data/finetune/tinker_mt/train_balanced.jsonl
  4. Evaluate it:
       uv run python scripts/eval_tinker_mt.py --model-path <tinker://.../weights/final> \\
         --data data/finetune/tinker_mt/validation.jsonl
MSG
