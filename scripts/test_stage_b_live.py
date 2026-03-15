#!/usr/bin/env python3
"""Test Stage B model live while training continues.

Uses the latest sampler weights checkpoint to sample from the model.

Usage:
    uv run python scripts/test_stage_b_live.py "Translate to Tuvaluan: Hello, how are you?"
    uv run python scripts/test_stage_b_live.py "Tali mai i te gana Tuvalu: What is the weather like today?"
    uv run python scripts/test_stage_b_live.py --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tv.common.tinker_runtime import (
    create_service_client,
    ensure_cookbook_on_path,
    get_renderer,
    get_sampling_params,
    require_tinker_api_key,
)


def get_latest_sampler_path() -> str:
    """Find latest sampler checkpoint from checkpoints.jsonl."""
    ckpt_file = REPO_ROOT / "logs" / "tinker" / "stage_b_llama8b" / "checkpoints.jsonl"
    if not ckpt_file.exists():
        raise SystemExit("No checkpoints found")

    sampler_path = None
    for line in ckpt_file.read_text().strip().split("\n"):
        entry = json.loads(line)
        if "sampler_path" in entry:
            sampler_path = entry["sampler_path"]

    if not sampler_path:
        raise SystemExit("No sampler checkpoints found — wait for step 500+")

    return sampler_path


def sample_one(prompt_text: str, sampling_client, renderer, sampling_params) -> str:
    """Send a prompt and get a response."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt = renderer.build_generation_prompt(messages)
    future = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1)
    result = future.result()
    output_tokens = result.sequences[0].tokens
    response_message, _ok = renderer.parse_response(output_tokens)
    if isinstance(response_message, dict):
        return str(response_message.get("content", ""))
    return str(response_message)


def main():
    parser = argparse.ArgumentParser(description="Test Stage B model live")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt to send")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    require_tinker_api_key()

    model_name = "Qwen/Qwen3-30B-A3B"
    _tokenizer, renderer, renderer_name = get_renderer(model_name)
    print(f"Renderer: {renderer_name}")

    sampler_path = get_latest_sampler_path()
    print(f"Sampler: {sampler_path}")

    service = create_service_client()
    sampling_client = service.create_sampling_client(model_path=sampler_path)
    sampling_params = get_sampling_params(
        renderer, max_tokens=args.max_tokens, temperature=args.temperature
    )
    print("Ready!\n")

    if args.interactive:
        while True:
            try:
                prompt_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not prompt_text:
                continue
            response = sample_one(prompt_text, sampling_client, renderer, sampling_params)
            print(f"Model: {response}\n")
    elif args.prompt:
        response = sample_one(args.prompt, sampling_client, renderer, sampling_params)
        print(f"Model: {response}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
