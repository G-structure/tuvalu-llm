#!/usr/bin/env python3
"""Build cross-lingual instruction data: English prompts → Tuvaluan responses.

Pairs English source examples with their synthetic TVL translations to create
examples where the user speaks English but requests a Tuvaluan response.

Usage:
    uv run python scripts/build_crosslingual_data.py
    uv run python scripts/build_crosslingual_data.py --limit 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tv.common.io import read_jsonl

ENGLISH_DIR = REPO_ROOT / "data" / "finetune" / "stage_b_sources" / "english_normalized"
SYNTHETIC_DIR = REPO_ROOT / "data" / "finetune" / "stage_b_synthetic_tvl" / "accepted"
OUTPUT_DIR = REPO_ROOT / "data" / "finetune" / "stage_b_crosslingual"

# Instruction variants to append/prepend — keeps the model robust to phrasing
RESPOND_IN_TVL = [
    "Respond in Tuvaluan.",
    "Please respond in Tuvaluan.",
    "Answer in Tuvaluan.",
    "Reply in Tuvaluan.",
    "(Respond in Tuvaluan)",
    "(Please answer in Tuvaluan.)",
    "Tali mai i te gana Tuvalu.",  # "Reply in Tuvaluan language"
    "Fai mai i te gana Tuvalu.",   # "Say it in Tuvaluan language"
]

SYSTEM_PROMPTS = [
    "You are a helpful assistant. The user may ask you to respond in Tuvaluan.",
    "You are a bilingual assistant fluent in English and Tuvaluan.",
    "You are a helpful assistant that can respond in both English and Tuvaluan.",
    None,  # no system prompt
]


def build_crosslingual_example(
    en_example: dict,
    tvl_example: dict,
    rng: random.Random,
) -> dict | None:
    """Create a cross-lingual example: English prompt → TVL response."""
    en_msgs = en_example.get("messages", [])
    tvl_msgs = tvl_example.get("messages", [])

    if len(en_msgs) < 2 or len(tvl_msgs) < 2:
        return None

    # Find user and assistant messages
    en_user_msgs = [m for m in en_msgs if m["role"] == "user"]
    tvl_asst_msgs = [m for m in tvl_msgs if m["role"] == "assistant"]

    if not en_user_msgs or not tvl_asst_msgs:
        return None

    instruction = rng.choice(RESPOND_IN_TVL)
    system = rng.choice(SYSTEM_PROMPTS)

    # Build new message list
    new_messages = []
    if system:
        new_messages.append({"role": "system", "content": system})

    # For multi-turn, interleave EN user with TVL assistant
    en_turns = [m for m in en_msgs if m["role"] in ("user", "assistant")]
    tvl_turns = [m for m in tvl_msgs if m["role"] in ("user", "assistant")]

    # Simple case: single user-assistant pair
    if len(en_user_msgs) == 1 and len(tvl_asst_msgs) >= 1:
        user_content = en_user_msgs[0]["content"]
        # Randomly place the instruction
        placement = rng.choice(["append", "prepend", "newline"])
        if placement == "append":
            user_content = user_content.rstrip() + " " + instruction
        elif placement == "prepend":
            user_content = instruction + " " + user_content
        else:
            user_content = user_content.rstrip() + "\n\n" + instruction

        new_messages.append({"role": "user", "content": user_content})
        new_messages.append({"role": "assistant", "content": tvl_asst_msgs[-1]["content"]})
    else:
        # Multi-turn: add instruction to first user message, pair EN user with TVL assistant
        first_user = True
        en_idx = 0
        tvl_idx = 0
        for turn in en_turns:
            if turn["role"] == "user":
                content = turn["content"]
                if first_user:
                    content = content.rstrip() + "\n\n" + instruction
                    first_user = False
                new_messages.append({"role": "user", "content": content})
                en_idx += 1
            elif turn["role"] == "assistant":
                # Use TVL assistant response if available
                if tvl_idx < len(tvl_asst_msgs):
                    new_messages.append({"role": "assistant", "content": tvl_asst_msgs[tvl_idx]["content"]})
                    tvl_idx += 1
                else:
                    break

    if len([m for m in new_messages if m["role"] == "assistant"]) == 0:
        return None

    return {
        "id": f"crosslingual-{en_example.get('id', '')}",
        "task_family": en_example.get("task_family", "chat"),
        "messages": new_messages,
        "metadata": {
            "source": "crosslingual",
            "source_dataset": en_example.get("metadata", {}).get("source", ""),
            "original_id": en_example.get("id", ""),
            "instruction_language": "en",
            "response_language": "tvl",
        },
    }


def process_dataset(
    en_file: Path,
    tvl_file: Path,
    out_file: Path,
    rng: random.Random,
    limit: int | None = None,
) -> dict:
    """Process one dataset pair, return stats."""
    # Index TVL examples by ID for fast lookup
    tvl_by_id: dict[str, dict] = {}
    for row in read_jsonl(tvl_file):
        tvl_by_id[row.get("id", "")] = row

    count = 0
    skipped = 0
    with out_file.open("w") as f:
        for en_row in read_jsonl(en_file):
            if limit and count >= limit:
                break
            en_id = en_row.get("id", "")
            tvl_row = tvl_by_id.get(en_id)
            if not tvl_row:
                skipped += 1
                continue

            result = build_crosslingual_example(en_row, tvl_row, rng)
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
            else:
                skipped += 1

    return {"accepted": count, "skipped": skipped, "output": str(out_file)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-lingual EN→TVL data")
    parser.add_argument("--limit", type=int, default=None, help="Max examples per dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find matching EN/TVL pairs
    en_files = sorted(ENGLISH_DIR.glob("*.jsonl"))
    stats = {}

    for en_file in en_files:
        name = en_file.stem
        tvl_file = SYNTHETIC_DIR / f"{name}.jsonl"
        if not tvl_file.exists():
            print(f"  {name}: no TVL translation, skipping")
            continue

        out_file = OUTPUT_DIR / f"{name}.jsonl"
        result = process_dataset(en_file, tvl_file, out_file, rng, limit=args.limit)
        stats[name] = result
        print(f"  {name}: {result['accepted']:,} cross-lingual examples")

    # Summary
    total = sum(s["accepted"] for s in stats.values())
    print(f"\nTotal: {total:,} cross-lingual examples in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
