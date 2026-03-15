"""Tests for Stage B source loading, including local private TVL chat."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tv.common.io import write_jsonl
from tv.training.synthetic.loaders import load_private_tvl_chat


def test_load_private_tvl_chat_from_messages(tmp_path: Path):
    input_path = tmp_path / "private_tvl_chat.jsonl"
    write_jsonl(input_path, [
        {
            "id": "chat-1",
            "thread_id": "thread-1",
            "messages": [
                {"role": "user", "content": "Talofa"},
                {"role": "assistant", "content": "Malo ni"},
            ],
            "metadata": {"is_private": True},
        }
    ])

    rows = list(load_private_tvl_chat(str(input_path)))
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "chat-1"
    assert row["task_family"] == "chat"
    assert row["messages"][-1]["role"] == "assistant"
    assert row["metadata"]["source_dataset"] == "private_tvl_chat"
    assert row["metadata"]["split_group"] == "thread-1"
    assert row["metadata"]["language_mode"] == "tvl"


def test_load_private_tvl_chat_from_prompt_completion(tmp_path: Path):
    input_path = tmp_path / "private_tvl_chat_pc.jsonl"
    write_jsonl(input_path, [
        {
            "id": "chat-2",
            "conversation_id": "conv-9",
            "prompt": "Can you help?",
            "completion": "Ioe, e mafai.",
            "language_mode": "mixed",
        }
    ])

    rows = list(load_private_tvl_chat(str(input_path)))
    assert len(rows) == 1
    row = rows[0]
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"
    assert row["metadata"]["split_group"] == "conv-9"
    assert row["metadata"]["language_mode"] == "mixed"
