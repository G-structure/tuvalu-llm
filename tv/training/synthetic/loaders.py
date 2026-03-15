"""Dataset loaders: download, normalize, and register HuggingFace datasets.

Each loader is a generator that yields normalized examples (common schema dicts).
Loaders are registered via the @register decorator from registry.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from tv.common.config import get_repo_root, resolve_path
from tv.common.io import read_jsonl
from tv.common.schema import make_example

from .normalize import generate_translate_mask, normalize_messages
from .registry import register

logger = logging.getLogger(__name__)


def _load_hf(
    path: str,
    name: str | None = None,
    split: str = "train",
    streaming: bool = True,
) -> Any:
    """Lazy HuggingFace datasets.load_dataset wrapper."""
    from datasets import load_dataset  # type: ignore

    return load_dataset(path, name, split=split, streaming=streaming)


def _limit_iter(ds: Any, limit: int | None) -> Iterator[Any]:
    """Iterate over a (possibly streaming) HF dataset, stopping after *limit* rows."""
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield row


def _iter_local_jsonl(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    """Read local JSONL rows with the same limit contract as HF loaders."""
    for i, row in enumerate(read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        yield row


# ---------------------------------------------------------------------------
# Chat loaders
# ---------------------------------------------------------------------------


@register("tasksource/tasksource-instruct-v0")
def load_tasksource(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """tasksource/tasksource-instruct-v0 -> task_family='chat'.

    Schema: {"inputs": str, "targets": str, "task": str, ...}
    Some rows use "prompt"/"completion" but "inputs"/"targets" is the dominant format.
    """
    ds = _load_hf("tasksource/tasksource-instruct-v0", split=split)
    for i, row in enumerate(_limit_iter(ds, limit)):
        user_text = row.get("inputs") or row.get("prompt") or ""
        asst_text = row.get("targets") or row.get("completion") or ""
        if not user_text or not asst_text:
            continue
        messages = [
            {"role": "user", "content": str(user_text)},
            {"role": "assistant", "content": str(asst_text)},
        ]
        mask = generate_translate_mask(messages, "chat")
        yield make_example(
            id=f"tasksource-{i}",
            task_family="chat",
            messages=messages,
            metadata={"source": "tasksource/tasksource-instruct-v0", "task": row.get("task", "")},
            translate_mask=mask,
        )


@register("HuggingFaceH4/ultrachat_200k")
def load_ultrachat(
    split: str = "train_sft",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """HuggingFaceH4/ultrachat_200k -> task_family='chat'.

    Schema: {"messages": [{"role": ..., "content": ...}], "prompt": str, ...}
    The "messages" field is already in chat format.
    """
    ds = _load_hf("HuggingFaceH4/ultrachat_200k", split=split)
    for i, row in enumerate(_limit_iter(ds, limit)):
        raw_messages = row.get("messages", [])
        if not raw_messages:
            continue
        messages = normalize_messages(raw_messages)
        mask = generate_translate_mask(messages, "chat")
        yield make_example(
            id=f"ultrachat-{i}",
            task_family="chat",
            messages=messages,
            metadata={"source": "HuggingFaceH4/ultrachat_200k"},
            translate_mask=mask,
        )


@register("private_tvl_chat")
def load_private_tvl_chat(
    input_path: str,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Normalize local private TVL/mixed chat JSONL into Stage B source schema.

    Expected input is either already in chat-style JSONL (`messages`) or a simpler
    two-turn form (`prompt`/`completion`, `user`/`assistant`).
    """
    repo_root = get_repo_root()
    path = resolve_path(input_path, repo_root)
    if not path.exists():
        raise FileNotFoundError(f"private_tvl_chat input not found: {path}")

    for i, row in enumerate(_iter_local_jsonl(path, limit)):
        raw_messages = row.get("messages")
        if raw_messages:
            messages = normalize_messages(raw_messages)
        elif row.get("prompt") and row.get("completion"):
            messages = [
                {"role": "user", "content": str(row["prompt"])},
                {"role": "assistant", "content": str(row["completion"])},
            ]
        elif row.get("user") and row.get("assistant"):
            messages = [
                {"role": "user", "content": str(row["user"])},
                {"role": "assistant", "content": str(row["assistant"])},
            ]
        else:
            logger.warning("Skipping private_tvl_chat row %d: no supported message fields", i)
            continue

        if not messages or messages[-1].get("role") != "assistant":
            logger.warning("Skipping private_tvl_chat row %d: final turn is not assistant", i)
            continue

        metadata = dict(row.get("metadata") or {})
        metadata.setdefault("source", "private_tvl_chat")
        metadata.setdefault("source_dataset", "private_tvl_chat")

        language_mode = (
            row.get("language_mode")
            or metadata.get("language_mode")
            or "tvl"
        )
        metadata["language_mode"] = language_mode

        split_group = (
            row.get("thread_id")
            or row.get("conversation_id")
            or metadata.get("thread_id")
            or metadata.get("conversation_id")
        )
        if split_group:
            metadata.setdefault("split_group", str(split_group))

        example_id = str(row.get("id") or f"private-tvl-chat-{i}")
        mask = generate_translate_mask(messages, "chat")
        yield make_example(
            id=example_id,
            task_family="chat",
            messages=messages,
            metadata=metadata,
            translate_mask=mask,
        )


# ---------------------------------------------------------------------------
# Math loaders
# ---------------------------------------------------------------------------


@register("openai/gsm8k")
def load_gsm8k(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """openai/gsm8k -> task_family='math'.

    Schema: {"question": str, "answer": str}
    The answer field contains chain-of-thought reasoning ending with ####<number>.
    """
    ds = _load_hf("openai/gsm8k", name="main", split=split)
    for i, row in enumerate(_limit_iter(ds, limit)):
        question = row.get("question", "")
        answer = row.get("answer", "")
        if not question or not answer:
            continue
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        # Math reasoning text is translatable, but the final #### number should be preserved
        mask = [{"translate": True}, {"translate": "selective"}]
        yield make_example(
            id=f"gsm8k-{i}",
            task_family="math",
            messages=messages,
            metadata={"source": "openai/gsm8k"},
            translate_mask=mask,
        )


# ---------------------------------------------------------------------------
# Tool / function-calling loaders
# ---------------------------------------------------------------------------


@register("Salesforce/xlam-function-calling-60k")
def load_xlam_fc(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Salesforce/xlam-function-calling-60k -> task_family='tool'.

    Schema: {"query": str, "answers": str (JSON list of tool calls),
             "tools": str (JSON list of tool schemas)}
    Preserve tool schemas in metadata; assistant output contains JSON tool calls.
    """
    ds = _load_hf("Salesforce/xlam-function-calling-60k", split=split)
    for i, row in enumerate(_limit_iter(ds, limit)):
        query = row.get("query", "")
        answers_raw = row.get("answers", "[]")
        tools_raw = row.get("tools", "[]")
        if not query:
            continue

        # Parse tool schemas and answers (stored as JSON strings)
        try:
            tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        except (json.JSONDecodeError, TypeError):
            tools = []
        try:
            answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        except (json.JSONDecodeError, TypeError):
            answers = []

        # Build system message with tool definitions
        system_content = "You have access to the following tools:\n" + json.dumps(tools, indent=2)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps(answers, indent=2)},
        ]
        # system=False (structural), user=True, assistant=False (JSON tool calls)
        mask = [{"translate": False}, {"translate": True}, {"translate": False}]
        yield make_example(
            id=f"xlam-fc-{i}",
            task_family="tool",
            messages=messages,
            metadata={
                "source": "Salesforce/xlam-function-calling-60k",
                "tools": tools,
            },
            translate_mask=mask,
        )


# ---------------------------------------------------------------------------
# Code loaders
# ---------------------------------------------------------------------------


@register("Muennighoff/mbpp")
def load_mbpp(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Muennighoff/mbpp -> task_family='code'.

    Schema: {"text": str, "code": str, "test_list": list[str], "task_id": int}
    The "text" is the prompt, "code" is the solution.
    """
    # google-research-datasets/mbpp has parquet (Muennighoff/mbpp uses deprecated scripts)
    ds = _load_hf("google-research-datasets/mbpp", split=split)
    for i, row in enumerate(_limit_iter(ds, limit)):
        prompt = row.get("text", "")
        code = row.get("code", "")
        if not prompt or not code:
            continue
        test_list = row.get("test_list", [])
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": code},
        ]
        # user text is translatable, code output is selective (docstrings translatable, code not)
        mask = [{"translate": True}, {"translate": "selective"}]
        yield make_example(
            id=f"mbpp-{row.get('task_id', i)}",
            task_family="code",
            messages=messages,
            metadata={
                "source": "Muennighoff/mbpp",
                "test_list": test_list,
            },
            translate_mask=mask,
        )


# ---------------------------------------------------------------------------
# QA loaders
# ---------------------------------------------------------------------------


@register("rajpurkar/squad")
def load_squad(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """rajpurkar/squad -> task_family='qa'.

    Schema: {"context": str, "question": str, "answers": {"text": [...], "answer_start": [...]},
             "id": str, "title": str}
    """
    ds = _load_hf("rajpurkar/squad", split=split)
    for i, row in enumerate(_limit_iter(ds, limit)):
        context = row.get("context", "")
        question = row.get("question", "")
        answers = row.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        if not context or not question or not answer_texts:
            continue
        answer = answer_texts[0]  # use first answer
        messages = [
            {"role": "system", "content": f"Answer the question based on the context below.\n\nContext: {context}"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        mask = generate_translate_mask(messages, "qa")
        yield make_example(
            id=f"squad-{row.get('id', i)}",
            task_family="qa",
            messages=messages,
            metadata={
                "source": "rajpurkar/squad",
                "title": row.get("title", ""),
                "all_answers": answer_texts,
            },
            translate_mask=mask,
        )


# ---------------------------------------------------------------------------
# Summarization loaders
# ---------------------------------------------------------------------------


@register("ccdv/cnn_dailymail")
def load_cnn_dailymail(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """ccdv/cnn_dailymail (3.0.0) -> task_family='summarization'.

    Schema: {"article": str, "highlights": str, "id": str}
    """
    # abisee/cnn_dailymail has parquet (ccdv uses deprecated scripts)
    ds = _load_hf("abisee/cnn_dailymail", name="3.0.0", split=split)
    for i, row in enumerate(_limit_iter(ds, limit)):
        article = row.get("article", "")
        highlights = row.get("highlights", "")
        if not article or not highlights:
            continue
        messages = [
            {"role": "user", "content": f"Summarize the following article:\n\n{article}"},
            {"role": "assistant", "content": highlights},
        ]
        mask = generate_translate_mask(messages, "summarization")
        yield make_example(
            id=f"cnn-dm-{row.get('id', i)}",
            task_family="summarization",
            messages=messages,
            metadata={"source": "ccdv/cnn_dailymail"},
            translate_mask=mask,
        )


# ---------------------------------------------------------------------------
# Story loaders
# ---------------------------------------------------------------------------


@register("roneneldan/TinyStories")
def load_tinystories(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """roneneldan/TinyStories -> task_family='chat'.

    Schema: {"text": str}
    Simple short stories (GPT-3.5/4 generated, small vocabulary).
    We frame as user asks for a story, assistant tells it.
    """
    ds = _load_hf("roneneldan/TinyStories", split=split)
    for i, row in enumerate(_limit_iter(ds, limit)):
        text = row.get("text", "").strip()
        if not text or len(text) < 50:
            continue
        messages = [
            {"role": "user", "content": "Tell me a short story."},
            {"role": "assistant", "content": text},
        ]
        # User prompt is simple/generic — translate both
        mask = [{"translate": True}, {"translate": True}]
        yield make_example(
            id=f"tinystories-{i}",
            task_family="chat",
            messages=messages,
            metadata={"source": "roneneldan/TinyStories"},
            translate_mask=mask,
        )


# ---------------------------------------------------------------------------
# TODO / partial loaders (registered but not fully implemented)
# ---------------------------------------------------------------------------


@register("meta-math/MetaMathQA")
def load_metamathqa(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """meta-math/MetaMathQA -> task_family='math'.

    TODO: Implement full loader. Expected schema:
    {"query": str, "response": str, "type": str, "original_question": str}
    """
    raise NotImplementedError(
        "meta-math/MetaMathQA loader not yet implemented. "
        "Expected fields: query, response, type, original_question."
    )
    yield  # type: ignore[misc]  # make it a generator


@register("NousResearch/hermes-function-calling-v1")
def load_hermes_fc(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """NousResearch/hermes-function-calling-v1 -> task_family='tool'.

    TODO: Implement full loader. Expected to have multi-turn function calling
    conversations with tool definitions.
    """
    raise NotImplementedError(
        "NousResearch/hermes-function-calling-v1 loader not yet implemented. "
        "Expected: multi-turn conversations with function calls."
    )
    yield  # type: ignore[misc]


@register("zai-org/AgentInstruct")
def load_agentinstruct(
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """zai-org/AgentInstruct -> task_family='chat'.

    TODO: Implement full loader. Expected to have instruction-following
    conversations.
    """
    raise NotImplementedError(
        "zai-org/AgentInstruct loader not yet implemented. "
        "Expected: instruction-following conversations."
    )
    yield  # type: ignore[misc]
