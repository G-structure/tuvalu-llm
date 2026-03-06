"""Tool-use mode switching for Stage B training data.

Two modes:
- SAFE (default): tool calls/results serialized as tagged text in assistant content.
  No special message types needed; works with any chat fine-tuning setup.
- NATIVE (experimental): uses Tinker's native tool message format. Behind config
  flag "tool_mode": "native".
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

SAFE_MODE: Literal["safe"] = "safe"
NATIVE_MODE: Literal["native"] = "native"
ToolMode = Literal["safe", "native"]

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_TOOL_RESULT_OPEN = "<tool_result>"
_TOOL_RESULT_CLOSE = "</tool_result>"

# Patterns for detecting tool-related content in messages
_TOOL_CALL_RE = re.compile(
    r"<tool_call>.*?</tool_call>", re.DOTALL
)
_TOOL_RESULT_RE = re.compile(
    r"<tool_result>.*?</tool_result>", re.DOTALL
)
_FUNCTION_CALL_RE = re.compile(
    r'"function_call"\s*:', re.DOTALL
)


def wrap_tool_call(call_json: str) -> str:
    """Wrap a tool call JSON string in safe-mode markers."""
    return f"{_TOOL_CALL_OPEN}\n{call_json}\n{_TOOL_CALL_CLOSE}"


def wrap_tool_result(result_json: str) -> str:
    """Wrap a tool result JSON string in safe-mode markers."""
    return f"{_TOOL_RESULT_OPEN}\n{result_json}\n{_TOOL_RESULT_CLOSE}"


def format_tool_message(message: dict[str, Any], mode: ToolMode) -> dict[str, Any]:
    """Format a message according to the chosen tool mode.

    In SAFE mode:
    - If the message has role "tool" or role "function", convert it to a "user"
      message with the content wrapped in <tool_result> tags.
    - If the message is an "assistant" message containing function_call or
      tool_calls metadata, wrap that data in <tool_call> tags in the content.

    In NATIVE mode:
    - Return the message unchanged (Tinker handles native tool messages).
    """
    if mode == NATIVE_MODE:
        return dict(message)

    role = message.get("role", "")
    out = dict(message)

    # Tool result messages -> user message with tagged content
    if role in ("tool", "function"):
        content = message.get("content", "")
        out = {
            "role": "user",
            "content": wrap_tool_result(content),
        }
        if "name" in message:
            out["content"] = f"[{message['name']}]\n{out['content']}"
        return out

    # Assistant messages with tool_calls metadata
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if tool_calls:
            parts = []
            text_content = message.get("content", "")
            if text_content:
                parts.append(text_content)
            for tc in tool_calls:
                tc_json = json.dumps(tc, ensure_ascii=False)
                parts.append(wrap_tool_call(tc_json))
            out = {"role": "assistant", "content": "\n".join(parts)}
            return out

        function_call = message.get("function_call")
        if function_call:
            parts = []
            text_content = message.get("content", "")
            if text_content:
                parts.append(text_content)
            fc_json = json.dumps(function_call, ensure_ascii=False)
            parts.append(wrap_tool_call(fc_json))
            out = {"role": "assistant", "content": "\n".join(parts)}
            return out

    return out


def format_messages(messages: list[dict[str, Any]], mode: ToolMode) -> list[dict[str, Any]]:
    """Format all messages in a conversation according to tool mode."""
    return [format_tool_message(msg, mode) for msg in messages]


def detect_tool_messages(messages: list[dict[str, Any]]) -> list[int]:
    """Return indices of messages that contain tool-related content.

    Detects:
    - Messages with role "tool" or "function"
    - Assistant messages with tool_calls or function_call keys
    - Messages whose content matches <tool_call> or <tool_result> patterns
    - Messages whose content matches "function_call" JSON patterns
    """
    indices: list[int] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("tool", "function"):
            indices.append(i)
            continue
        if role == "assistant" and (msg.get("tool_calls") or msg.get("function_call")):
            indices.append(i)
            continue
        if _TOOL_CALL_RE.search(content) or _TOOL_RESULT_RE.search(content):
            indices.append(i)
            continue
        if _FUNCTION_CALL_RE.search(content):
            indices.append(i)
            continue

    return indices
