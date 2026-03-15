"""Test selective translation masking/unmasking and quality validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tv.training.synthetic.selective_translate import (
    classify_message_content,
    mask_protected_spans,
    selective_translate_example,
    unmask_protected_spans,
)
from tv.training.synthetic.quality import (
    check_placeholder_leaks,
    validate_code_preservation,
    validate_json_preservation,
    validate_translation,
)


class TestMasking:
    def test_code_fence_masked(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        masked, ph_map = mask_protected_spans(text)
        assert "```" not in masked
        assert len(ph_map) >= 1
        # Unmask restores original
        restored = unmask_protected_spans(masked, ph_map)
        assert "```python\nprint('hello')\n```" in restored

    def test_inline_code_masked(self):
        text = "Use the `print()` function to output text."
        masked, ph_map = mask_protected_spans(text)
        assert "`print()`" not in masked
        restored = unmask_protected_spans(masked, ph_map)
        assert "`print()`" in restored

    def test_url_masked(self):
        text = "Visit https://example.com/api/v2 for docs."
        masked, ph_map = mask_protected_spans(text)
        assert "https://example.com/api/v2" not in masked
        restored = unmask_protected_spans(masked, ph_map)
        assert "https://example.com/api/v2" in restored

    def test_placeholder_masked(self):
        text = "Hello {name}, your id is {user_id}."
        masked, ph_map = mask_protected_spans(text)
        assert "{name}" not in masked
        assert "{user_id}" not in masked
        restored = unmask_protected_spans(masked, ph_map)
        assert "{name}" in restored
        assert "{user_id}" in restored

    def test_json_object_masked(self):
        text = 'The response is {"key": "value", "num": 42} as expected.'
        masked, ph_map = mask_protected_spans(text)
        assert '"key"' not in masked
        restored = unmask_protected_spans(masked, ph_map)
        assert '{"key": "value", "num": 42}' in restored

    def test_plain_text_not_masked(self):
        text = "This is just plain English text with no code or structure."
        masked, ph_map = mask_protected_spans(text)
        # Should have few or no placeholders for plain text
        assert masked.replace("__PH_", "").count("__") < 5

    def test_roundtrip_identity(self):
        """Mask then unmask with no translation = identity."""
        text = "Code: ```js\nconsole.log('hi')\n``` and url https://x.com"
        masked, ph_map = mask_protected_spans(text)
        restored = unmask_protected_spans(masked, ph_map)
        assert restored == text

    def test_latex_masked(self):
        text = "The equation is $$E = mc^2$$ as shown."
        masked, ph_map = mask_protected_spans(text)
        assert "$$E = mc^2$$" not in masked
        restored = unmask_protected_spans(masked, ph_map)
        assert "$$E = mc^2$$" in restored

    def test_file_path_masked(self):
        text = "Open the file /usr/local/bin/python3."
        masked, ph_map = mask_protected_spans(text)
        assert "/usr/local/bin/python3" not in masked
        restored = unmask_protected_spans(masked, ph_map)
        assert "/usr/local/bin/python3" in restored


class TestClassification:
    def test_tool_always_preserve(self):
        assert classify_message_content("anything", "tool", "tool") == "preserve"

    def test_system_translate(self):
        assert classify_message_content("You are a helpful assistant.", "system", "chat") == "translate"

    def test_system_with_tool_schema_preserve(self):
        content = '{"type": "function", "parameters": {"type": "object"}}'
        assert classify_message_content(content, "system", "tool") == "preserve"

    def test_user_translate(self):
        assert classify_message_content("What is the weather?", "user", "chat") == "translate"

    def test_user_with_code_selective(self):
        content = "Fix this code:\n```python\nx = 1\n```"
        assert classify_message_content(content, "user", "code") == "selective"

    def test_assistant_translate(self):
        assert classify_message_content("The answer is 42.", "assistant", "chat") == "translate"

    def test_assistant_with_code_selective(self):
        content = "Here's the solution:\n```python\nprint(42)\n```"
        assert classify_message_content(content, "assistant", "code") == "selective"

    def test_assistant_with_tool_call_preserve(self):
        content = '<tool_call>{"function": {"name": "search"}}</tool_call>'
        assert classify_message_content(content, "assistant", "tool") == "preserve"


class TestValidation:
    def test_placeholder_leak_detection(self):
        assert check_placeholder_leaks("normal text") == []
        assert len(check_placeholder_leaks("text with __PH_001__ leaked")) == 1
        assert len(check_placeholder_leaks("__PH_000__ and __PH_001__")) == 2

    def test_code_preservation_pass(self):
        orig = "text ```python\nprint(1)\n``` more"
        trans = "texte ```python\nprint(1)\n``` plus"
        assert validate_code_preservation(orig, trans) is True

    def test_code_preservation_fail(self):
        orig = "text ```python\nprint(1)\n``` more"
        trans = "texte ```python\nprint(2)\n``` plus"
        assert validate_code_preservation(orig, trans) is False

    def test_json_preservation_pass(self):
        orig = '{"name": "test", "value": 1}'
        trans = '{"name": "tesi", "value": 2}'
        assert validate_json_preservation(orig, trans) is True

    def test_json_preservation_fail_missing_key(self):
        orig = '{"name": "test", "value": 1}'
        trans = '{"name": "test"}'
        assert validate_json_preservation(orig, trans) is False

    def test_validate_translation_pass(self):
        original = {
            "messages": [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        translated = {
            "messages": [
                {"role": "user", "content": "Talofa te lalolagi"},
                {"role": "assistant", "content": "Talofa ia koe"},
            ]
        }
        accepted, reasons = validate_translation(original, translated)
        assert accepted is True
        assert reasons == []

    def test_validate_translation_message_count_mismatch(self):
        original = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        translated = {
            "messages": [
                {"role": "user", "content": "Talofa"},
            ]
        }
        accepted, reasons = validate_translation(original, translated)
        assert accepted is False
        assert any("message_count_mismatch" in r for r in reasons)

    def test_validate_translation_placeholder_leak(self):
        original = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"},
            ]
        }
        translated = {
            "messages": [
                {"role": "user", "content": "Talofa"},
                {"role": "assistant", "content": "The __PH_001__ leaked"},
            ]
        }
        accepted, reasons = validate_translation(original, translated)
        assert accepted is False
        assert any("placeholder_leak" in r for r in reasons)


class TestMaskUsage:
    """Test that selective_translate_example uses the translate_mask field."""

    @staticmethod
    def _mock_translate(text):
        return f"TRANSLATED:{text}"

    def test_explicit_mask_overrides_heuristic(self):
        """mask=[False, True] should translate only the second message."""
        example = {
            "task_family": "chat",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello world"},
            ],
            "translate_mask": [
                {"translate": False},
                {"translate": True},
            ],
        }
        result = selective_translate_example(example, self._mock_translate)
        # First message preserved (mask=False)
        assert result["messages"][0]["content"] == "You are helpful."
        # Second message translated (mask=True)
        assert result["messages"][1]["content"] == "TRANSLATED:Hello world"

    def test_selective_mask_preserves_code(self):
        """mask='selective' on a message with code block preserves the code."""
        code_content = "Here is code:\n```python\nprint('hello')\n```\nDone."
        example = {
            "task_family": "code",
            "messages": [
                {"role": "assistant", "content": code_content},
            ],
            "translate_mask": [
                {"translate": "selective"},
            ],
        }
        result = selective_translate_example(example, self._mock_translate)
        output = result["messages"][0]["content"]
        # Code block should be preserved (not translated)
        assert "```python\nprint('hello')\n```" in output
        # Surrounding text should be translated
        assert "TRANSLATED:" in output

    def test_preservation_metadata_emitted(self):
        """When selective masking finds placeholders, metadata records them."""
        code_content = "Check ```python\nx = 1\n``` and https://example.com."
        example = {
            "task_family": "code",
            "messages": [
                {"role": "assistant", "content": code_content},
            ],
            "translate_mask": [
                {"translate": "selective"},
            ],
        }
        result = selective_translate_example(example, self._mock_translate)
        meta = result.get("metadata", {})
        assert meta.get("selectively_translated") is True
        preservation = meta.get("preservation", {})
        assert "msg_0_placeholders" in preservation
        assert preservation["msg_0_placeholders"] > 0

    def test_no_mask_falls_back_to_heuristic(self):
        """Without translate_mask, old heuristic behavior works."""
        example = {
            "task_family": "chat",
            "messages": [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hi there"},
            ],
            # No translate_mask field
        }
        result = selective_translate_example(example, self._mock_translate)
        # Both should be translated via heuristic (user+assistant = translate)
        assert result["messages"][0]["content"] == "TRANSLATED:Hello world"
        assert result["messages"][1]["content"] == "TRANSLATED:Hi there"
