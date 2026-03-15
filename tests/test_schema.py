"""Test normalized example schema."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tv.common.schema import make_example, validate_example, TASK_FAMILIES


def test_make_example_valid():
    ex = make_example(
        id="test-1",
        task_family="chat",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    )
    assert ex["id"] == "test-1"
    assert ex["task_family"] == "chat"
    assert len(ex["messages"]) == 2
    assert ex["metadata"] == {}


def test_make_example_with_metadata():
    ex = make_example(
        id="test-2",
        task_family="tool",
        messages=[{"role": "user", "content": "Call func"}],
        metadata={"source": "test_dataset"},
    )
    assert ex["metadata"]["source"] == "test_dataset"


def test_make_example_with_translate_mask():
    ex = make_example(
        id="test-3",
        task_family="code",
        messages=[
            {"role": "user", "content": "Fix code"},
            {"role": "assistant", "content": "def foo(): pass"},
        ],
        translate_mask=[{"translate": True}, {"translate": "selective"}],
    )
    assert "translate_mask" in ex
    assert ex["translate_mask"][1]["translate"] == "selective"


def test_make_example_invalid_family():
    try:
        make_example(
            id="bad",
            task_family="nonexistent",
            messages=[{"role": "user", "content": "x"}],
        )
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_validate_example_valid():
    ex = make_example(
        id="v-1",
        task_family="math",
        messages=[
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
    )
    errors = validate_example(ex)
    assert errors == []


def test_validate_example_missing_id():
    ex = {"task_family": "chat", "messages": [{"role": "user", "content": "x"}]}
    errors = validate_example(ex)
    assert any("id" in e for e in errors)


def test_validate_example_bad_messages():
    ex = {"id": "x", "task_family": "chat", "messages": "not a list"}
    errors = validate_example(ex)
    assert any("messages" in e for e in errors)


def test_all_task_families():
    """All declared task families are valid."""
    for fam in TASK_FAMILIES:
        ex = make_example(
            id=f"test-{fam}",
            task_family=fam,
            messages=[{"role": "user", "content": "test"}],
        )
        assert validate_example(ex) == []
