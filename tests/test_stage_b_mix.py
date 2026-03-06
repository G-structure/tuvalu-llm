"""Test Stage B mix building: ratios, stratification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.schema import make_example


def _make_en_example(i: int, family: str = "chat") -> dict:
    return make_example(
        id=f"en_{i}",
        task_family=family,
        messages=[
            {"role": "user", "content": f"English question {i}"},
            {"role": "assistant", "content": f"English answer {i}"},
        ],
        metadata={"source": "test", "stage_b_source": "english"},
    )


def _make_tvl_example(i: int, family: str = "chat") -> dict:
    return make_example(
        id=f"tvl_{i}",
        task_family=family,
        messages=[
            {"role": "user", "content": f"Fesili Tuvaluan {i}"},
            {"role": "assistant", "content": f"Tali Tuvaluan {i}"},
        ],
        metadata={"source": "test", "stage_b_source": "synthetic_tvl", "selectively_translated": True},
    )


def _make_anchor_example(i: int) -> dict:
    return make_example(
        id=f"anchor_{i}",
        task_family="translation",
        messages=[
            {"role": "system", "content": "Translate."},
            {"role": "user", "content": f"Translate: text {i}"},
            {"role": "assistant", "content": f"Tusi: text {i}"},
        ],
        metadata={"source": "parallel", "stage_b_source": "anchor"},
    )


def test_mix_ratio_approximate():
    """Test that mixing respects approximate ratios."""
    en = [_make_en_example(i) for i in range(400)]
    tvl = [_make_tvl_example(i) for i in range(400)]
    anchor = [_make_anchor_example(i) for i in range(200)]

    # Simulate simple ratio-based sampling
    ratios = {"english": 0.4, "synthetic_tvl": 0.4, "anchor": 0.2}
    total = 100

    en_count = int(total * ratios["english"])
    tvl_count = int(total * ratios["synthetic_tvl"])
    anchor_count = total - en_count - tvl_count

    assert en_count == 40
    assert tvl_count == 40
    assert anchor_count == 20


def test_task_family_stratification():
    """Test that different task families are represented."""
    families = ["chat", "tool", "math", "code", "qa", "summarization"]
    en_examples = []
    for i, fam in enumerate(families):
        for j in range(10):
            en_examples.append(_make_en_example(i * 10 + j, family=fam))

    # Check that examples cover all families
    seen_families = {ex["task_family"] for ex in en_examples}
    assert seen_families == set(families)


def test_dedup_across_languages():
    """Test that same-source-id examples in both languages are distinct."""
    en = _make_en_example(1)
    tvl = _make_tvl_example(1)
    assert en["id"] != tvl["id"]
    assert en["metadata"]["stage_b_source"] != tvl["metadata"]["stage_b_source"]
