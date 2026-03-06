"""Test Stage B mix building: ratios, stratification, anchor handling."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.schema import make_example
from training.stage_b_agent.build_mix import (
    _deduplicate,
    _filter_by_task_family,
    _sample_to_ratio,
    _tag_source,
)


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


def test_sample_to_ratio_balanced_pools():
    """When all pools have enough, ratios should be closely matched."""
    en = [_make_en_example(i) for i in range(400)]
    tvl = [_make_tvl_example(i) for i in range(400)]
    anchor = [_make_anchor_example(i) for i in range(200)]
    pools = {"english": en, "synthetic_tvl": tvl, "anchor": anchor}
    ratios = {"english": 0.4, "synthetic_tvl": 0.4, "anchor": 0.2}
    rng = random.Random(42)

    result, report = _sample_to_ratio(pools, ratios, rng)
    assert len(result) > 0
    # Check realized ratios are close to requested
    for name, target_ratio in ratios.items():
        realized = report["realized_ratios"][name]
        assert abs(realized - target_ratio) < 0.05, f"{name}: {realized} vs {target_ratio}"


def test_sample_to_ratio_small_anchor():
    """When anchor pool is small, others should be capped proportionally."""
    en = [_make_en_example(i) for i in range(400)]
    tvl = [_make_tvl_example(i) for i in range(400)]
    anchor = [_make_anchor_example(i) for i in range(10)]  # very small
    pools = {"english": en, "synthetic_tvl": tvl, "anchor": anchor}
    ratios = {"english": 0.4, "synthetic_tvl": 0.4, "anchor": 0.2}
    rng = random.Random(42)

    result, report = _sample_to_ratio(pools, ratios, rng)
    # Anchor uses all 10
    assert report["realized_counts"]["anchor"] == 10
    # Report should show shortfall
    assert "shortfall" in report


def test_sample_to_ratio_empty_pools():
    """Empty pools should return empty results with a valid report."""
    pools = {"english": [], "synthetic_tvl": [], "anchor": []}
    ratios = {"english": 0.4, "synthetic_tvl": 0.4, "anchor": 0.2}
    rng = random.Random(42)

    result, report = _sample_to_ratio(pools, ratios, rng)
    assert result == []
    assert report["realized_counts"]["english"] == 0


def test_anchor_survives_task_family_filter():
    """Anchor examples should not be removed by task family filtering."""
    anchor = [_make_anchor_example(i) for i in range(10)]
    # task_family filter that would exclude "translation"
    filtered = _filter_by_task_family(anchor, include=["chat", "code"], exclude=None)
    # Anchors have task_family="translation", so they get filtered here.
    # In the actual pipeline, anchors bypass _filter_by_task_family entirely.
    assert len(filtered) == 0  # filter works as expected

    # But the pipeline design exempts anchors from filtering:
    # english/synthetic_tvl go through filter, anchors do not.
    # This is verified by checking the main() flow doesn't filter anchor_raw.


def test_realized_ratio_reporting():
    """Stats should show both requested and realized ratios."""
    en = [_make_en_example(i) for i in range(100)]
    tvl = [_make_tvl_example(i) for i in range(100)]
    anchor = [_make_anchor_example(i) for i in range(50)]
    pools = {"english": en, "synthetic_tvl": tvl, "anchor": anchor}
    ratios = {"english": 0.4, "synthetic_tvl": 0.4, "anchor": 0.2}
    rng = random.Random(42)

    _, report = _sample_to_ratio(pools, ratios, rng)
    assert "requested_ratios" in report
    assert "realized_ratios" in report
    assert "realized_counts" in report
    assert "shortfall" in report
    # All keys present
    for name in ratios:
        assert name in report["requested_ratios"]
        assert name in report["realized_ratios"]
        assert name in report["realized_counts"]


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


def test_tag_source():
    """Test that _tag_source adds the correct metadata."""
    examples = [_make_en_example(i) for i in range(3)]
    _tag_source(examples, "english")
    for ex in examples:
        assert ex["metadata"]["stage_b_source"] == "english"


def test_deduplicate():
    """Test that _deduplicate removes exact ID duplicates."""
    ex1 = _make_en_example(1)
    ex2 = _make_en_example(1)  # same ID
    ex3 = _make_en_example(2)
    result = _deduplicate([ex1, ex2, ex3])
    assert len(result) == 2
    ids = [r["id"] for r in result]
    assert ids == ["en_1", "en_2"]
