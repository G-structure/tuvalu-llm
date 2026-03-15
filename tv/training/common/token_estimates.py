"""Token counting and estimation utilities."""

from __future__ import annotations

from typing import Any


def estimate_tokens_chars(text: str) -> int:
    """Rough token estimate from character count (chars / 4)."""
    return max(1, len(text) // 4)


def estimate_example_tokens(example: dict[str, Any]) -> int:
    """Estimate total tokens for a normalized example."""
    total = 0
    for msg in example.get("messages", []):
        total += estimate_tokens_chars(msg.get("content", ""))
    return total


def estimate_dataset_tokens(examples: list[dict[str, Any]]) -> int:
    """Sum token estimates across a list of examples."""
    return sum(estimate_example_tokens(ex) for ex in examples)


def format_token_count(n: int) -> str:
    """Human-readable token count (e.g., '1.2M', '500K')."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def compute_budget_allocation(
    total_budget: int,
    dataset_weights: dict[str, float],
) -> dict[str, int]:
    """Allocate a token budget across datasets proportionally.

    Args:
        total_budget: Total token budget.
        dataset_weights: {dataset_name: relative_weight}.

    Returns:
        {dataset_name: allocated_tokens}.
    """
    total_weight = sum(dataset_weights.values())
    if total_weight == 0:
        return {k: 0 for k in dataset_weights}
    return {
        name: int(total_budget * weight / total_weight)
        for name, weight in dataset_weights.items()
    }
