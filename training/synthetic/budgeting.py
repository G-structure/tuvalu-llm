"""Token budget management for synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from training.common.token_estimates import format_token_count


@dataclass
class DatasetBudget:
    """Per-dataset token quota tracker."""

    name: str
    quota: int
    tokens_used: int = 0
    examples_count: int = 0

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.quota - self.tokens_used)

    @property
    def exhausted(self) -> bool:
        return self.tokens_used >= self.quota

    def record(self, n_tokens: int) -> None:
        self.tokens_used += n_tokens
        self.examples_count += 1


class BudgetManager:
    """Global token budget with per-dataset allocation.

    Args:
        total_budget: Total token budget across all datasets (default 200M).
        allocations: {dataset_name: token_quota} overrides.
            Datasets not in allocations get an equal share of the remainder.
    """

    def __init__(
        self,
        total_budget: int = 200_000_000,
        allocations: dict[str, int] | None = None,
    ) -> None:
        self.total_budget = total_budget
        self._budgets: dict[str, DatasetBudget] = {}
        if allocations:
            for name, quota in allocations.items():
                self._budgets[name] = DatasetBudget(name=name, quota=quota)

    def _ensure(self, name: str) -> DatasetBudget:
        if name not in self._budgets:
            # Auto-allocate: equal share of remaining budget
            used_quota = sum(b.quota for b in self._budgets.values())
            remaining = max(0, self.total_budget - used_quota)
            self._budgets[name] = DatasetBudget(name=name, quota=remaining)
        return self._budgets[name]

    def should_continue(self, dataset_name: str) -> bool:
        """Check if dataset_name still has budget."""
        budget = self._ensure(dataset_name)
        return not budget.exhausted

    def record_usage(self, dataset_name: str, n_tokens: int) -> None:
        """Record token usage for a dataset."""
        budget = self._ensure(dataset_name)
        budget.record(n_tokens)

    def get_report(self) -> dict[str, Any]:
        """Return per-dataset and global budget report."""
        datasets: dict[str, Any] = {}
        total_used = 0
        total_examples = 0
        for name, b in sorted(self._budgets.items()):
            datasets[name] = {
                "quota": b.quota,
                "tokens_used": b.tokens_used,
                "tokens_remaining": b.tokens_remaining,
                "examples": b.examples_count,
                "exhausted": b.exhausted,
                "tokens_used_fmt": format_token_count(b.tokens_used),
            }
            total_used += b.tokens_used
            total_examples += b.examples_count
        return {
            "total_budget": self.total_budget,
            "total_used": total_used,
            "total_remaining": max(0, self.total_budget - total_used),
            "total_examples": total_examples,
            "total_used_fmt": format_token_count(total_used),
            "datasets": datasets,
        }
