"""Synthetic Tuvaluan data generation runner.

Uses a trained Stage A translation model to selectively translate English
capability datasets into Tuvaluan. Writes accepted/rejected JSONL files,
stats, manifests, and token budget reports.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from ..common.config import get_repo_root, load_config, resolve_path
from ..common.io import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from ..common.manifests import create_manifest, save_manifest
from ..common.schema import validate_example
from ..common.token_estimates import (
    estimate_example_tokens,
    format_token_count,
)

logger = logging.getLogger(__name__)


class GenerationState:
    """Tracks generation progress for resume support."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.completed: dict[str, set[str]] = {}  # dataset -> set of completed ids
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            data = read_json(self.state_path)
            for ds, ids in data.get("completed", {}).items():
                self.completed[ds] = set(ids)
            logger.info(
                "Resumed generation state: %d datasets, %d total completed",
                len(self.completed),
                sum(len(v) for v in self.completed.values()),
            )

    def save(self) -> None:
        data = {
            "completed": {ds: sorted(ids) for ds, ids in self.completed.items()},
            "timestamp": time.time(),
        }
        write_json(self.state_path, data)

    def is_done(self, dataset: str, example_id: str) -> bool:
        return example_id in self.completed.get(dataset, set())

    def mark_done(self, dataset: str, example_id: str) -> None:
        if dataset not in self.completed:
            self.completed[dataset] = set()
        self.completed[dataset].add(example_id)


class BudgetTracker:
    """Tracks token usage against a global budget."""

    def __init__(self, total_budget: int, per_dataset_weights: dict[str, float]):
        self.total_budget = total_budget
        self.per_dataset_weights = per_dataset_weights
        self.used: dict[str, int] = {}
        total_weight = sum(per_dataset_weights.values()) or 1.0
        self.per_dataset_budget = {
            name: int(total_budget * w / total_weight)
            for name, w in per_dataset_weights.items()
        }

    def record(self, dataset: str, tokens: int) -> None:
        self.used[dataset] = self.used.get(dataset, 0) + tokens

    def should_continue(self, dataset: str) -> bool:
        budget = self.per_dataset_budget.get(dataset, self.total_budget)
        return self.used.get(dataset, 0) < budget

    def total_used(self) -> int:
        return sum(self.used.values())

    def get_report(self) -> dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "total_used": self.total_used(),
            "total_used_human": format_token_count(self.total_used()),
            "budget_remaining": self.total_budget - self.total_used(),
            "per_dataset": {
                name: {
                    "budget": self.per_dataset_budget.get(name, 0),
                    "used": self.used.get(name, 0),
                    "used_human": format_token_count(self.used.get(name, 0)),
                    "remaining": self.per_dataset_budget.get(name, 0) - self.used.get(name, 0),
                }
                for name in sorted(
                    set(list(self.per_dataset_budget.keys()) + list(self.used.keys()))
                )
            },
        }


def create_translate_fn(
    service_client: Any,
    renderer: Any,
    *,
    model_path: str | None = None,
    base_model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> Callable[[str], str]:
    """Create a translation function using a Tinker sampling client.

    Returns a callable that translates text using the Stage A model.
    """
    from ..common.tinker_runtime import create_sampling_client, get_sampling_params

    sampling_client = create_sampling_client(
        service_client,
        model_path=model_path,
        base_model=base_model,
    )
    sampling_params = get_sampling_params(
        renderer, max_tokens=max_tokens, temperature=temperature
    )

    system_prompt = (
        "You are a careful translator between English and Tuvaluan. "
        "Translate faithfully. Preserve names, numbers, punctuation, line breaks, "
        "and structure when possible. Output only the translation."
    )

    def translate(text: str) -> str:
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Translate from English to Tuvaluan:\n\n{text}",
            },
        ]
        prompt = renderer.build_generation_prompt(prompt_messages)
        result = sampling_client.sample(
            prompt, sampling_params=sampling_params, num_samples=1
        ).result()
        output_tokens = result.sequences[0].tokens
        response_message, _parse_ok = renderer.parse_response(output_tokens)
        if isinstance(response_message, dict):
            return str(response_message.get("content", ""))
        return str(response_message)

    return translate


def generate_synthetic_data(config: dict[str, Any]) -> dict[str, Any]:
    """Run synthetic data generation from config.

    This is the main entry point. It:
    1. Loads normalized English sources
    2. Sets up the Stage A translation model
    3. Selectively translates each example
    4. Validates outputs
    5. Writes accepted/rejected JSONL + stats

    Returns a summary dict.
    """
    repo_root = get_repo_root()

    # Resolve paths
    sources_dir = resolve_path(
        config.get("output", {}).get("sources_dir", "data/finetune/stage_b_sources"),
        repo_root,
    )
    synthetic_dir = resolve_path(
        config.get("output", {}).get("synthetic_dir", "data/finetune/stage_b_synthetic_tvl"),
        repo_root,
    )
    accepted_dir = synthetic_dir / "accepted"
    rejected_dir = synthetic_dir / "rejected"
    manifests_dir = synthetic_dir / "manifests"
    stats_dir = synthetic_dir / "stats"
    for d in [accepted_dir, rejected_dir, manifests_dir, stats_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Budget
    budget_config = config.get("budget", {})
    total_budget = budget_config.get("total_tokens", 200_000_000)
    per_dataset_weights = budget_config.get("per_dataset", {})
    budget = BudgetTracker(total_budget, per_dataset_weights)

    # Resume state
    state = GenerationState(synthetic_dir / "generation_state.json")

    # Set up translation
    from ..common.tinker_runtime import (
        create_service_client,
        get_renderer,
        require_tinker_api_key,
    )

    require_tinker_api_key()
    stage_a_config = config.get("stage_a_model", {})
    model_name = stage_a_config.get("base_model", "Qwen/Qwen3-30B-A3B-Base")
    _tokenizer, renderer, _renderer_name = get_renderer(model_name)
    service_client = create_service_client()

    translation_config = config.get("translation", {})
    translate_fn = create_translate_fn(
        service_client,
        renderer,
        model_path=stage_a_config.get("model_path"),
        base_model=model_name,
        max_tokens=translation_config.get("max_tokens", 1024),
        temperature=translation_config.get("temperature", 0.3),
    )
    tool_mode = translation_config.get("tool_mode", "safe")

    # Import selective translation (lazy to avoid circular deps)
    from .selective_translate import selective_translate_example
    from .quality import validate_translation

    # Process each dataset
    english_dir = sources_dir / "english_normalized"
    datasets_config = config.get("datasets", [])
    dataset_names = [
        d["name"].split("/")[-1]
        for d in datasets_config
        if d.get("enabled", True)
    ]

    total_accepted = 0
    total_rejected = 0
    per_dataset_stats: dict[str, dict[str, Any]] = {}

    for ds_name in dataset_names:
        source_file = english_dir / f"{ds_name}.jsonl"
        if not source_file.exists():
            logger.warning("Source file not found: %s, skipping", source_file)
            continue

        if not budget.should_continue(ds_name):
            logger.info("Budget exhausted for %s, skipping", ds_name)
            continue

        logger.info("Processing dataset: %s", ds_name)
        examples = read_jsonl(source_file)
        ds_accepted = 0
        ds_rejected = 0
        ds_tokens = 0

        accepted_path = accepted_dir / f"{ds_name}.jsonl"
        rejected_path = rejected_dir / f"{ds_name}.jsonl"

        for i, example in enumerate(examples):
            ex_id = example.get("id", f"{ds_name}_{i}")

            if state.is_done(ds_name, ex_id):
                continue

            if not budget.should_continue(ds_name):
                logger.info("Budget exhausted mid-dataset for %s at example %d", ds_name, i)
                break

            try:
                translated = selective_translate_example(
                    example, translate_fn, tool_mode=tool_mode
                )
                accepted, reasons = validate_translation(example, translated)

                tokens = estimate_example_tokens(translated)
                budget.record(ds_name, tokens)
                ds_tokens += tokens

                if accepted:
                    append_jsonl(accepted_path, translated)
                    ds_accepted += 1
                    total_accepted += 1
                else:
                    append_jsonl(
                        rejected_path,
                        {"example": translated, "reasons": reasons, "original_id": ex_id},
                    )
                    ds_rejected += 1
                    total_rejected += 1

            except Exception as exc:
                logger.warning("Error translating %s: %s", ex_id, exc)
                append_jsonl(
                    rejected_path,
                    {"original_id": ex_id, "reasons": ["translation_error", str(exc)]},
                )
                ds_rejected += 1
                total_rejected += 1

            state.mark_done(ds_name, ex_id)

            if (i + 1) % 100 == 0:
                state.save()
                logger.info(
                    "  %s: %d/%d processed, %d accepted, %d rejected, %s tokens",
                    ds_name,
                    i + 1,
                    len(examples),
                    ds_accepted,
                    ds_rejected,
                    format_token_count(ds_tokens),
                )

        per_dataset_stats[ds_name] = {
            "total": len(examples),
            "accepted": ds_accepted,
            "rejected": ds_rejected,
            "tokens": ds_tokens,
            "tokens_human": format_token_count(ds_tokens),
            "accept_rate": ds_accepted / max(ds_accepted + ds_rejected, 1),
        }

        state.save()

    # Final stats
    summary = {
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_tokens": budget.total_used(),
        "total_tokens_human": format_token_count(budget.total_used()),
        "accept_rate": total_accepted / max(total_accepted + total_rejected, 1),
        "per_dataset": per_dataset_stats,
        "budget_report": budget.get_report(),
    }

    write_json(stats_dir / "generation_stats.json", summary)
    save_manifest(
        create_manifest(stage="synthetic_generation", config=config, extra=summary),
        manifests_dir / "generation_manifest.json",
    )

    logger.info(
        "Generation complete: %d accepted, %d rejected, %s tokens",
        total_accepted,
        total_rejected,
        format_token_count(budget.total_used()),
    )

    return summary


def main(config: dict[str, Any]) -> dict[str, Any]:
    """Entry point for the generation runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return generate_synthetic_data(config)
