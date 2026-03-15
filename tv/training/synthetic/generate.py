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
from .naming import dataset_name_to_filename
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


SYSTEM_PROMPT = (
    "You are a careful translator between English and Tuvaluan. "
    "Translate faithfully. Preserve names, numbers, punctuation, line breaks, "
    "and structure when possible. Output only the translation."
)


class TranslationEngine:
    """Wraps a Tinker sampling client for both sync and fire-all translation."""

    def __init__(self, sampling_client: Any, renderer: Any, sampling_params: Any):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.sampling_params = sampling_params

    def _build_prompt(self, text: str) -> Any:
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate from English to Tuvaluan:\n\n{text}"},
        ]
        return self.renderer.build_generation_prompt(prompt_messages)

    def _parse_result(self, result: Any) -> str:
        output_tokens = result.sequences[0].tokens
        response_message, _parse_ok = self.renderer.parse_response(output_tokens)
        if isinstance(response_message, dict):
            return str(response_message.get("content", ""))
        return str(response_message)

    def translate(self, text: str) -> str:
        """Synchronous single translation (for backward compat)."""
        prompt = self._build_prompt(text)
        result = self.sampling_client.sample(
            prompt, sampling_params=self.sampling_params, num_samples=1
        ).result()
        return self._parse_result(result)

    def fire(self, text: str) -> Any:
        """Fire a translation request, return future (don't block)."""
        prompt = self._build_prompt(text)
        return self.sampling_client.sample(
            prompt, sampling_params=self.sampling_params, num_samples=1
        )

    def collect(self, future: Any) -> str:
        """Block on a future and parse the result."""
        return self._parse_result(future.result())


def create_translation_engine(
    service_client: Any,
    renderer: Any,
    *,
    model_path: str | None = None,
    base_model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> TranslationEngine:
    """Create a TranslationEngine from a Tinker service client."""
    from ..common.tinker_runtime import create_sampling_client, get_sampling_params

    sampling_client = create_sampling_client(
        service_client,
        model_path=model_path,
        base_model=base_model,
    )
    sampling_params = get_sampling_params(
        renderer, max_tokens=max_tokens, temperature=temperature
    )
    return TranslationEngine(sampling_client, renderer, sampling_params)


def create_translate_fn(
    service_client: Any,
    renderer: Any,
    *,
    model_path: str | None = None,
    base_model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> Callable[[str], str]:
    """Create a sync translation function (backward compat wrapper)."""
    engine = create_translation_engine(
        service_client, renderer,
        model_path=model_path, base_model=base_model,
        max_tokens=max_tokens, temperature=temperature,
    )
    return engine.translate


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

    # Resolve Stage A model path
    model_path = stage_a_config.get("model_path")
    if not model_path and stage_a_config.get("log_path"):
        log_path = resolve_path(stage_a_config["log_path"], repo_root)
        export_info_path = log_path / "export_info.json"
        if export_info_path.exists():
            export_info = read_json(export_info_path)
            model_path = export_info.get("model_path")
            logger.info("Resolved Stage A model from export_info.json: %s", model_path)
        else:
            from ..stage_a_mt.export import get_model_path

            try:
                model_path = get_model_path({"log_path": stage_a_config["log_path"]})
                logger.info("Resolved Stage A model from checkpoint: %s", model_path)
            except FileNotFoundError:
                logger.warning("Could not resolve Stage A model path, using base model")

    translation_config = config.get("translation", {})
    engine = create_translation_engine(
        service_client,
        renderer,
        model_path=model_path,
        base_model=model_name,
        max_tokens=translation_config.get("max_tokens", 1024),
        temperature=translation_config.get("temperature", 0.3),
    )
    tool_mode = translation_config.get("tool_mode", "safe")
    batch_size = config.get("output", {}).get("batch_size", 512)

    # Import selective translation helpers
    from .selective_translate import (
        classify_message_content,
        mask_protected_spans,
        unmask_protected_spans,
    )
    from .quality import validate_translation

    # Process each dataset
    english_dir = sources_dir / "english_normalized"
    datasets_config = config.get("datasets", [])
    dataset_entries = [
        (d["name"], dataset_name_to_filename(d["name"]))
        for d in datasets_config
        if d.get("enabled", True)
    ]

    total_accepted = 0
    total_rejected = 0
    per_dataset_stats: dict[str, dict[str, Any]] = {}

    for ds_name, ds_filename in dataset_entries:
        source_file = english_dir / f"{ds_filename}.jsonl"
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

        accepted_path = accepted_dir / f"{ds_filename}.jsonl"
        rejected_path = rejected_dir / f"{ds_filename}.jsonl"

        # Filter to pending examples
        pending: list[tuple[int, str, dict[str, Any]]] = []
        for i, example in enumerate(examples):
            ex_id = example.get("id", f"{ds_name}_{i}")
            if state.is_done(ds_name, ex_id):
                continue
            if not budget.should_continue(ds_name):
                break
            pending.append((i, ex_id, example))

        logger.info("  %s: %d pending examples (after resume filter)", ds_name, len(pending))

        # Process in batches using fire-all-futures
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]

            if not budget.should_continue(ds_name):
                logger.info("Budget exhausted for %s", ds_name)
                break

            # Phase 1: Pre-scan all messages, fire all translation futures
            # Each slot: (batch_idx, msg_idx, future, is_selective, ph_map)
            flight_plan: list[tuple[int, int, Any, bool, dict[str, str]]] = []
            batch_msg_metadata: list[list[str]] = []  # per-example, per-msg action

            for b_idx, (_orig_idx, _ex_id, example) in enumerate(batch):
                task_family = example.get("task_family", "chat")
                translate_mask = example.get("translate_mask")
                msg_actions: list[str] = []

                for m_idx, msg in enumerate(example.get("messages", [])):
                    content = msg.get("content", "")
                    role = msg.get("role", "")

                    # Determine action from mask or heuristic
                    if translate_mask and m_idx < len(translate_mask):
                        mask_action = translate_mask[m_idx].get("translate", True)
                        if mask_action is False:
                            msg_actions.append("preserve")
                            continue
                        elif mask_action is True:
                            action = "translate"
                        else:
                            action = "selective"
                    else:
                        action = classify_message_content(content, role, task_family)

                    msg_actions.append(action)

                    if action == "preserve" or not content or not isinstance(content, str):
                        continue

                    if action == "selective":
                        masked, ph_map = mask_protected_spans(content)
                        future = engine.fire(masked)
                        flight_plan.append((b_idx, m_idx, future, True, ph_map))
                    else:  # translate
                        future = engine.fire(content)
                        flight_plan.append((b_idx, m_idx, future, False, {}))

                batch_msg_metadata.append(msg_actions)

            logger.info(
                "  %s: fired %d translation requests for batch %d-%d",
                ds_name, len(flight_plan),
                batch_start, min(batch_start + batch_size, len(pending)),
            )

            # Phase 2: Collect all futures
            translations: dict[tuple[int, int], tuple[str, dict[str, str]]] = {}
            for b_idx, m_idx, future, is_selective, ph_map in flight_plan:
                try:
                    translated_text = engine.collect(future)
                    if is_selective:
                        translated_text = unmask_protected_spans(translated_text, ph_map)
                    translations[(b_idx, m_idx)] = (translated_text, ph_map)
                except Exception as exc:
                    translations[(b_idx, m_idx)] = (f"__ERROR__: {exc}", {})

            # Phase 3: Reassemble examples and validate
            for b_idx, (orig_idx, ex_id, example) in enumerate(batch):
                try:
                    result_example = dict(example)
                    translated_messages: list[dict[str, Any]] = []
                    preservation_metadata: dict[str, Any] = {}

                    for m_idx, msg in enumerate(example.get("messages", [])):
                        if (b_idx, m_idx) in translations:
                            result_msg = dict(msg)
                            text, ph_map = translations[(b_idx, m_idx)]
                            if text.startswith("__ERROR__:"):
                                raise RuntimeError(text)
                            result_msg["content"] = text
                            if ph_map:
                                preservation_metadata[f"msg_{m_idx}_placeholders"] = len(ph_map)
                            translated_messages.append(result_msg)
                        else:
                            translated_messages.append(dict(msg))

                    result_example["messages"] = translated_messages
                    meta = dict(result_example.get("metadata", {}))
                    meta["selectively_translated"] = True
                    meta["tool_mode"] = tool_mode
                    if preservation_metadata:
                        meta["preservation"] = preservation_metadata
                    result_example["metadata"] = meta

                    accepted, reasons = validate_translation(example, result_example)
                    tokens = estimate_example_tokens(result_example)
                    budget.record(ds_name, tokens)
                    ds_tokens += tokens

                    if accepted:
                        append_jsonl(accepted_path, result_example)
                        ds_accepted += 1
                        total_accepted += 1
                    else:
                        append_jsonl(
                            rejected_path,
                            {"example": result_example, "reasons": reasons, "original_id": ex_id},
                        )
                        ds_rejected += 1
                        total_rejected += 1

                except Exception as exc:
                    logger.warning("Error processing %s: %s", ex_id, exc)
                    append_jsonl(
                        rejected_path,
                        {"original_id": ex_id, "reasons": ["translation_error", str(exc)]},
                    )
                    ds_rejected += 1
                    total_rejected += 1

                state.mark_done(ds_name, ex_id)

            state.save()
            processed = min(batch_start + batch_size, len(pending))
            logger.info(
                "  %s: %d/%d processed, %d accepted, %d rejected, %s tokens",
                ds_name, processed, len(pending),
                ds_accepted, ds_rejected, format_token_count(ds_tokens),
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
        "stage_a_model_path": model_path,
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
