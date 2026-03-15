"""Build the Stage B mixed training dataset.

Stage B trains a bilingual capability adapter on openai/gpt-oss-120b BASE
(NOT Stage A weights). Stage A exists only to produce the synthetic TVL
dataset that feeds into this mix.

Primary data sources:
1. English capability data  (real upstream EN datasets, normalized)
2. Synthetic TVL translations (produced by Stage A adapter via selective translation)
3. Cross-lingual data (EN prompt + "respond in Tuvaluan" → TVL response)
4. Original TVL<->EN parallel anchor data (from Stage A training set)
5. Optional real TVL chat data (normalized local chat JSONL)

Default ratio: 30% English / 30% synthetic TVL / 20% cross-lingual / 20% anchor.

Outputs (under data/finetune/stage_b_mix/):
    train.jsonl        - full mixed training set
    train_pilot.jsonl  - ~5000 examples for shakeout runs
    validation.jsonl   - held-out validation set
    test.jsonl         - held-out test set
    stats.json         - dataset statistics
    manifest.json      - reproducibility manifest
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tv.common.config import get_repo_root, resolve_path
from tv.common.io import read_jsonl, write_json, write_jsonl
from tv.common.manifests import create_manifest, save_manifest
from tv.common.schema import TASK_FAMILIES, validate_example
from tv.common.token_estimates import estimate_dataset_tokens, format_token_count

from .tooling_modes import SAFE_MODE, ToolMode, detect_tool_messages, format_messages

DEFAULTS: dict[str, Any] = {
    # Input paths (relative to repo root)
    "english_dir": "data/finetune/stage_b_sources/english_normalized",
    "synthetic_tvl_dir": "data/finetune/stage_b_synthetic_tvl/accepted",
    "crosslingual_dir": "data/finetune/stage_b_crosslingual",
    "real_tvl_chat_dir": "data/finetune/stage_b_sources/real_tvl_chat",
    "anchor_path": "data/finetune/stage_a_mt/train_balanced.jsonl",
    # Output path
    "output_dir": "data/finetune/stage_b_mix",
    # Mix ratios
    "mix_ratios": {"english": 0.30, "synthetic_tvl": 0.30, "crosslingual": 0.20, "anchor": 0.20},
    # Splits
    "validation_fraction": 0.02,
    "test_fraction": 0.02,
    "pilot_size": 5000,
    # Filtering
    "include_task_families": None,  # None = all
    "exclude_task_families": None,
    # Tool mode
    "tool_mode": "safe",
    # Reproducibility
    "seed": 42,
}


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _load_jsonl_dir(dir_path: Path) -> list[dict[str, Any]]:
    """Load all JSONL files from a directory, sorted by filename."""
    rows: list[dict[str, Any]] = []
    if not dir_path.exists():
        return rows
    for path in sorted(dir_path.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def _filter_by_task_family(
    examples: list[dict[str, Any]],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter examples by task family inclusion/exclusion lists."""
    if include is not None:
        examples = [ex for ex in examples if ex.get("task_family") in include]
    if exclude is not None:
        examples = [ex for ex in examples if ex.get("task_family") not in exclude]
    return examples


def _tag_source(examples: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Add source tag to each example's metadata."""
    for ex in examples:
        meta = ex.get("metadata", {})
        meta["stage_b_source"] = source
        ex["metadata"] = meta
    return examples


def _deduplicate(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate examples by base id (strip language suffixes).

    Examples from the same source (same base id) in different languages
    are kept, but exact duplicates are removed.
    """
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for ex in examples:
        full_id = ex.get("id", "")
        if full_id in seen:
            continue
        seen.add(full_id)
        deduped.append(ex)
    return deduped


def _apply_tool_mode(examples: list[dict[str, Any]], mode: ToolMode) -> list[dict[str, Any]]:
    """Format tool messages in all examples according to tool mode."""
    out: list[dict[str, Any]] = []
    for ex in examples:
        msgs = ex.get("messages", [])
        tool_indices = detect_tool_messages(msgs)
        if tool_indices:
            ex = dict(ex)
            ex["messages"] = format_messages(msgs, mode)
        out.append(ex)
    return out


def _sample_to_ratio(
    pools: dict[str, list[dict[str, Any]]],
    ratios: dict[str, float],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample from pools according to target ratios.

    If a pool has fewer examples than its allocation, it uses all of them
    and redistributes the remaining budget to other pools.

    Returns (sampled_examples, ratio_report) where ratio_report contains:
    - requested_ratios: the input ratios (normalized)
    - realized_counts: {pool_name: actual_count}
    - realized_ratios: {pool_name: actual_count / total}
    - shortfall: {pool_name: target_count - actual_count} for pools that were short
    """
    total_weight = sum(ratios.values())
    normalized = {k: v / total_weight for k, v in ratios.items()}

    # Total available
    total_available = sum(len(v) for v in pools.values())
    if total_available == 0:
        return [], {
            "requested_ratios": normalized,
            "realized_counts": {k: 0 for k in ratios},
            "realized_ratios": {k: 0.0 for k in ratios},
            "shortfall": {},
        }

    # Iteratively allocate, capping pools that are too small
    allocation: dict[str, int] = {}
    remaining_budget = total_available
    remaining_ratios = dict(normalized)

    # Track ideal target for shortfall reporting
    ideal_allocation: dict[str, int] = {
        k: int(total_available * v) for k, v in normalized.items()
    }

    for _ in range(len(pools)):
        if not remaining_ratios:
            break
        ratio_sum = sum(remaining_ratios.values())
        new_remaining_ratios: dict[str, float] = {}
        for name, ratio in remaining_ratios.items():
            target = int(remaining_budget * ratio / ratio_sum)
            available = len(pools.get(name, []))
            if available <= target:
                allocation[name] = available
                remaining_budget -= available
            else:
                new_remaining_ratios[name] = ratio
        if not new_remaining_ratios or new_remaining_ratios == remaining_ratios:
            # Final pass: allocate remaining
            ratio_sum = sum(new_remaining_ratios.values())
            for name, ratio in new_remaining_ratios.items():
                target = int(remaining_budget * ratio / ratio_sum)
                allocation[name] = min(target, len(pools.get(name, [])))
            break
        remaining_ratios = new_remaining_ratios

    result: list[dict[str, Any]] = []
    for name, count in allocation.items():
        pool = pools.get(name, [])
        if count >= len(pool):
            result.extend(pool)
        else:
            result.extend(rng.sample(pool, count))

    # Build ratio report
    total_sampled = len(result)
    realized_counts = {name: allocation.get(name, 0) for name in ratios}
    realized_ratios = {
        name: count / total_sampled if total_sampled > 0 else 0.0
        for name, count in realized_counts.items()
    }
    shortfall = {
        name: ideal_allocation[name] - realized_counts[name]
        for name in ratios
        if realized_counts[name] < ideal_allocation[name]
    }

    ratio_report: dict[str, Any] = {
        "requested_ratios": normalized,
        "realized_counts": realized_counts,
        "realized_ratios": realized_ratios,
        "shortfall": shortfall,
    }

    return result, ratio_report


def _assign_split(
    example_id: str,
    val_frac: float,
    test_frac: float,
) -> str:
    """Deterministic split assignment by hash."""
    bucket = _stable_hash(example_id) % 10000
    test_cut = int(test_frac * 10000)
    val_cut = test_cut + int(val_frac * 10000)
    if bucket < test_cut:
        return "test"
    if bucket < val_cut:
        return "validation"
    return "train"


def _example_split_key(example: dict[str, Any]) -> str:
    """Return the deterministic split key for an example.

    If metadata carries `split_group` (for example a conversation/thread id),
    all examples in that group will land in the same split.
    """
    metadata = example.get("metadata", {})
    split_group = metadata.get("split_group")
    if split_group:
        return str(split_group)
    return str(example["id"])


def _split_examples(
    examples: list[dict[str, Any]],
    anchor_examples: list[dict[str, Any]],
    val_frac: float,
    test_frac: float,
) -> dict[str, list[dict[str, Any]]]:
    """Split examples into train/validation/test.

    Anchor examples always go to train (they have their own eval set in Stage A).
    Non-anchor examples are split by hash.
    """
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for ex in anchor_examples:
        splits["train"].append(ex)

    for ex in examples:
        source = ex.get("metadata", {}).get("stage_b_source", "")
        if source == "anchor":
            splits["train"].append(ex)
        else:
            split = _assign_split(ex["id"], val_frac, test_frac)
            splits[split].append(ex)

    return dict(splits)


def _summarize(examples: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Compute summary stats for a set of examples."""
    if not examples:
        return {"label": label, "count": 0}

    by_source = Counter(ex.get("metadata", {}).get("stage_b_source", "unknown") for ex in examples)
    by_family = Counter(ex.get("task_family", "unknown") for ex in examples)
    total_tokens = estimate_dataset_tokens(examples)

    return {
        "label": label,
        "count": len(examples),
        "by_source": dict(by_source),
        "by_task_family": dict(by_family),
        "total_tokens_est": total_tokens,
        "total_tokens_human": format_token_count(total_tokens),
    }


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the Stage B mixed dataset.

    Args:
        config: Configuration overrides. Missing keys use DEFAULTS.

    Returns:
        Stats dict.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    repo_root = get_repo_root()
    rng = random.Random(cfg["seed"])
    tool_mode: ToolMode = cfg.get("tool_mode", SAFE_MODE)

    # Resolve paths
    english_dir = resolve_path(cfg["english_dir"], repo_root)
    synthetic_tvl_dir = resolve_path(cfg["synthetic_tvl_dir"], repo_root)
    crosslingual_dir = resolve_path(cfg["crosslingual_dir"], repo_root)
    real_tvl_chat_dir = resolve_path(cfg["real_tvl_chat_dir"], repo_root)
    anchor_path = resolve_path(cfg["anchor_path"], repo_root)
    output_dir = resolve_path(cfg["output_dir"], repo_root)

    # Load data sources
    english_raw = _load_jsonl_dir(english_dir)
    synthetic_tvl_raw = _load_jsonl_dir(synthetic_tvl_dir)
    crosslingual_raw = _load_jsonl_dir(crosslingual_dir)
    real_tvl_chat_raw = _load_jsonl_dir(real_tvl_chat_dir)
    anchor_raw = read_jsonl(anchor_path) if anchor_path.exists() else []

    print(
        f"Loaded: {len(english_raw)} English, {len(synthetic_tvl_raw)} synthetic TVL, "
        f"{len(crosslingual_raw)} cross-lingual, "
        f"{len(real_tvl_chat_raw)} real TVL chat, {len(anchor_raw)} anchor"
    )

    # Tag sources
    _tag_source(english_raw, "english")
    _tag_source(synthetic_tvl_raw, "synthetic_tvl")
    _tag_source(crosslingual_raw, "crosslingual")
    _tag_source(real_tvl_chat_raw, "real_tvl_chat")
    _tag_source(anchor_raw, "anchor")

    # Ensure anchor examples have task_family="translation" for filtering purposes
    for ex in anchor_raw:
        if "task_family" not in ex:
            ex["task_family"] = "translation"

    # Filter by task family — anchors are exempt (they prevent catastrophic forgetting)
    include_families = cfg.get("include_task_families")
    exclude_families = cfg.get("exclude_task_families")
    english = _filter_by_task_family(english_raw, include_families, exclude_families)
    synthetic_tvl = _filter_by_task_family(synthetic_tvl_raw, include_families, exclude_families)
    crosslingual = _filter_by_task_family(crosslingual_raw, include_families, exclude_families)
    real_tvl_chat = _filter_by_task_family(real_tvl_chat_raw, include_families, exclude_families)

    # Validate examples
    for label, pool in [
        ("english", english),
        ("synthetic_tvl", synthetic_tvl),
        ("crosslingual", crosslingual),
        ("real_tvl_chat", real_tvl_chat),
        ("anchor", anchor_raw),
    ]:
        invalid = sum(1 for ex in pool if validate_example(ex))
        if invalid:
            print(f"  Warning: {invalid} invalid examples in {label}")

    # Deduplicate
    english = _deduplicate(english)
    synthetic_tvl = _deduplicate(synthetic_tvl)
    crosslingual = _deduplicate(crosslingual)
    real_tvl_chat = _deduplicate(real_tvl_chat)
    anchor = _deduplicate(anchor_raw)

    print(
        f"After dedup: {len(english)} English, {len(synthetic_tvl)} synthetic TVL, "
        f"{len(crosslingual)} cross-lingual, "
        f"{len(real_tvl_chat)} real TVL chat, {len(anchor)} anchor"
    )

    # Apply tool mode formatting
    english = _apply_tool_mode(english, tool_mode)
    synthetic_tvl = _apply_tool_mode(synthetic_tvl, tool_mode)
    crosslingual = _apply_tool_mode(crosslingual, tool_mode)
    real_tvl_chat = _apply_tool_mode(real_tvl_chat, tool_mode)

    # Split non-anchor data into train/val/test
    val_frac = cfg["validation_fraction"]
    test_frac = cfg["test_fraction"]

    split_pools: dict[str, dict[str, list[dict[str, Any]]]] = {
        "english": defaultdict(list),
        "synthetic_tvl": defaultdict(list),
        "crosslingual": defaultdict(list),
        "real_tvl_chat": defaultdict(list),
    }
    non_anchor_pools = {
        "english": english,
        "synthetic_tvl": synthetic_tvl,
        "crosslingual": crosslingual,
        "real_tvl_chat": real_tvl_chat,
    }

    for source_name, pool in non_anchor_pools.items():
        for ex in pool:
            split = _assign_split(_example_split_key(ex), val_frac, test_frac)
            split_pools[source_name][split].append(ex)

    # Build training mix from train splits only
    mix_ratios = cfg["mix_ratios"]
    active_non_anchor_sources = [
        name for name in split_pools
        if mix_ratios.get(name, 0) > 0
    ]
    train_pools = {name: split_pools[name]["train"] for name in active_non_anchor_sources}
    if mix_ratios.get("anchor", 0) > 0:
        train_pools["anchor"] = anchor
    train_all, ratio_report = _sample_to_ratio(train_pools, mix_ratios, rng)
    rng.shuffle(train_all)

    # Pilot subset
    pilot_size = min(cfg["pilot_size"], len(train_all))
    train_pilot = train_all[:pilot_size]

    # Validation and test combine all non-anchor sources (no anchor)
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for source_name in active_non_anchor_sources:
        validation.extend(split_pools[source_name]["validation"])
        test.extend(split_pools[source_name]["test"])

    # Sort deterministically
    train_all.sort(key=lambda x: _stable_hash(x["id"]))
    train_pilot.sort(key=lambda x: _stable_hash(x["id"]))
    validation.sort(key=lambda x: _stable_hash(x["id"]))
    test.sort(key=lambda x: _stable_hash(x["id"]))

    # Strip translate_mask (mixed bool/string types break PyArrow in datasets.load_dataset)
    for pool in (train_all, train_pilot, validation, test):
        for ex in pool:
            ex.pop("translate_mask", None)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_all)
    write_jsonl(output_dir / "train_pilot.jsonl", train_pilot)
    write_jsonl(output_dir / "validation.jsonl", validation)
    write_jsonl(output_dir / "test.jsonl", test)

    stats = {
        "train": _summarize(train_all, "train"),
        "train_pilot": _summarize(train_pilot, "train_pilot"),
        "validation": _summarize(validation, "validation"),
        "test": _summarize(test, "test"),
        "mix_ratios": ratio_report,
        "input_counts": {
            "english_raw": len(english_raw),
            "synthetic_tvl_raw": len(synthetic_tvl_raw),
            "crosslingual_raw": len(crosslingual_raw),
            "real_tvl_chat_raw": len(real_tvl_chat_raw),
            "anchor_raw": len(anchor_raw),
            "english_after_filter": len(english),
            "synthetic_tvl_after_filter": len(synthetic_tvl),
            "crosslingual_after_filter": len(crosslingual),
            "real_tvl_chat_after_filter": len(real_tvl_chat),
            "anchor_after_dedup": len(anchor),
        },
        "config": {
            k: v for k, v in cfg.items()
            if k not in ("english_dir", "synthetic_tvl_dir", "anchor_path", "output_dir")
        },
    }
    write_json(output_dir / "stats.json", stats)

    manifest = create_manifest(
        stage="stage_b_mix",
        config=cfg,
        extra={
            "english_dir": str(english_dir),
            "synthetic_tvl_dir": str(synthetic_tvl_dir),
            "crosslingual_dir": str(crosslingual_dir),
            "real_tvl_chat_dir": str(real_tvl_chat_dir),
            "anchor_path": str(anchor_path),
            "output_dir": str(output_dir),
            "train_count": len(train_all),
            "validation_count": len(validation),
            "test_count": len(test),
        },
    )
    save_manifest(manifest, output_dir / "manifest.json")

    print(f"\nStage B mix built:")
    print(f"  train: {len(train_all)} ({format_token_count(stats['train']['total_tokens_est'])} tokens)")
    print(f"  train_pilot: {len(train_pilot)}")
    print(f"  validation: {len(validation)}")
    print(f"  test: {len(test)}")
    print(f"  Output: {output_dir}")

    return stats
