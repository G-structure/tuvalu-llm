#!/usr/bin/env python3
"""Build Stage B source data: load, normalize, and write English examples.

Usage:
    uv run python scripts/build_stage_b_sources.py --config configs/stage_b_sources.json
    uv run python scripts/build_stage_b_sources.py --limit 50   # quick test run
    uv run python scripts/build_stage_b_sources.py --datasets gsm8k,squad --limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.config import get_repo_root, load_config
from training.common.io import write_json
from training.common.manifests import create_manifest, save_manifest
from training.common.token_estimates import estimate_example_tokens, format_token_count
from training.synthetic.budgeting import BudgetManager
from training.synthetic.naming import dataset_name_to_filename
from training.synthetic.registry import get_loader, list_datasets

# Force loader registration
import training.synthetic.loaders  # noqa: F401

logger = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
DEFAULT_OUTPUT = REPO_ROOT / "data" / "finetune" / "stage_b_sources"
DEFAULT_OUTPUT_SUBDIR = "english_normalized"

# Default datasets to load when no config is provided
DEFAULT_DATASETS = [
    "openai/gsm8k",
    "Salesforce/xlam-function-calling-60k",
    "Muennighoff/mbpp",
    "rajpurkar/squad",
    "ccdv/cnn_dailymail",
    "HuggingFaceH4/ultrachat_200k",
    "tasksource/tasksource-instruct-v0",
]


def _normalize_dataset_specs(raw_datasets: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize config/CLI dataset declarations into loader specs."""
    specs: list[dict[str, Any]] = []
    reserved = {"name", "enabled", "task_family", "output_subdir", "loader_kwargs", "alias"}

    for dataset in raw_datasets:
        if isinstance(dataset, str):
            specs.append({
                "name": dataset,
                "alias": dataset,
                "output_subdir": DEFAULT_OUTPUT_SUBDIR,
                "loader_kwargs": {},
            })
            continue

        if not dataset.get("enabled", True):
            continue

        loader_kwargs = dict(dataset.get("loader_kwargs", {}))
        for key, value in dataset.items():
            if key not in reserved:
                loader_kwargs.setdefault(key, value)

        name = dataset["name"]
        specs.append({
            "name": name,
            "alias": dataset.get("alias", name),
            "output_subdir": dataset.get("output_subdir", DEFAULT_OUTPUT_SUBDIR),
            "loader_kwargs": loader_kwargs,
        })

    return specs


def build_sources(
    datasets: list[dict[str, Any]],
    output_dir: Path,
    limit: int | None = None,
    budget_manager: BudgetManager | None = None,
) -> dict[str, dict]:
    """Load and write normalized English examples for each dataset.

    Returns per-dataset stats.
    """
    stats: dict[str, dict] = {}

    for spec in datasets:
        ds_name = spec["name"]
        ds_alias = spec.get("alias", ds_name)
        loader_kwargs = dict(spec.get("loader_kwargs", {}))
        output_subdir = spec.get("output_subdir", DEFAULT_OUTPUT_SUBDIR)
        dataset_dir = output_dir / output_subdir
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading %s (limit=%s)...", ds_name, limit)
        try:
            loader = get_loader(ds_name)
        except KeyError as e:
            logger.warning("Skipping unknown dataset: %s (%s)", ds_name, e)
            continue

        safe_name = dataset_name_to_filename(ds_name)
        out_path = dataset_dir / f"{safe_name}.jsonl"
        count = 0
        total_tokens = 0

        try:
            with out_path.open("w") as f:
                for example in loader(limit=limit, **loader_kwargs):
                    tokens = estimate_example_tokens(example)
                    if budget_manager and not budget_manager.should_continue(ds_alias):
                        logger.info("Budget exhausted for %s after %d examples", ds_name, count)
                        break
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
                    total_tokens += tokens
                    if budget_manager:
                        budget_manager.record_usage(ds_alias, tokens)
        except NotImplementedError as e:
            logger.warning("Skipping unimplemented loader %s: %s", ds_name, e)
            out_path.unlink(missing_ok=True)
            stats[ds_alias] = {"status": "not_implemented", "error": str(e)}
            continue
        except Exception:
            logger.exception("Error loading %s", ds_name)
            stats[ds_alias] = {"status": "error"}
            continue

        stats[ds_alias] = {
            "status": "ok",
            "examples": count,
            "tokens_est": total_tokens,
            "tokens_fmt": format_token_count(total_tokens),
            "output": str(out_path),
            "dataset_name": ds_name,
            "output_subdir": output_subdir,
        }
        logger.info("  -> %d examples, ~%s tokens -> %s", count, format_token_count(total_tokens), out_path)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage B English source data")
    parser.add_argument("--config", type=str, help="JSON config file with dataset list and budgets")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max examples per dataset (for testing)")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names to load (overrides config)",
    )
    parser.add_argument("--budget", type=int, default=200_000_000, help="Total token budget (default 200M)")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        print("Available datasets:")
        for name in list_datasets():
            print(f"  {name}")
        return

    # Determine dataset list
    if args.datasets:
        datasets = _normalize_dataset_specs([d.strip() for d in args.datasets.split(",")])
    elif args.config:
        config = load_config(args.config)
        raw_datasets = config.get("datasets", DEFAULT_DATASETS)
        datasets = _normalize_dataset_specs(raw_datasets)
    else:
        datasets = _normalize_dataset_specs(DEFAULT_DATASETS)

    output_dir = Path(args.output)
    # Pre-allocate equal budget per dataset to avoid first-come-takes-all bug
    per_dataset = args.budget // max(len(datasets), 1)
    allocations = {spec["alias"]: per_dataset for spec in datasets}
    budget_manager = BudgetManager(total_budget=args.budget, allocations=allocations)

    logger.info("Building Stage B sources: %d datasets, budget=%s", len(datasets), format_token_count(args.budget))
    if args.limit:
        logger.info("Limit: %d examples per dataset", args.limit)

    stats = build_sources(
        datasets=datasets,
        output_dir=output_dir,
        limit=args.limit,
        budget_manager=budget_manager,
    )

    # Write manifest
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest = create_manifest(
        stage="stage_b_sources",
        config={"datasets": datasets, "limit": args.limit, "budget": args.budget},
        extra={
            "dataset_stats": stats,
            "budget_report": budget_manager.get_report(),
        },
    )
    manifest_path = manifests_dir / "build_manifest.json"
    save_manifest(manifest, manifest_path)
    logger.info("Manifest written to %s", manifest_path)

    # Summary
    print("\n--- Stage B Sources Summary ---")
    for ds_name, ds_stats in stats.items():
        status = ds_stats.get("status", "unknown")
        if status == "ok":
            print(f"  {ds_name}: {ds_stats['examples']} examples, ~{ds_stats['tokens_fmt']} tokens")
        else:
            print(f"  {ds_name}: {status}")

    report = budget_manager.get_report()
    print(f"\nTotal: {report['total_examples']} examples, ~{report['total_used_fmt']} tokens")
    print(f"Budget: {format_token_count(report['total_used'])}/{format_token_count(report['total_budget'])}")


if __name__ == "__main__":
    main()
