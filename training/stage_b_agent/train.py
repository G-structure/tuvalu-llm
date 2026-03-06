"""Stage B trainer: bilingual capability adapter.

IMPORTANT: Stage B starts from a base/chat model, NOT from Stage A weights.
Stage A exists only to produce the synthetic TVL dataset. The adapter produced
by Stage B is the final shipping artifact.

Supported base models:
- openai/gpt-oss-120b (MoE, 117B/5.1B active) — uses gpt_oss renderer (harmony format)
- Qwen/Qwen3-30B-A3B (MoE, 30B/3B active) — uses qwen3 renderer (im_start/im_end format)
  NOTE: Use the chat variant (not -Base) for tool-calling support.

Training modes (ablation support):
- "mixed" (default): full 3-source mix (EN + synthetic TVL + anchor)
- "english_only": only English capability data (no TVL)
- "tvl_only": only synthetic TVL data (no English replay)

Config-driven: model_name, lora_rank, max_length, batch_size, epochs,
learning_rate, save_every, seed. Supports task family inclusion/exclusion
for ablation studies.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from training.common.checkpoints import get_last_checkpoint, save_checkpoint
from training.common.config import get_repo_root, resolve_path
from training.common.io import append_jsonl, read_jsonl, setup_run_dir, write_json
from training.common.manifests import create_manifest, save_manifest
from training.common.tb import TBLogger
from training.common.tinker_runtime import (
    create_lora_training_client,
    create_service_client,
    ensure_cookbook_on_path,
    get_adam_params,
    get_renderer,
    require_tinker_api_key,
    resume_training_client,
)
from training.common.token_estimates import estimate_dataset_tokens, format_token_count

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    # Model -- Stage B starts from BASE, not Stage A adapter
    "model_name": "openai/gpt-oss-120b",
    # LoRA
    "lora_rank": 32,
    # Training hyperparams
    "max_length": 2048,
    "batch_size": 64,
    "epochs": 2,
    "learning_rate": 2e-4,
    "save_every": 200,
    "seed": 42,
    "train_on_what": "ALL_ASSISTANT_MESSAGES",
    "ttl_seconds": 7 * 24 * 3600,
    # Data paths (relative to repo root)
    "train_data": "data/finetune/stage_b_mix/train.jsonl",
    "validation_data": "data/finetune/stage_b_mix/validation.jsonl",
    # Ablation mode: "mixed" | "english_only" | "tvl_only"
    "ablation_mode": "mixed",
    # Task family filtering
    "include_task_families": None,
    "exclude_task_families": None,
    # Output
    "output_dir": "out/stage_b_agent",
    "run_name": None,  # auto-generated if None
    # Resume
    "resume_from": None,  # path to checkpoint state dir
    "val_every": None,  # defaults to save_every; set 0 to disable periodic val
}


def _filter_by_ablation(
    examples: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """Filter examples by ablation mode."""
    if mode == "mixed":
        return examples
    if mode == "english_only":
        return [
            ex for ex in examples
            if ex.get("metadata", {}).get("stage_b_source") == "english"
        ]
    if mode == "tvl_only":
        return [
            ex for ex in examples
            if ex.get("metadata", {}).get("stage_b_source") in ("synthetic_tvl", "anchor")
        ]
    raise ValueError(f"Unknown ablation_mode: {mode!r}")


def _filter_by_task_family(
    examples: list[dict[str, Any]],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[dict[str, Any]]:
    if include is not None:
        examples = [ex for ex in examples if ex.get("task_family") in include]
    if exclude is not None:
        examples = [ex for ex in examples if ex.get("task_family") not in exclude]
    return examples


def _load_split(path: Path, split_name: str) -> Any:
    import datasets  # type: ignore

    ds = datasets.load_dataset("json", data_files={split_name: str(path)})
    assert isinstance(ds, datasets.DatasetDict)
    return ds[split_name]


def _get_train_on_what(name: str) -> Any:
    ensure_cookbook_on_path()
    from tinker_cookbook import renderers  # type: ignore

    try:
        return getattr(renderers.TrainOnWhat, name)
    except AttributeError as exc:
        raise SystemExit(f"Unknown TrainOnWhat value: {name}") from exc


def _mean_val_loss(
    *,
    dataset: Any,
    renderer: Any,
    training_client: Any,
    batch_size: int,
    max_length: int,
    train_on_what: Any,
) -> float:
    ensure_cookbook_on_path()
    from tinker_cookbook.supervised.common import compute_mean_nll  # type: ignore
    from tinker_cookbook.supervised.data import conversation_to_datum  # type: ignore

    losses: list[float] = []
    n_batches = math.ceil(len(dataset) / batch_size)
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(dataset))
        rows = dataset.select(range(start, end))
        batch = [
            conversation_to_datum(row["messages"], renderer, max_length, train_on_what)
            for row in rows
        ]
        result = training_client.forward(batch, "cross_entropy").result()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [d.loss_fn_inputs["weights"] for d in batch]
        losses.append(compute_mean_nll(logprobs, weights))
    return sum(losses) / len(losses) if losses else float("nan")


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run Stage B training.

    Args:
        config: Configuration overrides. Missing keys use DEFAULTS.

    Returns:
        Run summary dict.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    require_tinker_api_key()

    repo_root = get_repo_root()
    model_name = cfg["model_name"]
    lora_rank = cfg["lora_rank"]

    # Setup run directory
    out_base = resolve_path(cfg["output_dir"], repo_root)
    run_dir = setup_run_dir(out_base, cfg.get("run_name"))
    metrics_path = run_dir / "metrics.jsonl"
    log_path = str(run_dir / "checkpoints")

    logger.info("Stage B training run: %s", run_dir)
    logger.info("Base model: %s (LoRA rank=%d)", model_name, lora_rank)
    logger.info("IMPORTANT: Training from BASE model, NOT Stage A adapter")

    # Get renderer for the model
    _tokenizer, renderer, renderer_name = get_renderer(model_name)
    logger.info("Renderer: %s", renderer_name)

    train_on_what = _get_train_on_what(cfg.get("train_on_what", "ALL_ASSISTANT_MESSAGES"))

    # Load data as HF datasets (like Stage A)
    train_path = resolve_path(cfg["train_data"], repo_root)
    val_path = resolve_path(cfg["validation_data"], repo_root)

    if not train_path.exists():
        raise SystemExit(f"Training data not found: {train_path}")

    train_dataset = _load_split(train_path, "train")

    # Apply ablation filter
    ablation_mode = cfg["ablation_mode"]
    if ablation_mode != "mixed":
        # Convert to list, filter, then back to dataset
        all_rows = list(train_dataset)
        all_rows = _filter_by_ablation(all_rows, ablation_mode)
        # Apply task family filter
        include = cfg.get("include_task_families")
        exclude = cfg.get("exclude_task_families")
        all_rows = _filter_by_task_family(all_rows, include, exclude)
        if not all_rows:
            raise SystemExit("No training examples after filtering")
        import datasets  # type: ignore
        train_dataset = datasets.Dataset.from_list(all_rows)
    else:
        include = cfg.get("include_task_families")
        exclude = cfg.get("exclude_task_families")
        if include is not None or exclude is not None:
            all_rows = list(train_dataset)
            all_rows = _filter_by_task_family(all_rows, include, exclude)
            if not all_rows:
                raise SystemExit("No training examples after filtering")
            import datasets  # type: ignore
            train_dataset = datasets.Dataset.from_list(all_rows)

    train_dataset = train_dataset.shuffle(seed=cfg["seed"])
    logger.info("Training data: %d examples, ablation=%s", len(train_dataset), ablation_mode)

    val_dataset = None
    if val_path.exists():
        val_dataset = _load_split(val_path, "validation")
        if ablation_mode != "mixed":
            val_rows = list(val_dataset)
            val_rows = _filter_by_ablation(val_rows, ablation_mode)
            val_rows = _filter_by_task_family(val_rows, include, exclude)
            if val_rows:
                import datasets as _ds  # type: ignore
                val_dataset = _ds.Dataset.from_list(val_rows)
            else:
                val_dataset = None
        logger.info("Validation data: %d examples", len(val_dataset) if val_dataset else 0)

    tb = TBLogger(run_dir / "tb")

    # Save manifest
    manifest = create_manifest(
        stage="stage_b_agent_train",
        config=cfg,
        extra={
            "model_name": model_name,
            "renderer": renderer_name,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset) if val_dataset else 0,
            "ablation_mode": ablation_mode,
            "run_dir": str(run_dir),
        },
    )
    save_manifest(manifest, run_dir / "manifest.json")

    # Create or resume training client
    service = create_service_client()
    start_step = 0

    if cfg.get("resume_from"):
        logger.info("Resuming from: %s", cfg["resume_from"])
        training_client, start_step = resume_training_client(
            service, cfg["resume_from"]
        )
    else:
        training_client = create_lora_training_client(
            service, model_name, lora_rank=lora_rank,
        )

    ensure_cookbook_on_path()
    from tinker_cookbook.supervised.common import compute_mean_nll  # type: ignore
    from tinker_cookbook.supervised.data import conversation_to_datum  # type: ignore

    # Training loop (aligned with Stage A pattern)
    batch_size = cfg["batch_size"]
    max_length = cfg["max_length"]
    save_every = cfg["save_every"]
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    total_steps = steps_per_epoch * cfg["epochs"]

    logger.info(
        "Starting training: %d epochs, batch_size=%d, total_steps=%d, start_step=%d",
        cfg["epochs"], batch_size, total_steps, start_step,
    )

    final_metrics: dict[str, Any] = {}

    for global_step in range(start_step, total_steps):
        epoch = global_step // steps_per_epoch
        step_in_epoch = global_step % steps_per_epoch
        batch_start = step_in_epoch * batch_size
        batch_end = min(batch_start + batch_size, len(train_dataset))
        rows = train_dataset.select(range(batch_start, batch_end))
        batch = [
            conversation_to_datum(row["messages"], renderer, max_length, train_on_what)
            for row in rows
        ]
        if not batch:
            continue

        if save_every > 0 and global_step > 0 and global_step % save_every == 0:
            save_checkpoint(
                training_client=training_client,
                name=f"{global_step:06d}",
                log_path=log_path,
                kind="state",
                loop_state={"step": global_step},
                ttl_seconds=cfg["ttl_seconds"],
            )

        val_every = cfg["val_every"] if cfg["val_every"] is not None else save_every
        if (
            val_every > 0
            and val_dataset is not None
            and global_step > 0
            and global_step % val_every == 0
        ):
            val_nll = _mean_val_loss(
                dataset=val_dataset,
                renderer=renderer,
                training_client=training_client,
                batch_size=batch_size,
                max_length=max_length,
                train_on_what=train_on_what,
            )
            val_m = {"step": global_step, "validation_mean_nll": val_nll}
            append_jsonl(metrics_path, val_m)
            tb.log_scalars(val_m, step=global_step)
            logger.info("step=%d validation_mean_nll=%.4f", global_step, val_nll)

        start_time = time.time()
        lr_mult = max(0.0, 1.0 - (global_step / max(total_steps, 1)))
        current_lr = cfg["learning_rate"] * lr_mult
        adam_params = get_adam_params(current_lr)

        fwd_bwd_result = training_client.forward_backward(batch, loss_fn="cross_entropy").result()
        optim_result = training_client.optim_step(adam_params).result()

        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        metrics: dict[str, Any] = {
            "step": global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "learning_rate": current_lr,
            "train_mean_nll": train_nll,
            "num_sequences": len(batch),
            "num_tokens": sum(d.model_input.length for d in batch),
            "time_total": time.time() - start_time,
        }
        if getattr(optim_result, "metrics", None):
            metrics.update(optim_result.metrics)
        append_jsonl(metrics_path, metrics)
        tb.log_scalars(metrics, step=global_step)
        final_metrics = metrics

        if global_step % 10 == 0:
            logger.info(
                "step=%d/%d epoch=%d train_nll=%.4f lr=%.6g batch=%d tokens=%d",
                global_step,
                total_steps,
                epoch,
                train_nll,
                current_lr,
                len(batch),
                metrics["num_tokens"],
            )

    # Save final checkpoint
    save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"step": total_steps},
        ttl_seconds=cfg["ttl_seconds"],
    )

    tb.close()

    summary = {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "renderer": renderer_name,
        "ablation_mode": ablation_mode,
        "total_steps": total_steps,
        "train_examples": len(train_dataset),
        "final_checkpoint": log_path,
        "final_metrics": final_metrics,
    }
    write_json(run_dir / "summary.json", summary)

    logger.info("Stage B training complete. Run dir: %s", run_dir)
    return summary
