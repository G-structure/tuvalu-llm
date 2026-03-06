"""Stage B trainer: bilingual capability adapter.

IMPORTANT: Stage B starts from openai/gpt-oss-120b BASE, NOT from Stage A
weights. Stage A exists only to produce the synthetic TVL dataset. The adapter
produced by Stage B is the final shipping artifact.

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
    tokenizer, renderer, renderer_name = get_renderer(model_name)
    logger.info("Renderer: %s", renderer_name)

    # Load data
    train_path = resolve_path(cfg["train_data"], repo_root)
    val_path = resolve_path(cfg["validation_data"], repo_root)

    train_examples = read_jsonl(train_path)
    val_examples = read_jsonl(val_path) if val_path.exists() else []

    # Apply ablation filter
    ablation_mode = cfg["ablation_mode"]
    train_examples = _filter_by_ablation(train_examples, ablation_mode)
    val_examples = _filter_by_ablation(val_examples, ablation_mode)

    # Apply task family filter
    include = cfg.get("include_task_families")
    exclude = cfg.get("exclude_task_families")
    train_examples = _filter_by_task_family(train_examples, include, exclude)
    val_examples = _filter_by_task_family(val_examples, include, exclude)

    if not train_examples:
        raise SystemExit("No training examples after filtering")

    train_tokens = estimate_dataset_tokens(train_examples)
    logger.info(
        "Training data: %d examples (%s tokens), ablation=%s",
        len(train_examples), format_token_count(train_tokens), ablation_mode,
    )
    if val_examples:
        logger.info("Validation data: %d examples", len(val_examples))

    # Render examples to token sequences
    def render_example(ex: dict[str, Any]) -> list[int]:
        """Render a single example to token IDs via the renderer."""
        return renderer.render_messages(ex["messages"], tokenizer)

    train_rendered = []
    skipped = 0
    max_length = cfg["max_length"]
    for ex in train_examples:
        tokens = render_example(ex)
        if len(tokens) > max_length:
            skipped += 1
            continue
        train_rendered.append(tokens)

    if skipped:
        logger.info("Skipped %d examples exceeding max_length=%d", skipped, max_length)

    val_rendered = []
    for ex in val_examples:
        tokens = render_example(ex)
        if len(tokens) <= max_length:
            val_rendered.append(tokens)
    if val_rendered:
        logger.info("Rendered %d validation examples", len(val_rendered))

    tb = TBLogger(run_dir / "tb")

    # Save manifest
    manifest = create_manifest(
        stage="stage_b_agent_train",
        config=cfg,
        extra={
            "model_name": model_name,
            "renderer": renderer_name,
            "train_examples": len(train_rendered),
            "train_skipped": skipped,
            "val_examples": len(val_examples),
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

    adam_params = get_adam_params(cfg["learning_rate"])

    # Training loop
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    save_every = cfg["save_every"]
    total_steps = epochs * len(train_rendered) // batch_size

    logger.info(
        "Starting training: %d epochs, batch_size=%d, total_steps=%d, start_step=%d",
        epochs, batch_size, total_steps, start_step,
    )

    global_step = start_step
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        # Simple sequential batching
        for batch_start in range(0, len(train_rendered), batch_size):
            if global_step < start_step:
                global_step += 1
                continue

            batch = train_rendered[batch_start : batch_start + batch_size]
            if not batch:
                continue

            result = training_client.train_step(
                batch=batch,
                adam_params=adam_params,
            )

            loss = float(result.get("loss", 0.0))
            epoch_loss_sum += loss
            epoch_batches += 1
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = epoch_loss_sum / max(epoch_batches, 1)
                logger.info(
                    "step=%d/%d epoch=%d loss=%.4f avg_loss=%.4f",
                    global_step, total_steps, epoch, loss, avg_loss,
                )
                step_metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss,
                    "avg_loss": avg_loss,
                    "timestamp": time.time(),
                }
                append_jsonl(metrics_path, step_metrics)
                tb.log_scalars(step_metrics, step=global_step)

            if save_every and global_step % save_every == 0:
                save_checkpoint(
                    training_client,
                    name=f"step_{global_step}",
                    log_path=log_path,
                    loop_state={"step": global_step, "epoch": epoch},
                )

            val_every = cfg["val_every"] if cfg["val_every"] is not None else save_every
            if val_every and val_rendered and global_step % val_every == 0:
                val_loss_sum = 0.0
                val_batches = 0
                for vb_start in range(0, len(val_rendered), batch_size):
                    vb = val_rendered[vb_start : vb_start + batch_size]
                    if not vb:
                        continue
                    vr = training_client.forward(vb)
                    val_loss_sum += float(vr.get("loss", 0.0))
                    val_batches += 1
                val_loss = val_loss_sum / max(val_batches, 1)
                val_m = {"step": global_step, "val_loss": val_loss}
                append_jsonl(metrics_path, val_m)
                tb.log_scalars(val_m, step=global_step)
                logger.info("step=%d val_loss=%.4f", global_step, val_loss)

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        logger.info(
            "Epoch %d complete: avg_loss=%.4f time=%.1fs",
            epoch, avg_loss, epoch_time,
        )

        # Save epoch checkpoint
        save_checkpoint(
            training_client,
            name=f"epoch_{epoch}",
            log_path=log_path,
            kind="both",
            loop_state={"step": global_step, "epoch": epoch},
        )

    # Save final checkpoint
    save_checkpoint(
        training_client,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"step": global_step, "epoch": epochs - 1},
    )

    tb.close()

    summary = {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "renderer": renderer_name,
        "ablation_mode": ablation_mode,
        "total_steps": global_step,
        "train_examples": len(train_rendered),
        "final_checkpoint": log_path,
    }
    write_json(run_dir / "summary.json", summary)

    logger.info("Stage B training complete. Run dir: %s", run_dir)
    return summary
