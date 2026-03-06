"""Stage A: Train a Tuvaluan<->English LoRA translation adapter in Tinker.

Refactored from scripts/train_tinker_mt.py. Uses the training.common
foundation package for client setup, checkpointing, and metrics.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

from training.common.checkpoints import get_last_checkpoint, save_checkpoint
from training.common.config import get_repo_root, resolve_path
from training.common.io import append_jsonl, write_json
from training.common.manifests import create_manifest, save_manifest
from training.common.tinker_runtime import (
    create_lora_training_client,
    create_service_client,
    ensure_cookbook_on_path,
    get_adam_params,
    get_renderer,
    require_tinker_api_key,
    resume_training_client,
)

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "data": "data/finetune/stage_a_mt/train_balanced.jsonl",
    "val_data": "data/finetune/stage_a_mt/validation.jsonl",
    "model_name": "Qwen/Qwen3-30B-A3B-Base",
    "log_path": "logs/stage_a_mt",
    "base_url": None,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 2,
    "max_length": 2048,
    "lora_rank": 32,
    "train_on_what": "ALL_ASSISTANT_MESSAGES",
    "save_every": 100,
    "ttl_seconds": 7 * 24 * 3600,
    "seed": 17,
    "do_final_val_loss": False,
}


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "train.log"),
        ],
    )
    return metrics_path


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
    """Run Stage A training.

    Args:
        config: Configuration dict. Missing keys use DEFAULTS.

    Returns:
        Summary dict with final metrics and log path.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    repo_root = get_repo_root()
    data_path = resolve_path(cfg["data"], repo_root)
    val_path = resolve_path(cfg["val_data"], repo_root)
    log_path = resolve_path(cfg["log_path"], repo_root)

    metrics_path = _setup_logging(log_path)
    require_tinker_api_key()

    if not data_path.exists():
        raise SystemExit(f"Training data not found: {data_path}")
    if not val_path.exists():
        logger.warning("Validation data not found: %s", val_path)

    model_name = cfg["model_name"]
    logger.info("Loading tokenizer and renderer for %s", model_name)
    _tokenizer, renderer, renderer_name = get_renderer(model_name)

    train_on_what = _get_train_on_what(cfg["train_on_what"])
    train_dataset = _load_split(data_path, "train")
    train_dataset = train_dataset.shuffle(seed=cfg["seed"])
    logger.info("Loaded %d training examples", len(train_dataset))

    val_dataset = None
    if val_path.exists():
        val_dataset = _load_split(val_path, "validation")
        logger.info("Loaded %d validation examples", len(val_dataset))

    service_client = create_service_client(base_url=cfg["base_url"])

    checkpoint_info = get_last_checkpoint(str(log_path))
    if checkpoint_info:
        logger.info("Resuming from checkpoint: %s", checkpoint_info["state_path"])
        training_client, start_step = resume_training_client(
            service_client, str(log_path)
        )
    else:
        training_client = create_lora_training_client(
            service_client,
            model_name=model_name,
            lora_rank=cfg["lora_rank"],
        )
        start_step = 0

    ensure_cookbook_on_path()
    from tinker_cookbook.supervised.common import compute_mean_nll  # type: ignore
    from tinker_cookbook.supervised.data import conversation_to_datum  # type: ignore

    steps_per_epoch = math.ceil(len(train_dataset) / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    logger.info("Training for %d epochs, %d total steps", cfg["epochs"], total_steps)

    manifest = create_manifest(
        stage="stage_a_mt_train",
        config=cfg,
        extra={
            "data_path": str(data_path),
            "log_path": str(log_path),
            "total_steps": total_steps,
            "train_examples": len(train_dataset),
        },
    )
    save_manifest(manifest, log_path / "manifest.json")

    final_metrics: dict[str, Any] = {}

    for global_step in range(start_step, total_steps):
        epoch = global_step // steps_per_epoch
        step_in_epoch = global_step % steps_per_epoch
        batch_start = step_in_epoch * cfg["batch_size"]
        batch_end = min(batch_start + cfg["batch_size"], len(train_dataset))
        rows = train_dataset.select(range(batch_start, batch_end))
        batch = [
            conversation_to_datum(row["messages"], renderer, cfg["max_length"], train_on_what)
            for row in rows
        ]
        if not batch:
            continue

        if cfg["save_every"] > 0 and global_step > 0 and global_step % cfg["save_every"] == 0:
            save_checkpoint(
                training_client=training_client,
                name=f"{global_step:06d}",
                log_path=str(log_path),
                kind="state",
                loop_state={"step": global_step},
                ttl_seconds=cfg["ttl_seconds"],
            )

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

    save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=str(log_path),
        kind="both",
        loop_state={"step": total_steps},
        ttl_seconds=cfg["ttl_seconds"],
    )
    logger.info("Saved final checkpoint under %s", log_path)

    if cfg["do_final_val_loss"] and val_dataset is not None and len(val_dataset) > 0:
        logger.info("Computing final validation loss...")
        final_resume = get_last_checkpoint(str(log_path))
        if not final_resume:
            raise SystemExit("Expected a final checkpoint but none was found.")
        final_client = service_client.create_training_client_from_state(
            final_resume["state_path"]
        )
        val_nll = _mean_val_loss(
            dataset=val_dataset,
            renderer=renderer,
            training_client=final_client,
            batch_size=cfg["batch_size"],
            max_length=cfg["max_length"],
            train_on_what=train_on_what,
        )
        append_jsonl(metrics_path, {"step": total_steps, "validation_mean_nll": val_nll})
        logger.info("final validation_mean_nll=%.4f", val_nll)
        final_metrics["validation_mean_nll"] = val_nll

    logger.info("Training completed")

    return {
        "log_path": str(log_path),
        "total_steps": total_steps,
        "model_name": model_name,
        "final_metrics": final_metrics,
    }
