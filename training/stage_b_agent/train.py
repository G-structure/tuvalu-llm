"""Stage B trainer: bilingual capability adapter.

IMPORTANT: Stage B starts from a base/chat model, NOT from Stage A weights.
Stage A exists only to produce the synthetic TVL dataset. The adapter produced
by Stage B is the final shipping artifact.

Optimizations (matching Stage A):
- Fire-all-futures gen eval for fast chrF++/BLEU scoring
- Per-epoch data reshuffling with deterministic seeds
- Pipelined forward_backward + optim_step (1 clock cycle instead of 3)

Training modes (ablation support):
- "mixed" (default): full multi-source mix (EN + synthetic TVL + crosslingual + anchor)
- "english_only": only English capability data (no TVL)
- "tvl_only": only synthetic TVL data (no English replay)
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
from training.common.io import append_jsonl, read_jsonl, setup_run_dir, write_json
from training.common.manifests import create_manifest, save_git_diff, save_manifest
from training.common.metrics import compute_translation_metrics
from training.common.tb import TBLogger
from training.common.tinker_runtime import (
    create_lora_training_client,
    create_sampling_client,
    create_service_client,
    ensure_cookbook_on_path,
    get_adam_params,
    get_renderer,
    get_sampling_params,
    require_tinker_api_key,
    resume_training_client,
)
from training.common.token_estimates import estimate_dataset_tokens, format_token_count

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    # Model -- Stage B starts from BASE, not Stage A adapter
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    # LoRA
    "lora_rank": 32,
    # Training hyperparams
    "max_length": 2048,
    "batch_size": 128,
    "epochs": 3,
    "learning_rate": 2.83e-4,
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
    # Gen eval (fire-all parallel)
    "gen_eval_every": 500,
    "gen_eval_data": "data/finetune/stage_a_mt/test.jsonl",
    "gen_eval_parallel": 512,
    "gen_eval_max_tokens": 512,
    # Validation
    "val_every": 500,
    "val_max_examples": 200,
    # Output
    "output_dir": "logs/tinker/stage_b",
    "run_name": None,
    # Resume
    "resume_from": None,
}


def _filter_by_ablation(
    examples: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
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


def _run_gen_eval(
    *,
    dataset: Any,
    service_client: Any,
    training_client: Any,
    renderer: Any,
    step: int,
    log_path: str,
    parallel: int = 512,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Fire-all-futures gen eval: saves sampler weights, fires all requests, collects."""
    step_name = f"{step:06d}"

    # Save sampler weights for this eval
    ckpt_paths = save_checkpoint(
        training_client=training_client,
        name=f"gen_eval_{step_name}",
        log_path=log_path,
        kind="sampler",
    )
    sampler_path = ckpt_paths.get("sampler_path", "")
    logger.info("Saved sampler weights for gen eval: %s", sampler_path)

    sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    sampling_params = get_sampling_params(
        renderer, max_tokens=max_tokens, temperature=temperature
    )

    n = len(dataset)
    predictions: list[dict[str, Any]] = [None] * n  # type: ignore[list-item]

    # Fire ALL futures upfront
    all_futures: list[tuple[int, dict[str, Any], Any]] = []
    for idx in range(n):
        row = dataset[idx]
        messages = row["messages"]
        prompt = renderer.build_generation_prompt(messages[:-1])
        future = sampling_client.sample(
            prompt, sampling_params=sampling_params, num_samples=1
        )
        all_futures.append((idx, row, future))
    logger.info("Gen eval: fired %d sample requests", n)

    # Collect results
    for i, (idx, row, future) in enumerate(all_futures):
        result = future.result()
        output_tokens = result.sequences[0].tokens
        response_message, _ok = renderer.parse_response(output_tokens)
        pred_text = ""
        if isinstance(response_message, dict):
            pred_text = str(response_message.get("content", ""))
        else:
            pred_text = str(response_message)

        messages = row["messages"]
        predictions[idx] = {
            "prediction": pred_text,
            "reference": messages[-1]["content"],
            "direction": row.get("metadata", {}).get("direction"),
            "domain": row.get("metadata", {}).get("domain"),
        }
        if (i + 1) % parallel == 0 or (i + 1) == n:
            logger.info("Gen eval: scored %d / %d", i + 1, n)

    metrics = compute_translation_metrics(predictions)
    metrics["count"] = len(predictions)
    return metrics


def _mean_val_loss(
    *,
    dataset: Any,
    renderer: Any,
    training_client: Any,
    batch_size: int,
    max_length: int,
    train_on_what: Any,
    max_examples: int | None = None,
) -> float:
    ensure_cookbook_on_path()
    from tinker_cookbook.supervised.common import compute_mean_nll  # type: ignore
    from tinker_cookbook.supervised.data import conversation_to_datum  # type: ignore

    if max_examples and len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))

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
    """Run Stage B training."""
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    require_tinker_api_key()

    repo_root = get_repo_root()
    model_name = cfg["model_name"]
    lora_rank = cfg["lora_rank"]

    # Setup run directory
    out_base = resolve_path(cfg["output_dir"], repo_root)
    out_base.mkdir(parents=True, exist_ok=True)
    log_path = str(out_base)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_base / "train.log"),
        ],
    )

    metrics_path = out_base / "metrics.jsonl"

    logger.info("Stage B training: %s", out_base)
    logger.info("Base model: %s (LoRA rank=%d)", model_name, lora_rank)
    logger.info("IMPORTANT: Training from BASE model, NOT Stage A adapter")

    # Renderer
    _tokenizer, renderer, renderer_name = get_renderer(model_name)
    logger.info("Renderer: %s", renderer_name)

    train_on_what = _get_train_on_what(cfg.get("train_on_what", "ALL_ASSISTANT_MESSAGES"))

    # Load data
    train_path = resolve_path(cfg["train_data"], repo_root)
    val_path = resolve_path(cfg["validation_data"], repo_root)

    if not train_path.exists():
        raise SystemExit(f"Training data not found: {train_path}")

    train_dataset_raw = _load_split(train_path, "train")

    # Apply ablation + task family filters
    ablation_mode = cfg["ablation_mode"]
    include = cfg.get("include_task_families")
    exclude = cfg.get("exclude_task_families")
    if ablation_mode != "mixed" or include is not None or exclude is not None:
        all_rows = list(train_dataset_raw)
        all_rows = _filter_by_ablation(all_rows, ablation_mode)
        all_rows = _filter_by_task_family(all_rows, include, exclude)
        if not all_rows:
            raise SystemExit("No training examples after filtering")
        import datasets  # type: ignore
        train_dataset_raw = datasets.Dataset.from_list(all_rows)

    logger.info("Loaded %d training examples (ablation=%s)", len(train_dataset_raw), ablation_mode)

    val_dataset = None
    if val_path.exists():
        val_dataset = _load_split(val_path, "validation")
        logger.info("Loaded %d validation examples", len(val_dataset))

    # Gen eval dataset
    gen_eval_dataset = None
    gen_eval_every = cfg.get("gen_eval_every")
    if gen_eval_every and gen_eval_every > 0:
        gen_eval_path = resolve_path(cfg["gen_eval_data"], repo_root)
        if gen_eval_path.exists():
            gen_eval_dataset = _load_split(gen_eval_path, "eval")
            logger.info("Loaded %d gen eval examples", len(gen_eval_dataset))
        else:
            logger.warning("Gen eval data not found: %s — skipping gen eval", gen_eval_path)

    tb = TBLogger(out_base / "tb")

    # Save manifest
    manifest = create_manifest(
        stage="stage_b_agent_train",
        config=cfg,
        extra={
            "model_name": model_name,
            "renderer": renderer_name,
            "train_examples": len(train_dataset_raw),
            "val_examples": len(val_dataset) if val_dataset else 0,
            "ablation_mode": ablation_mode,
        },
    )
    save_manifest(manifest, out_base / "manifest.json")
    save_git_diff(out_base)

    # Create or resume training client
    service = create_service_client()
    start_step = 0

    last_ckpt = get_last_checkpoint(log_path, key="state_path")
    if last_ckpt:
        logger.info("Resuming from checkpoint: %s", last_ckpt["state_path"])
        training_client, _info = resume_training_client(service, last_ckpt["state_path"])
        start_step = last_ckpt.get("step", 0)
    elif cfg.get("resume_from"):
        logger.info("Resuming from: %s", cfg["resume_from"])
        training_client, _info = resume_training_client(service, cfg["resume_from"])
    else:
        training_client = create_lora_training_client(
            service, model_name, lora_rank=lora_rank,
        )

    ensure_cookbook_on_path()
    from tinker_cookbook.supervised.common import compute_mean_nll  # type: ignore
    from tinker_cookbook.supervised.data import conversation_to_datum  # type: ignore

    # Training loop
    batch_size = cfg["batch_size"]
    max_length = cfg["max_length"]
    save_every = cfg["save_every"]
    val_every = cfg.get("val_every") or save_every
    steps_per_epoch = math.ceil(len(train_dataset_raw) / batch_size)
    total_steps = steps_per_epoch * cfg["epochs"]

    logger.info(
        "Training for %d epochs, %d total steps (start_step=%d, batch_size=%d)",
        cfg["epochs"], total_steps, start_step, batch_size,
    )

    current_epoch = -1
    train_dataset = train_dataset_raw  # reshuffled per epoch
    final_metrics: dict[str, Any] = {}

    for global_step in range(start_step, total_steps):
        epoch = global_step // steps_per_epoch
        step_in_epoch = global_step % steps_per_epoch

        # Reshuffle at the start of each epoch
        if epoch != current_epoch:
            epoch_seed = cfg["seed"] + epoch
            train_dataset = train_dataset_raw.shuffle(seed=epoch_seed)
            logger.info("Epoch %d: reshuffled training data (seed=%d)", epoch, epoch_seed)
            current_epoch = epoch

        # Save checkpoint
        if save_every > 0 and global_step > 0 and global_step % save_every == 0:
            save_checkpoint(
                training_client=training_client,
                name=f"{global_step:06d}",
                log_path=log_path,
                kind="state",
                loop_state={"step": global_step},
                ttl_seconds=cfg["ttl_seconds"],
            )

        # Validation NLL
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
                max_examples=cfg.get("val_max_examples"),
            )
            logger.info("step=%d validation_mean_nll=%.4f", global_step, val_nll)
            val_m = {"step": global_step, "validation_mean_nll": val_nll}
            append_jsonl(metrics_path, val_m)
            tb.log_scalars(val_m, step=global_step)

        # Gen eval (fire-all parallel)
        if (
            gen_eval_every
            and gen_eval_every > 0
            and gen_eval_dataset is not None
            and global_step > 0
            and global_step % gen_eval_every == 0
        ):
            gen_metrics = _run_gen_eval(
                dataset=gen_eval_dataset,
                service_client=service,
                training_client=training_client,
                renderer=renderer,
                step=global_step,
                log_path=log_path,
                parallel=cfg.get("gen_eval_parallel", 512),
                max_tokens=cfg.get("gen_eval_max_tokens", 512),
            )
            gen_m = {
                "step": global_step,
                "gen_eval_chrf_pp": gen_metrics["chrf_pp"],
                "gen_eval_bleu": gen_metrics["bleu"],
                "gen_eval_exact_match": gen_metrics["exact_match"],
                "gen_eval_count": gen_metrics["count"],
            }
            append_jsonl(metrics_path, gen_m)
            tb.log_scalars(gen_m, step=global_step)
            logger.info(
                "step=%d gen_eval chrF++=%.2f BLEU=%.2f exact=%.3f",
                global_step, gen_metrics["chrf_pp"], gen_metrics["bleu"],
                gen_metrics["exact_match"],
            )

        # Training step — pipelined (1 clock cycle)
        batch_start = step_in_epoch * batch_size
        batch_end = min(batch_start + batch_size, len(train_dataset))
        rows = train_dataset.select(range(batch_start, batch_end))
        batch = [
            conversation_to_datum(row["messages"], renderer, max_length, train_on_what)
            for row in rows
        ]
        if not batch:
            continue

        start_time = time.time()
        lr_mult = max(0.0, 1.0 - (global_step / max(total_steps, 1)))
        current_lr = cfg["learning_rate"] * lr_mult
        adam_params = get_adam_params(current_lr)

        # Submit both before awaiting either (1 clock cycle instead of 3)
        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()

        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        metrics: dict[str, Any] = {
            "step": global_step,
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_mean_nll": train_nll,
            "batch": len(batch),
            "tokens": sum(d.model_input.length for d in batch),
            "time_total": time.time() - start_time,
        }
        append_jsonl(metrics_path, metrics)
        tb.log_scalars(metrics, step=global_step)
        final_metrics = metrics

        if global_step % 10 == 0:
            logger.info(
                "step=%d/%d epoch=%d train_nll=%.4f lr=%.6g batch=%d tokens=%d",
                global_step, total_steps, epoch,
                train_nll, current_lr, len(batch),
                metrics["tokens"],
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
    logger.info("Saved final checkpoint to %s", log_path)

    tb.close()

    summary = {
        "model_name": model_name,
        "renderer": renderer_name,
        "ablation_mode": ablation_mode,
        "total_steps": total_steps,
        "train_examples": len(train_dataset_raw),
        "final_metrics": final_metrics,
    }
    write_json(out_base / "summary.json", summary)

    logger.info("Stage B training complete.")
    return summary
