"""Stage A: Evaluate a trained translation adapter.

Refactored from scripts/eval_tinker_mt.py. Uses training.common for
client setup and metrics computation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from training.common.config import get_repo_root, resolve_path
from training.common.io import write_json, write_jsonl
from training.common.manifests import create_manifest, save_manifest
from training.common.metrics import (
    compute_grouped_metrics,
    compute_translation_metrics,
)
from training.common.tinker_runtime import (
    create_sampling_client,
    create_service_client,
    get_renderer,
    get_sampling_params,
    require_tinker_api_key,
)

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "data": "data/finetune/stage_a_mt/test.jsonl",
    "model_path": None,
    "base_model": None,
    "model_name": "Qwen/Qwen3-30B-A3B-Base",
    "out_dir": "logs/stage_a_mt/evals/latest",
    "base_url": None,
    "max_tokens": 512,
    "temperature": 0.0,
    "limit": None,
}


def _load_split(path: Path) -> Any:
    import datasets  # type: ignore

    ds = datasets.load_dataset("json", data_files={"eval": str(path)})
    assert isinstance(ds, datasets.DatasetDict)
    return ds["eval"]


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run Stage A evaluation.

    Args:
        config: Configuration dict. Missing keys use DEFAULTS.

    Returns:
        Metrics summary dict.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    repo_root = get_repo_root()
    data_path = resolve_path(cfg["data"], repo_root)
    out_dir = resolve_path(cfg["out_dir"], repo_root)

    require_tinker_api_key()
    if not data_path.exists():
        raise SystemExit(f"Evaluation data not found: {data_path}")
    if not cfg["model_path"] and not cfg["base_model"]:
        raise SystemExit("Provide either model_path or base_model in config.")

    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "eval.log"),
        ],
    )

    model_name = cfg["model_name"]
    _tokenizer, renderer, renderer_name = get_renderer(model_name)

    dataset = _load_split(data_path)
    if cfg["limit"] is not None:
        dataset = dataset.select(range(min(cfg["limit"], len(dataset))))
    logger.info("Loaded %d eval examples", len(dataset))

    service_client = create_service_client(base_url=cfg["base_url"])
    sampling_client = create_sampling_client(
        service_client,
        model_path=cfg["model_path"],
        base_model=cfg["base_model"],
    )
    sampling_params = get_sampling_params(
        renderer,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
    )

    predictions: list[dict[str, Any]] = []
    parse_success_count = 0
    for idx, row in enumerate(dataset):
        messages = row["messages"]
        prompt = renderer.build_generation_prompt(messages[:-1])
        result = sampling_client.sample(
            prompt, sampling_params=sampling_params, num_samples=1
        ).result()
        output_tokens = result.sequences[0].tokens
        response_message, parse_success = renderer.parse_response(output_tokens)
        if parse_success:
            parse_success_count += 1

        prediction_text = ""
        if isinstance(response_message, dict):
            prediction_text = str(response_message.get("content", ""))
        else:
            prediction_text = str(response_message)

        record = {
            "id": row.get("id", idx),
            "prediction": prediction_text,
            "reference": messages[-1]["content"],
            "direction": row.get("metadata", {}).get("direction"),
            "domain": row.get("metadata", {}).get("domain"),
            "content_type": row.get("metadata", {}).get("content_type"),
            "parse_success": bool(parse_success),
        }
        predictions.append(record)

        if idx % 25 == 0:
            logger.info("Scored %d / %d", idx + 1, len(dataset))

    write_jsonl(out_dir / "predictions.jsonl", predictions)

    overall = compute_translation_metrics(predictions)
    by_direction = compute_grouped_metrics(predictions, "direction")
    by_domain = compute_grouped_metrics(predictions, "domain")
    by_content_type = compute_grouped_metrics(predictions, "content_type")

    summary = {
        "model_path": cfg["model_path"],
        "base_model": cfg["base_model"],
        "model_name": model_name,
        "count": len(predictions),
        "parse_success_rate": parse_success_count / len(predictions) if predictions else 0,
        "overall": overall,
        "by_direction": by_direction,
        "by_domain": by_domain,
        "by_content_type": by_content_type,
    }
    write_json(out_dir / "metrics.json", summary)

    manifest = create_manifest(
        stage="stage_a_mt_eval",
        config=cfg,
        extra={
            "data_path": str(data_path),
            "out_dir": str(out_dir),
            "count": len(predictions),
            "overall": overall,
        },
    )
    save_manifest(manifest, out_dir / "manifest.json")

    print(f"Evaluation complete: {len(predictions)} examples")
    if overall.get("chrf_pp") is not None:
        print(f"  chrF++: {overall['chrf_pp']:.2f}  BLEU: {overall['bleu']:.2f}  "
              f"Exact: {overall['exact_match']:.3f}")

    return summary
