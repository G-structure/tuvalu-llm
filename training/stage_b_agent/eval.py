"""Stage B evaluation: translation regression, capability, bilingual comparison.

Supports multiple base models (gpt-oss-120b, Qwen3-30B-A3B, etc.) — renderer
and tokenizer are auto-selected via get_renderer() based on model_name.

Evaluation dimensions:
1. Translation regression: compare Stage B adapter on MT test set vs Stage A baseline.
2. Capability smoke tests: per-task-family generation quality checks.
3. Bilingual comparison: same held-out tasks in EN vs synthetic TVL.
4. Preservation metrics: JSON parse rate, schema validity, code exact-match,
   placeholder leak rate (via training.common.metrics.compute_preservation_metrics).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.common.config import get_repo_root, resolve_path
from training.common.io import read_jsonl, write_json
from training.common.manifests import create_manifest, save_manifest
from training.common.metrics import (
    compute_grouped_metrics,
    compute_preservation_metrics,
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
    # Model to evaluate
    "model_name": "openai/gpt-oss-120b",
    "model_path": None,  # path to Stage B adapter weights; if None, uses base model
    # Stage A adapter for regression comparison (optional)
    "stage_a_model_path": None,
    # Test data
    "mt_test_data": "data/finetune/stage_a_mt/test.jsonl",
    "capability_test_data": "data/finetune/stage_b_mix/test.jsonl",
    # Generation params
    "max_tokens": 512,
    "temperature": 0.0,
    # Output
    "output_dir": "out/stage_b_eval",
    "run_name": None,
    # Batching
    "eval_batch_size": 16,
}


def _extract_prompt_and_reference(example: dict[str, Any]) -> tuple[list[dict], str]:
    """Split an example into prompt messages and expected reference.

    The last assistant message is the reference. Everything before it is the prompt.
    """
    messages = example.get("messages", [])
    if not messages:
        return [], ""

    # Find last assistant message as reference
    ref_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            ref_idx = i
            break

    if ref_idx is None:
        return messages, ""

    prompt = messages[:ref_idx]
    reference = messages[ref_idx].get("content", "")
    return prompt, reference


def _generate_predictions(
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    examples: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Generate predictions for a set of examples using Stage A's known-good pattern."""
    predictions: list[dict[str, Any]] = []
    parse_success_count = 0

    for i, ex in enumerate(examples):
        prompt_msgs, reference = _extract_prompt_and_reference(ex)
        if not prompt_msgs:
            continue

        try:
            prompt = renderer.build_generation_prompt(prompt_msgs)
            result = sampling_client.sample(
                prompt, sampling_params=sampling_params, num_samples=1,
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
        except Exception as e:
            logger.warning("Generation failed for %s: %s", ex.get("id"), e)
            prediction_text = ""
            parse_success = False

        predictions.append({
            "id": ex.get("id", ""),
            "task_family": ex.get("task_family", "unknown"),
            "prediction": prediction_text,
            "reference": reference,
            "metadata": ex.get("metadata", {}),
            "parse_success": bool(parse_success),
        })

        if i % 25 == 0:
            logger.info("Generated %d / %d predictions", i + 1, len(examples))

    return predictions


def _run_translation_regression(
    service: Any,
    renderer: Any,
    sampling_params: Any,
    mt_test: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate translation quality on MT test set, optionally comparing to Stage A."""
    report: dict[str, Any] = {}
    batch_size = cfg["eval_batch_size"]

    # Stage B model predictions
    if cfg.get("model_path"):
        stage_b_client = create_sampling_client(service, model_path=cfg["model_path"])
    else:
        stage_b_client = create_sampling_client(service, base_model=cfg["model_name"])

    logger.info("Evaluating Stage B on %d MT test examples", len(mt_test))
    stage_b_preds = _generate_predictions(
        stage_b_client, renderer, sampling_params, mt_test, batch_size,
    )
    report["stage_b"] = compute_translation_metrics(stage_b_preds)
    report["stage_b_by_direction"] = compute_grouped_metrics(stage_b_preds, "direction")

    # Stage A baseline (optional)
    if cfg.get("stage_a_model_path"):
        logger.info("Evaluating Stage A baseline for regression comparison")
        stage_a_client = create_sampling_client(service, model_path=cfg["stage_a_model_path"])
        stage_a_preds = _generate_predictions(
            stage_a_client, renderer, sampling_params, mt_test, batch_size,
        )
        report["stage_a"] = compute_translation_metrics(stage_a_preds)
        report["stage_a_by_direction"] = compute_grouped_metrics(stage_a_preds, "direction")

        # Compute regression delta
        if report["stage_b"].get("chrf_pp") and report["stage_a"].get("chrf_pp"):
            report["regression_delta"] = {
                "chrf_pp": report["stage_b"]["chrf_pp"] - report["stage_a"]["chrf_pp"],
                "bleu": report["stage_b"].get("bleu", 0) - report["stage_a"].get("bleu", 0),
            }

    return report


def _run_capability_smoke(
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    capability_test: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    """Run per-task-family capability smoke tests."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in capability_test:
        family = ex.get("task_family", "unknown")
        by_family[family].append(ex)

    report: dict[str, Any] = {}
    for family, examples in sorted(by_family.items()):
        logger.info("Capability smoke test: %s (%d examples)", family, len(examples))
        preds = _generate_predictions(
            sampling_client, renderer, sampling_params, examples, batch_size,
        )
        report[family] = {
            "translation_metrics": compute_translation_metrics(preds),
            "preservation_metrics": compute_preservation_metrics(preds),
            "count": len(preds),
        }

    return report


def _run_bilingual_comparison(
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    capability_test: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    """Compare performance on EN vs synthetic TVL versions of the same tasks."""
    en_examples = [
        ex for ex in capability_test
        if ex.get("metadata", {}).get("stage_b_source") == "english"
    ]
    tvl_examples = [
        ex for ex in capability_test
        if ex.get("metadata", {}).get("stage_b_source") == "synthetic_tvl"
    ]

    report: dict[str, Any] = {"english_count": len(en_examples), "tvl_count": len(tvl_examples)}

    if en_examples:
        en_preds = _generate_predictions(
            sampling_client, renderer, sampling_params, en_examples, batch_size,
        )
        report["english"] = compute_translation_metrics(en_preds)

    if tvl_examples:
        tvl_preds = _generate_predictions(
            sampling_client, renderer, sampling_params, tvl_examples, batch_size,
        )
        report["synthetic_tvl"] = compute_translation_metrics(tvl_preds)

    return report


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run Stage B evaluation suite.

    Args:
        config: Configuration overrides.

    Returns:
        Full evaluation report dict.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    require_tinker_api_key()

    repo_root = get_repo_root()
    model_name = cfg["model_name"]

    # Setup output directory
    from training.common.io import setup_run_dir
    out_base = resolve_path(cfg["output_dir"], repo_root)
    run_dir = setup_run_dir(out_base, cfg.get("run_name"))

    logger.info("Stage B evaluation run: %s", run_dir)

    # Get renderer
    _tokenizer, renderer, renderer_name = get_renderer(model_name)
    sampling_params = get_sampling_params(
        renderer, max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
    )

    # Load test data
    mt_test_path = resolve_path(cfg["mt_test_data"], repo_root)
    cap_test_path = resolve_path(cfg["capability_test_data"], repo_root)

    mt_test = read_jsonl(mt_test_path) if mt_test_path.exists() else []
    capability_test = read_jsonl(cap_test_path) if cap_test_path.exists() else []

    logger.info("MT test: %d, Capability test: %d", len(mt_test), len(capability_test))

    service = create_service_client()
    batch_size = cfg["eval_batch_size"]

    # Create sampling client for Stage B
    if cfg.get("model_path"):
        stage_b_sampling = create_sampling_client(service, model_path=cfg["model_path"])
    else:
        stage_b_sampling = create_sampling_client(service, base_model=model_name)

    report: dict[str, Any] = {
        "model_name": model_name,
        "renderer": renderer_name,
        "model_path": cfg.get("model_path"),
        "timestamp": time.time(),
    }

    # 1. Translation regression
    if mt_test:
        logger.info("=== Translation Regression ===")
        report["translation_regression"] = _run_translation_regression(
            service, renderer, sampling_params, mt_test, cfg,
        )

    # 2. Capability smoke tests
    if capability_test:
        logger.info("=== Capability Smoke Tests ===")
        report["capability_smoke"] = _run_capability_smoke(
            stage_b_sampling, renderer, sampling_params,
            capability_test, batch_size,
        )

    # 3. Bilingual comparison
    if capability_test:
        logger.info("=== Bilingual Comparison ===")
        report["bilingual_comparison"] = _run_bilingual_comparison(
            stage_b_sampling, renderer, sampling_params,
            capability_test, batch_size,
        )

    # 4. Preservation metrics (aggregated across all capability predictions)
    if capability_test:
        logger.info("=== Preservation Metrics ===")
        all_preds = _generate_predictions(
            stage_b_sampling, renderer, sampling_params,
            capability_test, batch_size,
        )
        report["preservation"] = compute_preservation_metrics(all_preds)

    # Write report
    write_json(run_dir / "eval_report.json", report)

    manifest = create_manifest(
        stage="stage_b_agent_eval",
        config=cfg,
        extra={
            "run_dir": str(run_dir),
            "mt_test_count": len(mt_test),
            "capability_test_count": len(capability_test),
        },
    )
    save_manifest(manifest, run_dir / "manifest.json")

    # Print summary
    print(f"\nStage B Evaluation Report: {run_dir}")
    if "translation_regression" in report:
        tr = report["translation_regression"]
        sb = tr.get("stage_b", {})
        print(f"  MT (Stage B): chrF++={sb.get('chrf_pp', 'N/A'):.1f}, "
              f"BLEU={sb.get('bleu', 'N/A'):.1f}")
        if "regression_delta" in tr:
            d = tr["regression_delta"]
            print(f"  Regression vs Stage A: chrF++ {d['chrf_pp']:+.1f}, BLEU {d['bleu']:+.1f}")
    if "capability_smoke" in report:
        print("  Capability smoke tests:")
        for family, data in sorted(report["capability_smoke"].items()):
            tm = data.get("translation_metrics", {})
            print(f"    {family}: n={data['count']}, chrF++={tm.get('chrf_pp', 'N/A')}")
    if "preservation" in report:
        p = report["preservation"]
        print(f"  Preservation: JSON={p.get('json_parse_rate', 'N/A')}, "
              f"code={p.get('code_exact_match_rate', 'N/A')}, "
              f"leak={p.get('placeholder_leak_rate', 'N/A')}")

    logger.info("Stage B evaluation complete. Report: %s", run_dir / "eval_report.json")
    return report
