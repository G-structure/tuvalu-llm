#!/usr/bin/env python3
"""Evaluate a Tinker MT model on held-out Tuvaluan↔English examples.

Metrics:
- chrF++ (via sacrebleu CHRF with word_order=2)
- BLEU
- Exact match after whitespace normalization
- Per-direction and per-domain breakdowns

Example:
    uv run python scripts/eval_tinker_mt.py \
        --model-path tinker://.../weights/final \
        --data data/finetune/tinker_mt/test.jsonl \
        --out-dir logs/tinker/evals/test-final
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def add_cookbook_to_path(repo_root: Path) -> None:
    cookbook_root = repo_root / "vendor" / "tinker-cookbook"
    if cookbook_root.exists():
        sys.path.insert(0, str(cookbook_root))


REPO_ROOT = Path(__file__).resolve().parent.parent
add_cookbook_to_path(REPO_ROOT)

import datasets  # type: ignore  # noqa: E402
import sacrebleu  # type: ignore  # noqa: E402
import tinker  # type: ignore  # noqa: E402
from tinker_cookbook import model_info, renderers  # type: ignore  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # type: ignore  # noqa: E402

logger = logging.getLogger("eval_tinker_mt")
logging.getLogger("httpx").setLevel(logging.WARN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "finetune" / "tinker_mt" / "test.jsonl",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/gpt-oss-120b",
        help="Used for tokenizer + renderer selection.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "tinker" / "evals" / "latest",
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def normalize_ws(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def load_split(path: Path) -> datasets.Dataset:
    ds = datasets.load_dataset("json", data_files={"eval": str(path)})
    assert isinstance(ds, datasets.DatasetDict)
    return ds["eval"]


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}

    refs = [[r["reference"] for r in rows]]
    hyps = [r["prediction"] for r in rows]
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    bleu = sacrebleu.metrics.BLEU(effective_order=True)
    exact = sum(
        normalize_ws(r["prediction"]) == normalize_ws(r["reference"])
        for r in rows
    )
    return {
        "count": len(rows),
        "chrf_pp": chrf.corpus_score(hyps, refs).score,
        "bleu": bleu.corpus_score(hyps, refs).score,
        "exact_match": exact / len(rows),
    }


def main() -> None:
    args = parse_args()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY must be set in the environment.")
    if not args.data.exists():
        raise SystemExit(f"Evaluation data not found: {args.data}")
    if not args.model_path and not args.base_model:
        raise SystemExit("Provide either --model-path or --base-model.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.out_dir / "eval.log"),
        ],
    )

    tokenizer = get_tokenizer(args.model_name)
    renderer_name = model_info.get_recommended_renderer_name(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer %s", renderer_name)

    dataset = load_split(args.data)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    logger.info("Loaded %d eval examples", len(dataset))

    service_client = tinker.ServiceClient(base_url=args.base_url)
    if args.model_path:
        sampling_client = service_client.create_sampling_client(model_path=args.model_path)
        logger.info("Sampling from model path: %s", args.model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=args.base_model)
        logger.info("Sampling from base model: %s", args.base_model)

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=renderer.get_stop_sequences(),
    )

    predictions: list[dict[str, Any]] = []
    parse_success_counter = Counter()
    for idx, row in enumerate(dataset):
        messages = row["messages"]
        prompt = renderer.build_generation_prompt(messages[:-1])
        result = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
        output_tokens = result.sequences[0].tokens
        response_message, parse_success = renderer.parse_response(output_tokens)
        parse_success_counter[str(bool(parse_success))] += 1

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

    with (args.out_dir / "predictions.jsonl").open("w") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    overall = compute_metrics(predictions)
    by_direction: dict[str, Any] = {}
    by_domain: dict[str, Any] = {}
    by_content_type: dict[str, Any] = {}

    grouped_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_content: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped_direction[str(row.get("direction", "unknown"))].append(row)
        grouped_domain[str(row.get("domain", "unknown"))].append(row)
        grouped_content[str(row.get("content_type", "unknown"))].append(row)

    for key, rows in grouped_direction.items():
        by_direction[key] = compute_metrics(rows)
    for key, rows in grouped_domain.items():
        by_domain[key] = compute_metrics(rows)
    for key, rows in grouped_content.items():
        by_content_type[key] = compute_metrics(rows)

    summary = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "model_name": args.model_name,
        "count": len(predictions),
        "parse_success": dict(parse_success_counter),
        "overall": overall,
        "by_direction": by_direction,
        "by_domain": by_domain,
        "by_content_type": by_content_type,
    }
    with (args.out_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
