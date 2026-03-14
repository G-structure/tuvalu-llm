#!/usr/bin/env python3
"""Benchmark TVL model against leading models and Google Translate.

Evaluates on a stratified ~0.5M token budget across:
  - Translation (EN↔TVL): chrF++, BLEU, exact match
  - Chat / instruction following (TVL): chrF++ vs reference
  - QA (TVL): chrF++ vs reference
  - Summarization (TVL): chrF++ vs reference

Models compared via OpenRouter API:
  - TVL fine-tune (our model, via Tinker or local backend)
  - GPT-4o, Claude Sonnet, Gemini 2.5 Flash, Llama 4 Scout (via OpenRouter)
  - Google Translate (via Cloud Translation API, translation tasks only)

Usage:
    # Full benchmark
    uv run python scripts/benchmark_eval.py --openrouter-key sk-... --google-key AIza...

    # Our model only (no API costs)
    uv run python scripts/benchmark_eval.py --our-model-only

    # Specific models
    uv run python scripts/benchmark_eval.py --models tvl,gpt-4o,google-translate

    # Dry run (show token budget, no API calls)
    uv run python scripts/benchmark_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))
from training.common.metrics import (
    compute_grouped_metrics,
    compute_translation_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token budget & sampling
# ---------------------------------------------------------------------------

TARGET_TOKENS = 500_000  # 0.5M total token budget per model
MAX_OUTPUT_TOKENS = 256
CHARS_PER_TOKEN = 3.5  # rough estimate

# How many examples per task slice (balanced across tasks)
TASK_BUDGET = {
    "translation_en_to_tvl": 250,
    "translation_tvl_to_en": 250,
    "chat_tvl": 300,
    "qa_tvl": 150,
    "summarization_tvl": 50,
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

OPENROUTER_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "gemini-2.5-flash": "google/gemini-2.5-flash-preview",
    "llama-4-scout": "meta-llama/llama-4-scout",
}

ALL_MODEL_KEYS = ["tvl"] + list(OPENROUTER_MODELS.keys()) + ["google-translate"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def sample_translation_data(direction: str, n: int) -> list[dict]:
    """Sample n translation examples for a direction from Stage A test set."""
    path = REPO_ROOT / "data" / "finetune" / "stage_a_mt" / "test.jsonl"
    all_rows = load_jsonl(path)
    filtered = [r for r in all_rows if r.get("metadata", {}).get("direction") == direction]
    random.shuffle(filtered)
    return filtered[:n]


def sample_task_data(task_family: str, source: str, n: int) -> list[dict]:
    """Sample n examples from Stage B test set for a task family + source."""
    path = REPO_ROOT / "data" / "finetune" / "stage_b_mix" / "test.jsonl"
    all_rows = load_jsonl(path)
    filtered = [
        r
        for r in all_rows
        if r.get("task_family") == task_family
        and r.get("metadata", {}).get("stage_b_source") == source
    ]
    random.shuffle(filtered)
    return filtered[:n]


def build_eval_set() -> dict[str, list[dict]]:
    """Build the stratified evaluation set within token budget."""
    log.info("Building evaluation set...")
    slices: dict[str, list[dict]] = {}

    slices["translation_en_to_tvl"] = sample_translation_data(
        "en_to_tvl", TASK_BUDGET["translation_en_to_tvl"]
    )
    slices["translation_tvl_to_en"] = sample_translation_data(
        "tvl_to_en", TASK_BUDGET["translation_tvl_to_en"]
    )
    # TVL tasks: prefer synthetic_tvl and crosslingual (actual Tuvaluan content)
    slices["chat_tvl"] = sample_task_data("chat", "synthetic_tvl", TASK_BUDGET["chat_tvl"] // 2) + sample_task_data(
        "chat", "crosslingual", TASK_BUDGET["chat_tvl"] // 2
    )
    slices["qa_tvl"] = sample_task_data("qa", "synthetic_tvl", TASK_BUDGET["qa_tvl"] // 2) + sample_task_data(
        "qa", "crosslingual", TASK_BUDGET["qa_tvl"] // 2
    )
    slices["summarization_tvl"] = sample_task_data(
        "summarization", "synthetic_tvl", TASK_BUDGET["summarization_tvl"]
    )

    total_examples = sum(len(v) for v in slices.values())
    total_chars = sum(
        sum(len(m["content"]) for m in row["messages"]) for rows in slices.values() for row in rows
    )
    est_tokens = int(total_chars / CHARS_PER_TOKEN) + total_examples * MAX_OUTPUT_TOKENS

    for name, rows in slices.items():
        log.info("  %s: %d examples", name, len(rows))
    log.info("Total: %d examples, ~%dK tokens/model", total_examples, est_tokens // 1000)

    return slices


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


def call_openrouter(
    model_id: str,
    messages: list[dict],
    api_key: str,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Call OpenRouter chat completion API."""
    import urllib.error
    import urllib.request

    payload = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tvl-chat.pages.dev",
            "X-Title": "TVL Benchmark",
        },
        method="POST",
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            if e.code == 429:
                wait = 2 ** (attempt + 1)
                log.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            log.error("OpenRouter HTTP %d: %s", e.code, error_body[:200])
            raise
        except urllib.error.URLError as e:
            log.error("OpenRouter network error: %s", e.reason)
            raise
    raise RuntimeError("OpenRouter: max retries exceeded")


def call_tvl_model(
    messages: list[dict],
    backend_url: str = "https://api.cyberneticphysics.com/tvl-chat",
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Call our TVL model via the VPS backend."""
    import urllib.request

    payload = json.dumps(
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()

    req = urllib.request.Request(
        f"{backend_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    return body.get("content", "")


def call_google_translate(
    text: str,
    source_lang: str,
    target_lang: str,
    api_key: str,
) -> str:
    """Call Google Cloud Translation API v2."""
    import urllib.parse
    import urllib.request

    # Map our lang codes to Google's
    lang_map = {"tvl": "tvl", "en": "en"}
    src = lang_map.get(source_lang, source_lang)
    tgt = lang_map.get(target_lang, target_lang)

    params = urllib.parse.urlencode(
        {
            "q": text,
            "source": src,
            "target": tgt,
            "key": api_key,
            "format": "text",
        }
    )

    req = urllib.request.Request(
        f"https://translation.googleapis.com/language/translate/v2?{params}",
        method="POST",
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
            translations = body.get("data", {}).get("translations", [])
            if translations:
                return translations[0]["translatedText"]
            return ""
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            if e.code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            # Google may not support tvl — return empty
            if e.code == 400 and "unsupported" in error_body.lower():
                return "[unsupported_language]"
            log.error("Google Translate HTTP %d: %s", e.code, error_body[:200])
            raise
    raise RuntimeError("Google Translate: max retries exceeded")


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def extract_source_and_ref(example: dict) -> tuple[list[dict], str]:
    """Extract prompt messages and reference from an example.

    Returns (messages_without_final_assistant, reference_text).
    """
    messages = example["messages"]
    reference = messages[-1]["content"]
    prompt = messages[:-1]  # everything except the reference
    return prompt, reference


def extract_translation_text(example: dict) -> tuple[str, str, str]:
    """Extract source text, source_lang, target_lang from a translation example."""
    meta = example.get("metadata", {})
    direction = meta.get("direction", "")
    if direction == "en_to_tvl":
        return meta.get("en", ""), "en", "tvl"
    else:
        return meta.get("tvl", ""), "tvl", "en"


def run_model_on_slice(
    model_key: str,
    task_name: str,
    examples: list[dict],
    *,
    openrouter_key: str | None = None,
    google_key: str | None = None,
    tvl_backend_url: str = "https://api.cyberneticphysics.com/tvl-chat",
    parallel: int = 8,
) -> list[dict]:
    """Run a model on a slice of examples, return prediction records."""
    predictions = []
    is_translation = task_name.startswith("translation_")

    def process_one(example: dict) -> dict | None:
        prompt, reference = extract_source_and_ref(example)

        try:
            if model_key == "tvl":
                prediction = call_tvl_model(prompt, backend_url=tvl_backend_url)
            elif model_key == "google-translate":
                if not is_translation:
                    return None  # Google Translate only for translation tasks
                source_text, src_lang, tgt_lang = extract_translation_text(example)
                prediction = call_google_translate(source_text, src_lang, tgt_lang, google_key)
            elif model_key in OPENROUTER_MODELS:
                prediction = call_openrouter(
                    OPENROUTER_MODELS[model_key], prompt, openrouter_key
                )
            else:
                return None
        except Exception as e:
            log.warning("  %s failed on %s: %s", model_key, example.get("id", "?"), e)
            return None

        meta = example.get("metadata", {})
        return {
            "id": example.get("id", ""),
            "prediction": prediction,
            "reference": reference,
            "direction": meta.get("direction", ""),
            "domain": meta.get("domain", ""),
            "task_family": example.get("task_family", "translation"),
            "stage_b_source": meta.get("stage_b_source", ""),
        }

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(process_one, ex): i for i, ex in enumerate(examples)}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                predictions.append(result)
            done += 1
            if done % 50 == 0:
                log.info("  %s / %s: %d/%d", model_key, task_name, done, len(examples))

    return predictions


# ---------------------------------------------------------------------------
# Results & reporting
# ---------------------------------------------------------------------------


def compute_slice_metrics(predictions: list[dict]) -> dict[str, Any]:
    """Compute metrics for a slice of predictions."""
    if not predictions:
        return {"count": 0}
    overall = compute_translation_metrics(predictions)

    # If translation, also compute per-direction
    directions = set(p.get("direction", "") for p in predictions)
    if directions - {""}:
        by_direction = compute_grouped_metrics(predictions, "direction")
        overall["by_direction"] = by_direction

    return overall


def print_results_table(results: dict[str, dict[str, dict]]) -> None:
    """Print a formatted comparison table."""
    models = list(results.keys())
    tasks = list(next(iter(results.values())).keys())

    # Header
    print("\n" + "=" * 100)
    print("TVL BENCHMARK RESULTS")
    print("=" * 100)

    for task in tasks:
        print(f"\n{'─' * 80}")
        print(f"  {task}")
        print(f"{'─' * 80}")
        print(f"  {'Model':<25} {'chrF++':>8} {'BLEU':>8} {'Exact%':>8} {'Count':>8}")
        print(f"  {'─' * 61}")

        # Sort by chrF++ descending
        model_scores = []
        for model in models:
            m = results[model].get(task, {})
            if m.get("count", 0) > 0:
                model_scores.append((model, m))

        model_scores.sort(key=lambda x: x[1].get("chrf_pp", 0), reverse=True)

        for model, m in model_scores:
            chrf = f"{m.get('chrf_pp', 0):.1f}" if m.get("chrf_pp") is not None else "—"
            bleu = f"{m.get('bleu', 0):.1f}" if m.get("bleu") is not None else "—"
            exact = f"{m.get('exact_match', 0) * 100:.1f}" if m.get("exact_match") is not None else "—"
            count = str(m.get("count", 0))
            print(f"  {model:<25} {chrf:>8} {bleu:>8} {exact:>8} {count:>8}")

    # Summary table: average chrF++ across all tasks
    print(f"\n{'=' * 80}")
    print("  OVERALL (mean chrF++ across tasks)")
    print(f"{'─' * 80}")
    model_avgs = []
    for model in models:
        scores = [
            results[model][t].get("chrf_pp", 0)
            for t in tasks
            if results[model].get(t, {}).get("count", 0) > 0
        ]
        if scores:
            avg = sum(scores) / len(scores)
            model_avgs.append((model, avg, len(scores)))
    model_avgs.sort(key=lambda x: x[1], reverse=True)
    for model, avg, n_tasks in model_avgs:
        print(f"  {model:<25} {avg:>8.1f}  ({n_tasks} tasks)")
    print("=" * 80)


def save_results(results: dict, predictions: dict, output_dir: Path) -> None:
    """Save results and predictions to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for model, model_preds in predictions.items():
        with (output_dir / f"predictions_{model}.jsonl").open("w") as f:
            for task, preds in model_preds.items():
                for p in preds:
                    p["_task"] = task
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

    log.info("Results saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="TVL Benchmark Evaluation")
    parser.add_argument("--openrouter-key", type=str, default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--google-key", type=str, default=os.environ.get("GOOGLE_TRANSLATE_KEY"))
    parser.add_argument(
        "--tvl-backend",
        type=str,
        default="https://api.cyberneticphysics.com/tvl-chat",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated model keys. Available: {','.join(ALL_MODEL_KEYS)}",
    )
    parser.add_argument("--our-model-only", action="store_true", help="Only evaluate TVL model")
    parser.add_argument("--dry-run", action="store_true", help="Show token budget, no API calls")
    parser.add_argument("--parallel", type=int, default=8, help="Concurrent requests per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "eval" / "benchmark"),
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Determine which models to run
    if args.our_model_only:
        model_keys = ["tvl"]
    elif args.models:
        model_keys = [m.strip() for m in args.models.split(",")]
    else:
        model_keys = list(ALL_MODEL_KEYS)

    # Validate keys
    for key in model_keys:
        if key not in ALL_MODEL_KEYS:
            sys.exit(f"Unknown model key: {key}. Available: {', '.join(ALL_MODEL_KEYS)}")
    if any(k in OPENROUTER_MODELS for k in model_keys) and not args.openrouter_key:
        sys.exit("Set OPENROUTER_API_KEY or pass --openrouter-key for OpenRouter models")
    if "google-translate" in model_keys and not args.google_key:
        sys.exit("Set GOOGLE_TRANSLATE_KEY or pass --google-key for Google Translate")

    # Build eval set
    eval_set = build_eval_set()

    if args.dry_run:
        total = sum(len(v) for v in eval_set.values())
        print(f"\nDry run: {total} examples across {len(eval_set)} tasks")
        print(f"Models: {', '.join(model_keys)}")
        print(f"Estimated tokens/model: ~{TARGET_TOKENS // 1000}K")
        print(f"Total API calls: ~{total * len(model_keys)}")
        return

    # Run evaluation
    all_results: dict[str, dict[str, dict]] = {}
    all_predictions: dict[str, dict[str, list[dict]]] = {}

    for model_key in model_keys:
        log.info("=" * 60)
        log.info("Evaluating: %s", model_key)
        log.info("=" * 60)

        model_results: dict[str, dict] = {}
        model_preds: dict[str, list[dict]] = {}

        for task_name, examples in eval_set.items():
            if not examples:
                log.warning("  Skipping %s: no examples", task_name)
                continue

            # Google Translate only handles translation tasks
            if model_key == "google-translate" and not task_name.startswith("translation_"):
                continue

            log.info("  Running %s on %s (%d examples)...", model_key, task_name, len(examples))
            t0 = time.time()

            preds = run_model_on_slice(
                model_key,
                task_name,
                examples,
                openrouter_key=args.openrouter_key,
                google_key=args.google_key,
                tvl_backend_url=args.tvl_backend,
                parallel=args.parallel,
            )

            metrics = compute_slice_metrics(preds)
            elapsed = time.time() - t0

            model_results[task_name] = metrics
            model_preds[task_name] = preds

            chrf = metrics.get("chrf_pp", 0)
            bleu = metrics.get("bleu", 0)
            log.info(
                "  %s / %s: chrF++=%.1f BLEU=%.1f (%d/%d ok, %.0fs)",
                model_key,
                task_name,
                chrf,
                bleu,
                len(preds),
                len(examples),
                elapsed,
            )

        all_results[model_key] = model_results
        all_predictions[model_key] = model_preds

    # Print and save
    print_results_table(all_results)
    save_results(all_results, all_predictions, Path(args.output_dir))


if __name__ == "__main__":
    main()
