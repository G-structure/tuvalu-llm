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

MAX_OUTPUT_TOKENS = 256
MAX_OUTPUT_TOKENS_THINKING = 2048  # thinking models need more headroom
CHARS_PER_TOKEN = 3.5  # rough estimate

# Preset budgets (selected via --budget flag)
BUDGET_PRESETS = {
    "full": {
        "target_tokens": 500_000,
        "tasks": {
            "translation_en_to_tvl": 250,
            "translation_tvl_to_en": 250,
            "textbook_en_to_tvl": 46,
            "textbook_tvl_to_en": 46,
            "chat_tvl": 250,
            "qa_tvl": 120,
            "summarization_tvl": 40,
        },
    },
    "tiny": {
        "target_tokens": 10_000,
        "tasks": {
            "translation_en_to_tvl": 5,
            "translation_tvl_to_en": 5,
            "textbook_en_to_tvl": 5,
            "textbook_tvl_to_en": 5,
            "chat_tvl": 5,
            "qa_tvl": 3,
            "summarization_tvl": 2,
        },
    },
}

TARGET_TOKENS = BUDGET_PRESETS["full"]["target_tokens"]
TASK_BUDGET = BUDGET_PRESETS["full"]["tasks"]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

OPENROUTER_MODELS = {
    "gpt-5.4": "openai/gpt-5.4",
    "qwen3-30b": "qwen/qwen3-30b-a3b-thinking-2507",
    "claude-sonnet": "anthropic/claude-sonnet-4.6",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
}

# Models that use thinking/reasoning tokens — need higher max_tokens
THINKING_MODELS = {"qwen3-30b"}

# Tinker-direct models (called via SDK, not via VPS backend)
TINKER_MODELS = {
    "tvl-stage-a": {
        "model_name": "Qwen/Qwen3-30B-A3B",
        "sampler_path": "tinker://391bb502-f0c1-509e-bcd5-68bc35c82db2:train:0/sampler_weights/final",
        "label": "Stage A (step 7851)",
    },
}

ALL_MODEL_KEYS = ["tvl"] + list(TINKER_MODELS.keys()) + list(OPENROUTER_MODELS.keys()) + ["google-translate"]


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


def load_textbook_data(direction: str, n: int) -> list[dict]:
    """Load children's textbook pairs (held-out, never in training) and format as eval examples."""
    seed_dir = REPO_ROOT / "data" / "external" / "stage_a_seed"
    textbook_files = [
        "unstruct_pai_vau.jsonl",
        "unstruct_toku_atufenua.jsonl",
        "unstruct_tepapa_activity.jsonl",
    ]
    sys_prompt = (
        "You are a careful translator between Tuvaluan and English. "
        "Translate faithfully. Preserve names, numbers, punctuation, line breaks, "
        "and structure when possible. Output only the translation."
    )
    all_examples = []
    for fname in textbook_files:
        path = seed_dir / fname
        if not path.exists():
            continue
        rows = load_jsonl(path)
        for row in rows:
            tvl_text = row.get("tvl", "")
            en_text = row.get("en", "")
            if not tvl_text or not en_text:
                continue
            domain = row.get("domain", "textbook")
            row_id = row.get("id", fname.replace(".jsonl", ""))
            if direction == "en_to_tvl":
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Translate the following English text into Tuvaluan.\n\n{en_text}"},
                    {"role": "assistant", "content": tvl_text},
                ]
            else:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Translate the following Tuvaluan text into English.\n\n{tvl_text}"},
                    {"role": "assistant", "content": en_text},
                ]
            all_examples.append({
                "id": f"{row_id}::{direction}",
                "messages": messages,
                "metadata": {
                    "direction": direction,
                    "domain": domain,
                    "tvl": tvl_text,
                    "en": en_text,
                },
            })
    random.shuffle(all_examples)
    return all_examples[:n]


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
    # Children's textbook data (held-out, never in training)
    if "textbook_en_to_tvl" in TASK_BUDGET:
        slices["textbook_en_to_tvl"] = load_textbook_data(
            "en_to_tvl", TASK_BUDGET["textbook_en_to_tvl"]
        )
    if "textbook_tvl_to_en" in TASK_BUDGET:
        slices["textbook_tvl_to_en"] = load_textbook_data(
            "tvl_to_en", TASK_BUDGET["textbook_tvl_to_en"]
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
            msg = body["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or ""
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


# ---------------------------------------------------------------------------
# Tinker direct client (lazy singleton)
# ---------------------------------------------------------------------------

_tinker_clients: dict[str, Any] = {}


def _get_tinker_client(model_key: str) -> tuple[Any, Any, Any]:
    """Lazily init and cache a Tinker sampling client + renderer."""
    if model_key in _tinker_clients:
        return _tinker_clients[model_key]

    cfg = TINKER_MODELS[model_key]
    sys.path.insert(0, str(REPO_ROOT / "vendor" / "tinker-cookbook"))
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    model_name = cfg["model_name"]
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(model_path=cfg["sampler_path"])
    log.info("Tinker client ready: %s → %s", model_key, cfg["sampler_path"])

    result = (renderer, sampling_client, tinker)
    _tinker_clients[model_key] = result
    return result


def call_tinker_direct(
    model_key: str,
    messages: list[dict],
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Call a Tinker model directly via SDK."""
    renderer, sampling_client, tinker = _get_tinker_client(model_key)

    params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    prompt = renderer.build_generation_prompt(messages)
    future = sampling_client.sample(prompt, sampling_params=params, num_samples=1)
    result = future.result()
    output_tokens = result.sequences[0].tokens
    response_message, _ok = renderer.parse_response(output_tokens)
    if isinstance(response_message, dict):
        return str(response_message.get("content", ""))
    return str(response_message)


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
    is_translation = task_name.startswith("translation_") or task_name.startswith("textbook_")
    out_tokens = MAX_OUTPUT_TOKENS_THINKING if model_key in THINKING_MODELS else MAX_OUTPUT_TOKENS

    def process_one(example: dict) -> dict | None:
        prompt, reference = extract_source_and_ref(example)

        try:
            if model_key == "tvl":
                prediction = call_tvl_model(prompt, backend_url=tvl_backend_url)
            elif model_key in TINKER_MODELS:
                prediction = call_tinker_direct(model_key, prompt)
            elif model_key == "google-translate":
                if not is_translation:
                    return None  # Google Translate only for translation tasks
                source_text, src_lang, tgt_lang = extract_translation_text(example)
                prediction = call_google_translate(source_text, src_lang, tgt_lang, google_key)
            elif model_key in OPENROUTER_MODELS:
                prediction = call_openrouter(
                    OPENROUTER_MODELS[model_key], prompt, openrouter_key,
                    max_tokens=out_tokens,
                )
            else:
                return None
        except Exception as e:
            log.warning("  %s failed on %s: %s", model_key, example.get("id", "?"), e)
            return None

        meta = example.get("metadata", {})
        return {
            "id": example.get("id", ""),
            "prediction": prediction or "",
            "reference": reference,
            "direction": meta.get("direction", ""),
            "domain": meta.get("domain", ""),
            "task_family": example.get("task_family", "translation"),
            "stage_b_source": meta.get("stage_b_source", ""),
        }

    # Tinker SDK clients use a lock internally; limit parallelism for direct models
    effective_parallel = 1 if model_key in TINKER_MODELS else parallel
    with ThreadPoolExecutor(max_workers=effective_parallel) as pool:
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


def compute_slice_metrics(predictions: list[dict], task_name: str = "") -> dict[str, Any]:
    """Compute metrics for a slice of predictions."""
    if not predictions:
        return {"count": 0}
    overall = compute_translation_metrics(predictions)

    is_translation = task_name.startswith("translation_") or task_name.startswith("textbook_")

    # Per-domain breakdown for translation tasks
    if is_translation:
        domains = set(p.get("domain", "") for p in predictions)
        if domains - {""}:
            overall["by_domain"] = compute_grouped_metrics(predictions, "domain")

    # Per-source breakdown for non-translation tasks
    if not is_translation:
        sources = set(p.get("stage_b_source", "") for p in predictions)
        if sources - {""}:
            overall["by_source"] = compute_grouped_metrics(predictions, "stage_b_source")

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
# D1 upload
# ---------------------------------------------------------------------------

ACCOUNT_ID = "8f86f0b518afefff58d515fe2a253b33"
DATABASE_ID = "7087ac6b-6417-48a4-9c7f-1d108057cd51"
D1_API_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}"
    f"/d1/database/{DATABASE_ID}/query"
)


def d1_exec(sql: str, params: list, token: str) -> None:
    """Execute a single SQL statement against D1."""
    import urllib.request

    payload = json.dumps({"sql": sql, "params": params}).encode()
    req = urllib.request.Request(
        D1_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    if not body.get("success"):
        raise RuntimeError(f"D1 error: {body.get('errors')}")


def upload_to_d1(
    run_id: str,
    budget: str,
    results: dict[str, dict],
    predictions: dict[str, dict[str, list[dict]]],
    token: str,
) -> None:
    """Upload benchmark results and predictions to D1."""
    log.info("Uploading results to D1 (run_id=%s)...", run_id)

    # Upload per-model results
    for model_key, model_results in results.items():
        d1_exec(
            "INSERT INTO eval_runs (run_id, model_key, budget, results_json) VALUES (?, ?, ?, ?)",
            [run_id, model_key, budget, json.dumps(model_results)],
            token,
        )

    # Upload predictions in chunks
    for model_key, model_preds in predictions.items():
        for task, preds in model_preds.items():
            CHUNK = 10
            for i in range(0, len(preds), CHUNK):
                chunk = preds[i : i + CHUNK]
                placeholders = ", ".join(["(?, ?, ?, ?, ?, ?, ?)"] * len(chunk))
                params = []
                for p in chunk:
                    meta = {
                        k: p.get(k)
                        for k in ("direction", "domain", "task_family", "stage_b_source")
                        if p.get(k)
                    }
                    params.extend([
                        run_id,
                        model_key,
                        task,
                        p.get("id", ""),
                        p.get("prediction", ""),
                        p.get("reference", ""),
                        json.dumps(meta),
                    ])
                d1_exec(
                    f"INSERT INTO eval_predictions (run_id, model_key, task, example_id, prediction, reference, metadata_json) VALUES {placeholders}",
                    params,
                    token,
                )

    log.info("Uploaded to D1: %d models, run %s", len(results), run_id)


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
    parser.add_argument(
        "--budget",
        type=str,
        default="full",
        choices=list(BUDGET_PRESETS.keys()),
        help="Token budget preset: full (~500K) or tiny (~10K)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show token budget, no API calls")
    parser.add_argument("--upload", action="store_true", help="Upload results to D1 database")
    parser.add_argument("--cf-token", type=str, default=os.environ.get("CLOUDFLARE_API_TOKEN"))
    parser.add_argument("--parallel", type=int, default=8, help="Concurrent requests per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "eval" / "benchmark"),
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Apply budget preset
    global TARGET_TOKENS, TASK_BUDGET
    preset = BUDGET_PRESETS[args.budget]
    TARGET_TOKENS = preset["target_tokens"]
    TASK_BUDGET = preset["tasks"]

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
            if model_key == "google-translate" and not (task_name.startswith("translation_") or task_name.startswith("textbook_")):
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

            metrics = compute_slice_metrics(preds, task_name)
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

    # Upload to D1
    if args.upload:
        if not args.cf_token:
            log.error("Set CLOUDFLARE_API_TOKEN or --cf-token for D1 upload")
        else:
            from datetime import datetime

            run_id = f"{datetime.utcnow().strftime('%Y-%m-%d_%H%M')}_{args.budget}"
            upload_to_d1(run_id, args.budget, all_results, all_predictions, args.cf_token)


if __name__ == "__main__":
    main()
