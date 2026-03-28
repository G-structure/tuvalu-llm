"""Compare translation quality across models on held-out children's book data.

Translates paragraphs from "The Gifts of Pai and Vau" (held-out, never in
any training data) through multiple models and computes chrF++ scores.

Models tested:
- TVL Stage B (our model, served at api.cyberneticphysics.com)
- OpenAI GPT-5.4
- OpenAI GPT-5.4 Nano
- Claude Sonnet 4.6

Usage:
    uv run python scripts/compare_textbook_models.py
    uv run python scripts/compare_textbook_models.py --direction tvl_to_en
    uv run python scripts/compare_textbook_models.py --direction en_to_tvl
"""

import argparse
import json
import os
import time
from pathlib import Path

import httpx
import sacrebleu

# ─── Config ───

DATA_FILE = Path(__file__).resolve().parent.parent / "data/external/stage_a_seed/unstruct_pai_vau.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval/textbook_comparison"

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a careful translator between Tuvaluan and English. "
    "Translate faithfully. Preserve names, numbers, punctuation, and structure. "
    "Output only the translation, nothing else."
)

MODELS = {
    "tvl-stage-b": {
        "type": "cybernetics",
        "endpoint": "https://api.cyberneticphysics.com/tvl-chat/api/chat",
        "label": "TVL Stage B (ours)",
    },
    "gpt-5.4": {
        "type": "openrouter",
        "model_id": "openai/gpt-5.4",
        "label": "GPT-5.4",
    },
    "gpt-5.4-nano": {
        "type": "openrouter",
        "model_id": "openai/gpt-5.4-nano",
        "label": "GPT-5.4 Nano",
    },
    "claude-sonnet": {
        "type": "openrouter",
        "model_id": "anthropic/claude-sonnet-4.6",
        "label": "Claude Sonnet 4.6",
    },
}


def get_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found")
    return key


def load_data() -> list[dict]:
    records = []
    with open(DATA_FILE) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def translate_openrouter(
    client: httpx.Client, api_key: str, model_id: str,
    text: str, direction: str, max_retries: int = 3,
) -> str | None:
    if direction == "en_to_tvl":
        user_msg = f"Translate this English text to Tuvaluan:\n\n{text}"
    else:
        user_msg = f"Translate this Tuvaluan text to English:\n\n{text}"

    for attempt in range(max_retries):
        try:
            resp = client.post(
                OPENROUTER_BASE,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://tuvalugpt.tv",
                    "X-Title": "TVL Textbook Comparison",
                },
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"    API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
                continue
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"    Error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def translate_cybernetics(
    client: httpx.Client, text: str, direction: str, max_retries: int = 3,
) -> str | None:
    if direction == "en_to_tvl":
        user_msg = f"Translate this English text to Tuvaluan:\n\n{text}"
    else:
        user_msg = f"Translate this Tuvaluan text to English:\n\n{text}"

    for attempt in range(max_retries):
        try:
            resp = client.post(
                MODELS["tvl-stage-b"]["endpoint"],
                json={
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            if resp.status_code != 200:
                print(f"    Cybernetics error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
                continue
            data = resp.json()
            return data.get("content", "").strip()
        except Exception as e:
            print(f"    Error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def compute_chrf(predictions: list[str], references: list[str]) -> float:
    chrf = sacrebleu.corpus_chrf(
        predictions, [references], word_order=2,
    )
    return round(chrf.score, 1)


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = sacrebleu.corpus_bleu(
        predictions, [references], use_effective_order=True,
    )
    return round(bleu.score, 1)


def main():
    parser = argparse.ArgumentParser(description="Compare models on Pai & Vau textbook")
    parser.add_argument(
        "--direction", default="en_to_tvl",
        choices=["en_to_tvl", "tvl_to_en"],
        help="Translation direction (default: en_to_tvl)",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        help="Models to evaluate",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    print(f"Loaded {len(data)} paragraphs from 'The Gifts of Pai and Vau'")
    print(f"Direction: {args.direction}")
    print(f"Models: {', '.join(args.models)}")
    print()

    # Determine source and reference
    if args.direction == "en_to_tvl":
        sources = [r["en"] for r in data]
        references = [r["tvl"] for r in data]
    else:
        sources = [r["tvl"] for r in data]
        references = [r["en"] for r in data]

    api_key = get_openrouter_key()
    client = httpx.Client(http2=True)

    all_results = {}

    for model_key in args.models:
        model = MODELS[model_key]
        print(f"{'='*60}")
        print(f"Model: {model['label']}")
        print(f"{'='*60}")

        predictions = []
        for i, source in enumerate(sources):
            preview = source[:60].replace("\n", " ")
            print(f"  [{i+1}/{len(sources)}] {preview}...", end="", flush=True)

            if model["type"] == "openrouter":
                pred = translate_openrouter(
                    client, api_key, model["model_id"],
                    source, args.direction,
                )
            else:
                pred = translate_cybernetics(client, source, args.direction)

            if pred:
                predictions.append(pred)
                print(f" OK ({len(pred)} chars)")
            else:
                predictions.append("")
                print(f" FAILED")

            time.sleep(0.5)  # rate limit courtesy

        # Compute metrics
        chrf = compute_chrf(predictions, references)
        bleu = compute_bleu(predictions, references)

        all_results[model_key] = {
            "label": model["label"],
            "direction": args.direction,
            "chrf": chrf,
            "bleu": bleu,
            "n": len(predictions),
            "predictions": [
                {
                    "id": data[i]["id"],
                    "source": sources[i],
                    "reference": references[i],
                    "prediction": predictions[i],
                }
                for i in range(len(predictions))
            ],
        }

        print(f"\n  chrF++: {chrf}")
        print(f"  BLEU:   {bleu}")
        print()

    # Save results
    out_file = OUTPUT_DIR / f"pai_vau_{args.direction}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY — Pai & Vau ({args.direction})")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'chrF++':>8} {'BLEU':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8}")
    for key in args.models:
        r = all_results[key]
        print(f"{r['label']:<25} {r['chrf']:>8} {r['bleu']:>8}")

    # Print side-by-side for first 3 paragraphs
    print(f"\n{'='*60}")
    print("SAMPLE TRANSLATIONS (first 3 paragraphs)")
    print(f"{'='*60}")
    for i in range(min(3, len(data))):
        print(f"\n--- Paragraph {i+1} ---")
        print(f"SOURCE:    {sources[i][:200]}...")
        print(f"REFERENCE: {references[i][:200]}...")
        for key in args.models:
            pred = all_results[key]["predictions"][i]["prediction"]
            print(f"{all_results[key]['label']:<25}: {pred[:200]}...")
        print()

    client.close()


if __name__ == "__main__":
    main()
