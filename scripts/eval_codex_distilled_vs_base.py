"""Compare a codex-distilled Stage B adapter against the zero-shot base
model on the same held-out test prompts.

Generates assistant responses for every test prompt under two policies:

1. ``base`` — the unmodified base model (e.g. Llama-3.2-1B-Instruct)
   sampled at the same temperature as the SFT training data.
2. ``distilled`` — the same base model + the codex-distilled Stage B
   LoRA adapter, loaded via the tinker SamplingClient.

For each (prompt, base_answer, distilled_answer) triple, score with the
same NativeChatVerifier the harvest used (langid, format compliance,
length, protected-term recall when relevant). Emit a per-prompt
comparison row + a summary table.

Usage::

    uv run python scripts/eval_codex_distilled_vs_base.py \\
        --test data/finetune/stage_b_synthetic_tvl/codex_nc500/test.jsonl \\
        --base meta-llama/Llama-3.2-1B-Instruct \\
        --distilled-uri tinker://run/<run_id>/checkpoint/<step> \\
        --out runs/eval_codex_nc500.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_test_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_prompt(ex: dict) -> str:
    """Pull the user-message text out of a normalized example."""
    for m in ex.get("messages") or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def extract_gold(ex: dict) -> str:
    """Pull the assistant message (i.e., the teacher's answer)."""
    for m in (ex.get("messages") or [])[::-1]:
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


async def sample_completion(
    *,
    client,
    renderer,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Sample one response via the tinker SamplingClient."""
    import tinker

    messages = [{"role": "user", "content": prompt}]
    rendered = renderer.build_supervised_example(
        messages=messages,
        train_on_what="ALL_ASSISTANT_MESSAGES",
    )
    # Slice the prompt portion (everything before the last assistant turn)
    prompt_tokens = rendered["input_tokens"]
    model_input = tinker.ModelInput(
        chunks=[tinker.EncodedTextChunk(tokens=list(prompt_tokens))]
    )
    params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=[renderer.end_of_text_token_id] if hasattr(renderer, "end_of_text_token_id") else [],
    )
    result = await client.sample_async(model_input, 1, params)
    seq = result.sequences[0]
    return renderer.decode(seq.tokens)


def score_response(
    *,
    response: str,
    gold: str,
    verifier_kind: str = "native_chat",
) -> dict:
    """Quick langid + format + length scoring (verifier-equivalent metrics)."""
    from tv.data.codex.verifiers._common import (
        langid_score_en,
        langid_score_tvl,
        chrf_plus_plus,
    )
    import re

    fesili_re = re.compile(r"FESILI\s*:\s*(.+?)(?:\n|$)", re.DOTALL | re.IGNORECASE)
    tali_re = re.compile(r"TALI\s*:\s*(.+?)(?:\Z|\nFESILI)", re.DOTALL | re.IGNORECASE)

    format_q = bool(fesili_re.search(response))
    format_a = bool(tali_re.search(response))
    return {
        "len": len(response),
        "tvl_score": langid_score_tvl(response),
        "en_score": langid_score_en(response),
        "format_fesili": format_q,
        "format_tali": format_a,
        "format_ok": format_q and format_a,
        "chrf_vs_gold": chrf_plus_plus(response, gold) if gold else 0.0,
    }


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument(
        "--base", required=True,
        help="Tinker base model name, e.g. meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--distilled-uri",
        default=None,
        help="Tinker sampling client URI for the SFT'd model (skip to eval only the base)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n", type=int, default=25, help="number of test rows to score")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("eval_codex_distilled")

    from tv.common.tinker_runtime import (
        create_sampling_client,
        create_service_client,
        ensure_cookbook_on_path,
        get_renderer,
        require_tinker_api_key,
    )
    require_tinker_api_key()
    ensure_cookbook_on_path()

    rows = load_test_rows(args.test)[: args.n]
    log.info("loaded %d test rows", len(rows))

    tokenizer, renderer, _ = get_renderer(args.base)
    service = create_service_client()

    base_client = await create_sampling_client(service, args.base, lora_uri=None)
    distilled_client = None
    if args.distilled_uri:
        distilled_client = await create_sampling_client(
            service, args.base, lora_uri=args.distilled_uri,
        )

    out_rows: list[dict] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for i, ex in enumerate(rows):
            prompt = extract_prompt(ex)
            gold = extract_gold(ex)
            try:
                base_resp = await sample_completion(
                    client=base_client, renderer=renderer,
                    prompt=prompt, max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:
                log.warning("base sample failed on %d: %s", i, exc)
                base_resp = ""
            distilled_resp = ""
            if distilled_client is not None:
                try:
                    distilled_resp = await sample_completion(
                        client=distilled_client, renderer=renderer,
                        prompt=prompt, max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                except Exception as exc:
                    log.warning("distilled sample failed on %d: %s", i, exc)

            row = {
                "idx": i,
                "task_id": ex.get("id"),
                "prompt": prompt,
                "gold": gold,
                "base_response": base_resp,
                "distilled_response": distilled_resp,
                "base_metrics": score_response(response=base_resp, gold=gold),
                "distilled_metrics": (
                    score_response(response=distilled_resp, gold=gold)
                    if distilled_resp else None
                ),
            }
            out_rows.append(row)
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            log.info(
                "[%d/%d] base.tvl=%.2f distilled.tvl=%s base.chrf=%.2f distilled.chrf=%s",
                i + 1, len(rows),
                row["base_metrics"]["tvl_score"],
                f"{row['distilled_metrics']['tvl_score']:.2f}" if row["distilled_metrics"] else "n/a",
                row["base_metrics"]["chrf_vs_gold"],
                f"{row['distilled_metrics']['chrf_vs_gold']:.2f}" if row["distilled_metrics"] else "n/a",
            )

    # Summary
    def agg(rows: list[dict], key: str, sub: str) -> float:
        vals = [r[key][sub] for r in rows if r.get(key)]
        return statistics.fmean(vals) if vals else 0.0

    n = len(out_rows)
    base_tvl = agg(out_rows, "base_metrics", "tvl_score")
    base_chrf = agg(out_rows, "base_metrics", "chrf_vs_gold")
    base_fmt = sum(1 for r in out_rows if r["base_metrics"]["format_ok"]) / max(n, 1)
    summary: dict = {
        "n": n,
        "base": {
            "tvl_score_mean": base_tvl,
            "chrf_vs_gold_mean": base_chrf,
            "format_ok_frac": base_fmt,
        },
    }
    if distilled_client is not None:
        distilled_tvl = agg(out_rows, "distilled_metrics", "tvl_score")
        distilled_chrf = agg(out_rows, "distilled_metrics", "chrf_vs_gold")
        distilled_fmt = sum(
            1 for r in out_rows if r["distilled_metrics"] and r["distilled_metrics"]["format_ok"]
        ) / max(n, 1)
        summary["distilled"] = {
            "tvl_score_mean": distilled_tvl,
            "chrf_vs_gold_mean": distilled_chrf,
            "format_ok_frac": distilled_fmt,
        }
        summary["delta"] = {
            "tvl_score": distilled_tvl - base_tvl,
            "chrf_vs_gold": distilled_chrf - base_chrf,
            "format_ok_frac": distilled_fmt - base_fmt,
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
