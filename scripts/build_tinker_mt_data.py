#!/usr/bin/env python3
"""Build Tinker-ready Tuvaluan↔English MT chat datasets from aligned JSONL.

This script turns the current `data/aligned/*.jsonl` files into conversation-style
JSONL files suitable for Tinker supervised fine-tuning.

Outputs:
    data/finetune/tinker_mt/
        train_full.jsonl
        train_balanced.jsonl
        validation.jsonl
        test.jsonl
        rejected.jsonl
        stats.json

Design choices:
- Keeps the aligned JSONL files as the canonical source of truth.
- Generates BOTH directions from each accepted pair.
- Uses deterministic splits to avoid leakage:
  * Bible: split by held-out books, not random verses.
  * Articles: split by `doc_id` when present.
  * Daily text: split by `date`.
  * Everything else: split by deterministic hash of a group key.
- Drops lower-confidence alignment rows by default for the first MT adapter.
- Emits a balanced train file that caps Bible share so the adapter is not almost
  entirely scripture-shaped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALIGNED_DIR = DATA_DIR / "aligned"
OUT_DIR = DATA_DIR / "finetune" / "tinker_mt"

SYSTEM_PROMPT = (
    "You are a careful translator between Tuvaluan and English. "
    "Translate faithfully. Preserve names, numbers, punctuation, line breaks, "
    "and structure when possible. Output only the translation."
)

TVL_TO_EN_TEMPLATES = [
    "Translate from Tuvaluan to English:\n\n{source}",
    "Translate the following Tuvaluan text into English. Preserve formatting and do not add commentary.\n\n{source}",
    "Convert this Tuvaluan text to natural English while keeping the original structure when possible.\n\n{source}",
]

EN_TO_TVL_TEMPLATES = [
    "Translate from English to Tuvaluan:\n\n{source}",
    "Translate the following English text into Tuvaluan. Preserve formatting and do not add commentary.\n\n{source}",
    "Convert this English text to Tuvaluan while keeping the original structure when possible.\n\n{source}",
]

TEST_BOOKS = {8, 57, 65}   # Ruth, Philemon, Jude
VALID_BOOKS = {31, 63, 64} # Obadiah, 2 John, 3 John


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ALIGNED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=4096)
    parser.add_argument("--ratio-min", type=float, default=0.4)
    parser.add_argument("--ratio-max", type=float, default=2.5)
    parser.add_argument(
        "--allow-low-confidence-articles",
        action="store_true",
        help="Keep document-level article fallbacks with confidence below the main threshold.",
    )
    parser.add_argument(
        "--bible-max-train-share",
        type=float,
        default=0.70,
        help="Cap the share of bible_verse examples in train_balanced.jsonl.",
    )
    parser.add_argument(
        "--non-bible-val-frac",
        type=float,
        default=0.05,
        help="Validation fraction for non-Bible group-hash splits.",
    )
    parser.add_argument(
        "--non-bible-test-frac",
        type=float,
        default=0.05,
        help="Test fraction for non-Bible group-hash splits.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def normalize_for_hash(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_preserve_structure(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def row_quality_reasons(
    row: dict[str, Any],
    *,
    min_confidence: float,
    min_chars: int,
    max_chars: int,
    ratio_min: float,
    ratio_max: float,
    allow_low_conf_article: bool,
) -> list[str]:
    reasons: list[str] = []

    tvl = normalize_preserve_structure(str(row.get("tvl", "")))
    en = normalize_preserve_structure(str(row.get("en", "")))
    if not tvl or not en:
        reasons.append("empty_text")

    tvl_chars = int(row.get("tvl_chars") or len(tvl))
    en_chars = int(row.get("en_chars") or len(en))
    if tvl_chars < min_chars or en_chars < min_chars:
        reasons.append("too_short")
    if tvl_chars > max_chars or en_chars > max_chars:
        reasons.append("too_long")

    ratio = row.get("length_ratio")
    if ratio is None or ratio == 0:
        if en_chars > 0:
            ratio = tvl_chars / en_chars
    if ratio and (ratio < ratio_min or ratio > ratio_max):
        reasons.append("bad_length_ratio")

    confidence = float(row.get("alignment_confidence") or 0.0)
    if confidence < min_confidence:
        is_low_conf_article = (
            row.get("content_type") == "article_paragraph"
            and row.get("alignment_method") == "document_level"
            and allow_low_conf_article
        )
        if not is_low_conf_article:
            reasons.append("low_alignment_confidence")

    return reasons


def group_key(row: dict[str, Any]) -> str:
    content_type = row.get("content_type")
    if content_type == "bible_verse":
        return f"bible_book_{row.get('book_num')}"
    if row.get("doc_id"):
        return f"doc_{row['doc_id']}"
    if row.get("date"):
        return f"date_{row['date']}"
    if row.get("pub_code"):
        return f"pub_{row['pub_code']}"
    return f"row_{row.get('id', '')}"


def assign_split(
    row: dict[str, Any],
    *,
    non_bible_val_frac: float,
    non_bible_test_frac: float,
) -> str:
    if row.get("content_type") == "bible_verse":
        book_num = int(row.get("book_num") or 0)
        if book_num in TEST_BOOKS:
            return "test"
        if book_num in VALID_BOOKS:
            return "validation"
        return "train"

    key = group_key(row)
    bucket = stable_hash(key) % 10000
    test_cut = int(non_bible_test_frac * 10000)
    val_cut = test_cut + int(non_bible_val_frac * 10000)
    if bucket < test_cut:
        return "test"
    if bucket < val_cut:
        return "validation"
    return "train"


def choose_template(row_id: str, direction: str) -> tuple[str, int]:
    if direction == "tvl_to_en":
        templates = TVL_TO_EN_TEMPLATES
    else:
        templates = EN_TO_TVL_TEMPLATES
    idx = stable_hash(f"{row_id}::{direction}") % len(templates)
    return templates[idx], idx


def build_example(row: dict[str, Any], direction: str) -> dict[str, Any]:
    if direction == "tvl_to_en":
        src = normalize_preserve_structure(str(row["tvl"]))
        tgt = normalize_preserve_structure(str(row["en"]))
        src_lang = "tvl"
        tgt_lang = "en"
    elif direction == "en_to_tvl":
        src = normalize_preserve_structure(str(row["en"]))
        tgt = normalize_preserve_structure(str(row["tvl"]))
        src_lang = "en"
        tgt_lang = "tvl"
    else:
        raise ValueError(direction)

    template, template_idx = choose_template(str(row["id"]), direction)
    metadata = dict(row)
    metadata.update(
        {
            "direction": direction,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "template_idx": template_idx,
        }
    )
    return {
        "id": f"{row['id']}::{direction}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": template.format(source=src)},
            {"role": "assistant", "content": tgt},
        ],
        "metadata": metadata,
    }


def downsample_bible_examples(
    train_examples: list[dict[str, Any]],
    *,
    bible_max_share: float,
) -> list[dict[str, Any]]:
    if not 0 < bible_max_share < 1:
        return train_examples

    bible = [x for x in train_examples if x["metadata"].get("content_type") == "bible_verse"]
    non_bible = [x for x in train_examples if x["metadata"].get("content_type") != "bible_verse"]
    if not bible or not non_bible:
        return train_examples

    max_bible = math.floor((bible_max_share / (1.0 - bible_max_share)) * len(non_bible))
    if len(bible) <= max_bible:
        return train_examples

    bible_sorted = sorted(bible, key=lambda x: stable_hash(x["id"]))
    kept_bible = bible_sorted[:max_bible]
    combined = non_bible + kept_bible
    return sorted(combined, key=lambda x: stable_hash(x["id"]))


def summarize_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_direction = Counter(x["metadata"]["direction"] for x in rows)
    by_domain = Counter(x["metadata"].get("domain", "unknown") for x in rows)
    by_content_type = Counter(x["metadata"].get("content_type", "unknown") for x in rows)
    tokenish_lengths = [
        len(normalize_for_hash(x["messages"][-1]["content"])) / 4.0
        for x in rows
    ]
    return {
        "examples": len(rows),
        "by_direction": dict(by_direction),
        "by_domain": dict(by_domain),
        "by_content_type": dict(by_content_type),
        "target_chars_mean": round(
            statistics.mean(len(x["messages"][-1]["content"]) for x in rows), 1
        ) if rows else 0,
        "target_tokens_est_mean": round(statistics.mean(tokenish_lengths), 1) if rows else 0,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    del rng  # deterministic hashing is used instead of pseudo-random assignment.

    aligned_paths = sorted(args.input_dir.glob("*.jsonl"))
    if not aligned_paths:
        raise SystemExit(f"No JSONL files found in {args.input_dir}")

    raw_rows: list[dict[str, Any]] = []
    for path in aligned_paths:
        raw_rows.extend(read_jsonl(path))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_pair_hashes: set[str] = set()

    for row in raw_rows:
        row = dict(row)
        row["tvl"] = normalize_preserve_structure(str(row.get("tvl", "")))
        row["en"] = normalize_preserve_structure(str(row.get("en", "")))

        reasons = row_quality_reasons(
            row,
            min_confidence=args.min_confidence,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
            allow_low_conf_article=args.allow_low_confidence_articles,
        )

        dedup_hash = hashlib.sha256(
            (
                normalize_for_hash(row["tvl"]) + "|||" + normalize_for_hash(row["en"])
            ).encode("utf-8")
        ).hexdigest()
        if dedup_hash in seen_pair_hashes:
            reasons.append("duplicate_pair")
        else:
            seen_pair_hashes.add(dedup_hash)

        if reasons:
            rejected.append({"row": row, "reasons": reasons})
            continue
        accepted.append(row)

    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        split = assign_split(
            row,
            non_bible_val_frac=args.non_bible_val_frac,
            non_bible_test_frac=args.non_bible_test_frac,
        )
        splits[split].append(row)

    rendered: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, rows in splits.items():
        for row in rows:
            rendered[split].append(build_example(row, "tvl_to_en"))
            rendered[split].append(build_example(row, "en_to_tvl"))

    train_full = sorted(rendered["train"], key=lambda x: stable_hash(x["id"]))
    train_balanced = downsample_bible_examples(
        train_full,
        bible_max_share=args.bible_max_train_share,
    )
    validation = sorted(rendered["validation"], key=lambda x: stable_hash(x["id"]))
    test = sorted(rendered["test"], key=lambda x: stable_hash(x["id"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_full.jsonl", train_full)
    write_jsonl(args.output_dir / "train_balanced.jsonl", train_balanced)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "test.jsonl", test)
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)

    stats = {
        "source_files": [str(p.name) for p in aligned_paths],
        "input_rows": len(raw_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "split_row_counts": {k: len(v) for k, v in splits.items()},
        "train_full": summarize_examples(train_full),
        "train_balanced": summarize_examples(train_balanced),
        "validation": summarize_examples(validation),
        "test": summarize_examples(test),
        "rejection_reasons": dict(
            Counter(reason for item in rejected for reason in item["reasons"])
        ),
        "config": {
            "min_confidence": args.min_confidence,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "ratio_min": args.ratio_min,
            "ratio_max": args.ratio_max,
            "bible_max_train_share": args.bible_max_train_share,
            "non_bible_val_frac": args.non_bible_val_frac,
            "non_bible_test_frac": args.non_bible_test_frac,
        },
    }
    with (args.output_dir / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
