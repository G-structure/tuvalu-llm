"""Build Stage A MT chat datasets from aligned JSONL.

Refactored from scripts/build_tinker_mt_data.py. Turns aligned data files
into conversation-style JSONL for Tinker supervised fine-tuning.

Outputs:
    data/finetune/stage_a_mt/
        train_full.jsonl
        train_balanced.jsonl
        validation.jsonl
        test.jsonl
        rejected.jsonl
        stats.json
        manifest.json

Design:
- Keeps aligned JSONL files as canonical source of truth.
- Generates BOTH directions from each accepted pair.
- Deterministic splits (Bible by book, articles by doc_id, daily text by date,
  everything else by hash).
- Quality filtering + dedup.
- Balanced train file that caps Bible share.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.common.config import get_repo_root, resolve_path
from training.common.io import read_jsonl, write_json, write_jsonl
from training.common.manifests import create_manifest, save_manifest
from training.common.token_estimates import estimate_dataset_tokens, estimate_example_tokens, format_token_count

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

DEFAULT_TEST_BOOKS = {8, 57, 65}  # Ruth, Philemon, Jude
DEFAULT_VALID_BOOKS = {31, 63, 64}  # Obadiah, 2 John, 3 John

# Default config values
DEFAULTS: dict[str, Any] = {
    "input_dir": "data/aligned",
    "output_dir": "data/finetune/stage_a_mt",
    "seed": 17,
    "min_confidence": 0.8,
    "min_chars": 10,
    "max_chars": 4096,
    "ratio_min": 0.4,
    "ratio_max": 2.5,
    "allow_low_confidence_articles": False,
    "bible_max_train_share": 0.70,
    "non_bible_val_frac": 0.05,
    "non_bible_test_frac": 0.05,
}


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _normalize_for_hash(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_preserve_structure(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _row_quality_reasons(
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

    tvl = _normalize_preserve_structure(str(row.get("tvl", "")))
    en = _normalize_preserve_structure(str(row.get("en", "")))
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


def _group_key(row: dict[str, Any]) -> str:
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


def _assign_split(
    row: dict[str, Any],
    *,
    non_bible_val_frac: float,
    non_bible_test_frac: float,
    test_books: set[int] | None = None,
    validation_books: set[int] | None = None,
) -> str:
    if test_books is None:
        test_books = DEFAULT_TEST_BOOKS
    if validation_books is None:
        validation_books = DEFAULT_VALID_BOOKS

    if row.get("content_type") == "bible_verse":
        book_num = int(row.get("book_num") or 0)
        if book_num in test_books:
            return "test"
        if book_num in validation_books:
            return "validation"
        return "train"

    key = _group_key(row)
    bucket = _stable_hash(key) % 10000
    test_cut = int(non_bible_test_frac * 10000)
    val_cut = test_cut + int(non_bible_val_frac * 10000)
    if bucket < test_cut:
        return "test"
    if bucket < val_cut:
        return "validation"
    return "train"


def _choose_template(row_id: str, direction: str) -> tuple[str, int]:
    if direction == "tvl_to_en":
        templates = TVL_TO_EN_TEMPLATES
    else:
        templates = EN_TO_TVL_TEMPLATES
    idx = _stable_hash(f"{row_id}::{direction}") % len(templates)
    return templates[idx], idx


def _build_example(row: dict[str, Any], direction: str) -> dict[str, Any]:
    if direction == "tvl_to_en":
        src = _normalize_preserve_structure(str(row["tvl"]))
        tgt = _normalize_preserve_structure(str(row["en"]))
        src_lang = "tvl"
        tgt_lang = "en"
    elif direction == "en_to_tvl":
        src = _normalize_preserve_structure(str(row["en"]))
        tgt = _normalize_preserve_structure(str(row["tvl"]))
        src_lang = "en"
        tgt_lang = "tvl"
    else:
        raise ValueError(direction)

    template, template_idx = _choose_template(str(row["id"]), direction)
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


def _downsample_bible_examples(
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

    bible_sorted = sorted(bible, key=lambda x: _stable_hash(x["id"]))
    kept_bible = bible_sorted[:max_bible]
    combined = non_bible + kept_bible
    return sorted(combined, key=lambda x: _stable_hash(x["id"]))


def _summarize_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_direction = Counter(x["metadata"]["direction"] for x in rows)
    by_domain = Counter(x["metadata"].get("domain", "unknown") for x in rows)
    by_content_type = Counter(x["metadata"].get("content_type", "unknown") for x in rows)
    tokenish_lengths = [
        len(_normalize_for_hash(x["messages"][-1]["content"])) / 4.0
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
        "total_tokens_est": estimate_dataset_tokens(rows),
    }


def _format_budget_label(budget: int) -> str:
    """Human-readable label for a token budget (e.g., 2000000 -> '2m')."""
    if budget >= 1_000_000 and budget % 1_000_000 == 0:
        return f"{budget // 1_000_000}m"
    if budget >= 1_000 and budget % 1_000 == 0:
        return f"{budget // 1_000}k"
    return str(budget)


def build_pilot_subset(
    examples: list[dict[str, Any]],
    *,
    token_budget: int,
) -> list[dict[str, Any]]:
    """Build a deterministic pilot subset within a token budget.

    Selection: sort by stable hash of id, then accumulate until budget is reached.
    Token counting uses estimate_example_tokens (full sequence).
    """
    sorted_examples = sorted(examples, key=lambda x: _stable_hash(x["id"]))
    subset: list[dict[str, Any]] = []
    total_tokens = 0
    for ex in sorted_examples:
        ex_tokens = estimate_example_tokens(ex)
        if total_tokens + ex_tokens > token_budget and subset:
            break
        subset.append(ex)
        total_tokens += ex_tokens
    return subset


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build Stage A MT dataset.

    Args:
        config: Configuration dict. Missing keys use DEFAULTS.

    Returns:
        Stats dict summarizing what was built.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    repo_root = get_repo_root()
    input_dir = resolve_path(cfg["input_dir"], repo_root)
    output_dir = resolve_path(cfg["output_dir"], repo_root)

    aligned_paths = sorted(input_dir.glob("*.jsonl"))
    if not aligned_paths:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    raw_rows: list[dict[str, Any]] = []
    for path in aligned_paths:
        raw_rows.extend(read_jsonl(path))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_pair_hashes: set[str] = set()

    for row in raw_rows:
        row = dict(row)
        row["tvl"] = _normalize_preserve_structure(str(row.get("tvl", "")))
        row["en"] = _normalize_preserve_structure(str(row.get("en", "")))

        reasons = _row_quality_reasons(
            row,
            min_confidence=cfg["min_confidence"],
            min_chars=cfg["min_chars"],
            max_chars=cfg["max_chars"],
            ratio_min=cfg["ratio_min"],
            ratio_max=cfg["ratio_max"],
            allow_low_conf_article=cfg["allow_low_confidence_articles"],
        )

        dedup_hash = hashlib.sha256(
            (
                _normalize_for_hash(row["tvl"]) + "|||" + _normalize_for_hash(row["en"])
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

    # Config-driven holdout books (fall back to module defaults)
    test_books = set(cfg["test_books"]) if "test_books" in cfg else None
    validation_books = set(cfg["validation_books"]) if "validation_books" in cfg else None

    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        split = _assign_split(
            row,
            non_bible_val_frac=cfg["non_bible_val_frac"],
            non_bible_test_frac=cfg["non_bible_test_frac"],
            test_books=test_books,
            validation_books=validation_books,
        )
        splits[split].append(row)

    rendered: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, rows in splits.items():
        for row in rows:
            rendered[split].append(_build_example(row, "tvl_to_en"))
            rendered[split].append(_build_example(row, "en_to_tvl"))

    train_full = sorted(rendered["train"], key=lambda x: _stable_hash(x["id"]))
    train_balanced = _downsample_bible_examples(
        train_full,
        bible_max_share=cfg["bible_max_train_share"],
    )
    validation = sorted(rendered["validation"], key=lambda x: _stable_hash(x["id"]))
    test = sorted(rendered["test"], key=lambda x: _stable_hash(x["id"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train_full.jsonl", train_full)
    write_jsonl(output_dir / "train_balanced.jsonl", train_balanced)
    write_jsonl(output_dir / "validation.jsonl", validation)
    write_jsonl(output_dir / "test.jsonl", test)
    write_jsonl(output_dir / "rejected.jsonl", rejected)

    # Pilot subset (optional)
    pilot_budget = cfg.get("pilot_token_budget")
    train_pilot = None
    pilot_file = None
    if pilot_budget is not None:
        pilot_budget = int(pilot_budget)
        budget_label = _format_budget_label(pilot_budget)
        pilot_file = f"train_pilot_{budget_label}.jsonl"
        train_pilot = build_pilot_subset(train_balanced, token_budget=pilot_budget)
        write_jsonl(output_dir / pilot_file, train_pilot)
        print(f"  Pilot subset: {len(train_pilot)} examples -> {pilot_file}")

    stats = {
        "source_files": [str(p.name) for p in aligned_paths],
        "input_rows": len(raw_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "split_row_counts": {k: len(v) for k, v in splits.items()},
        "train_full": _summarize_examples(train_full),
        "train_balanced": _summarize_examples(train_balanced),
        "validation": _summarize_examples(validation),
        "test": _summarize_examples(test),
        "rejection_reasons": dict(
            Counter(reason for item in rejected for reason in item["reasons"])
        ),
        "config": {k: v for k, v in cfg.items() if k not in ("input_dir", "output_dir")},
    }
    if train_pilot is not None:
        stats["train_pilot"] = _summarize_examples(train_pilot)
        stats["train_pilot"]["file"] = pilot_file
    write_json(output_dir / "stats.json", stats)

    manifest = create_manifest(
        stage="stage_a_mt_build_data",
        config=cfg,
        extra={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "input_rows": len(raw_rows),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
        },
    )
    save_manifest(manifest, output_dir / "manifest.json")

    print(f"Stage A data built: {len(train_full)} train_full, "
          f"{len(train_balanced)} train_balanced, "
          f"{len(validation)} validation, {len(test)} test, "
          f"{len(rejected)} rejected")
    print(f"  Token estimates: train_full={format_token_count(stats['train_full']['total_tokens_est'])}, "
          f"train_balanced={format_token_count(stats['train_balanced']['total_tokens_est'])}")
    print(f"  Output: {output_dir}")

    return stats
