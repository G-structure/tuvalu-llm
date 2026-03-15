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
- Applies macron correction and glottal normalization from clean_pipeline.
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

from tv.corpus.render import downsample_bible_examples as _shared_downsample_bible_examples
from tv.corpus.splits import assign_row_split as _shared_assign_row_split
from tv.corpus.splits import stable_hash as _shared_stable_hash
from tv.common.config import get_repo_root, resolve_path
from tv.common.io import read_jsonl, write_json, write_jsonl
from tv.common.manifests import create_manifest, save_manifest
from tv.common.token_estimates import estimate_dataset_tokens, estimate_example_tokens, format_token_count

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
    return _shared_stable_hash(value)


def _normalize_for_hash(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Zero-width and invisible characters to strip
_INVISIBLE_CHARS = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f"   # zero-width spaces/joiners/marks
    "\ufeff"                              # BOM / zero-width no-break space
    "\u00ad"                              # soft hyphen
    "\u2060"                              # word joiner
    "\u2028\u2029"                        # line/paragraph separator
    "]"
)

# HTML entities that might survive scraping
_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&nbsp;": " ", "&quot;": '"', "&#39;": "'",
    "&apos;": "'",
    "&mdash;": "—", "&ndash;": "–", "&hellip;": "…",
    "&lsquo;": "\u2018", "&rsquo;": "\u2019",
    "&ldquo;": "\u201c", "&rdquo;": "\u201d",
    "&prime;": "\u2032", "&Prime;": "\u2033",
    "&trade;": "\u2122", "&reg;": "\u00ae",
    "&copy;": "\u00a9", "&para;": "\u00b6",
    "&sect;": "\u00a7", "&deg;": "\u00b0",
    "&frac12;": "\u00bd", "&frac14;": "\u00bc",
    "&frac34;": "\u00be", "&times;": "\u00d7",
}

# Inline publication cross-references: (;w18.067 ¶16) or trailing —w16.0718 ¶4-5.
_INLINE_PUB_REF_RE = re.compile(
    r"\s*\(;?[a-z]{1,15}[\-.]?[^)]{0,80}¶[\d\-]+[^)]{0,40}\)"
    r"|[—\-]\s*[a-z]{1,6}\d{2}\.\d+\s*¶[\d\-]+\.?"
)

# Scripture reference stubs: () empty parens, (Faitau te.) / (Read.)
_SCRIPTURE_STUB_RE = re.compile(
    r"\s*\((?:Faitau\s+te|Read)\.\)"
    r"|\s*\(\)"
)

# Trailing reference stubs: —. / —;read. / —Compare. / bare —
_TRAILING_REF_STUB_RE = re.compile(
    r"—[\s;,]*(?:"
    r"[Ff]aitau(?:\s+te)?|[Rr]ead"
    r"|[Ff]akatusa(?:\s+ki\s+te)?|[Cc]ompare"
    r"|[Oo]noono(?:\s+ki\s+te)?|[Ss]ee"
    r"|,?\s*(?:ftn|footnote|fml)"
    r"|,?\s*(?:NW;?|Tusi\s+Paia[^.]*)"
    r")?\.?\s*$"
)

# Inline ",NW." translation edition markers
_INLINE_NW_RE = re.compile(r",\s*NW\.?")

# Fix missing spaces at sentence boundaries (e.g., "Night.And" → "Night. And")
_MISSING_SPACE_RE = re.compile(r"([.!?])([A-ZĀĒĪŌŪÀa-z])")


def _normalize_preserve_structure(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE_CHARS.sub("", text)
    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    # Strip pub refs, scripture stubs, trailing refs, NW markers
    text = _INLINE_PUB_REF_RE.sub("", text)
    text = _SCRIPTURE_STUB_RE.sub("", text)
    text = _TRAILING_REF_STUB_RE.sub("", text)
    text = _TRAILING_REF_STUB_RE.sub("", text)  # apply twice for chained stubs
    text = _INLINE_NW_RE.sub("", text)
    text = text.replace(" ()", "").replace("()", "")
    text = _MISSING_SPACE_RE.sub(r"\1 \2", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    text = text.lstrip(". ")
    return text


# ── Glottal stop normalization ───────────────────────────────────────────────
# Normalize variant glottal marks to U+2035 (reversed prime, corpus standard)
_GLOTTAL_VARIANTS = str.maketrans({
    "\u02cb": "\u2035",  # modifier letter grave accent
    "\u02bb": "\u2035",  # modifier letter turned comma
    "\u0060": "\u2035",  # grave accent
    "\u2019": "\u2035",  # right single quotation mark
    "\u0027": "\u2035",  # ASCII apostrophe
})


# ── Dictionary-guided macron correction ──────────────────────────────────────
_MACRON_MAP: dict[str, str] | None = None
_MACRON_TO_BARE = str.maketrans("āēīōūĀĒĪŌŪ", "aeiouAEIOU")


def _load_macron_map() -> dict[str, str]:
    """Build macron correction map from dictionary data.

    Only includes corrections where:
    - The bare (macron-stripped) form does NOT appear as a separate dictionary entry
    - There is exactly one macronized form (unambiguous)
    - The word is at least 3 characters (skip short function words)
    """
    global _MACRON_MAP
    if _MACRON_MAP is not None:
        return _MACRON_MAP

    import json

    repo_root = get_repo_root()
    aligned_dir = repo_root / "data" / "aligned"

    all_entries: set[str] = set()
    macron_words: dict[str, set[str]] = {}

    for fname in ("tuvalu_dictionary.jsonl", "tuvalu_app.jsonl"):
        fpath = aligned_dir / fname
        if not fpath.exists():
            continue
        for line in open(fpath):
            r = json.loads(line)
            tvl = r["tvl"].strip()
            if " " in tvl or "/" in tvl or "volume_up" in tvl:
                continue
            all_entries.add(tvl.lower())
            if any(c in tvl for c in "āēīōūĀĒĪŌŪ"):
                bare = tvl.lower().translate(_MACRON_TO_BARE)
                if bare != tvl.lower():
                    macron_words.setdefault(bare, set()).add(tvl.lower())

    _MACRON_MAP = {}
    for bare, forms in macron_words.items():
        if len(forms) == 1 and len(bare) >= 3 and bare not in all_entries:
            _MACRON_MAP[bare] = next(iter(forms))

    return _MACRON_MAP


def _apply_macron_correction(text: str) -> str:
    """Apply dictionary-guided macron corrections to TVL text."""
    macron_map = _load_macron_map()
    if not macron_map:
        return text

    words = text.split()
    corrected = False
    for i, word in enumerate(words):
        prefix = word[:len(word) - len(word.lstrip(".,;:!?()\"—‵\u201c\u201d\u2018\u2019"))]
        suffix = word[len(word) - len(word.rstrip(".,;:!?()\"—‵\u201c\u201d\u2018\u2019")):] if word.rstrip(".,;:!?()\"—‵\u201c\u201d\u2018\u2019") != word else ""
        core = word[len(prefix):len(word) - len(suffix)] if suffix else word[len(prefix):]
        lookup = core.lower()

        if lookup in macron_map:
            replacement = macron_map[lookup]
            if core[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            if core.isupper():
                replacement = replacement.upper()
            words[i] = prefix + replacement + suffix
            corrected = True

    return " ".join(words) if corrected else text


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
    content_type = row.get("content_type", "")
    is_dict = content_type in ("word", "expression")

    # Dictionary entries are naturally short — use relaxed bounds
    effective_min_chars = 1 if is_dict else min_chars
    if tvl_chars < effective_min_chars or en_chars < effective_min_chars:
        reasons.append("too_short")
    if tvl_chars > max_chars or en_chars > max_chars:
        reasons.append("too_long")

    ratio = row.get("length_ratio")
    if ratio is None or ratio == 0:
        if en_chars > 0:
            ratio = tvl_chars / en_chars
    effective_ratio_min = 0.005 if is_dict else ratio_min
    effective_ratio_max = 20.0 if is_dict else ratio_max
    if ratio and (ratio < effective_ratio_min or ratio > effective_ratio_max):
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
    return _shared_assign_row_split(
        row,
        non_bible_val_frac=non_bible_val_frac,
        non_bible_test_frac=non_bible_test_frac,
        test_books=test_books,
        validation_books=validation_books,
        include_pub_code=True,
    )


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
    return _shared_downsample_bible_examples(
        train_examples,
        bible_max_share=bible_max_share,
    )


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

    macron_corrections = 0
    for row in raw_rows:
        row = dict(row)
        tvl = _normalize_preserve_structure(str(row.get("tvl", "")))
        # Glottal stop normalization
        tvl = tvl.translate(_GLOTTAL_VARIANTS)
        # Macron correction for non-dictionary text
        if row.get("content_type") not in ("word", "expression"):
            tvl_corrected = _apply_macron_correction(tvl)
            if tvl_corrected != tvl:
                macron_corrections += 1
                tvl = tvl_corrected
        row["tvl"] = tvl
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

    if macron_corrections:
        print(f"  Macron corrections applied to {macron_corrections:,} rows")

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
