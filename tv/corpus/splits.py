#!/usr/bin/env python3
"""Leak-proof train/test split pipeline with cross-source decontamination.

Reads data/cleaned/cleaned.jsonl (immutable), writes split files to data/splits/.

Phases:
  1. Assign splits (deterministic, doc-level grouping)
  2. Build held-out n-gram index from test+val Bible verses
  3. Cross-source decontamination (quarantine contaminated training examples)
  4. Validation (verify zero leakage post-quarantine)
  5. Write outputs

Usage:
    uv run python scripts/build_splits.py
    uv run python scripts/build_splits.py --dry-run
    uv run python scripts/build_splits.py --ngram-size 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CLEANED_FILE = DATA_DIR / "cleaned" / "cleaned.jsonl"
OUTPUT_DIR = DATA_DIR / "splits"

# ── Split configuration ──────────────────────────────────────────────────────

SPLIT_CONFIG = {
    # Bible book assignments
    "test_books": {8, 57, 65},  # Ruth, Philemon, Jude
    "validation_books": {31, 37, 56, 63, 64},  # Obadiah, Haggai, Titus, 2 John, 3 John

    # Non-Bible hash split fractions
    "non_bible_test_frac": 0.05,
    "non_bible_val_frac": 0.05,

    # Decontamination
    "ngram_size": 10,
    "containment_threshold": 0.6,
    "contamination_warn_pct": 1.0,
    "contamination_fail_pct": 5.0,
}


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _extract_ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


# ── Phase 1: Assign splits ───────────────────────────────────────────────────

def group_key(row: dict[str, Any], *, include_pub_code: bool = False) -> str:
    content_type = row.get("content_type")
    if content_type == "bible_verse":
        return f"bible_book_{row.get('book_num')}"
    if row.get("doc_id"):
        return f"doc_{row['doc_id']}"
    if row.get("date"):
        return f"date_{row['date']}"
    if include_pub_code and row.get("pub_code"):
        return f"pub_{row['pub_code']}"
    return f"row_{row.get('id', '')}"


def assign_row_split(
    row: dict[str, Any],
    *,
    non_bible_val_frac: float,
    non_bible_test_frac: float,
    test_books: set[int],
    validation_books: set[int],
    include_pub_code: bool = False,
) -> str:
    if row.get("content_type") == "bible_verse":
        book_num = int(row.get("book_num") or 0)
        if book_num in test_books:
            return "test"
        if book_num in validation_books:
            return "validation"
        return "train"

    key = group_key(row, include_pub_code=include_pub_code)
    bucket = stable_hash(key) % 10000
    test_cut = int(non_bible_test_frac * 10000)
    val_cut = test_cut + int(non_bible_val_frac * 10000)
    if bucket < test_cut:
        return "test"
    if bucket < val_cut:
        return "validation"
    return "train"


def assign_split(row: dict[str, Any], config: dict) -> str:
    return assign_row_split(
        row,
        non_bible_val_frac=config["non_bible_val_frac"],
        non_bible_test_frac=config["non_bible_test_frac"],
        test_books=set(config["test_books"]),
        validation_books=set(config["validation_books"]),
        include_pub_code=False,
    )


# ── Phase 2: Build held-out n-gram index ──────────────────────────────────────

def build_heldout_index(
    heldout_rows: list[dict[str, Any]], ngram_size: int
) -> tuple[set[tuple[str, ...]], set[str], list[dict]]:
    """Build n-gram set and exact-hash set from held-out Bible verses.

    Returns (ngram_set, exact_hash_set, short_verse_tokens).
    short_verse_tokens only includes verses shorter than ngram_size words
    (longer ones are already caught by the n-gram check).
    """
    ngram_set: set[tuple[str, ...]] = set()
    exact_hashes: set[str] = set()
    short_verse_tokens: list[dict] = []

    for row in heldout_rows:
        if row.get("content_type") != "bible_verse":
            continue
        for side in ("tvl", "en"):
            text = str(row.get(side, ""))
            exact_hashes.add(_text_hash(text))
            tokens = _tokenize(text)
            ngrams = _extract_ngrams(tokens, ngram_size)
            ngram_set.update(ngrams)
            # Only keep short verses for containment check
            if len(tokens) < ngram_size and len(tokens) >= 4:
                short_verse_tokens.append({
                    "id": row["id"],
                    "side": side,
                    "tokens": tokens,
                    "token_str": " ".join(tokens),
                })

    return ngram_set, exact_hashes, short_verse_tokens


# ── Phase 3: Cross-source decontamination ─────────────────────────────────────

def check_contamination(
    row: dict[str, Any],
    ngram_set: set[tuple[str, ...]],
    exact_hashes: set[str],
    short_verse_tokens: list[dict],
    ngram_size: int,
    containment_threshold: float,
) -> list[dict[str, Any]]:
    """Check a training row for contamination against held-out data.

    Returns list of contamination reasons (empty = clean).
    """
    reasons: list[dict[str, Any]] = []

    for side in ("tvl", "en"):
        text = str(row.get(side, ""))

        # Level 1: Exact match
        if _text_hash(text) in exact_hashes:
            reasons.append({
                "level": "exact_match",
                "side": side,
                "detail": "exact text match with held-out verse",
            })
            continue

        tokens = _tokenize(text)
        if not tokens:
            continue

        # Level 2: N-gram overlap
        row_ngrams = _extract_ngrams(tokens, ngram_size)
        if row_ngrams:
            overlap = row_ngrams & ngram_set
            if overlap:
                reasons.append({
                    "level": "ngram_overlap",
                    "side": side,
                    "matching_ngrams": len(overlap),
                    "total_ngrams": len(row_ngrams),
                    "sample": " ".join(next(iter(overlap))),
                })
                continue

        # Level 3: Containment check — only for short verses (< ngram_size words)
        # that wouldn't be caught by the n-gram check above.
        token_str = " ".join(tokens)
        for sv in short_verse_tokens:
            if sv["side"] != side:
                continue
            v_str = sv["token_str"]
            if v_str in token_str:
                reasons.append({
                    "level": "high_containment",
                    "side": side,
                    "verse_id": sv["id"],
                    "containment": 1.0,
                })
                break
            # Check partial: >threshold fraction of verse tokens appear contiguously
            v_toks = sv["tokens"]
            best = 0
            for start in range(len(tokens)):
                k = 0
                while (
                    start + k < len(tokens)
                    and k < len(v_toks)
                    and tokens[start + k] == v_toks[k]
                ):
                    k += 1
                best = max(best, k)
            if best / len(v_toks) > containment_threshold:
                reasons.append({
                    "level": "high_containment",
                    "side": side,
                    "verse_id": sv["id"],
                    "containment": round(best / len(v_toks), 3),
                })
                break

    return reasons


def decontaminate(
    train_rows: list[dict[str, Any]],
    ngram_set: set[tuple[str, ...]],
    exact_hashes: set[str],
    short_verse_tokens: list[dict],
    config: dict,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan training rows and quarantine contaminated ones.

    Returns (clean_train, quarantined, contamination_details).
    """
    clean: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    ngram_size = config["ngram_size"]
    containment_threshold = config["containment_threshold"]

    # Check non-Bible training rows for all contamination types
    bible_train = [r for r in train_rows if r.get("content_type") == "bible_verse"]
    non_bible_train = [r for r in train_rows if r.get("content_type") != "bible_verse"]

    for i, row in enumerate(non_bible_train):
        if i > 0 and i % 25000 == 0:
            print(f"    ... checked {i}/{len(non_bible_train)} ({len(quarantined)} quarantined)")
        reasons = check_contamination(
            row, ngram_set, exact_hashes, short_verse_tokens,
            ngram_size, containment_threshold,
        )
        if reasons:
            quarantined.append(row)
            details.append({"id": row["id"], "reasons": reasons})
        else:
            clean.append(row)

    # Bible training rows: only check exact text matches against held-out text
    # (n-gram/containment not needed — split is structural by book)
    clean_bible: list[dict[str, Any]] = []
    for row in bible_train:
        exact_reasons: list[dict[str, Any]] = []
        for side in ("tvl", "en"):
            text = str(row.get(side, ""))
            if _text_hash(text) in exact_hashes:
                exact_reasons.append({
                    "level": "exact_match",
                    "side": side,
                    "detail": "Bible training verse text matches held-out non-Bible text",
                })
        if exact_reasons:
            quarantined.append(row)
            details.append({"id": row["id"], "reasons": exact_reasons})
        else:
            clean_bible.append(row)

    clean_all = clean_bible + clean
    return clean_all, quarantined, details


# ── Phase 4: Validation ───────────────────────────────────────────────────────

def validate_splits(
    splits: dict[str, list[dict[str, Any]]],
    ngram_set: set[tuple[str, ...]],
    config: dict,
) -> list[str]:
    """Run validation checks. Returns list of failure messages (empty = pass)."""
    failures: list[str] = []
    ngram_size = config["ngram_size"]

    train = splits.get("train", [])
    test = splits.get("test", [])
    val = splits.get("validation", [])

    # Check 1: No doc_id overlap
    def _doc_ids(rows: list[dict]) -> set[str]:
        return {r["doc_id"] for r in rows if r.get("doc_id")}

    train_docs = _doc_ids(train)
    test_docs = _doc_ids(test)
    val_docs = _doc_ids(val)
    if train_docs & test_docs:
        failures.append(f"doc_id overlap train/test: {train_docs & test_docs}")
    if train_docs & val_docs:
        failures.append(f"doc_id overlap train/val: {train_docs & val_docs}")

    # Check 2: No Bible book overlap
    def _book_nums(rows: list[dict]) -> set[int]:
        return {
            int(r["book_num"])
            for r in rows
            if r.get("content_type") == "bible_verse" and r.get("book_num")
        }

    train_books = _book_nums(train)
    test_books = _book_nums(test)
    val_books = _book_nums(val)
    if train_books & test_books:
        failures.append(f"Bible book overlap train/test: {train_books & test_books}")
    if train_books & val_books:
        failures.append(f"Bible book overlap train/val: {train_books & val_books}")

    # Check 3: No date overlap (daily text)
    def _dates(rows: list[dict]) -> set[str]:
        return {r["date"] for r in rows if r.get("date")}

    train_dates = _dates(train)
    test_dates = _dates(test)
    val_dates = _dates(val)
    if train_dates & test_dates:
        failures.append(f"Date overlap train/test: {len(train_dates & test_dates)} dates")
    if train_dates & val_dates:
        failures.append(f"Date overlap train/val: {len(train_dates & val_dates)} dates")

    # Check 4: No exact text overlap
    train_hashes: set[str] = set()
    for row in train:
        train_hashes.add(_text_hash(str(row.get("tvl", ""))))
        train_hashes.add(_text_hash(str(row.get("en", ""))))

    exact_leaks = 0
    for row in test + val:
        for side in ("tvl", "en"):
            if _text_hash(str(row.get(side, ""))) in train_hashes:
                exact_leaks += 1
    if exact_leaks:
        failures.append(f"Exact text overlap: {exact_leaks} matches between train and test/val")

    # Check 5: No n-gram overlap (post-quarantine)
    # Build n-gram set from test+val Bible verses
    heldout_bible = [
        r for r in test + val if r.get("content_type") == "bible_verse"
    ]
    heldout_ngrams: set[tuple[str, ...]] = set()
    for row in heldout_bible:
        for side in ("tvl", "en"):
            tokens = _tokenize(str(row.get(side, "")))
            heldout_ngrams.update(_extract_ngrams(tokens, ngram_size))

    # Check non-Bible training rows for n-gram overlap
    ngram_leaks = 0
    for row in train:
        if row.get("content_type") == "bible_verse":
            continue
        for side in ("tvl", "en"):
            tokens = _tokenize(str(row.get(side, "")))
            row_ngrams = _extract_ngrams(tokens, ngram_size)
            if row_ngrams & heldout_ngrams:
                ngram_leaks += 1
                break
    if ngram_leaks:
        failures.append(
            f"N-gram overlap: {ngram_leaks} training examples share {ngram_size}-grams "
            f"with held-out Bible verses"
        )

    # Check 6: Minimum sizes
    if len(test) < 300:
        failures.append(f"Test set too small: {len(test)} < 300")
    if len(val) < 100:
        failures.append(f"Validation set too small: {len(val)} < 100")

    # Check 7: Per-domain representation in test
    test_domains = Counter(r.get("content_type", "unknown") for r in test)
    for required in ("bible_verse", "article_paragraph", "daily_text"):
        if test_domains.get(required, 0) == 0:
            failures.append(f"Test set missing domain: {required}")

    return failures


# ── Phase 5: Write outputs ────────────────────────────────────────────────────

def write_split_report(
    splits: dict[str, list[dict[str, Any]]],
    quarantined: list[dict[str, Any]],
    contamination_details: list[dict[str, Any]],
    validation_failures: list[str],
    config: dict,
    output_dir: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "config": {k: list(v) if isinstance(v, set) else v for k, v in config.items()},
        "splits": {},
        "quarantined": len(quarantined),
        "validation_passed": len(validation_failures) == 0,
        "validation_failures": validation_failures,
    }

    for split_name, rows in splits.items():
        by_ct = Counter(r.get("content_type", "unknown") for r in rows)
        by_domain = Counter(r.get("domain", "unknown") for r in rows)
        total_tvl_chars = sum(r.get("tvl_chars", 0) for r in rows)
        total_en_chars = sum(r.get("en_chars", 0) for r in rows)
        report["splits"][split_name] = {
            "count": len(rows),
            "by_content_type": dict(by_ct),
            "by_domain": dict(by_domain),
            "total_tvl_chars": total_tvl_chars,
            "total_en_chars": total_en_chars,
        }

    quarantine_reasons = Counter()
    for d in contamination_details:
        for r in d["reasons"]:
            quarantine_reasons[r["level"]] += 1
    report["quarantine_reasons"] = dict(quarantine_reasons)

    return report


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build leak-proof train/test splits")
    parser.add_argument("--input", type=Path, default=CLEANED_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ngram-size", type=int, default=None)
    parser.add_argument("--containment-threshold", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = dict(SPLIT_CONFIG)
    if args.ngram_size is not None:
        config["ngram_size"] = args.ngram_size
    if args.containment_threshold is not None:
        config["containment_threshold"] = args.containment_threshold

    print(f"Reading {args.input} ...")
    all_rows = read_jsonl(args.input)
    print(f"  {len(all_rows)} records loaded")

    # Phase 1: Assign splits
    print("\nPhase 1: Assigning splits ...")
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        split = assign_split(row, config)
        splits[split].append(row)

    for name in ("train", "validation", "test"):
        ct = Counter(r.get("content_type") for r in splits[name])
        print(f"  {name}: {len(splits[name])} ({dict(ct)})")

    # Phase 2: Build held-out n-gram index
    print(f"\nPhase 2: Building held-out n-gram index (n={config['ngram_size']}) ...")
    heldout_all = splits["test"] + splits["validation"]
    heldout_bible = [r for r in heldout_all if r.get("content_type") == "bible_verse"]
    ngram_set, bible_exact_hashes, short_verse_tokens = build_heldout_index(
        heldout_bible, config["ngram_size"]
    )
    # Also build exact hashes from ALL held-out rows (catches cross-doc text duplication)
    all_heldout_hashes: set[str] = set(bible_exact_hashes)
    for row in heldout_all:
        for side in ("tvl", "en"):
            all_heldout_hashes.add(_text_hash(str(row.get(side, ""))))
    print(f"  {len(heldout_bible)} held-out Bible verses")
    print(f"  {len(ngram_set)} unique {config['ngram_size']}-grams")
    print(f"  {len(all_heldout_hashes)} exact hashes (all held-out text)")
    print(f"  {len(short_verse_tokens)} short verses for containment check")

    # Phase 3: Cross-source decontamination
    print("\nPhase 3: Decontaminating training set ...")
    clean_train, quarantined, contamination_details = decontaminate(
        splits["train"], ngram_set, all_heldout_hashes, short_verse_tokens, config,
    )
    print(f"  {len(splits['train'])} train rows -> {len(clean_train)} clean + {len(quarantined)} quarantined")
    if contamination_details:
        reason_counts = Counter()
        for d in contamination_details:
            for r in d["reasons"]:
                reason_counts[r["level"]] += 1
        for level, count in reason_counts.most_common():
            print(f"    {level}: {count}")

    splits["train"] = clean_train

    # Phase 4: Validation
    print("\nPhase 4: Validating splits ...")
    failures = validate_splits(splits, ngram_set, config)
    if failures:
        print("  VALIDATION FAILURES:")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  All checks passed")

    # Summary
    print("\n── Final split sizes ──")
    total = 0
    for name in ("train", "validation", "test"):
        ct = Counter(r.get("content_type") for r in splits[name])
        print(f"  {name:12s}: {len(splits[name]):>7,} ({dict(ct)})")
        total += len(splits[name])
    print(f"  {'quarantined':12s}: {len(quarantined):>7,}")
    print(f"  {'total':12s}: {total + len(quarantined):>7,}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Phase 5: Write outputs
    print(f"\nPhase 5: Writing to {args.output_dir} ...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(args.output_dir / "train.jsonl", splits["train"])
    write_jsonl(args.output_dir / "validation.jsonl", splits["validation"])
    write_jsonl(args.output_dir / "test.jsonl", splits["test"])
    write_jsonl(args.output_dir / "quarantined.jsonl", quarantined)
    write_jsonl(args.output_dir / "contamination_details.jsonl", contamination_details)

    report = write_split_report(
        splits, quarantined, contamination_details, failures, config, args.output_dir,
    )
    write_json(args.output_dir / "split_report.json", report)

    print("  Done.")
    if failures:
        print(f"\nWARNING: {len(failures)} validation failures detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()


_stable_hash = stable_hash
_group_key = group_key
