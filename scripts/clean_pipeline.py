"""Clean parallel corpus data — immutable input, new output.

Reads from data/aligned/ (never modified), writes cleaned data to data/cleaned/.

Usage:
    uv run python scripts/clean_pipeline.py
    uv run python scripts/clean_pipeline.py --dry-run
    uv run python scripts/clean_pipeline.py --profile strict
"""

import re
import sys
import json
import hashlib
import argparse
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALIGNED_DIR = DATA_DIR / "aligned"
CLEANED_DIR = DATA_DIR / "cleaned"

# ── Cleaning profiles ──────────────────────────────────────────────────────────

PROFILES = {
    "balanced": {
        "min_chars": 10,          # minimum chars on BOTH sides
        "max_chars": 8192,        # maximum chars on either side
        "ratio_min": 0.2,         # tvl_chars / en_chars lower bound
        "ratio_max": 5.0,         # tvl_chars / en_chars upper bound
        "bible_ratio_min": 0.4,   # tighter ratio for verse-aligned data
        "bible_ratio_max": 2.5,
        "strip_metadata": True,
        "strip_identical": True,
        "strip_truncated_daily": True,
    },
    "strict": {
        "min_chars": 20,
        "max_chars": 4096,
        "ratio_min": 0.3,
        "ratio_max": 3.0,
        "bible_ratio_min": 0.5,
        "bible_ratio_max": 2.0,
        "strip_metadata": True,
        "strip_identical": True,
        "strip_truncated_daily": True,
    },
    "lenient": {
        "min_chars": 5,
        "max_chars": 16384,
        "ratio_min": 0.1,
        "ratio_max": 10.0,
        "bible_ratio_min": 0.3,
        "bible_ratio_max": 3.0,
        "strip_metadata": True,
        "strip_identical": True,
        "strip_truncated_daily": True,
    },
}

# ── Text normalization ─────────────────────────────────────────────────────────

# Zero-width and invisible characters to strip
INVISIBLE_CHARS = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f"   # zero-width spaces/joiners/marks
    "\ufeff"                              # BOM / zero-width no-break space
    "\u00ad"                              # soft hyphen
    "\u2060"                              # word joiner
    "\u2028\u2029"                        # line/paragraph separator
    "]"
)

# HTML entities that might survive scraping
HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&nbsp;": " ", "&quot;": '"', "&#39;": "'",
    "&apos;": "'",
}


def normalize_text(text: str) -> str:
    """Normalize text: NFC, strip invisible chars, fix entities, collapse whitespace."""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Strip invisible characters
    text = INVISIBLE_CHARS.sub("", text)

    # Replace HTML entities
    for entity, replacement in HTML_ENTITIES.items():
        text = text.replace(entity, replacement)

    # Normalize whitespace (collapse runs, strip)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Metadata / boilerplate detection ──────────────────────────────────────────

# Picture captions: [Picture on page 5], [Picture Credit Line on page 5]
PICTURE_CAPTION_RE = re.compile(
    r"^\[(?:Picture|Pictures|Picture Credit Line|Pikitia)s?\s", re.IGNORECASE
)

# Box/chart/diagram markers
BOX_CHART_RE = re.compile(
    r"^\[(?:Box|Chart|Diagram|Graph|Map|Footnote|Table)\s", re.IGNORECASE
)

# Photo credit lines
PHOTO_CREDIT_RE = re.compile(r"^(?:Photo|Image|Picture)\s+[Cc]redit", re.IGNORECASE)

# Copyright notices
COPYRIGHT_RE = re.compile(r"^©\s*\d{4}")

# Page number references
PAGE_NUMBER_RE = re.compile(r"^\[(?:p|page|pp)\.\s*\d+", re.IGNORECASE)

# Chapter/section headers (both EN and TVL)
HEADER_RE = re.compile(
    r"^(?:CHAPTER|MATAUPU\s+E|SECTION|PART|TE\s+VAEGA\s+E)\s+\d+",
    re.IGNORECASE,
)

# Footnote markers (standalone)
FOOTNOTE_MARKER_RE = re.compile(r"^[*†‡§]\s*$")

# Page markers
PAGE_MARKER_RE = re.compile(r"^(?:PAGECHAPTER|MATAUPUTE ITULAU)$")


def is_metadata(text: str) -> bool:
    """Check if text is metadata/boilerplate rather than translatable content."""
    t = text.strip()
    if not t:
        return True
    if PICTURE_CAPTION_RE.match(t):
        return True
    if BOX_CHART_RE.match(t):
        return True
    if PHOTO_CREDIT_RE.match(t):
        return True
    if COPYRIGHT_RE.match(t):
        return True
    if PAGE_NUMBER_RE.match(t):
        return True
    if HEADER_RE.match(t):
        return True
    if FOOTNOTE_MARKER_RE.match(t):
        return True
    if PAGE_MARKER_RE.match(t):
        return True
    return False


# ── Rejection reasons ─────────────────────────────────────────────────────────

def classify_rejection(record: dict, profile: dict) -> str | None:
    """Return rejection reason or None if record passes all filters.

    Rejection reasons (checked in priority order):
        duplicate_id       — same record ID seen before (handled externally)
        duplicate_content  — same (tvl, en) text seen before (handled externally)
        empty_text         — either side is empty after normalization
        metadata           — either side is metadata/boilerplate
        identical_pair     — tvl == en (untranslated content)
        too_short          — both sides below min_chars
        too_long           — either side above max_chars
        bad_ratio          — length ratio outside bounds
        truncated_daily    — daily text with truncated TVL (May 2025 bug)
    """
    tvl = record.get("_tvl_clean", "")
    en = record.get("_en_clean", "")

    # Empty text
    if not tvl or not en:
        return "empty_text"

    # Metadata
    if profile["strip_metadata"] and (is_metadata(tvl) or is_metadata(en)):
        return "metadata"

    # Identical pair (untranslated)
    if profile["strip_identical"] and tvl == en:
        return "identical_pair"

    tvl_chars = len(tvl)
    en_chars = len(en)

    # Too short (both sides must be below threshold)
    if tvl_chars < profile["min_chars"] and en_chars < profile["min_chars"]:
        return "too_short"

    # Too long
    if tvl_chars > profile["max_chars"] or en_chars > profile["max_chars"]:
        return "too_long"

    # Length ratio
    if en_chars > 0:
        ratio = tvl_chars / en_chars
        content_type = record.get("content_type", "")

        if content_type == "bible_verse":
            ratio_min = profile["bible_ratio_min"]
            ratio_max = profile["bible_ratio_max"]
        else:
            ratio_min = profile["ratio_min"]
            ratio_max = profile["ratio_max"]

        if ratio < ratio_min or ratio > ratio_max:
            return "bad_ratio"

    # Truncated daily text (May 2025 bug: TVL has only theme, missing commentary)
    if profile["strip_truncated_daily"]:
        if record.get("content_type") == "daily_text":
            dt = record.get("date", "")
            if dt and dt.startswith("2025-05"):
                if en_chars > 0 and tvl_chars / en_chars < 0.2:
                    return "truncated_daily"

    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_records(source_dir: Path) -> list[dict]:
    """Load all JSONL files from source directory."""
    records = []
    for jsonl_path in sorted(source_dir.glob("*.jsonl")):
        source_name = jsonl_path.stem
        with open(jsonl_path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["_source_file"] = source_name
                    record["_source_line"] = line_no
                    records.append(record)
                except json.JSONDecodeError:
                    print(f"  WARNING: invalid JSON in {jsonl_path.name}:{line_no}")
    return records


def content_hash(tvl: str, en: str) -> str:
    """Hash normalized text pair for deduplication."""
    combined = tvl.strip().lower() + "|||" + en.strip().lower()
    return hashlib.sha256(combined.encode()).hexdigest()


def run_pipeline(records: list[dict], profile: dict) -> tuple[list[dict], list[dict]]:
    """Run the cleaning pipeline. Returns (accepted, rejected) lists."""
    accepted = []
    rejected = []

    rejection_counts = Counter()
    seen_ids = {}       # id -> index in accepted
    seen_hashes = {}    # content_hash -> id

    for record in records:
        rid = record.get("id", "")

        # ── Stage 1: Normalize text ──
        tvl_clean = normalize_text(record.get("tvl", ""))
        en_clean = normalize_text(record.get("en", ""))
        record["_tvl_clean"] = tvl_clean
        record["_en_clean"] = en_clean

        # ── Stage 2: Deduplicate by record ID ──
        if rid in seen_ids:
            record["_rejection_reason"] = "duplicate_id"
            rejected.append(record)
            rejection_counts["duplicate_id"] += 1
            continue
        seen_ids[rid] = len(accepted)

        # ── Stage 3: Deduplicate by content hash ──
        chash = content_hash(tvl_clean, en_clean)
        if chash in seen_hashes:
            record["_rejection_reason"] = "duplicate_content"
            rejected.append(record)
            rejection_counts["duplicate_content"] += 1
            continue
        seen_hashes[chash] = rid

        # ── Stage 4: Quality filters ──
        reason = classify_rejection(record, profile)
        if reason:
            record["_rejection_reason"] = reason
            rejected.append(record)
            rejection_counts[reason] += 1
            continue

        # ── Stage 5: Rebuild clean record (no internal fields) ──
        clean_record = {
            "id": rid,
            "tvl": tvl_clean,
            "en": en_clean,
            "content_type": record.get("content_type"),
            "domain": record.get("domain"),
            "alignment_method": record.get("alignment_method"),
            "alignment_confidence": record.get("alignment_confidence"),
            "doc_id": record.get("doc_id"),
            "source_url_tvl": record.get("source_url_tvl"),
            "source_url_en": record.get("source_url_en"),
            "book_num": record.get("book_num"),
            "book_name": record.get("book_name"),
            "chapter": record.get("chapter"),
            "verse": record.get("verse"),
            "date": record.get("date"),
            "pub_code": record.get("pub_code"),
            "tvl_chars": len(tvl_clean),
            "en_chars": len(en_clean),
            "length_ratio": round(len(tvl_clean) / len(en_clean), 3)
                if len(en_clean) > 0 else 0,
        }
        accepted.append(clean_record)

    return accepted, rejected, rejection_counts


def generate_report(
    total_input: int,
    accepted: list[dict],
    rejected: list[dict],
    rejection_counts: Counter,
    profile_name: str,
    profile: dict,
) -> dict:
    """Generate a cleaning report."""
    # Stats on accepted
    tvl_chars_list = [r["tvl_chars"] for r in accepted]
    en_chars_list = [r["en_chars"] for r in accepted]
    ratios = [r["length_ratio"] for r in accepted if r["length_ratio"] > 0]

    def safe_median(vals):
        if not vals:
            return 0
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    # Content type breakdown
    ct_counts = Counter(r.get("content_type", "unknown") for r in accepted)
    source_counts = Counter(r.get("_source_file", "unknown")
                            for r in accepted + rejected)

    # Domain breakdown
    domain_counts = Counter(r.get("domain", "unknown") for r in accepted)

    report = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile_name,
        "profile_settings": profile,
        "input": {
            "total_records": total_input,
            "source_files": dict(source_counts),
        },
        "output": {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "acceptance_rate": round(len(accepted) / total_input * 100, 1)
                if total_input else 0,
        },
        "rejections": {
            reason: count
            for reason, count in rejection_counts.most_common()
        },
        "accepted_stats": {
            "by_content_type": dict(ct_counts.most_common()),
            "by_domain": dict(domain_counts.most_common()),
            "tvl_chars": {
                "min": min(tvl_chars_list) if tvl_chars_list else 0,
                "max": max(tvl_chars_list) if tvl_chars_list else 0,
                "mean": round(sum(tvl_chars_list) / len(tvl_chars_list), 1)
                    if tvl_chars_list else 0,
                "median": safe_median(tvl_chars_list),
            },
            "en_chars": {
                "min": min(en_chars_list) if en_chars_list else 0,
                "max": max(en_chars_list) if en_chars_list else 0,
                "mean": round(sum(en_chars_list) / len(en_chars_list), 1)
                    if en_chars_list else 0,
                "median": safe_median(en_chars_list),
            },
            "length_ratio": {
                "min": round(min(ratios), 3) if ratios else 0,
                "max": round(max(ratios), 3) if ratios else 0,
                "mean": round(sum(ratios) / len(ratios), 3) if ratios else 0,
                "median": round(safe_median(ratios), 3),
            },
        },
        "total_chars": sum(tvl_chars_list) + sum(en_chars_list),
        "estimated_tokens": round(
            (sum(tvl_chars_list) + sum(en_chars_list)) / 3.8
        ),
    }
    return report


def print_report(report: dict):
    """Print a human-readable cleaning report."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("CLEANING REPORT")
    print(sep)

    print(f"\nProfile: {report['profile']}")
    print(f"Timestamp: {report['timestamp']}")

    print(f"\n{'Input':>20s}: {report['input']['total_records']:,} records")
    print(f"{'Accepted':>20s}: {report['output']['accepted']:,} records")
    print(f"{'Rejected':>20s}: {report['output']['rejected']:,} records")
    print(f"{'Acceptance rate':>20s}: {report['output']['acceptance_rate']}%")

    print(f"\nRejection reasons:")
    for reason, count in sorted(
        report["rejections"].items(), key=lambda x: -x[1]
    ):
        pct = count / report["input"]["total_records"] * 100
        print(f"  {reason:>25s}: {count:>8,} ({pct:5.1f}%)")

    print(f"\nAccepted by content_type:")
    for ct, count in report["accepted_stats"]["by_content_type"].items():
        print(f"  {ct:>25s}: {count:>8,}")

    print(f"\nAccepted by domain:")
    for dom, count in report["accepted_stats"]["by_domain"].items():
        print(f"  {dom:>25s}: {count:>8,}")

    stats = report["accepted_stats"]
    print(f"\nCharacter stats (accepted):")
    for field in ["tvl_chars", "en_chars"]:
        s = stats[field]
        print(f"  {field}: min={s['min']}, max={s['max']}, "
              f"mean={s['mean']}, median={s['median']}")
    s = stats["length_ratio"]
    print(f"  ratio: min={s['min']}, max={s['max']}, "
          f"mean={s['mean']}, median={s['median']}")

    total_chars = report["total_chars"]
    est_tokens = report["estimated_tokens"]
    print(f"\n{'Total chars':>20s}: {total_chars:,}")
    print(f"{'Estimated tokens':>20s}: ~{est_tokens/1e6:.1f}M")

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Clean parallel corpus data (immutable input → new output)."
    )
    parser.add_argument(
        "--profile", choices=PROFILES.keys(), default="balanced",
        help="Cleaning profile (default: balanced)",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=ALIGNED_DIR,
        help="Input directory (default: data/aligned)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=CLEANED_DIR,
        help="Output directory (default: data/cleaned)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze and report only, don't write output files",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]

    # ── Load ──
    print(f"Loading records from {args.input_dir}...")
    records = load_records(args.input_dir)
    total_input = len(records)
    print(f"  Loaded {total_input:,} records")

    # ── Clean ──
    print(f"\nRunning cleaning pipeline (profile: {args.profile})...")
    accepted, rejected, rejection_counts = run_pipeline(records, profile)

    # ── Report ──
    report = generate_report(
        total_input, accepted, rejected, rejection_counts,
        args.profile, profile,
    )
    print_report(report)

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # ── Write output ──
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write accepted records
    accepted_path = args.output_dir / "cleaned.jsonl"
    with open(accepted_path, "w") as f:
        for record in accepted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(accepted):,} accepted records → {accepted_path}")

    # Write rejected records (with reasons)
    rejected_path = args.output_dir / "rejected.jsonl"
    with open(rejected_path, "w") as f:
        for record in rejected:
            # Build a slim rejected record
            out = {
                "id": record.get("id", ""),
                "rejection_reason": record.get("_rejection_reason", "unknown"),
                "tvl": record.get("tvl", "")[:200],  # truncate for space
                "en": record.get("en", "")[:200],
                "content_type": record.get("content_type"),
                "source_file": record.get("_source_file"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rejected):,} rejected records → {rejected_path}")

    # Write report
    report_path = args.output_dir / "cleaning_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote cleaning report → {report_path}")

    # Write per-rejection-reason samples (10 examples each)
    samples_path = args.output_dir / "rejection_samples.jsonl"
    reason_samples = defaultdict(list)
    for record in rejected:
        reason = record.get("_rejection_reason", "unknown")
        if len(reason_samples[reason]) < 10:
            reason_samples[reason].append({
                "id": record.get("id", ""),
                "reason": reason,
                "tvl": record.get("tvl", "")[:300],
                "en": record.get("en", "")[:300],
                "tvl_chars": record.get("tvl_chars", len(record.get("tvl", ""))),
                "en_chars": record.get("en_chars", len(record.get("en", ""))),
            })
    with open(samples_path, "w") as f:
        for reason in sorted(reason_samples):
            for sample in reason_samples[reason]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote rejection samples → {samples_path}")


if __name__ == "__main__":
    main()
