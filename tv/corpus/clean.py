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

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ALIGNED_DIR = DATA_DIR / "aligned"
CLEANED_DIR = DATA_DIR / "cleaned"

# ── Cleaning profiles ──────────────────────────────────────────────────────────

PROFILES = {
    "balanced": {
        "min_chars": 10,          # minimum chars on EITHER side
        "max_chars": 8192,        # maximum chars on either side
        "ratio_min": 0.2,         # tvl_chars / en_chars lower bound
        "ratio_max": 5.0,         # tvl_chars / en_chars upper bound
        "bible_ratio_min": 0.4,   # tighter ratio for verse-aligned data
        "bible_ratio_max": 2.5,
        "dict_min_chars": 1,      # dictionary entries are naturally short
        "dict_ratio_min": 0.005,  # single-char TVL grammar words vs long EN gloss
        "dict_ratio_max": 20.0,
        "strip_metadata": True,
        "strip_identical": True,
        "strip_truncated_daily": True,
        "strip_pub_refs": True,   # strip inline ¶ pub refs from text
        "boilerplate_max": 10,    # max times same EN text can appear (0=off)
        "strip_ref_only": True,   # remove ref-only pairs (¶ cross-refs)
    },
    "strict": {
        "min_chars": 20,
        "max_chars": 4096,
        "ratio_min": 0.3,
        "ratio_max": 3.0,
        "bible_ratio_min": 0.5,
        "bible_ratio_max": 2.0,
        "dict_min_chars": 2,
        "dict_ratio_min": 0.01,
        "dict_ratio_max": 10.0,
        "strip_metadata": True,
        "strip_identical": True,
        "strip_truncated_daily": True,
        "strip_pub_refs": True,
        "boilerplate_max": 5,
        "strip_ref_only": True,
    },
    "lenient": {
        "min_chars": 5,
        "max_chars": 16384,
        "ratio_min": 0.1,
        "ratio_max": 10.0,
        "bible_ratio_min": 0.3,
        "bible_ratio_max": 3.0,
        "dict_min_chars": 1,
        "dict_ratio_min": 0.005,
        "dict_ratio_max": 20.0,
        "strip_metadata": True,
        "strip_identical": True,
        "strip_truncated_daily": True,
        "strip_pub_refs": False,
        "boilerplate_max": 20,
        "strip_ref_only": False,
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

# Inline publication cross-references to strip from text:
#   (;w18.067 ¶16)  or  (;cl chap. 18 ¶20-24)  or  (ip-1198 ¶20)
#   (;ijwbvarticle 5 ¶4-7-mwbr) or (;it"Ariel" ¶1;it"Ariel" No. 3)
#   also: trailing refs like —w16.0718 ¶4-5.
INLINE_PUB_REF_RE = re.compile(
    r"\s*\(;?"                        # opening paren, optional semicolon
    r"[a-z]{1,15}[\-.]?"             # pub code (w, cl, fg, ip, ijwbqarticle...)
    r"[^)]{0,80}"                     # up to 80 chars of ref detail
    r"¶[\d\-]+"                       # paragraph marker ¶ + number/range
    r"[^)]{0,40}"                     # optional trailing ref (;second ref, -mwbr)
    r"\)"                             # closing paren
    r"|"                              # OR
    r"[—\-]\s*[a-z]{1,6}\d{2}\.\d+"  # trailing dash-ref: —w16.0718
    r"\s*¶[\d\-]+\.?"                 # ¶ marker + range
)

# Scripture reference stubs left after stripping:
#   () — empty parens from removed scripture citations
#   (Faitau te.) / (Read.) — instruction stubs ("Read the...")
#   Trailing —. / —;read. / —Compare. / —,ftn. / —,NW. etc.
SCRIPTURE_STUB_RE = re.compile(
    r"\s*\((?:Faitau\s+te|Read)\.\)"   # (Faitau te.) / (Read.)
    r"|\s*\(\)"                         # empty ()
)

# Trailing scripture reference stubs at end of text:
#   —.  —;read.  —Compare.  —, ftn.  —,NW;.  —Faitau te.  —,Tusi Paia,Samoa.
#   Also bare — at end (stripped ref left only the em-dash)
TRAILING_REF_STUB_RE = re.compile(
    r"—"                               # em-dash
    r"[\s;,]*"                         # optional separator (space, semicolon, comma)
    r"(?:"
    r"[Ff]aitau(?:\s+te)?|[Rr]ead"    # read instructions
    r"|[Ff]akatusa(?:\s+ki\s+te)?|[Cc]ompare"  # compare instructions
    r"|[Oo]noono(?:\s+ki\s+te)?|[Ss]ee"  # see instructions
    r"|,?\s*(?:ftn|footnote|fml)"      # footnote refs
    r"|,?\s*(?:NW;?|Tusi\s+Paia[^.]*)"  # translation edition refs
    r")?"                              # all instruction text is optional
    r"\.?\s*$"                         # optional period + end (catches both —. and bare —)
)

# Inline ",NW." translation edition markers mid-text: "3:27,NW. A te..."
INLINE_NW_RE = re.compile(r",\s*NW\.?")

# Fix missing spaces at sentence boundaries (e.g., "Night.And" → "Night. And")
MISSING_SPACE_RE = re.compile(r"([.!?])([A-ZĀĒĪŌŪÀa-z])")

# Normalize glottal mark variants to U+2035 (reversed prime, the standard form)
GLOTTAL_VARIANTS = str.maketrans({
    "\u02cb": "\u2035",  # modifier letter grave accent → reversed prime
    "\u02bb": "\u2035",  # modifier letter turned comma → reversed prime
    "\u0060": "\u2035",  # grave accent → reversed prime
    "\u2019": "\u2035",  # right single quotation mark → reversed prime
    "\u0027": "\u2035",  # ASCII apostrophe → reversed prime
})


# ── Dictionary-guided macron correction ──────────────────────────────────────

_MACRON_MAP = None  # lazy-loaded


def _load_macron_map() -> dict[str, str]:
    """Build macron correction map from dictionary data.

    Only includes corrections where:
    - The bare (macron-stripped) form does NOT appear as a separate dictionary entry
    - There is exactly one macronized form (unambiguous)
    - The word is at least 3 characters (skip short function words)

    Returns {bare_lower: macronized_lower} mapping.
    """
    global _MACRON_MAP
    if _MACRON_MAP is not None:
        return _MACRON_MAP

    MACRON_TO_BARE = str.maketrans("āēīōūĀĒĪŌŪ", "aeiouAEIOU")

    all_entries = set()   # all tvl headwords (bare + macronized)
    macron_words = {}     # bare -> set of macronized forms

    dict_files = [
        ALIGNED_DIR / "tuvalu_dictionary.jsonl",
        ALIGNED_DIR / "tuvalu_app.jsonl",
    ]

    for fpath in dict_files:
        if not fpath.exists():
            continue
        for line in open(fpath):
            r = json.loads(line)
            tvl = r["tvl"].strip()
            # Single words only
            if " " in tvl or "/" in tvl or "volume_up" in tvl:
                continue
            all_entries.add(tvl.lower())

            if any(c in tvl for c in "āēīōūĀĒĪŌŪ"):
                bare = tvl.lower().translate(MACRON_TO_BARE)
                if bare != tvl.lower():
                    if bare not in macron_words:
                        macron_words[bare] = set()
                    macron_words[bare].add(tvl.lower())

    # Only keep unambiguous, non-homograph, 3+ char corrections
    _MACRON_MAP = {}
    for bare, forms in macron_words.items():
        if len(forms) == 1 and len(bare) >= 3 and bare not in all_entries:
            _MACRON_MAP[bare] = list(forms)[0]

    return _MACRON_MAP


def apply_macron_correction(text: str) -> str:
    """Apply dictionary-guided macron corrections to TVL text.

    Replaces bare (macron-less) word forms with their canonical macronized
    spelling from the dictionary, but only for unambiguous cases where the
    bare form is not itself a valid dictionary word.
    """
    macron_map = _load_macron_map()
    if not macron_map:
        return text

    words = text.split()
    corrected = False
    for i, word in enumerate(words):
        # Strip punctuation for lookup, preserve it for output
        stripped = word.strip(".,;:!?()\"—‵""''")
        prefix = word[:len(word) - len(word.lstrip(".,;:!?()\"—‵""''"))]
        suffix = word[len(word) - len(word.rstrip(".,;:!?()\"—‵""''")):]  if word.rstrip(".,;:!?()\"—‵""''") != word else ""

        core = word[len(prefix):len(word) - len(suffix)] if suffix else word[len(prefix):]
        lookup = core.lower()

        if lookup in macron_map:
            replacement = macron_map[lookup]
            # Preserve original capitalization
            if core[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            if core.isupper():
                replacement = replacement.upper()
            words[i] = prefix + replacement + suffix
            corrected = True

    return " ".join(words) if corrected else text


def normalize_text(text: str, strip_pub_refs: bool = False) -> str:
    """Normalize text: NFC, strip invisible chars, fix entities, collapse whitespace.

    If strip_pub_refs is True, also:
    - Remove inline ¶ publication cross-references
    - Clean up scripture reference stubs (empty parens, —., Read instructions)
    - Fix missing spaces at sentence boundaries
    - Normalize glottal mark variants
    - Strip leading periods
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Strip invisible characters
    text = INVISIBLE_CHARS.sub("", text)

    # Replace HTML entities
    for entity, replacement in HTML_ENTITIES.items():
        text = text.replace(entity, replacement)

    if strip_pub_refs:
        # Strip inline publication cross-references
        text = INLINE_PUB_REF_RE.sub("", text)

        # Clean up scripture reference stubs (empty parens, instruction parens)
        text = SCRIPTURE_STUB_RE.sub("", text)

        # Strip trailing reference stubs (—. / —;read. / —Compare. / bare — etc.)
        # Apply twice: raw text can have chained stubs like —.—Read.
        text = TRAILING_REF_STUB_RE.sub("", text)
        text = TRAILING_REF_STUB_RE.sub("", text)

        # Strip inline ",NW." translation edition markers
        text = INLINE_NW_RE.sub("", text)

        # Clean up any remaining empty parentheses from stripped citations
        text = text.replace(" ()", "").replace("()", "")

        # Fix missing spaces at sentence boundaries
        text = MISSING_SPACE_RE.sub(r"\1 \2", text)

        # Normalize glottal mark variants
        text = text.translate(GLOTTAL_VARIANTS)

    # Normalize whitespace (collapse runs, strip)
    text = re.sub(r"\s+", " ", text).strip()

    # Strip leading periods (artifact of incomplete ref stripping)
    if strip_pub_refs:
        text = text.lstrip(". ")

    return text


# ── Metadata / boilerplate detection ──────────────────────────────────────────

# Picture captions: [Picture on page 5], [Picture Credit Line on page 5]
PICTURE_CAPTION_RE = re.compile(
    r"^\[(?:Picture|Pictures|Picture Credit Line|Pikitia)s?\s", re.IGNORECASE
)

# Box/chart/diagram/credit/blurb markers
BOX_CHART_RE = re.compile(
    r"^\[(?:Box|Chart|Diagram|Graph|Map|Footnote|Table|Credit Line|Blurb)\s?",
    re.IGNORECASE,
)

# TVL image page references: [Ata i te itulau e 5], [Fakamatalaga o te Ata...]
TVL_IMAGE_REF_RE = re.compile(
    r"^\[(?:Ata\s+i\s+te\s+itulau|Fakamatalaga\s+o\s+te\s+Ata)\s", re.IGNORECASE
)

# Photo credit lines
PHOTO_CREDIT_RE = re.compile(r"^(?:Photo|Image|Picture)\s+[Cc]redit", re.IGNORECASE)

# Copyright notices (also catch "Copyright (c)" and "Mountain High Maps(R) Copyright")
COPYRIGHT_RE = re.compile(
    r"(?:^©\s*\d{4}|Copyright\s*\(?[cC©]\)?\s*\d{4}|Maps?\(R\)\s*Copyright)",
    re.IGNORECASE,
)

# Page number references
PAGE_NUMBER_RE = re.compile(r"^\[(?:p|page|pp)\.\s*\d+", re.IGNORECASE)

# Chapter/section headers (both EN and TVL, including abbreviations)
HEADER_RE = re.compile(
    r"^(?:CHAPTER|CH(?:APT)?\.?\s+|MATAUPU\s+E|SECTION|PART|TE\s+VAEGA\s+E)\s*\d+",
    re.IGNORECASE,
)

# Footnote markers (standalone)
FOOTNOTE_MARKER_RE = re.compile(r"^[*†‡§]\s*$")

# Page markers
PAGE_MARKER_RE = re.compile(r"^(?:PAGECHAPTER|MATAUPUTE ITULAU)$")

# Song structure markers (standalone chorus/refrain labels — not actual lyrics)
SONG_MARKER_RE = re.compile(
    r"^\((?:CHORUS|TE\s+TALI|REFRAIN|BRIDGE)\)$", re.IGNORECASE
)

# Recurring section headers (exact matches, case-insensitive)
SECTION_HEADER_EXACT = {
    "also in this issue", "regular features",
    "how would you answer?", "theocratic ministry school review",
    "do you remember?", "review discussion",
    "bible reading:", "for fully formatted text, see publication",
    "(for fully formatted text, see publication)",
}

# See-also / cross-reference only: (See also.), (See paragraph 2), (Ke onoono...)
SEE_ALSO_RE = re.compile(
    r"^\(?(?:See\s+(?:also|paragraph|paragraphs|the\s+chart|the\s+box)|"
    r"Ke\s+onoono\s+ki)\b",
    re.IGNORECASE,
)

# Publication disclaimers
DISCLAIMER_RE = re.compile(
    r"(?:not\s+for\s+sale|"
    r"provided\s+as\s+part\s+of\s+a\s+worldwide|"
    r"supported\s+by\s+voluntary\s+donation)",
    re.IGNORECASE,
)

# Standalone "credit line" bracket
CREDIT_LINE_RE = re.compile(r"^\[Credit\s+Line\]$", re.IGNORECASE)


def is_metadata(text: str) -> bool:
    """Check if text is metadata/boilerplate rather than translatable content."""
    t = text.strip()
    if not t:
        return True
    if PICTURE_CAPTION_RE.match(t):
        return True
    if BOX_CHART_RE.match(t):
        return True
    if TVL_IMAGE_REF_RE.match(t):
        return True
    if PHOTO_CREDIT_RE.match(t):
        return True
    if COPYRIGHT_RE.search(t):
        return True
    if PAGE_NUMBER_RE.match(t):
        return True
    if HEADER_RE.match(t):
        return True
    if FOOTNOTE_MARKER_RE.match(t):
        return True
    if PAGE_MARKER_RE.match(t):
        return True
    if SONG_MARKER_RE.match(t):
        return True
    if CREDIT_LINE_RE.match(t):
        return True
    if t.lower() in SECTION_HEADER_EXACT:
        return True
    # Short see-also references (no substantive content)
    if SEE_ALSO_RE.match(t) and len(t) < 120:
        return True
    # Publication disclaimers (can be up to ~400 chars with combined boilerplate)
    if DISCLAIMER_RE.search(t) and len(t) < 400:
        return True
    return False


# ── Reference-only detection ─────────────────────────────────────────────────

# Pairs that are purely publication cross-references, not real translations.
# Match patterns like: "fymata. 5 ¶15- 28, pokisi ite itu. 61(30 minu.)"
# or "clchap. 18 ¶20-24,box on p. 188(30 min.)"
REF_ONLY_RE = re.compile(
    r"^(?:"
    r"(?:Napa|No\.?)\s+\d+:"                     # "No. 2:" / "Napa 3:"
    r"|[a-z]{1,4}\w*\.*\s*\d+"                   # any pub code + number
    r"|Bible\s+Reading:"                          # "Bible Reading:"
    r"|(?:\d+[\s,]+)*¶"                           # leading numbers + ¶
    r")",
    re.IGNORECASE,
)

# Meeting schedule items: "(30 min.)btchap. 21 ¶14-22" or "(5 minu.)be-E28 ¶3"
# Also: "Congregation Bible Study:(30 min.)btchap. 1 ¶16-21"
MEETING_SCHEDULE_RE = re.compile(
    r"^(?:"
    r"\(\d+\s*(?:min|minu)\.\)"                   # starts with duration
    r"|[^¶]{0,60}\(\d+\s*(?:min|minu)\.\)"        # prefix + duration
    r")",
    re.IGNORECASE,
)


def is_ref_only(text: str) -> bool:
    """Check if text is a publication cross-reference or meeting schedule item."""
    if "¶" not in text:
        return False
    t = text.strip()
    if len(t) > 120:
        return False
    if REF_ONLY_RE.match(t):
        return True
    if MEETING_SCHEDULE_RE.match(t):
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
        too_short          — EITHER side below min_chars
        too_long           — either side above max_chars
        bad_ratio          — length ratio outside bounds
        ref_only           — both sides are just publication cross-references
        boilerplate_en     — same EN text repeated too many times (2-pass)
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
    content_type = record.get("content_type", "")
    is_dict = content_type in ("word", "expression")

    # Too short — EITHER side below threshold (fixed: was AND, now OR)
    min_chars = profile.get("dict_min_chars", 1) if is_dict else profile["min_chars"]
    if tvl_chars < min_chars or en_chars < min_chars:
        return "too_short"

    # Too long
    if tvl_chars > profile["max_chars"] or en_chars > profile["max_chars"]:
        return "too_long"

    # Length ratio
    if en_chars > 0:
        ratio = tvl_chars / en_chars

        if content_type == "bible_verse":
            ratio_min = profile["bible_ratio_min"]
            ratio_max = profile["bible_ratio_max"]
        elif is_dict:
            ratio_min = profile.get("dict_ratio_min", profile["ratio_min"])
            ratio_max = profile.get("dict_ratio_max", profile["ratio_max"])
        else:
            ratio_min = profile["ratio_min"]
            ratio_max = profile["ratio_max"]

        if ratio < ratio_min or ratio > ratio_max:
            return "bad_ratio"

    # Song lyrics — line-level alignment is wrong (independently composed, not translations)
    pub_code = record.get("pub_code", "")
    if profile.get("strip_metadata") and pub_code in ("sjj", "snnw"):
        return "song_lyrics"

    # Reference-only pairs (publication cross-refs with ¶ markers)
    # Either side being ref-only is sufficient — these are meeting schedule items
    # or pub assignment references, not real translation pairs
    if profile.get("strip_ref_only") and (is_ref_only(tvl) or is_ref_only(en)):
        return "ref_only"

    # Boilerplate EN — checked in 2-pass mode (see run_pipeline)
    # Marker is set externally before classify_rejection is called
    if record.get("_boilerplate_en"):
        return "boilerplate_en"

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
    """Run the cleaning pipeline. Returns (accepted, rejected) lists.

    Uses a 2-pass approach:
      Pass 1: Normalize all text and count EN frequencies (for boilerplate detection).
      Pass 2: Apply dedup, quality filters, and boilerplate rejection.
    """
    strip_pub_refs = profile.get("strip_pub_refs", False)
    boilerplate_max = profile.get("boilerplate_max", 0)

    # ── Pass 1: Normalize, apply macron corrections, count EN frequencies ──
    en_freq = Counter()
    macron_corrections = 0
    for record in records:
        tvl_clean = normalize_text(record.get("tvl", ""), strip_pub_refs=strip_pub_refs)
        en_clean = normalize_text(record.get("en", ""), strip_pub_refs=strip_pub_refs)

        # Apply macron correction to non-dictionary TVL text
        content_type = record.get("content_type", "")
        if content_type not in ("word", "expression"):
            tvl_corrected = apply_macron_correction(tvl_clean)
            if tvl_corrected != tvl_clean:
                macron_corrections += 1
                tvl_clean = tvl_corrected

        record["_tvl_clean"] = tvl_clean
        record["_en_clean"] = en_clean
        en_freq[en_clean] += 1

    if macron_corrections:
        print(f"  Macron corrections applied to {macron_corrections:,} records")

    # Mark records whose EN text is repeated above threshold
    if boilerplate_max > 0:
        boilerplate_en_texts = {
            en_text for en_text, count in en_freq.items()
            if count > boilerplate_max
        }
        for record in records:
            en = record.get("_en_clean", "")
            if en in boilerplate_en_texts:
                record["_boilerplate_en"] = True

    # ── Pass 2: Dedup + quality filters ──
    accepted = []
    rejected = []

    rejection_counts = Counter()
    seen_ids = {}       # id -> index in accepted
    seen_hashes = {}    # content_hash -> id

    for record in records:
        rid = record.get("id", "")
        tvl_clean = record["_tvl_clean"]
        en_clean = record["_en_clean"]

        # ── Stage 1: Deduplicate by record ID ──
        if rid in seen_ids:
            record["_rejection_reason"] = "duplicate_id"
            rejected.append(record)
            rejection_counts["duplicate_id"] += 1
            continue
        seen_ids[rid] = len(accepted)

        # ── Stage 2: Deduplicate by content hash ──
        chash = content_hash(tvl_clean, en_clean)
        if chash in seen_hashes:
            record["_rejection_reason"] = "duplicate_content"
            rejected.append(record)
            rejection_counts["duplicate_content"] += 1
            continue
        seen_hashes[chash] = rid

        # ── Stage 3: Quality filters ──
        reason = classify_rejection(record, profile)
        if reason:
            record["_rejection_reason"] = reason
            rejected.append(record)
            rejection_counts[reason] += 1
            continue

        # ── Stage 4: Rebuild clean record (no internal fields) ──
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
            "category": record.get("category"),
            "subcategory": record.get("subcategory"),
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
