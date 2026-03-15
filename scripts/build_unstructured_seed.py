#!/usr/bin/env python3
"""Build reproducible unstructured seed artifacts for en↔tvl mining.

This script is intentionally conservative: it only emits high-confidence phrase
pairs for Stage A and keeps names/term candidates for Stage B separate.

Usage:
  uv run --extra ocr python scripts/build_unstructured_seed.py
  uv run --extra ocr python scripts/build_unstructured_seed.py --dry-run
  uv run --extra ocr python scripts/build_unstructured_seed.py --extract-ocr-terms
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Make repo package imports resilient when running from the scripts directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tv.common.io import write_json, write_jsonl
from tv.common.manifests import create_manifest, save_manifest

EXTRACTOR_VERSION = "unstructured-seed-v3"

TATOEBA_MIN_CHARS = 3
NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9`ʻ’’ʹ\-’]+(?:\s+[A-Z][A-Za-z0-9`ʻ’’ʹ\-’]+){0,3}\b")

# Part-of-speech markers used to detect new dictionary entries
_POS_MARKERS = r"(?:n\.|v\.|adj\.|adv\.|prep\.|conj\.|pron\.|int\.|excl\.|num\.|aux\.|art\.|abbr\.)"
# A line starts a new entry if first word is followed by a POS marker, source bracket, or numbered def
_ENTRY_START_RE = re.compile(
    r"^(?P<headword>[`\u02bb\-]?[a-zA-Z\u0101\u0113\u012b\u014d\u016b\u0100\u0112\u012a\u014c\u016a\u0300-\u0301`\u02bb’\u02b9\-]+(?:,\s*[`\u02bb\-]?[a-zA-Z\u0101\u0113\u012b\u014d\u016b\u0100\u0112\u012a\u014c\u016a`\u02bb’\u02b9\-]+)*)"
    r"\s+"
    r"(?P<body>"
    r"(?:"
    r"(?:\[[^\]]+\]\s+)?"  # optional [Bib.], [Eng.], etc.
    r"(?:\d+\.\s+)?"       # optional numbered definition
    rf"{_POS_MARKERS}"     # POS marker required
    r"|"
    r"\d+\.\s+"            # OR starts with numbered definition
    r"|"
    r"\[(?:Bib|Eng|Sam|Music|Niutao|Nanumea|Nanumaga|Funafuti|Vaitupu|Nui|Nukufetau|Nukulaelae|Niulakita)\.\]"  # source bracket (not [cf.])
    r")"
    r".*)",
    re.DOTALL,
)
# Proper name pair: "Abraham Apelaamo" (both capitalized, no POS marker)
_NAME_PAIR_RE = re.compile(
    r"^(?P<headword>[A-Z][a-zA-Z\u0101\u0113\u012b\u014d\u016b\-]+)\s+(?P<body>[A-Z][a-zA-Z\u0101\u0113\u012b\u014d\u016b\-]+)$"
)
# "(Bible name)" entry pattern: "Bethany (Bible name) Petania"
_BIBLE_NAME_ENTRY_RE = re.compile(
    r"^(?P<headword>[A-Z][a-zA-Z]+)\s+\(Bible\s+name\)\s+(?P<body>\S+.*)$"
)

# --- Body parsing helpers (v3) ---

# Regex to strip POS prefix from body text
_POS_STRIP_RE = re.compile(
    r"^(?:n\.|v\.|adj\.|adv\.|prep\.|conj\.|pron\.|int\.|excl\.|num\.|aux\.|art\.|abbr\.)\s*"
)

# Source/dialect bracket at start of body: [Bib.], [Eng.], [Sam.], [O], [N.B.], [Gr.], etc.
# Matches any bracket at position 0 that is ≤40 chars and doesn't look like a
# cross-reference (cf., pl., biblical refs) — those are handled by _REF_BRACKET_RE.
_SOURCE_BRACKET_RE = re.compile(
    r"^\[(?!cf\.|pl\.|Lit\.\s|Toke\.\s|It'?s\b)[^\]]{1,40}\]\s*"
)

# Reference brackets to remove: [cf. ...], [biblical refs], [pl. ...], [Lit. "..."], [Toke. ...]
_REF_BRACKET_RE = re.compile(
    r"\[(?:cf\.|pl\.|Lit\.\s|Toke\.\s|It'?s\b|[^\]]*\d+:\d+)[^\]]*\]"
)

# Parenthetical English glosses in en_tvl bodies: (concerning), (of speech), etc.
_ENGLISH_PAREN_RE = re.compile(r"\([^)]*\)")

# Detect TVL text via strong markers (macron vowels, stress-mark backtick, distinctive function words)
_TVL_STRONG_RE = re.compile(r"[\u0101\u0113\u012b\u014d\u016b\u0100\u0112\u012a\u014c\u016a]|`[a-zA-Z]")
_TVL_FUNC_WORD_RE = re.compile(
    r"\b(?:ne|ko|te|ki|se|ke|ei|mai|atu|toku|taku|tena|koe|koutou|ten[āa]|tenei|kon[āa]|pel[āa]|fua|eiloa|faka\w{3,})\b",
    re.IGNORECASE,
)

# Lines that signal appendix / non-dictionary content
_APPENDIX_MARKERS = {
    "an introduction to tuvaluan",
    "bible book abbreviations",
    "the calendar",
    "days of the week:",
    "months of the year:",
    "everyday expressions",
    "books of the bible",
}


def _looks_like_tvl(text: str) -> bool:
    """Heuristic: does this text look like Tuvaluan rather than English?

    Uses density-based thresholds: longer text needs more TVL markers to avoid
    false positives from English definitions that embed a single TVL word
    (e.g. "conversion (much more than `fuliga)").
    """
    strong = len(_TVL_STRONG_RE.findall(text))
    func = len(_TVL_FUNC_WORD_RE.findall(text))
    words = len(text.split())
    if words == 0:
        return False
    # For longer text (>10 words), require higher density of TVL markers
    # to avoid flagging English definitions with embedded TVL words
    if words > 10:
        if strong >= 2:
            return True
        if func >= 3:
            return True
        if strong >= 1 and func >= 2:
            return True
        return False
    # For shorter text, 1 strong marker or 2 function words is enough
    if strong >= 1:
        return True
    if func >= 2:
        return True
    if func >= 1 and words <= 4:
        return True
    return False


def _strip_ref_brackets(text: str) -> str:
    """Remove cross-reference and biblical citation brackets."""
    return _REF_BRACKET_RE.sub("", text)


def _split_numbered_senses(text: str) -> list[str]:
    """Split '1. def one 2. def two 3. def three' into individual senses."""
    parts = re.split(r"(?:^|\s)(\d+)\.\s", text)
    if len(parts) <= 1:
        return [text.strip()] if text.strip() else []
    senses = []
    for i in range(2, len(parts), 2):
        sense = parts[i].strip()
        if sense:
            senses.append(sense)
    return senses if senses else [text.strip()]


def _extract_definition_text(text: str) -> str:
    """Extract the English definition, removing TVL example sentences and their translations."""
    parts = re.split(r"(?<=[.!?:;])\s+", text)
    definition_parts: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _looks_like_tvl(part):
            break
        definition_parts.append(part)
    result = " ".join(definition_parts)
    result = result.rstrip(".,;: ")
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _clean_en_tvl_body(body: str) -> str:
    """Clean an en_tvl body: strip POS, English parens, ref brackets."""
    cleaned = body.strip()
    # Strip leading POS markers iteratively — handles compound (v.adj.) and sequential (n. v.)
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _POS_STRIP_RE.sub("", cleaned).strip()
    # Strip POS markers that appear after semicolons in multi-POS entries
    # e.g. "fakaūsu; adj. ūsu" → "fakaūsu; ūsu"
    cleaned = re.sub(
        r";\s*(?:n\.|v\.|adj\.|adv\.|prep\.|conj\.|pron\.|int\.|excl\.|num\.|aux\.|art\.|abbr\.)\s*",
        "; ",
        cleaned,
    )
    # Remove English parentheticals
    cleaned = _ENGLISH_PAREN_RE.sub("", cleaned)
    # Re-strip POS after paren removal — e.g. "n. (bring to a halt) v. fakatu" → "v. fakatu" → "fakatu"
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _POS_STRIP_RE.sub("", cleaned).strip()
    cleaned = _strip_ref_brackets(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(".,;: ")
    return cleaned


def _parse_tvl_en_body(body: str) -> list[str]:
    """Parse a tvl_en body into a list of clean English definitions.

    Handles source brackets, numbered senses, POS markers, example
    sentences, and cross-references.
    """
    cleaned = _SOURCE_BRACKET_RE.sub("", body).strip()
    cleaned = _strip_ref_brackets(cleaned)

    senses = _split_numbered_senses(cleaned)
    definitions: list[str] = []
    for sense in senses:
        sense = sense.strip()  # strip whitespace before POS removal (e.g. after bracket removal)
        # Strip POS markers and source brackets iteratively — handles:
        # - compound POS: adj./v.  - sequential POS: adj. v.
        # - POS before bracket: n. [O] armpit  - bracket before POS: [Eng.] n. Easter
        prev = None
        while prev != sense:
            prev = sense
            sense = re.sub(
                r"^/?" + _POS_MARKERS + r"\s*", "", sense
            ).strip()
            sense = _SOURCE_BRACKET_RE.sub("", sense).strip()
        if not sense:
            continue
        defn = _extract_definition_text(sense)
        if defn and len(defn) >= 2:
            definitions.append(defn)
    return definitions


FLORA_FAUNA_KEYWORDS = {
    "plant",
    "tree",
    "flower",
    "fruit",
    "bird",
    "fish",
    "turtle",
    "crab",
    "dog",
    "cat",
    "insect",
    "animal",
    "flora",
    "fauna",
    "pandanus",
    "sea",
    "reef",
    "beach",
    "ocean",
    "island",
}


class SectionTracker:
    """Track which section (tvl_en or en_tvl) we're in.

    Only triggers on short standalone section header lines, not inline mentions.
    Only returns a value when the section CHANGES (page headers that repeat
    the current section are suppressed).
    Ignores switches in the first ~400 lines (front matter / preface).
    """
    # The actual dictionary content starts after front matter (~400 lines
    # in the non-layout extraction). Before that, title page / TOC / preface
    # contain "Tuvaluan-English" / "English-Tuvaluan" strings that are NOT
    # section boundaries.
    MIN_LINE_FOR_SECTION = 400

    def __init__(self) -> None:
        self.current: str | None = None

    def update(self, line: str, line_no: int = 0) -> str | None:
        if line_no < self.MIN_LINE_FOR_SECTION:
            return None

        text = _normalize(line).lower().replace("\u2014", "-")  # em-dash
        # Only match short standalone lines (< 40 chars) to avoid preface sentences
        if len(text) > 40:
            return None

        new_section: str | None = None
        if text.strip() in ("tuvaluan-english", "tuvaluan-english dictionary"):
            new_section = "tvl_en"
        elif text.strip() in ("english-tuvaluan", "english-tuvaluan dictionary"):
            new_section = "en_tvl"

        # Only fire when section actually changes
        if new_section and new_section != self.current:
            self.current = new_section
            return self.current
        return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\f", " ").replace("\r", " ")).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unstructured en↔tvl seed artifacts")
    parser.add_argument(
        "--asset-dir",
        default="unstruct_lang_data",
        help="Directory containing unstructured assets",
    )
    parser.add_argument(
        "--stage-a-output",
        default="data/external/stage_a_seed",
        help="Output directory for Stage A seed candidates",
    )
    parser.add_argument(
        "--stage-b-output",
        default="data/external/stage_b_seed",
        help="Output directory for Stage B term candidates",
    )
    parser.add_argument(
        "--tatoeba-file",
        default="Tatoeba-v2023-04-12-en&tvl.tsv",
        help="Tatoeba file name under --asset-dir",
    )
    parser.add_argument(
        "--dictionary-pdf",
        default="DICTIONARY_Tuv_Palagi.pdf",
        help="Dictionary PDF name under --asset-dir",
    )
    parser.add_argument(
        "--dictionary-text",
        default="data/external/raw/DICTIONARY_Tuv_Palagi.txt",
        help="Extracted dictionary text path (generated from pdftotext if missing)",
    )
    parser.add_argument(
        "--ocr-dir",
        default="data/external/ocr_scans",
        help="Directory with OCR page JSONL files",
    )
    parser.add_argument(
        "--max-dict-entries",
        type=int,
        default=None,
        help="Optional hard cap per dictionary section for debugging",
    )
    parser.add_argument(
        "--ocr-conf-min",
        type=float,
        default=65.0,
        help="Minimum OCR page mean confidence for term mining",
    )
    parser.add_argument(
        "--ocr-term-min-freq",
        type=int,
        default=2,
        help="Minimum frequency threshold for OCR term candidates",
    )
    parser.add_argument(
        "--extract-ocr-terms",
        action="store_true",
        help="Mine conservative name/term candidates from OCR output",
    )
    parser.add_argument(
        "--extract-terms",
        action="store_true",
        help="Emit conservative dictionary-derived flora/fauna/name candidates",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run label used in manifest metadata only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print stats without writing artifacts",
    )
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=20,
        help="When --dry-run, print sample candidate rows from each source",
    )
    return parser.parse_args()


def _run_pdftotext(pdf_path: Path, out_path: Path, *, layout: bool = False) -> None:
    """Generate plain-text dictionary input via pdftotext."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd.extend([str(pdf_path), str(out_path)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"pdftotext failed for {pdf_path}: {result.stderr.strip() or result.stdout.strip()}"
        )


def _looks_like_noise_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return True
    if text in {"TUVALUAN", "ENGLISH", "DICTIONARY", "TUVALUAN—ENGLISH", "ENGLISH—TUVALUAN"}:
        return True
    if re.fullmatch(r"[\-]+", text):
        return True
    if re.fullmatch(r"\d+", text):
        return True
    if re.search(r"^Page\b|\bTable of Contents\b", text, re.IGNORECASE):
        return True
    if text.startswith("Eng") and "Dictionary" in text:
        return True
    # Section headers / letter dividers (single uppercase letter)
    if re.fullmatch(r"[A-Z]", text):
        return True
    # "Tuvaluan-English" or "English-Tuvaluan" section headers
    if re.fullmatch(r"(?:Tuvaluan|English)[\-—](?:Tuvaluan|English)", text):
        return True
    # Appendix / non-dictionary content boundaries
    low = text.lower()
    if low in _APPENDIX_MARKERS:
        return True
    if text.startswith("An Introduction to Tuvaluan"):
        return True
    if text.startswith("Bible Book Abbreviation"):
        return True
    # Appendix title fragments (appear as separate uppercase lines)
    if text in {"AN", "INTRODUCTION", "TO TUVALUAN", "AN INTRODUCTION"}:
        return True
    # Bible abbreviation table lines: "Genesis: Gen.", "Exodus: Ex.", etc.
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s+\S.*", text):
        return True
    # Digit-prefixed Bible abbreviation lines: "1 Samuel: 1 Sam.", "2 Kings: 1 Ki."
    if re.fullmatch(r"\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s+\S.*", text):
        return True
    # Digit-prefixed standalone book names: "1 Samuelu", "2 Tupu", "3 Ioane"
    if re.fullmatch(r"\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text):
        return True
    return False


def _try_parse_entry_start(line: str) -> tuple[str, str] | None:
    """Try to parse line as the start of a new dictionary entry.

    Returns (headword, body_start) if this line begins a new entry, else None.
    Uses POS markers, source brackets, and numbered definitions to identify entries.
    """
    text = line.strip()
    if not text or _looks_like_noise_line(text):
        return None

    # Try main pattern: headword + POS/bracket/number
    m = _ENTRY_START_RE.match(text)
    if m:
        headword = m.group("headword").strip().rstrip(",")
        body = m.group("body").strip()
        # Headword must contain at least one letter (reject "----------")
        if headword and body and len(headword) <= 60 and any(c.isalpha() for c in headword):
            return headword, body

    # Try name pair pattern: "Abraham Apelaamo"
    m = _NAME_PAIR_RE.match(text)
    if m:
        return m.group("headword"), m.group("body")

    # Try (Bible name) pattern: "Bethany (Bible name) Petania"
    m = _BIBLE_NAME_ENTRY_RE.match(text)
    if m:
        return m.group("headword"), m.group("body")

    return None


def _is_name_like(text: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z`ʻ‘’ʹ]+", text)]
    if not words:
        return False
    if len(words) > 3:
        return False
    return all(w.istitle() for w in words)


def _is_flora_fauna_like(text: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return bool(tokens & FLORA_FAUNA_KEYWORDS)


def extract_tatoeba(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats: dict[str, int] = Counter()
    stats["lines_total"] = 0

    if not path.exists():
        stats["missing_file"] = 1
        return rows, rejected, dict(stats)

    source_name = path.as_posix()
    with path.open(encoding="utf-8", errors="replace") as f:
        for idx, raw in enumerate(f, start=1):
            stats["lines_total"] += 1
            line = raw.strip("\n")
            if not line:
                stats["empty"] += 1
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                stats["invalid_columns"] += 1
                rejected.append(
                    {
                        "source_file": source_name,
                        "source_row": idx,
                        "reason": "invalid_tsv_columns",
                        "raw": line[:200],
                    }
                )
                continue

            en = _normalize(parts[0])
            tvl = _normalize(parts[1])

            # Fix known OCR typo in Tatoeba source
            tvl = tvl.replace("tamaflne", "tamafine")

            if len(en) < TATOEBA_MIN_CHARS or len(tvl) < TATOEBA_MIN_CHARS:
                stats["too_short"] += 1
                rejected.append(
                    {
                        "source_file": source_name,
                        "source_row": idx,
                        "reason": "too_short",
                        "en": en,
                        "tvl": tvl,
                    }
                )
                continue

            tvl_chars = len(tvl)
            en_chars = len(en)
            rows.append(
                {
                    "id": f"unstruct:tatoeba:{idx:04d}",
                    "tvl": tvl,
                    "en": en,
                    "content_type": "translation_phrase",
                    "domain": "tatoeba",
                    "alignment_method": "tatoeba_pair",
                    "alignment_confidence": 1.0,
                    "doc_id": None,
                    "source_url_tvl": source_name,
                    "source_url_en": source_name,
                    "book_num": None,
                    "chapter": None,
                    "verse": None,
                    "date": None,
                    "pub_code": "unstruct_tatoeba",
                    "tvl_chars": tvl_chars,
                    "en_chars": en_chars,
                    "length_ratio": tvl_chars / en_chars if en_chars else 0,
                    "metadata": {
                        "source_file": path.name,
                        "source_row": idx,
                        "source_table_row": idx,
                        "parse_mode": "tatoeba_seed",
                        "ocr_conf_mean": None,
                        "ocr_conf_p50": None,
                        "text_confidence": "high",
                        "extractor_version": EXTRACTOR_VERSION,
                    },
                }
            )

    return rows, rejected, dict(stats)


def extract_dictionary(
    asset_dir: Path,
    dictionary_pdf: Path,
    dictionary_text: Path,
    max_entries: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Parse dictionary PDF into headword→definition pairs.

    Uses pdftotext WITHOUT -layout to get sequential reading order
    (left column then right column per page), avoiding the column-merge
    problem that produced garbage entries in v1.

    Stream-based: accumulates continuation lines until the next headword.
    """
    # Re-extract without -layout for clean reading order
    nolayout_text = dictionary_text.parent / (dictionary_text.stem + "_nolayout.txt")
    if not nolayout_text.exists():
        if not dictionary_pdf.exists():
            raise FileNotFoundError(f"Missing dictionary PDF: {dictionary_pdf}")
        nolayout_text.parent.mkdir(parents=True, exist_ok=True)
        _run_pdftotext(dictionary_pdf, nolayout_text, layout=False)

    # Use split('\n') not splitlines() — the latter splits on \f (form feed)
    # creating 509 phantom empty lines that shift all line numbers.
    lines = nolayout_text.read_text(encoding="utf-8", errors="replace").split("\n")
    section = SectionTracker()
    current_section: str | None = None

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats: dict[str, int] = Counter()

    counters = {"tvl_en": 0, "en_tvl": 0}

    # Stream state: accumulate entry text
    current_headword: str | None = None
    current_body_parts: list[str] = []
    current_start_line: int = 0

    def _append_row(
        row_id: str,
        tvl: str,
        en: str,
        conf: float,
        parse_mode: str = "dictionary_stream_v3",
    ) -> None:
        """Append a single cleaned row to the output."""
        tvl_n = _normalize(tvl)
        en_n = _normalize(en)
        tvl_chars = len(tvl_n)
        en_chars = len(en_n)

        if tvl_chars < 3 or en_chars < 3:
            stats["too_short_cleaned"] += 1
            return

        rows.append({
            "id": row_id,
            "tvl": tvl_n,
            "en": en_n,
            "content_type": "translation_phrase",
            "domain": "dictionary",
            "alignment_method": "dictionary_entry",
            "alignment_confidence": conf,
            "doc_id": None,
            "source_url_tvl": dictionary_pdf.as_posix(),
            "source_url_en": dictionary_pdf.as_posix(),
            "book_num": None,
            "chapter": None,
            "verse": None,
            "date": None,
            "pub_code": "unstruct_dictionary",
            "tvl_chars": tvl_chars,
            "en_chars": en_chars,
            "length_ratio": tvl_chars / en_chars if en_chars else 0,
            "metadata": {
                "source_file": nolayout_text.name,
                "source_row": current_start_line,
                "source_section": current_section,
                "source_cell": None,
                "source_pdf_page": None,
                "parse_mode": parse_mode,
                "ocr_conf_mean": None,
                "ocr_conf_p50": None,
                "text_confidence": "medium",
                "extractor_version": EXTRACTOR_VERSION,
            },
        })

    def _emit_entry() -> None:
        """Emit the accumulated entry as one or more cleaned rows."""
        nonlocal current_headword, current_body_parts, current_start_line
        if current_headword is None or current_section is None:
            return

        body = _normalize(" ".join(current_body_parts))
        lemma = current_headword

        if not body or len(body) < 3:
            stats["too_short"] += 1
            rejected.append({
                "source_file": nolayout_text.name,
                "source_row": current_start_line,
                "source_section": current_section,
                "reason": "too_short",
                "lemma": lemma,
                "body": body,
            })
            return

        if any(tok in body.lower() for tok in {"see also", "index", "publisher"}):
            stats["noise_body"] += 1
            return

        # Reject excessively long bodies (appendix contamination)
        if len(body) > 2000:
            stats["body_too_long"] += 1
            rejected.append({
                "source_file": nolayout_text.name,
                "source_row": current_start_line,
                "source_section": current_section,
                "reason": "body_too_long",
                "lemma": lemma,
                "body": body[:200],
            })
            return

        stats["entries_parsed"] += 1

        if current_section == "en_tvl":
            # --- English headword → Tuvaluan translations ---
            cleaned_tvl = _clean_en_tvl_body(body)
            if not cleaned_tvl or len(cleaned_tvl) < 2:
                stats["empty_after_clean"] += 1
                return

            counters["en_tvl"] += 1
            if max_entries and counters["en_tvl"] > max_entries:
                return

            # Confidence: name pairs get lower score
            is_name = _is_name_like(lemma) and _is_name_like(cleaned_tvl)
            conf = 0.65 if is_name else 0.80

            _append_row(
                row_id=f"unstruct:dict_en_tvl:{counters['en_tvl']:06d}",
                tvl=cleaned_tvl,
                en=lemma,
                conf=conf,
            )
        else:
            # --- Tuvaluan headword → English definitions ---
            definitions = _parse_tvl_en_body(body)
            if not definitions:
                # Fallback: strip brackets and POS iteratively
                fallback = _SOURCE_BRACKET_RE.sub("", body).strip()
                fallback = _strip_ref_brackets(fallback).strip()
                prev = None
                while prev != fallback:
                    prev = fallback
                    fallback = re.sub(
                        r"^/?" + _POS_MARKERS + r"\s*", "", fallback
                    ).strip()
                    fallback = _SOURCE_BRACKET_RE.sub("", fallback).strip()
                fallback = fallback.rstrip(".,;: ")
                if fallback and len(fallback) >= 2:
                    definitions = [fallback]
                else:
                    stats["empty_after_clean"] += 1
                    return

            for defn in definitions:
                counters["tvl_en"] += 1
                if max_entries and counters["tvl_en"] > max_entries:
                    return

                is_name = _is_name_like(lemma) and _is_name_like(defn)
                conf = 0.65 if is_name else 0.85

                _append_row(
                    row_id=f"unstruct:dict_tvl_en:{counters['tvl_en']:06d}",
                    tvl=lemma,
                    en=defn,
                    conf=conf,
                )

    in_appendix = False

    for line_no, line in enumerate(lines, start=1):
        # Detect appendix boundaries — stop parsing dictionary entries
        line_clean = line.replace("\f", " ").strip()
        if (
            line_clean.startswith("An Introduction to Tuvaluan")
            or line_clean == "Bible Book Abbreviations"
            or (current_section == "tvl_en" and re.match(r"^Genesis:\s+Gen\b", line_clean))
        ):
            if current_section is not None:
                _emit_entry()
                current_headword = None
                current_body_parts = []
                in_appendix = True
                stats["appendix_boundary_hit"] += 1

        if in_appendix:
            # Check if we've switched back to a dictionary section
            switched = section.update(line, line_no=line_no)
            if switched:
                in_appendix = False
                current_section = switched
                stats[f"section_switch_{current_section}"] += 1
            continue

        switched = section.update(line, line_no=line_no)
        if switched:
            _emit_entry()
            current_headword = None
            current_body_parts = []
            current_section = switched
            stats[f"section_switch_{current_section}"] += 1
            continue

        if current_section is None:
            continue

        if _looks_like_noise_line(line):
            # Letter dividers (single uppercase letter) mark entry boundaries —
            # emit the current entry so it doesn't absorb text from the next letter.
            text_stripped = line.strip()
            if re.fullmatch(r"[A-Z]", text_stripped):
                _emit_entry()
                current_headword = None
                current_body_parts = []
            continue

        # Try to detect a new entry start
        parsed = _try_parse_entry_start(line)
        if parsed:
            # Emit previous entry
            _emit_entry()
            # Start new entry
            current_headword = parsed[0]
            current_body_parts = [parsed[1]]
            current_start_line = line_no
        elif current_headword is not None:
            # Continuation of current entry
            text = line.strip()
            if text:
                current_body_parts.append(text)

    # Emit last entry
    _emit_entry()

    stats["entries_tvl_en"] = counters["tvl_en"]
    stats["entries_en_tvl"] = counters["en_tvl"]
    return rows, rejected, dict(stats)


def _iter_ocr_pages(ocr_dir: Path) -> list[dict[str, Any]]:
    page_rows: list[dict[str, Any]] = []
    for jsonl_path in sorted(ocr_dir.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                row["_source_file"] = jsonl_path.name
                page_rows.append(row)
    return page_rows


def extract_ocr_terms(
    ocr_dir: Path,
    conf_min: float,
    min_freq: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not ocr_dir.exists():
        return [], {"ocr_dir_missing": 1}

    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = Counter()
    candidate_counts: dict[tuple[str, str], dict[str, Any]] = {}

    for row in _iter_ocr_pages(ocr_dir):
        stats["ocr_pages_scanned"] += 1
        if row.get("status") != "ok":
            continue

        conf = float(row.get("conf_mean") or 0.0)
        if conf < conf_min:
            stats["ocr_pages_low_conf"] += 1
            continue

        raw_text = _normalize(str(row.get("text", "")))
        if len(raw_text) < 40:
            continue

        source_pdf = row.get("pdf", "")
        page = row.get("page")

        for match in NAME_PATTERN.finditer(raw_text):
            term = match.group(0).strip()
            if len(term) < 3:
                continue
            lowered = term.lower()
            if lowered in {"the", "and", "for", "from", "with", "this", "that", "they", "there"}:
                continue

            if term[0].islower():
                continue

            if _is_flora_fauna_like(lowered):
                term_type = "flora_fauna"
            elif _is_name_like(term):
                term_type = "person_place"
            else:
                term_type = "other"

            key = (term, term_type)
            bucket = candidate_counts.setdefault(
                key,
                {
                    "count": 0,
                    "max_conf": 0.0,
                    "first_source": source_pdf,
                    "first_page": page,
                },
            )
            bucket["count"] += 1
            if conf > bucket["max_conf"]:
                bucket["max_conf"] = conf

    for (term, term_type), data in candidate_counts.items():
        if data["count"] < min_freq:
            continue
        rows.append(
            {
                "id": f"unstruct:ocr_term:{len(rows):05d}",
                "term": term,
                "term_type": term_type,
                "content_type": "unstructured_term",
                "domain": "news_term_mining",
                "alignment_method": "ocr_term_candidate",
                "alignment_confidence": min(0.95, data["max_conf"] / 100),
                "source": data["first_source"],
                "source_page": data["first_page"],
                "evidence_count": data["count"],
                "evidence_max_conf": data["max_conf"],
                "metadata": {
                    "source_file": data["first_source"],
                    "source_page_hint": data["first_page"],
                    "candidate_count": data["count"],
                    "extractor_version": EXTRACTOR_VERSION,
                },
            }
        )

    stats["ocr_terms_kept"] = len(rows)
    stats["ocr_terms_candidate_cells"] = len(candidate_counts)
    return rows, dict(stats)


def _build_dictionary_term_rows(dict_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = set()

    for row in dict_rows:
        en = row.get("en", "")
        tvl = row.get("tvl", "")
        section = row.get("metadata", {}).get("source_section")

        term_type = None
        term = None

        if _is_name_like(en):
            term_type = "person_place"
            term = en
        elif _is_flora_fauna_like(en):
            term_type = "flora_fauna"
            term = en
        elif _is_flora_fauna_like(tvl):
            term_type = "flora_fauna"
            term = tvl

        if not term_type:
            continue

        key = (term_type, term.lower())
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "id": f"unstruct:dict_term:{len(rows):05d}",
                "term": term,
                "term_type": term_type,
                "source_section": section,
                "alignment_method": "dictionary_term_candidate",
                "alignment_confidence": row.get("alignment_confidence", 0.7),
                "source_url_tvl": row.get("source_url_tvl"),
                "source_url_en": row.get("source_url_en"),
                "metadata": {
                    "source_file": row.get("metadata", {}).get("source_file"),
                    "source_row": row.get("metadata", {}).get("source_row"),
                    "alignment_confidence_source": row.get("alignment_confidence", 0.7),
                    "extractor_version": EXTRACTOR_VERSION,
                },
            }
        )

    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    dropped = 0

    for row in rows:
        key = (_normalize(row["tvl"]), _normalize(row["en"]))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(row)

    return deduped, dropped


def main() -> None:
    args = _parse_args()
    asset_dir = Path(args.asset_dir).expanduser()
    stage_a_output = Path(args.stage_a_output).expanduser()
    stage_b_output = Path(args.stage_b_output).expanduser()

    tatoeba_path = asset_dir / args.tatoeba_file
    dictionary_pdf = asset_dir / args.dictionary_pdf
    dictionary_text = Path(args.dictionary_text).expanduser()

    seed_rows: dict[str, list[dict[str, Any]]] = {
        "unstruct_tatoeba": [],
        "unstruct_dictionary_tvl_en": [],
        "unstruct_dictionary_en_tvl": [],
    }
    stage_a_rejected: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    # 1) Tatoeba
    print("[1/3] building Tatoeba seed")
    tatoeba_rows, tatoeba_rej, tatoeba_stats = extract_tatoeba(tatoeba_path)
    seed_rows["unstruct_tatoeba"] = tatoeba_rows
    stage_a_rejected.extend(tatoeba_rej)
    summary["tatoeba"] = {
        "rows": len(tatoeba_rows),
        "rejected": len(tatoeba_rej),
        **tatoeba_stats,
    }

    # 2) Dictionary parse
    print("[2/3] parsing dictionary text")
    dict_rows, dict_rej, dict_stats = extract_dictionary(
        asset_dir=asset_dir,
        dictionary_pdf=dictionary_pdf,
        dictionary_text=dictionary_text,
        max_entries=args.max_dict_entries,
    )

    seed_rows["unstruct_dictionary_tvl_en"] = [r for r in dict_rows if r["metadata"]["source_section"] == "tvl_en"]
    seed_rows["unstruct_dictionary_en_tvl"] = [r for r in dict_rows if r["metadata"]["source_section"] == "en_tvl"]
    stage_a_rejected.extend(dict_rej)

    summary["dictionary"] = {
        "rows": len(dict_rows),
        "rejected": len(dict_rej),
        **dict_stats,
    }

    # 3) Optional OCR term mining
    ocr_rows: list[dict[str, Any]] = []
    if args.extract_ocr_terms:
        print("[3/3] mining OCR term candidates")
        ocr_rows, ocr_stats = extract_ocr_terms(
            ocr_dir=Path(args.ocr_dir).expanduser(),
            conf_min=args.ocr_conf_min,
            min_freq=args.ocr_term_min_freq,
        )
        summary["ocr_terms"] = ocr_stats

    # Optional dictionary term extraction for Stage B name / flora/fauna support
    dict_term_rows: list[dict[str, Any]] = []
    if args.extract_terms:
        dict_term_rows = _build_dictionary_term_rows(dict_rows)
        summary["dictionary_terms"] = {
            "rows": len(dict_term_rows),
        }

    # Deduplicate Stage A rows by normalized pair
    all_stage_a = []
    for key in seed_rows:
        all_stage_a.extend(seed_rows[key])
    deduped_stage_a, dropped_pairs = _dedupe_rows(all_stage_a)

    final_stage_a = defaultdict(list)
    for row in deduped_stage_a:
        sec = row["metadata"].get("source_section", "misc")
        if sec == "tvl_en":
            final_stage_a["unstruct_dictionary_tvl_en"].append(row)
        elif sec == "en_tvl":
            final_stage_a["unstruct_dictionary_en_tvl"].append(row)
        else:
            final_stage_a["unstruct_tatoeba"].append(row)

    for source in list(seed_rows.keys()):
        if source not in final_stage_a:
            final_stage_a[source] = seed_rows[source]

    stage_a_summary = {
        "rows_input": len(all_stage_a),
        "rows_deduped": len(deduped_stage_a),
        "rows_dedropped_pair": dropped_pairs,
    }

    total_stage_b = len(ocr_rows) + len(dict_term_rows)

    print("[summary]")
    print(json.dumps({
        "stage_a": stage_a_summary,
        "sources": summary,
        "dict_term_rows": len(dict_term_rows),
        "ocr_term_rows": len(ocr_rows),
        "total_stage_b_rows": total_stage_b,
    }, indent=2))

    if args.dry_run:
        for name, rows in final_stage_a.items():
            print(f"\n{name}: {len(rows)} rows (sample {args.dry_run_limit})")
            for row in rows[: args.dry_run_limit]:
                print(f"  - {row['id']} :: {row['tvl']} || {row['en']}")
        if ocr_rows:
            print(f"\nocr_terms: {len(ocr_rows)} rows")
            for row in ocr_rows[: args.dry_run_limit]:
                print(f"  - {row['id']} :: {row['term']} :: {row['term_type']}")
        if dict_term_rows:
            print(f"\ndictionary_terms: {len(dict_term_rows)} rows")
            for row in dict_term_rows[: args.dry_run_limit]:
                print(f"  - {row['id']} :: {row['term']} :: {row['term_type']}")
        return

    # Write Stage A outputs
    stage_a_output.mkdir(parents=True, exist_ok=True)
    for name, rows in final_stage_a.items():
        out_path = stage_a_output / f"{name}.jsonl"
        write_jsonl(out_path, rows)
    rejected_path = stage_a_output / "rejected.jsonl"
    write_jsonl(rejected_path, stage_a_rejected)

    stage_a_stats = {
        "stage": "unstructured_stage_a_seed",
        "version": EXTRACTOR_VERSION,
        "rows": {
            "tatoeba": len(final_stage_a["unstruct_tatoeba"]),
            "dictionary_tvl_en": len(final_stage_a["unstruct_dictionary_tvl_en"]),
            "dictionary_en_tvl": len(final_stage_a["unstruct_dictionary_en_tvl"]),
            "deduped_total": len(deduped_stage_a),
            "rejected": len(stage_a_rejected),
            "dropped_duplicate_pairs": dropped_pairs,
        },
        "source_summary": summary,
        "run_name": args.run_name,
    }

    write_json(stage_a_output / "stats.json", stage_a_stats)

    manifest_output_files: dict[str, str] = {
        "unstruct_tatoeba": str(stage_a_output / "unstruct_tatoeba.jsonl"),
        "unstruct_dictionary_tvl_en": str(
            stage_a_output / "unstruct_dictionary_tvl_en.jsonl"
        ),
        "unstruct_dictionary_en_tvl": str(
            stage_a_output / "unstruct_dictionary_en_tvl.jsonl"
        ),
        "rejected": str(rejected_path),
    }
    data_files = [
        stage_a_output / "unstruct_tatoeba.jsonl",
        stage_a_output / "unstruct_dictionary_tvl_en.jsonl",
        stage_a_output / "unstruct_dictionary_en_tvl.jsonl",
        rejected_path,
        stage_a_output / "stats.json",
    ]

    manifest = create_manifest(
        stage="stage_a_seed_build",
        config={
            "asset_dir": str(asset_dir),
            "tatoeba_file": tatoeba_path.name,
            "dictionary_pdf": str(dictionary_pdf),
            "dictionary_text": str(dictionary_text),
            "extract_terms": args.extract_terms,
            "extract_ocr_terms": args.extract_ocr_terms,
            "max_dict_entries": args.max_dict_entries,
            "run_name": args.run_name,
        },
        extra={
            "output_summary": stage_a_stats,
            "output_files": manifest_output_files,
        },
        data_files=data_files,
    )
    save_manifest(manifest, stage_a_output / "manifest.json")

    # Write Stage B term outputs (if requested)
    if args.extract_ocr_terms or args.extract_terms:
        stage_b_output.mkdir(parents=True, exist_ok=True)
        out_rows = []
        if args.extract_ocr_terms:
            out_rows.extend(ocr_rows)
            write_jsonl(stage_b_output / "unstruct_ocr_terms.jsonl", ocr_rows)
        if args.extract_terms:
            out_rows.extend(dict_term_rows)
            write_jsonl(stage_b_output / "unstruct_dictionary_terms.jsonl", dict_term_rows)

        stage_b_stats = {
            "stage": "unstructured_stage_b_terms",
            "version": EXTRACTOR_VERSION,
            "rows": {
                "ocr_terms": len(ocr_rows),
                "dictionary_terms": len(dict_term_rows),
                "total": total_stage_b,
            },
            "source_summary": summary,
            "run_name": args.run_name,
        }
        write_json(stage_b_output / "stats.json", stage_b_stats)
        manifest_output_files: dict[str, Path] = {}
        if args.extract_ocr_terms:
            manifest_output_files["ocr_terms"] = stage_b_output / "unstruct_ocr_terms.jsonl"
        if args.extract_terms:
            manifest_output_files["dictionary_terms"] = (
                stage_b_output / "unstruct_dictionary_terms.jsonl"
            )

        manifest = create_manifest(
            stage="stage_b_seed_build",
            config={
                "ocr_dir": str(Path(args.ocr_dir).expanduser()),
                "ocr_conf_min": args.ocr_conf_min,
                "ocr_term_min_freq": args.ocr_term_min_freq,
                "extract_ocr_terms": args.extract_ocr_terms,
                "extract_terms": args.extract_terms,
                "run_name": args.run_name,
            },
            extra={
                "output_summary": stage_b_stats,
                "output_files": {k: str(v) for k, v in manifest_output_files.items()},
            },
            data_files=[*manifest_output_files.values(), stage_b_output / "stats.json"],
        )
        save_manifest(manifest, stage_b_output / "manifest.json")

    print("\nWrote Stage A seed artifacts:", stage_a_output)
    if args.extract_ocr_terms or args.extract_terms:
        print("Wrote Stage B term artifacts:", stage_b_output)


if __name__ == "__main__":
    main()
