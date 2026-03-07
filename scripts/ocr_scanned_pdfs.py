#!/usr/bin/env python3
"""OCR scanned PDFs into recoverable text and confidence-aware page JSONL.

Usage:
  uv run --extra ocr python scripts/ocr_scanned_pdfs.py \
    --inputs unstruct_lang_data/The_magical_garlands_of_Nukufetau.pdf \
             unstruct_lang_data/Tuvalu_News_Sheets_66-99.pdf \
    --output-dir data/external/ocr_scans
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytesseract
from pytesseract import TesseractNotFoundError
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


DEFAULT_PSMS = (6, 11, 4, 3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR scanned PDFs and write page-level + merged text artifacts."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input PDF files to OCR.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/external/ocr_scans",
        help="Directory for OCR outputs.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI for pdf2image.")
    parser.add_argument(
        "--lang",
        default="eng",
        help="OCR language code(s), comma-separated for pytesseract.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip PDFs already processed.")
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=200,
        help="If pdf2text text length exceeds this, OCR is skipped by default.",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="OCR all pages even if pdftotext output already looks usable.",
    )
    return parser.parse_args()


def _pdftotext_chars(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout or ""


def _preprocess_page(image: Image.Image) -> Image.Image:
    img = image.convert("L")
    img = img.resize((img.width * 2, img.height * 2))
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.75)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img


def _safe_mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return statistics.mean(values)


def _ocr_image(image: Image.Image, lang: str, psm: int) -> dict[str, Any]:
    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(
        image,
        lang=lang,
        config=config,
    )
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    confidences = []
    words = 0
    for conf in data.get("conf", []):
        try:
            c = int(float(conf))
        except (TypeError, ValueError):
            continue
        if c >= 0:
            confidences.append(c)
            words += 1
    return {
        "text": (text or "").strip(),
        "conf_mean": _safe_mean(confidences),
        "conf_p50": statistics.median(confidences) if confidences else 0.0,
        "words": words,
        "psm": psm,
    }


def _run_page_ocr(image: Image.Image, lang: str) -> tuple[str, dict[str, Any]]:
    best: dict[str, Any] | None = None
    for psm in DEFAULT_PSMS:
        candidate = _ocr_image(image, lang, psm)
        if best is None or candidate["conf_mean"] > best["conf_mean"]:
            best = candidate

    assert best is not None
    return best["text"], best


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_text(path: Path, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, text in enumerate(pages, start=1):
            f.write(f"\f\n--- page {idx} ---\n{text}\n\n")


def _process_pdf(pdf_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    stem = pdf_path.stem.replace(" ", "_")
    base = out_dir / stem
    pages_path = base.with_suffix(".jsonl")
    text_path = base.with_suffix(".txt")
    manifest_path = base.with_suffix(".manifest.json")

    if args.skip_existing and pages_path.exists() and text_path.exists():
        return {
            "pdf": str(pdf_path),
            "status": "skipped_existing",
            "pages": 0,
        }

    text_guess = _pdftotext_chars(pdf_path)
    should_ocr = args.force_ocr or len(text_guess.strip()) < args.min_text_chars

    if not should_ocr:
        text_path.write_text(text_guess.strip(), encoding="utf-8")
        return {
            "pdf": str(pdf_path),
            "status": "skipped_ocr_due_to_text_layer",
            "pages": 0,
            "extracted_chars": len(text_guess),
            "output": {
                "text": str(text_path),
            },
        }

    pages = convert_from_path(str(pdf_path), dpi=args.dpi, fmt="png")
    page_rows = []
    merged_pages: list[str] = []
    confs = []
    word_counts = []
    psm_counts = Counter()

    for page_no, page in enumerate(pages, start=1):
        image = _preprocess_page(page)
        txt, meta = _run_page_ocr(image, args.lang)
        merged_pages.append(txt)

        psm_counts[meta["psm"]] += 1
        confs.append(meta["conf_mean"])
        word_counts.append(meta["words"])
        page_rows.append(
            {
                "pdf": str(pdf_path),
                "page": page_no,
                "status": "ok",
                "text": txt,
                "engine": "pytesseract",
                "lang": args.lang,
                "psm": meta["psm"],
                "conf_mean": meta["conf_mean"],
                "conf_p50": meta["conf_p50"],
                "words": meta["words"],
            }
        )

    _write_jsonl(pages_path, page_rows)
    _write_text(text_path, merged_pages)
    merged_text = "\n".join(merged_pages)
    manifest = {
        "pdf": str(pdf_path),
        "status": "ocr",
        "pages": len(pages),
        "dpi": args.dpi,
        "lang": args.lang,
        "ocr_engine": "pytesseract",
        "text_chars": len(merged_text),
        "word_count_total": sum(word_counts),
        "conf_mean": _safe_mean(confs),
        "conf_p50": statistics.median(confs) if confs else 0.0,
        "psm_mode_counts": dict(psm_counts),
        "outputs": {
            "pages_jsonl": str(pages_path),
            "full_text": str(text_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [Path(p).expanduser() for p in args.inputs]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        print("Missing input files:", ", ".join(missing), file=sys.stderr)
        return 2

    if not all(p.suffix.lower() == ".pdf" for p in inputs):
        print("Only PDF inputs are supported.", file=sys.stderr)
        return 2

    # fail fast if the engine is missing
    try:
        _ = pytesseract.get_tesseract_version()
    except TesseractNotFoundError:
        print("Tesseract binary not found. Install it and retry.", file=sys.stderr)
        print("On macOS: brew install tesseract", file=sys.stderr)
        return 2

    results = []
    for p in inputs:
        result = _process_pdf(p, out_dir, args)
        print(json.dumps(result, ensure_ascii=False))
        results.append(result)

    summary = {
        "inputs": [str(p) for p in inputs],
        "results": results,
        "ocr_pages": sum(r.get("pages", 0) for r in results),
    }
    (out_dir / "ocr_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
