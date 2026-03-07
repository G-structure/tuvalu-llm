#!/usr/bin/env python3
"""End-to-end orchestrator for unstructured TVL-English mining.

This script runs OCR extraction, seed mining, and optional Stage A build in a
reproducible order, writing a compact run manifest under
`data/external/pipeline_runs/<run_name>/manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ts_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_command(cmd: list[str], description: str, *, dry_run: bool = False) -> None:
    print(f"\n[{description}]")
    print(" ".join(cmd))
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({description}) -> {completed.returncode}")


def _chunked_input_specs(asset_dir: Path) -> dict[Path, list[tuple[int, int]]]:
    return {
        asset_dir / "Tuvalu_News_Sheets_66-99.pdf": [
            (1, 80),
            (81, 160),
            (161, 240),
            (241, 320),
            (321, 383),
        ],
        asset_dir / "Tuvalu_News_Sheets_Part 2.pdf": [
            (1, 50),
            (51, 100),
            (101, 150),
            (151, 200),
        ],
    }


def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _make_chunks(
    pdf_path: Path,
    ranges: list[tuple[int, int]],
    chunk_dir: Path,
    *,
    force: bool,
    dry_run: bool,
) -> list[Path]:
    chunk_paths: list[Path] = []
    stem = pdf_path.stem.replace(" ", "_")

    for start, end in ranges:
        if start > end:
            raise ValueError(f"Invalid chunk range for {pdf_path.name}: {start}-{end}")

        out_path = chunk_dir / f"{stem}-p{start}-{end}.pdf"
        chunk_paths.append(out_path)

        if out_path.exists() and not force:
            print(f"[skip] chunk exists: {out_path}")
            continue

        _run_command(
            [
                "qpdf",
                "--empty",
                "--pages",
                str(pdf_path),
                f"{start}-{end}",
                "--",
                str(out_path),
            ],
            description=f"chunk {pdf_path.name} pages {start}-{end}",
            dry_run=dry_run,
        )

    return chunk_paths


def _collect_ocr_inputs(
    asset_dir: Path,
    magic_pdf: str,
    *,
    chunk_large: bool,
    chunk_dir: Path,
    force_chunks: bool,
    skip_non_chunked_large: bool,
    dry_run: bool,
) -> list[Path]:
    inputs = [asset_dir / magic_pdf]

    for pdf_path, ranges in _chunked_input_specs(asset_dir).items():
        if chunk_large:
            if not pdf_path.exists() and not dry_run:
                raise FileNotFoundError(str(pdf_path))
            inputs.extend(
                _make_chunks(
                    pdf_path=pdf_path,
                    ranges=ranges,
                    chunk_dir=chunk_dir,
                    force=force_chunks,
                    dry_run=dry_run,
                )
            )
            continue

        if skip_non_chunked_large:
            print(f"[skip] full PDF for {pdf_path.name} because --skip-non-chunked-large is set")
            continue

        if pdf_path.exists() or dry_run:
            inputs.append(pdf_path)

    return inputs


def _write_manifest(path: Path, run_name: str, steps: list[dict[str, Any]]) -> None:
    payload = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
    }

    # Include minimal git context if available for reproducibility.
    try:
        import training.common.manifests as manifest_lib

        payload["git_hash"] = manifest_lib.get_git_hash()
        payload["git_dirty"] = manifest_lib.get_git_dirty()
    except Exception:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run unstructured data mining for TVL-English assets in one reproducible sequence."
    )
    p.add_argument("--run-name", default=None, help="Label for this run manifest and logs.")
    p.add_argument("--asset-dir", default="unstruct_lang_data", help="Directory containing source assets.")
    p.add_argument("--raw-dir", default="data/external/raw", help="Raw extraction output directory.")
    p.add_argument("--ocr-dir", default="data/external/ocr_scans", help="OCR artifact output directory.")
    p.add_argument("--stage-a-output", default="data/external/stage_a_seed", help="Stage A seed output directory.")
    p.add_argument("--stage-b-output", default="data/external/stage_b_seed", help="Stage B term output directory.")
    p.add_argument(
        "--seed-output",
        default="data/finetune/stage_a_mt/unstructured_seed",
        help="Output for build_stage_a_mt_data.py.",
    )

    p.add_argument("--skip-stage-a-build", action="store_true", help="Skip Stage A dataset build from seed rows.")
    p.add_argument("--skip-ocr", action="store_true", help="Skip OCR. Use existing files in ocr_dir.")
    p.add_argument("--skip-seed", action="store_true", help="Skip seed extraction step.")
    p.add_argument("--dry-run", action="store_true", help="Print all command steps without executing them.")

    p.add_argument("--magic-pdf", default="The_magical_garlands_of_Nukufetau.pdf", help="Non-chunked PDF to run OCR on.")
    p.add_argument("--ocr-dpi", type=int, default=300, help="OCR rasterization DPI.")
    p.add_argument("--ocr-lang", default="eng", help="Tesseract language code(s).")
    p.add_argument("--ocr-min-text-chars", type=int, default=200, help="OCR skip threshold when pdftotext output is large enough.")
    p.add_argument("--ocr-force", action="store_true", help="Force OCR even when text layer appears present.")
    p.add_argument("--ocr-skip-existing", action="store_true", help="Skip PDFs already processed by OCR script.")

    p.add_argument("--no-chunk-large", action="store_true", help="Disable deterministic chunking for known large PDFs.")
    p.add_argument("--force-chunks", action="store_true", help="Rebuild chunk PDFs even if they already exist.")
    p.add_argument("--skip-non-chunked-large", action="store_true", help="Do not use full PDFs for chunk targets when not chunking.")

    p.add_argument("--extract-ocr-terms", action="store_true", default=False, help="Mine person/place/flora terms from OCR artifacts.")
    p.add_argument("--extract-dictionary-terms", action="store_true", default=False, help="Mine dictionary term candidates for Stage B.")
    p.add_argument("--ocr-term-min-freq", type=int, default=2, help="Minimum frequency for OCR term candidates.")
    p.add_argument("--max-dict-entries", type=int, default=None, help="Optional per-section hard cap in dictionary parsing.")

    p.add_argument("--stage-a-min-confidence", type=float, default=0.75, help="Build Stage A --min-confidence.")
    p.add_argument("--stage-a-min-chars", type=int, default=1, help="Build Stage A --min-chars.")
    p.add_argument("--stage-a-max-chars", type=int, default=2048, help="Build Stage A --max-chars.")
    p.add_argument("--stage-a-ratio-min", type=float, default=0.2, help="Build Stage A --ratio-min.")
    p.add_argument("--stage-a-ratio-max", type=float, default=3.0, help="Build Stage A --ratio-max.")

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    asset_dir = (REPO_ROOT / args.asset_dir).resolve()
    raw_dir = (REPO_ROOT / args.raw_dir).resolve()
    ocr_dir = (REPO_ROOT / args.ocr_dir).resolve()
    chunk_dir = ocr_dir / "chunks"
    stage_a_output = (REPO_ROOT / args.stage_a_output).resolve()
    stage_b_output = (REPO_ROOT / args.stage_b_output).resolve()
    seed_output = (REPO_ROOT / args.seed_output).resolve()
    pipeline_run_dir = REPO_ROOT / "data" / "external" / "pipeline_runs" / (args.run_name or _ts_label())

    _ensure_dirs(raw_dir, ocr_dir, stage_a_output, stage_b_output, seed_output, pipeline_run_dir)

    run_summary: list[dict[str, Any]] = []
    run_name = args.run_name or _ts_label()

    # Step 1: OCR
    if not args.skip_ocr:
        inputs = _collect_ocr_inputs(
            asset_dir=asset_dir,
            magic_pdf=args.magic_pdf,
            chunk_large=not args.no_chunk_large,
            chunk_dir=chunk_dir,
            force_chunks=args.force_chunks,
            skip_non_chunked_large=args.skip_non_chunked_large,
            dry_run=args.dry_run,
        )

        if not inputs:
            raise RuntimeError("No OCR input files found. Check --asset-dir and source file names.")

        ocr_cmd = [
            sys.executable,
            "scripts/ocr_scanned_pdfs.py",
            "--output-dir",
            str(ocr_dir),
            "--dpi",
            str(args.ocr_dpi),
            "--lang",
            args.ocr_lang,
            "--min-text-chars",
            str(args.ocr_min_text_chars),
            "--inputs",
            *[str(p) for p in inputs],
        ]
        if args.ocr_force:
            ocr_cmd.append("--force-ocr")
        if args.ocr_skip_existing:
            ocr_cmd.append("--skip-existing")

        _run_command(ocr_cmd, "OCR scan", dry_run=args.dry_run)
        run_summary.append({"step": "ocr", "status": "ok", "command": ocr_cmd})

    else:
        run_summary.append({"step": "ocr", "status": "skipped"})

    # Step 2: seed build
    if not args.skip_seed:
        seed_cmd = [
            sys.executable,
            "scripts/build_unstructured_seed.py",
            "--asset-dir",
            str(asset_dir),
            "--stage-a-output",
            str(stage_a_output),
            "--stage-b-output",
            str(stage_b_output),
            "--ocr-dir",
            str(ocr_dir),
            "--dictionary-text",
            str(raw_dir / "DICTIONARY_Tuv_Palagi.txt"),
            "--ocr-term-min-freq",
            str(args.ocr_term_min_freq),
            "--run-name",
            run_name,
        ]

        # The dictionary extractor is safe but not free, so keep off unless asked.
        # This mirrors current production use and avoids accidental term leakage.
        if args.extract_ocr_terms:
            seed_cmd.append("--extract-ocr-terms")
        if args.extract_dictionary_terms:
            seed_cmd.append("--extract-terms")
        if args.max_dict_entries is not None:
            seed_cmd.extend(["--max-dict-entries", str(args.max_dict_entries)])

        _run_command(seed_cmd, "Build unstructured seed", dry_run=args.dry_run)
        run_summary.append({"step": "seed", "status": "ok", "command": seed_cmd})
    else:
        run_summary.append({"step": "seed", "status": "skipped"})

    # Step 3: optional Stage A build
    if not args.skip_stage_a_build:
        stage_a_cmd = [
            sys.executable,
            "scripts/build_stage_a_mt_data.py",
            "--input-dir",
            str(stage_a_output),
            "--output-dir",
            str(seed_output),
            "--min-confidence",
            str(args.stage_a_min_confidence),
            "--min-chars",
            str(args.stage_a_min_chars),
            "--max-chars",
            str(args.stage_a_max_chars),
            "--ratio-min",
            str(args.stage_a_ratio_min),
            "--ratio-max",
            str(args.stage_a_ratio_max),
        ]
        _run_command(stage_a_cmd, "Build Stage A from seed", dry_run=args.dry_run)
        run_summary.append({"step": "stage_a_build", "status": "ok", "command": stage_a_cmd})

    _write_manifest(
        pipeline_run_dir / "manifest.json",
        run_name=run_name,
        steps=run_summary,
    )

    if args.dry_run:
        print(f"\nDry run complete. Manifest would be written to {pipeline_run_dir / 'manifest.json'}")
    else:
        print(f"\nPipeline complete for run '{run_name}'")
        print(f"Manifest: {pipeline_run_dir / 'manifest.json'}")
        print(f"Stage A seed: {stage_a_output}")
        print(f"Stage B seed: {stage_b_output}")
        print(f"Stage A dataset: {seed_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
