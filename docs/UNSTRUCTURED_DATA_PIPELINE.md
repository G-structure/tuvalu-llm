# Unstructured Data Processing Playbook (TVL-English)

Canonical reference for unstructured assets, conversion, and reproducible ingestion into the ML pipeline.

Backlinks:

- [docs/DATASET_COLLECTION_AND_ML_PIPELINE.md](DATASET_COLLECTION_AND_ML_PIPELINE.md)
- [docs/UNSTRUCTURED_DATA_MINING_PLAYBOOK.md](UNSTRUCTURED_DATA_MINING_PLAYBOOK.md)
- [docs/TVL_EN_TINKER_PLAN.md](TVL_EN_TINKER_PLAN.md)
- [docs/TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)

Last checked: `2026-03-07`.

## 1) What is in `unstruct_lang_data/`

```
unstruct_lang_data/
  DICTIONARY_Tuv_Palagi.pdf
  Tatoeba-v2023-04-12-en&tvl.tsv
  The_magical_garlands_of_Nukufetau.pdf
  Tuvalu_News_Sheets_66-99.pdf
  Tuvalu_News_Sheets_Part 2.pdf
```

Current usable observations in this checkout:

- `Tatoeba-v2023-04-12-en&tvl.tsv`
  - 14 usable pairs.
- `DICTIONARY_Tuv_Palagi.pdf`
  - extractable text exists in `data/external/raw/DICTIONARY_Tuv_Palagi.txt`.
- `The_magical_garlands_of_Nukufetau.pdf`
  - OCR artifacts present in `data/external/ocr_scans`.
- `Tuvalu_News_Sheets_66-99.pdf`
  - chunked OCR artifacts present for pages 1-383 in deterministic chunks.
- `Tuvalu_News_Sheets_Part 2.pdf`
  - chunked OCR artifacts present for pages 1-200 in deterministic chunks.

## 2) Required toolchain

```bash
brew install tesseract poppler qpdf
uv sync --extra ocr
```

## 3) Current unstructured dataset state

### Outputs that already exist

- `data/external/raw/DICTIONARY_Tuv_Palagi.txt`
- `data/external/ocr_scans/*.jsonl|*.txt|*.manifest.json`
- `data/external/stage_a_seed/` (dictionary + OCR-augmented seed files now present)
- `data/external/stage_b_seed/` (OCR + dictionary term candidates now present)
- `data/finetune/stage_a_mt/unstructured_seed/*`

### Important note on current seed coverage

Latest completed full run (`unstruct-manual-20260307-skipocr`) produced:

- `data/external/stage_a_seed/stats.json`:
  - `dictionary_tvl_en: 30917`
  - `deduped_total: 30931`
  - `rejected: 2381`
- `data/external/stage_b_seed/stats.json`:
  - `ocr_terms: 3222`
  - `dictionary_terms: 1076`
  - `total: 4298`
- `data/finetune/stage_a_mt/unstructured_seed` now has Stage A splits generated from that seed set.

## 4) Reproducible file layout (important)

- `data/external/raw/` — extracted raw text (dictionary via `pdftotext`).
- `data/external/ocr_scans/` — OCR outputs:
  - `<source>.jsonl`
  - `<source>.txt`
  - `<source>.manifest.json`
  - `ocr_batch_summary.json`
  - `chunks/` (deterministic PDF chunks for rescopes)
- `data/external/stage_a_seed/` — seed rows for Stage A and seed QA:
  - `unstruct_tatoeba.jsonl`
  - `unstruct_dictionary_tvl_en.jsonl`
  - `unstruct_dictionary_en_tvl.jsonl`
  - `rejected.jsonl`
  - `stats.json`
  - `manifest.json`
- `data/external/stage_b_seed/` — names/terms candidate rows:
  - `unstruct_ocr_terms.jsonl`
  - `unstruct_dictionary_terms.jsonl`
  - `stats.json`
  - `manifest.json`
- `data/finetune/stage_a_mt/unstructured_seed/` — Stage A split artifacts built from unstructured seed.
- `data/external/pipeline_runs/<run_name>/manifest.json` — per-run orchestration manifest from the new script.

## 5) Preferred end-to-end playbook

For a full run:

```bash
uv run --extra ocr python scripts/run_unstructured_datamining.py \
  --run-name unstruct-playbook-2026-03-07 \
  --asset-dir unstruct_lang_data \
  --ocr-dir data/external/ocr_scans \
  --raw-dir data/external/raw \
  --stage-a-output data/external/stage_a_seed \
  --stage-b-output data/external/stage_b_seed \
  --seed-output data/finetune/stage_a_mt/unstructured_seed \
  --extract-ocr-terms \
  --extract-dictionary-terms
```

Useful variant for quick seed-only checks:

```bash
uv run --extra ocr python scripts/run_unstructured_datamining.py \
  --skip-ocr \
  --extract-ocr-terms
```

If you only want deterministic chunk production for audit/retry:

```bash
uv run --extra ocr python scripts/run_unstructured_datamining.py \
  --skip-seed --skip-stage-a-build
```

## 6) Pipeline internals

### 6.1 OCR step

`scripts/ocr_scanned_pdfs.py` decides page-wise whether to OCR based on pdftotext length. It writes JSONL rows with `conf_mean`, `conf_p50`, `status`, `page`, and `pdf` fields, plus page-level/aggregate manifests.

### 6.2 Seed extraction

`scripts/build_unstructured_seed.py` consumes:

- Tatoeba TSV,
- dictionary text (auto-generated from PDF by pdftotext if needed),
- OCR pages (optional).

It emits Stage A/Stage B artifact families in the paths above and writes run manifests.

### 6.3 Stage A build

`scripts/build_stage_a_mt_data.py` can be run against `data/external/stage_a_seed` to produce `data/finetune/stage_a_mt/unstructured_seed`.

### 6.4 Where command history is stored

`data/external/pipeline_runs/<run_name>/manifest.json` records:

- run label,
- UTC timestamp,
- commands used for each stage,
- git hash/dirty state when available.

## 7) Stage A vs Stage B usage map

- Stage A input rows should feed directly into `build_stage_a_mt_data.py` and are constrained by existing Stage A quality gates.
- Stage B term rows are not yet mixed automatically into final Stage B mix; consume them through a later glossary/candidate-review step.
- Name/place/flora/fauna candidate extraction is now available in seed output for future Stage B integration.

## 8) QA checklist (copy/paste)

```bash
jq '.status, .pages, .conf_mean, .text_chars' data/external/ocr_scans/*.manifest.json
wc -l unstruct_lang_data/Tatoeba-v2023-04-12-en&tvl.tsv
wc -l data/external/stage_a_seed/*.jsonl
cat data/external/stage_a_seed/manifest.json
cat data/external/stage_b_seed/manifest.json
jq '.input_rows, .accepted_rows' data/finetune/stage_a_mt/unstructured_seed/manifest.json
```

## 9) Open tasks

- Add a strict dictionary sentence aligner (current parser is conservative).
- Re-run unstructured seed build to populate `unstruct_dictionary_tvl_en.jsonl` / `unstruct_dictionary_en_tvl.jsonl` in this working tree.
- Decide how dictionary/ocr term candidates are merged into Stage B (manual review first vs automatic).
