# Unstructured Data Mining Playbook (TVL-English)

This page is the reproducible runbook for all non-JW/seed assets stored under `unstruct_lang_data/` and the scripts that convert them into ML-ready artifacts.

Last updated: 2026-03-07

## 1) Why this exists

`unstruct_lang_data/` contains language assets that are not part of the canonical web scrape but can still add value to Stage A and Stage B training:

- short sentence-aligned pairs (`Tatoeba-v2023-04-12-en&tvl.tsv`),
- bilingual dictionary entries (`DICTIONARY_Tuv_Palagi.pdf`),
- scanned local publications (`The_magical_garlands_of_Nukufetau.pdf`, `Tuvalu_News_Sheets_66-99.pdf`, `Tuvalu_News_Sheets_Part 2.pdf`).

The goal is to convert these into:

- seed parallel rows for Stage A (`data/external/stage_a_seed/*.jsonl`),
- conservative name/place/flora/fauna term artifacts for Stage B enrichment (`data/external/stage_b_seed/*.jsonl`),
- reproducible manifests and stats for every run.

## 2) Current repository snapshot

- `unstruct_lang_data/Tatoeba-v2023-04-12-en&tvl.tsv`
  - 14 usable lines (header + 13 usable pairs after parser gates).
- `data/external/raw/DICTIONARY_Tuv_Palagi.txt`
  - extracted text from the dictionary PDF (machine-readable source).
- `data/external/ocr_scans/`
  - OCR JSONL/text/manifest artifacts are present for:
    - `The_magical_garlands_of_Nukufetau`
    - `Tuvalu_News_Sheets_66-99` (chunked)
    - `Tuvalu_News_Sheets_Part 2` (chunked)
- `data/external/stage_a_seed/` (most recently run as `unstruct-manual-20260307-skipocr`)
  - `unstruct_tatoeba.jsonl`: 14 rows
  - `unstruct_dictionary_tvl_en.jsonl`: 30,917 rows
  - `unstruct_dictionary_en_tvl.jsonl`: 0 rows
  - `rejected.jsonl`: 2,381 rows
  - `stats.json`: `deduped_total 30,931`, `dropped_duplicate_pairs 215`
- `data/external/stage_b_seed/` (same run)
  - `unstruct_ocr_terms.jsonl`: 3,222 rows
  - `unstruct_dictionary_terms.jsonl`: 1,076 rows
  - `stats.json`: `total 4,298`

## 3) Output structure and metadata contracts

### Stage A seed artifacts (`data/external/stage_a_seed/`)

- `unstruct_tatoeba.jsonl`
- `unstruct_dictionary_tvl_en.jsonl`
- `unstruct_dictionary_en_tvl.jsonl`
- `rejected.jsonl`
- `stats.json`
- `manifest.json`

Every Stage A candidate row should include canonical alignment fields used by `scripts/build_stage_a_mt_data.py`:

- `id`
- `tvl`, `en`
- `content_type`
- `domain`
- `alignment_method`
- `alignment_confidence`
- `doc_id`
- `source_url_tvl`, `source_url_en`
- `pub_code`
- `tvl_chars`, `en_chars`, `length_ratio`
- `metadata` (free-form provenance block)

Recommended metadata keys in unstructured output:

- `source_file`, `source_row`
- `source_section` for dictionary rows (`tvl_en` or `en_tvl`)
- `source_slice` when applicable
- `source_page` / `source_pdf_page` if OCR-derived
- `parse_mode`
- `extractor_version`

### Stage B term artifacts (`data/external/stage_b_seed/`)

- `unstruct_ocr_terms.jsonl`
- `unstruct_dictionary_terms.jsonl`
- `stats.json`
- `manifest.json`

Term rows include:

- `id`, `term`, `term_type`
- `content_type`, `domain`, `alignment_method`, `alignment_confidence`
- `source`, `source_page` (when available)
- `evidence_count`, `evidence_max_conf`
- `metadata` (`extractor_version`, source hints)

`term_type` values currently emitted by the script:

- `person_place`
- `flora_fauna`
- `other`

## 4) Reproducible pipeline (copy/paste)

### 4.1 Install requirements

```bash
cd /Users/cuboniks/Projects/tv

brew install tesseract poppler qpdf
uv sync --extra ocr
```

### 4.2 Full deterministic run (recommended)

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

What this does:

1. Generates missing OCR chunk PDFs for large scans (`66-99`, `Part 2`) with deterministic ranges.
2. Runs OCR with `scripts/ocr_scanned_pdfs.py` over: `The_magical_garlands_of_Nukufetau.pdf` + chunked news PDFs.
3. Builds seed artifacts using `scripts/build_unstructured_seed.py`:
   - unstructured Stage A rows from Tatoeba + dictionary split,
   - optional Stage B term candidates from OCR + dictionary.
4. Builds a Stage A training slice from the seed input:
   - `data/finetune/stage_a_mt/unstructured_seed/*`.
5. Writes a small run manifest at `data/external/pipeline_runs/<run_name>/manifest.json`.

Latest accomplished run:

- Run name: `unstruct-manual-20260307-skipocr`
- Command used (checkpointed run): `uv run --extra ocr python scripts/run_unstructured_datamining.py --run-name unstruct-manual-20260307-skipocr --skip-ocr --extract-ocr-terms --extract-dictionary-terms`
- Resulting totals:
  - Stage A deduped rows: 30,931
  - Stage B terms: 4,298 total
  - Stage A split output: generated under `data/finetune/stage_a_mt/unstructured_seed/*`
  - Orchestration manifest: `data/external/pipeline_runs/unstruct-manual-20260307-skipocr/manifest.json`

### 4.3 Command-level control

- Skip stages while debugging:

```bash
uv run --extra ocr python scripts/run_unstructured_datamining.py --skip-ocr --skip-stage-a-build
```

- Force deterministic dictionary-only parse and seed build:

```bash
uv run --extra ocr python scripts/run_unstructured_datamining.py \
  --skip-ocr --extract-ocr-terms \
  --extract-dictionary-terms --max-dict-entries 200
```

- Run a dry-pass (print plan only):

```bash
uv run --extra ocr python scripts/run_unstructured_datamining.py --dry-run
```

## 5) Quality checks / acceptance gates

Use these checks after each full run:

```bash
# 1) confirm run manifest exists
ls -l data/external/pipeline_runs/<run_name>/manifest.json

# 2) confirm OCR artifacts and counts
jq '.inputs|length, .ocr_pages' data/external/ocr_scans/ocr_batch_summary.json
for f in data/external/ocr_scans/*.manifest.json; do
  echo "== $f"
  jq '{pdf: .pdf, pages: .pages, conf_mean: .conf_mean, conf_p50: .conf_p50, text_chars: .text_chars}' "$f"
done

# 3) confirm seed counts and schema fields
python3 - <<'PY'
import json,glob
from pathlib import Path
for path in glob.glob('data/external/stage_a_seed/*.jsonl'):
    p = Path(path)
    with p.open(encoding='utf-8') as f:
      rows=[json.loads(x) for x in f if x.strip()]
    print(p.name, len(rows))
PY

# 4) sanity-check Stage A seed build
jq '.input_rows, .accepted_rows' data/finetune/stage_a_mt/unstructured_seed/manifest.json
```

## 6) Where to continue

- Stage A: `uv run python scripts/build_stage_a_mt_data.py --input-dir data/external/stage_a_seed ...` is the normal entrypoint.
- Stage B: term candidate files are intentionally not directly used in current Stage B mix yet; consume them through a glossary/augmentation step when you add a dedicated loader.

## 7) Known gaps / follow-up

- No automatic parser currently merges scanned-page sentences into parallel `tvl/en` sentence pairs.
- Dictionary parsing is conservative; many dictionary rows are intentionally dropped if noisy.
- Stage B integration is still manual: term rows are exported as a seed artifact, not yet merged into the default Stage B mix without curation.

For the high-level strategy and broader corpus schema, see `docs/UNSTRUCTURED_DATA_PIPELINE.md` and `docs/DATASET_COLLECTION_AND_ML_PIPELINE.md`.
