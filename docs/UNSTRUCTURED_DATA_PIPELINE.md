# Unstructured Data Processing Playbook (TVL-English)

This is the dedicated playbook for **`unstruct_lang_data/`** assets and how to
turn them into ML-ready signals.

Backlinks:

- [docs/DATASET_COLLECTION_AND_ML_PIPELINE.md](DATASET_COLLECTION_AND_ML_PIPELINE.md)
- [docs/TVL_EN_TINKER_PLAN.md](TVL_EN_TINKER_PLAN.md)
- [docs/TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)

Last checked: `2026-03-06`.

## 1) Scope and why this exists

The repository’s primary training graph is built from
`data/aligned/bible_verses.jsonl`, `data/aligned/articles.jsonl`,
and `data/aligned/daily_text.jsonl`.  
`unstruct_lang_data/` contains **four extra assets** that are not currently
part of that canonical aligned layer.

The goal of this pipeline is to:

- extract usable language pairs from these external sources,
- preserve provenance and quality metadata,
- and feed them as an explicit **seed** into Stage A and Stage B workflows
  without silently mixing unvetted text into the canonical data stream.

## 2) What assets are in `unstruct_lang_data/` today

Repo files:

```text
unstruct_lang_data/
  DICTIONARY_Tuv_Palagi.pdf
  Tatoeba-v2023-04-12-en&tvl.tsv
  The_magical_garlands_of_Nukufetau.pdf
  Tuvalu_News_Sheets_66-99.pdf
  Tuvalu_News_Sheets_Part 2.pdf
```

Current status snapshot (based on local files in this checkout):

- `Tatoeba-v2023-04-12-en&tvl.tsv`  
  Small but high-confidence seed with 14 usable pairs (plus header).
- `DICTIONARY_Tuv_Palagi.pdf`  
  Rich bilingual dictionary-like source (machine-readable text extraction works
  with `pdftotext`).
- `The_magical_garlands_of_Nukufetau.pdf`  
  scanned PDF, OCR needed (already processed successfully).
- `Tuvalu_News_Sheets_66-99.pdf`  
  scanned/legacy PDF, high page count, should be chunked before OCR.
- `Tuvalu_News_Sheets_Part 2.pdf`  
  scanned/legacy PDF, likely high value for names and named entities.

## 3) Required toolchain and environment

Python and external requirements:

```bash
brew install tesseract poppler qpdf
uv sync --extra ocr
uv run --extra ocr python -c "import pytesseract, pdf2image, PIL; print('ocr deps ok')"
```

Notes:

- `tesseract` is required for OCR text extraction.
- `pdftotext` (`poppler`) is required for machine-readable PDFs.
- `qpdf` is strongly recommended for splitting long PDFs in stable chunks.

## 4) Reproducible output locations

Use these canonical directories when adding new artifacts:

- `data/external/raw/` — raw text dumps from `pdftotext` and pre-parse artifacts.
- `data/external/ocr_scans/` — OCR artifacts produced by
  `scripts/ocr_scanned_pdfs.py`.
- `data/external/stage_a_seed/` — candidate aligned rows to be evaluated before
  feeding Stage A.
- `data/external/stage_b_seed/` — candidate names/terminology/lexicon rows for later
  Stage B enrichment work.

Existing OCR outputs already available in this checkout:

- `data/external/ocr_scans/The_magical_garlands_of_Nukufetau.{jsonl,txt,manifest.json}`
- `data/external/ocr_scans/Tuvalu_News_Sheets_66-99-p1-3.{jsonl,txt,manifest.json}`
- `data/external/ocr_scans/Tuvalu_News_Sheets_66-99-p1-5.{jsonl,txt,manifest.json}`
- `data/external/ocr_scans/ocr_batch_summary.json`

## 5) Exact command playbook

### 5.1 Inspect the raw asset set

```bash
ls -lh unstruct_lang_data
wc -l unstruct_lang_data/Tatoeba-v2023-04-12-en\&tvl.tsv
```

### 5.2 Extract machine-readable text assets first

#### Tatoeba TSV (direct seed pairs)

```bash
mkdir -p data/external/raw
cp unstruct_lang_data/Tatoeba-v2023-04-12-en\&tvl.tsv data/external/raw/
head -n 5 data/external/raw/Tatoeba-v2023-04-12-en\&tvl.tsv
```

#### Dictionary PDF (`pdftotext`)

```bash
mkdir -p data/external/raw
pdftotext -layout unstruct_lang_data/DICTIONARY_Tuv_Palagi.pdf \
  data/external/raw/DICTIONARY_Tuv_Palagi.txt

# quick sanity
rg -n "Tuvaluan-English|English-Tuvaluan" data/external/raw/DICTIONARY_Tuv_Palagi.txt | head
```

### 5.3 OCR scanned PDFs (baseline)

Use this for any page-scan-only input:

```bash
mkdir -p data/external/ocr_scans
uv run --extra ocr python scripts/ocr_scanned_pdfs.py \
  --inputs \
    unstruct_lang_data/The_magical_garlands_of_Nukufetau.pdf \
    unstruct_lang_data/Tuvalu_News_Sheets_Part\ 2.pdf \
    unstruct_lang_data/Tuvalu_News_Sheets_66-99.pdf \
  --output-dir data/external/ocr_scans \
  --lang eng \
  --dpi 300
```

Important behavior:

- Script auto-runs `pdftotext` first.
- If extracted text is longer than `--min-text-chars` (default 200), OCR is skipped
  unless `--force-ocr` is passed.
- Outputs:
  - `<pdf_stem>.jsonl` (page-by-page records + confidence)
  - `<pdf_stem>.txt` (merged page text with page separators)
  - `<pdf_stem>.manifest.json` (per-file summary)
- Batch summary: `data/external/ocr_scans/ocr_batch_summary.json`

### 5.4 OCR chunking pattern for large PDFs (recommended for 66-99/news)

`Tuvalu_News_Sheets_66-99.pdf` is large and noisy. Process in deterministic chunks
to recover from crashes and allow selective retries.

```bash
mkdir -p data/external/ocr_scans/chunks

qpdf --empty --pages unstruct_lang_data/Tuvalu_News_Sheets_66-99.pdf 1-100 -- \
  data/external/ocr_scans/chunks/Tuvalu_News_Sheets_66-99-p1-100.pdf
qpdf --empty --pages unstruct_lang_data/Tuvalu_News_Sheets_66-99.pdf 101-200 -- \
  data/external/ocr_scans/chunks/Tuvalu_News_Sheets_66-99-p101-200.pdf
qpdf --empty --pages unstruct_lang_data/Tuvalu_News_Sheets_66-99.pdf 201-300 -- \
  data/external/ocr_scans/chunks/Tuvalu_News_Sheets_66-99-p201-300.pdf
qpdf --empty --pages unstruct_lang_data/Tuvalu_News_Sheets_66-99.pdf 301-383 -- \
  data/external/ocr_scans/chunks/Tuvalu_News_Sheets_66-99-p301-383.pdf

uv run --extra ocr python scripts/ocr_scanned_pdfs.py \
  --inputs data/external/ocr_scans/chunks/Tuvalu_News_Sheets_66-99-p1-100.pdf \
           data/external/ocr_scans/chunks/Tuvalu_News_Sheets_66-99-p101-200.pdf \
           data/external/ocr_scans/chunks/Tuvalu_News_Sheets_66-99-p201-300.pdf \
           data/external/ocr_scans/chunks/Tuvalu_News_Sheets_66-99-p301-383.pdf \
  --output-dir data/external/ocr_scans
```

### 5.5 Inspect OCR quality before reuse

```bash
for f in data/external/ocr_scans/*.manifest.json; do
  echo "--- $f"
  jq '{pdf: .pdf, pages: .pages, conf_mean: .conf_mean, conf_p50: .conf_p50, text_chars: .text_chars}' "$f"
done
```

Example rule:

- keep pages when median confidence (`conf_p50`) is materially above ~70;
- re-run specific chunks with `--force-ocr` if page quality is low.

### 5.6 Normalize and validate candidate aligned rows for ML

Create a separate, reviewed dataset in `data/external/stage_a_seed/` and keep only
rows that pass these hard gates before training:

- non-empty `tvl` and `en`
- `id` field present (required by Stage A renderer)
- `content_type` and `alignment_method` present
- `alignment_confidence` filled
- `tvl_chars`, `en_chars`, `length_ratio` computed
- page/row provenance attached

Recommended JSONL schema for each row:

```json
{
  "id": "unstruct:dictionary:eng:000123",
  "tvl": "Ko ...",
  "en": "...",
  "content_type": "external_dictionary_term",
  "domain": "dictionary",
  "alignment_method": "dictionary_entry",
  "alignment_confidence": 0.95,
  "doc_id": null,
  "source_url_tvl": "unstruct_lang_data/DICTIONARY_Tuv_Palagi.pdf",
  "source_url_en": "unstruct_lang_data/DICTIONARY_Tuv_Palagi.pdf",
  "pub_code": "unstruct_dict",
  "tvl_chars": 57,
  "en_chars": 41,
  "length_ratio": 1.39,
  "metadata": {
    "source_file": "DICTIONARY_Tuv_Palagi.pdf",
    "source_row": 123,
    "source_slice": "English-Tuvaluan",
    "parse_mode": "dictionary_term",
    "ocr_conf_mean": null,
    "ocr_conf_p50": null,
    "text_confidence": "high",
    "extractor_version": "manual-v1"
  }
}
```

Then build Stage A from this dedicated folder to isolate impact:

```bash
uv run python scripts/build_stage_a_mt_data.py \
  --input-dir data/external/stage_a_seed \
  --output-dir data/finetune/stage_a_mt/unstructured_seed \
  --min-confidence 0.75 \
  --ratio-min 0.2 \
  --ratio-max 3.0 \
  --min-chars 8 \
  --max-chars 2048
```

You can merge this set into normal Stage A later by explicit file-level concatenation
once the seed corpus is approved.

## 6) Stage A vs Stage B usage map

### Stage A seed use

Primary use: bootstrap lexical/phrase-level parallel capacity and recover names that
are underrepresented in `data/aligned`.

- Best candidates: Tatoeba TSV + dictionary entries + manually approved OCR sentence rows.
- Keep strict confidence and char ratio gates; route rejected rows to a visible
  `rejected` audit JSONL.

### Stage B use

Primary use: glossary and named-entity coverage for bilingual capability:

- person names
- place names
- flora/fauna names and local terms
- culturally specific terms from `news sheets` and `magical garlands`.

Keep these as structured term rows with explicit extraction provenance and use them
to build a post-translation consistency list or evaluation basket.

## 7) Current baseline + what is still missing

Current checkpoint in this repo:

- OCR script is implemented and tested.
- A full OCR run exists for `The_magical_garlands_of_Nukufetau.pdf`.
- Partial chunked OCR exists for `Tuvalu_News_Sheets_66-99.pdf` (`-p1-3`, `-p1-5`
  sample batches).
- No automatic stage-to-stage ingest script yet bridges unstructured outputs into
  `data/aligned` or `data/finetune/stage_a_mt`.

Open tasks:

- add parser/enrichment script for `DICTIONARY_Tuv_Palagi.pdf` row types,
- build a dedicated QA + curation step for OCR pages before Stage A injection,
- add names/terms extraction (with confidence thresholds) into a dedicated Stage B
  glossary artifact.

## 8) Troubleshooting checklist

- `TesseractNotFoundError` → install `tesseract` and verify `$PATH`.
- `pdftotext` returns empty text → confirm scanner PDF and route via OCR script.
- OCR page text appears garbled → increase DPI (`--dpi 400`), rerun chunk with
  `--force-ocr`.
- Page-only PDF but no output file → check permissions and that `pdf2image` + poppler
  are installed.
- Stage A build crashes on row parsing → ensure every candidate row has an `id` key.

## 9) Minimal command replay sequence (copy-paste)

```bash
brew install tesseract poppler qpdf
uv sync --extra ocr
mkdir -p data/external/raw data/external/ocr_scans data/external/stage_a_seed data/external/stage_b_seed

pdftotext -layout unstruct_lang_data/DICTIONARY_Tuv_Palagi.pdf data/external/raw/DICTIONARY_Tuv_Palagi.txt
uv run --extra ocr python scripts/ocr_scanned_pdfs.py \
  --inputs unstruct_lang_data/The_magical_garlands_of_Nukufetau.pdf \
           unstruct_lang_data/Tuvalu_News_Sheets_Part\ 2.pdf \
  --output-dir data/external/ocr_scans
```

Then continue with chunked OCR for
`unstruct_lang_data/Tuvalu_News_Sheets_66-99.pdf` if needed, quality check with
`jq`, and only then create `data/external/stage_a_seed/*.jsonl` plus
`data/finetune/stage_a_mt/unstructured_seed`.
