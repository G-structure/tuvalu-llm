# Data collection and ML dataset pipeline (Tuvaluan-English)

## Docs that currently cover this area

- `README.md`
  - Collection strategy, scrape notes, aligned JSONL examples, and historical experiment logs.
- `docs/UNSTRUCTURED_DATA_PIPELINE.md`
  - Unstructured asset inventory (`unstruct_lang_data`), OCR workflow, and integration playbook.
- `tv2en.md`
  - URL mapping and cross-language pairing evidence used in scraper/alignment.
- `docs/TRAINING_PIPELINE.md`
  - Stage A → synthetic generation → Stage B flow, config inventory, and run commands.
- `docs/TVL_EN_TINKER_PLAN.md`
  - Strategic scaffold, split guidance, and dataset quality rationale.
- `docs/SELECTIVE_TRANSLATION_SPEC.md`
  - Structure preservation rules used for synthetic translation.
- `scripts/` + `training/`
  - Source of truth for exact schema, filters, and transforms.

## Source-of-truth structure (actual runtime artifacts)

1. Raw HTML
   - `data/raw/wol_tvl/*.html`
   - `data/raw/wol_en/*.html`
   - Includes per-page saves for resumable re-processing.

2. Aligned canonical pairs (`en`<->`tvl`) — **immutable raw data**
   - `data/aligned/bible_verses.jsonl`
   - `data/aligned/articles.jsonl`
   - `data/aligned/daily_text.jsonl`

3. Cleaned data (output of `scripts/clean_pipeline.py`)
   - `data/cleaned/cleaned.jsonl` — deduplicated, filtered pairs
   - `data/cleaned/rejected.jsonl` — rejected pairs with reasons
   - `data/cleaned/cleaning_report.json` — statistics
   - `data/cleaned/rejection_samples.jsonl` — examples per rejection category

5. Stage A dataset outputs
   - `data/finetune/stage_a_mt/train_full.jsonl`
   - `data/finetune/stage_a_mt/train_balanced.jsonl`
   - `data/finetune/stage_a_mt/validation.jsonl`
   - `data/finetune/stage_a_mt/test.jsonl`
   - `data/finetune/stage_a_mt/rejected.jsonl`
   - `data/finetune/stage_a_mt/stats.json`
   - `data/finetune/stage_a_mt/manifest.json`

6. Stage B (presently absent until run)
   - `data/finetune/stage_b_sources/english_normalized/*.jsonl`
   - `data/finetune/stage_b_synthetic_tvl/accepted/*.jsonl`
   - `data/finetune/stage_b_synthetic_tvl/rejected/*.jsonl`
   - `data/finetune/stage_b_mix/*.jsonl`

## Aligned row schema (`data/aligned/*.jsonl`)

Every row has this cross-language pair contract:

- `id` (string)
- `tvl` / `en`
- `content_type` (`bible_verse`, `article_paragraph`, `daily_text`)
- `domain` (`bible`, `book`, `daily_text`)
- `alignment_method` (`verse_number`, `paragraph_position`, `document_level`, `date`)
- `alignment_confidence`
- `doc_id`, `source_url_tvl`, `source_url_en`
- `book_num`, `book_name`, `chapter`, `verse`
- `date` (`YYYY-MM-DD` for daily text)
- `pub_code`
- `tvl_chars`, `en_chars`, `length_ratio`

## Alignment/collection behavior that is currently implemented

### Bible
- Scraper: `scripts/scrape_bible.py`
- Key: verse id pattern `v{bookNo}-{chapter}-{verse}-{part}` on WOL pages.
- Method: align by verse number across both languages.
- Confidence: fixed `1.0`.

### Articles
- Scraper: `scripts/scrape_articles.py`
- Key extraction: `docId`-scoped paragraph pairing via `data-pid` in `article#article`.
- Method A: paragraph-position alignment on shared `data-pid`.
- Method B: document-level fallback when paragraph mismatch is severe.
- Quality fallback conditions:
  - `alignment_method == document_level` when mismatch ratio `> 0.2` and overlapping pids are `< 0.8` of shorter stream.
  - fallback rows get confidence `0.6`, doc-level `id` format `article_{doc_id}_doc`.
- Paragraph filters:
  - Skip pairs with both sides `< 20` chars.
  - Skip ratios outside `0.15 .. 7.0`.
- Confidence for paragraph-level alignment:
  - `0.9` when pid order/sets match exactly.
  - `0.8` otherwise.

### Daily text
- Scraper: `scripts/scrape_daily_text.py`
- Method: date alignment (`YYYY-MM-DD`) between TVL and EN pages.
- Confidence: fixed `1.0`.
- Extractor reads adjacent dates from each fetch page to reduce requests.

## Stage A transformation (`scripts/build_stage_a_mt_data.py`)

Input: aligned JSONL rows (`data/aligned/*`).

Output rows keep
- `id` and a two-turn `messages` translation example.
- `metadata` containing all aligned row fields above.
- `metadata.direction` = `tvl_to_en` or `en_to_tvl`.
- `metadata.source_lang`, `metadata.target_lang`, `metadata.template_idx`.

Quality/rejection logic:
- `too_short` or `too_long` by `min_chars` / `max_chars`.
- `bad_length_ratio` outside configured ratio bounds.
- `low_alignment_confidence` for confidence under threshold (except optional low-confidence article doc-level override).
- `empty_text` when either side is empty.
- `duplicate_pair` via normalized hash dedupe.

Split policy:
- Bible by book holdouts (default Val `{31,63,64}`, Test `{8,57,65}`)
- Articles by `doc_id`
- Daily text by `date`
- Others by deterministic hash buckets

## Stage A dataset balancing and artifacts

- `train_full`: all accepted pairs, both directions.
- `train_balanced`: capped bible fraction by default (`bible_max_train_share = 0.70`) for diversity.
- `validation.jsonl` / `test.jsonl`: fixed deterministic split output.
- Optional pilot file is emitted when configured with `pilot_token_budget`.

## Stage B source/synthetic/mix contracts

### Stage B source layer (`scripts/build_stage_b_sources.py`)
- Uses registered dataset loaders (`training/synthetic/loaders.py`).
- Each source example is normalized to common schema (`id`, `task_family`, `messages`, `metadata`, optional `translate_mask`).
- `stage_b_sources` manifest includes budget use, dataset stats, and completed datasets.

### Stage B synthetic TVL (`scripts/generate_stage_b_synthetic_tvl.py`)
- Uses Stage A translation with selective mask+unmask.
- Preserves placeholders/JSON/code/tool/message structure with explicit checks in `training/synthetic/quality.py`.
- Writes `accepted`/`rejected` JSONL with reason logs.

### Stage B mix (`scripts/build_stage_b_mix.py`)
- Pools: `english`, `synthetic_tvl`, `anchor`.
- Defaults: 40% / 40% / 20% mix ratio, 2% val/test split on non-anchor.
- Anchor (`stage_a_mt/train_balanced.jsonl`) is always in train.
- Added `metadata.stage_b_source` on every row.

## External multilingual assets (`unstruct_lang_data/`)

The folder contains four high-value assets and one low-value pair set for this stage:

- `unstruct_lang_data/Tatoeba-v2023-04-12-en&tvl.tsv`
  - 15 rows total (14 usable sentence pairs + header).
  - Directly usable as aligned en↔tvl pairs after `id` normalization.
  - Best stage: **Stage A seed augmentation** (parallel corpus + potential bilingual glossary extraction from aligned examples).
- `unstruct_lang_data/DICTIONARY_Tuv_Palagi.pdf` (11,384+ English headwords; 11,553 Tuvaluan definitions)
  - PDF text is machine-readable via `pdftotext`.
  - Contains both `Tuvaluan-English` and `English-Tuvaluan` sections.
  - Best stages:
    - **Stage A**: can generate safe lexical/term pairs from consistent entries.
    - **Stage B**: useful for preservation glossary (names, flora/fauna terms, place terms, tool-domain terms).
  - Suggested metadata if ingested:
    - `metadata.source="unstruct_lang_data/DICTIONARY_Tuv_Palagi.pdf"`
    - `metadata.source_section="Tuvaluan-English" | "English-Tuvaluan"`
    - `metadata.source_row` and `metadata.parse_mode` (`dictionary_term`)
    - `metadata.entry_type` (`single`, `multiple`, `example`, `sense`)
- `unstruct_lang_data/The_magical_garlands_of_Nukufetau.pdf`
  - Producer indicates scanned capture (`PFU ScanSnap`) and plain text extraction is currently unusable.
  - Utility depends on OCR.
- `unstruct_lang_data/Tuvalu_News_Sheets_66-99.pdf`
- `unstruct_lang_data/Tuvalu_News_Sheets_Part 2.pdf`
  - Both were created with `AccuSoft ImageGear` and `pdftotext` only returns artifact/noise text.
  - Potentially high name/place/value for named-entity mining if OCR is run.

Suggested ingestion workflow for external assets:

1. Add a dedicated conversion step under `scripts/` that outputs standardized JSONL entries with provenance metadata.
2. Keep a strict quality gate for external additions:
   - de-dup against existing `data/aligned` rows,
   - minimum length checks,
   - enforce minimum lexical fidelity for dictionary-derived lines,
   - store parse confidence / source span in metadata.
3. Add external rows to a separate Stage A seed bucket first (`data/external/*`) so impact is measurable before merging.
4. For news PDFs, run OCR only if character confidence/quality is sufficient; otherwise treat output as name-harvest candidates only.
5. If OCR is successful, route names to:
   - name-preservation test set for Stage A,
   - term list / glossary for Stage B prompt-guidance and post-edit checks.

## Current dataset state (March 2026)

### Raw aligned data (`data/aligned/` — immutable, never modify)

| File | Pairs | Chars (both langs) | Est. tokens | Coverage |
|---|---|---|---|---|
| `bible_verses.jsonl` | 30,838 | 8.9M | ~2.4M | All 66 books, 1,189 chapters |
| `articles.jsonl` | 275,430 | 55.6M | ~14.6M | 7,255 docIds (contains ~118k duplicates from overlapping crawls) |
| `daily_text.jsonl` | 3,432 | 7.0M | ~1.8M | 3,287 dates (2017-2025), 0 gaps |
| **Total** | **309,700** | **71.6M** | **~18.8M** | |

### Cleaned data (`data/cleaned/` — output of `scripts/clean_pipeline.py`)

| File | Pairs | Chars (both langs) | Est. tokens |
|---|---|---|---|
| `cleaned.jsonl` | 178,371 | 96.1M | ~25.3M |
| `rejected.jsonl` | 131,329 | — | — |

Cleaning pipeline applies (in order): text normalization, ID dedup, content hash dedup, metadata removal, identical-pair removal, length/ratio filtering.

| Rejection reason | Count | % |
|---|---|---|
| `duplicate_id` | 118,553 | 38.3% |
| `duplicate_content` | 11,075 | 3.6% |
| `metadata` | 1,049 | 0.3% |
| `identical_pair` | 406 | 0.1% |
| `bad_ratio` | 246 | 0.1% |

Run: `uv run python scripts/clean_pipeline.py [--profile balanced|strict|lenient] [--dry-run]`

### Article sources breakdown

DocIds were harvested from:
- 22 direct publication codes (bh, bhs, bm, bt, fg, gf, hf, jl, jy, kr, lff, lffi, lmd, lv, lvs, my, sjj, snnw, th, wt, yc, ypq)
- 6 WOL library categories crawled recursively:
  - Watchtower (faleleoleo-maluga): ~6,500 docIds
  - Awake! (ala-mai): ~3,850 docIds
  - Meeting workbooks (tusi-mo-fakatasiga): ~3,900 docIds
  - Books (tusi): covered by pub codes
  - Ministry (te-tou-galuega-talai): all duplicates of above
  - Brochures (polosiua-ki-te-tama-tusi): all duplicates of above

### Scraping and cleaning playbook

For complete reproduction instructions (all phases including cleaning), see [docs/SCRAPING_PLAYBOOK.md](SCRAPING_PLAYBOOK.md).

## Current checked-in state

- Raw aligned data at `data/aligned/` is current and complete (309,700 pairs).
- Cleaned data at `data/cleaned/` is current (178,371 pairs after dedup + filtering).
- Stage A artifacts in `data/finetune/stage_a_mt/` were built from an earlier ~50k pair snapshot; needs rebuild with full cleaned dataset.
- Stage B source/synthetic/mix directories are not present until those steps are executed.
