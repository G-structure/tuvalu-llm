# scripts/

All scripts in the tv2en project. Run everything with `uv run` unless noted otherwise.

## Quick Reference

| Phase | What to run | Purpose |
|-------|-------------|---------|
| **Scraping** | `scrape_bible.py`, `scrape_articles.py`, `scrape_daily_text.py` | Build raw parallel corpus from JW.org |
| **Cleaning** | `clean_pipeline.py` | Thin CLI wrapper for `tv.corpus.clean` |
| **Splits** | `build_splits.py` → `validate_splits.py` | Thin CLI wrapper for `tv.corpus.splits` plus validation |
| **Training data** | `render_training_data.py` | Thin CLI wrapper for `tv.corpus.render` |
| **Training** | `train_stage_a_translation.py` | Fine-tune translation LoRA |
| **Local MLX prep** | `prepare_local_mlx_training.py` | Export Stage A / Stage B runs for local MLX-LM |
| **Eval** | `eval_stage_a_translation.py` | chrF++/BLEU on test set |
| **Football** | `pipeline_football.py` | Scrape → translate → deploy football news |

---

## Scraping (JW.org Parallel Corpus)

All scrapers use `fetch.py` under the hood, which routes HTTP through Docker `curl-impersonate` to bypass TLS fingerprint detection. Raw data lands in `data/aligned/` and is **immutable** — never modify these files.

### `fetch.py`
Library module (not a CLI script). Shared HTTP fetcher with caching, rate limiting, and retries via Docker `lwthiker/curl-impersonate:0.6-ff`. All other scrapers import from here.

### `scrape_sitemap.py`
Parses `jw.org/tvl/sitemap.xml` and classifies all 7,103 URLs by type (bible, magazine, book, etc.).

```bash
uv run scripts/scrape_sitemap.py
# Output: data/raw/sitemap_tvl.json
```

### `scrape_bible.py`
Verse-aligned TVL/EN Bible extraction from WOL. All 66 books, 1,189 chapters, 30,838 pairs.

```bash
uv run scripts/scrape_bible.py --pilot     # 3 chapters (test)
uv run scripts/scrape_bible.py --full      # all books
uv run scripts/scrape_bible.py --book 1    # single book
# Output: data/aligned/bible_verses.jsonl
```

### `scrape_articles.py`
Paragraph-aligned WOL articles by docId. Supports recursive library crawling. 275,430 pairs across 7,255 unique docIds.

```bash
uv run scripts/scrape_articles.py --pilot              # small test
uv run scripts/scrape_articles.py --pub w              # Watchtower
uv run scripts/scrape_articles.py --library             # crawl all library categories
uv run scripts/scrape_articles.py --docids 12345 67890  # specific docs
# Output: data/aligned/articles.jsonl
```

### `scrape_daily_text.py`
Date-aligned daily texts with a 3-day page optimization (each page yields 3 dates, reducing fetches by ~3x).

```bash
uv run scripts/scrape_daily_text.py --year 2024
uv run scripts/scrape_daily_text.py --range 2024-01-01 2024-06-30
# Output: data/aligned/daily_text.jsonl
```

---

## Cleaning

### `clean_pipeline.py`

CLI wrapper that delegates the implementation to `tv.corpus.clean`.
Deduplication, metadata filtering, length ratio checks, and normalization. Input from `data/aligned/` is never modified.

```bash
uv run scripts/clean_pipeline.py                          # balanced (default)
uv run scripts/clean_pipeline.py --profile strict         # stricter filtering
uv run scripts/clean_pipeline.py --dry-run                # report only, no writes
# Output: data/cleaned/cleaned.jsonl, rejected.jsonl, cleaning_report.json
```

Three profiles: `balanced` (default), `strict`, `lenient`.

### `stats.py`
Prints dataset statistics (character counts, length ratios, quality issues, duplicates) to terminal.

```bash
uv run scripts/stats.py
```

---

## Training Data Preparation

### `build_splits.py`

CLI wrapper that delegates the implementation to `tv.corpus.splits`.
Creates leak-proof train/val/test splits via a 5-phase pipeline: doc-level splitting → n-gram indexing → cross-source decontamination → validation → output.

```bash
uv run scripts/build_splits.py
uv run scripts/build_splits.py --dry-run  # preview without writing
# Output: data/splits/{train,validation,test,quarantined}.jsonl, split_report.json
```

### `validate_splits.py`
CI-friendly validator for split files. Checks doc_id overlap, Bible book overlap, date overlap, exact text leakage, and n-gram contamination.

```bash
uv run scripts/validate_splits.py
# Exit 0 = pass, 1 = fail
```

### `render_training_data.py`

CLI wrapper that delegates the implementation to `tv.corpus.render`.
Renders decontaminated splits into chat-formatted JSONL for Tinker. Creates both TVL→EN and EN→TVL directions with template variation, optional unstructured seed merging, and Bible downsampling.

```bash
uv run scripts/render_training_data.py
uv run scripts/render_training_data.py --include-unstructured  # add unstructured seed pairs
uv run scripts/render_training_data.py --dry-run
# Output: data/finetune/stage_a_mt_v2/{train_full,train_balanced,validation,test}.jsonl
```

### `build_stage_a_mt_data.py`
CLI wrapper that builds Stage A MT chat datasets from aligned JSONL. Delegates to `tv.training.stage_a_mt.build_data`. Largely superseded by `render_training_data.py` for the v2 pipeline.

```bash
uv run scripts/build_stage_a_mt_data.py --config configs/stage_a_translation_qwen30b_base.json
```

### `build_stage_b_sources.py`
Downloads and normalizes English capability datasets (GSM8K, XLAM, MBPP, SQuAD, CNN/DailyMail, UltraChat, etc.) for Stage B synthetic generation with token budgeting.
It can also normalize local private chat JSONL into a separate Stage B pool when configured with the `private_tvl_chat` loader and `output_subdir: "real_tvl_chat"`.

```bash
uv run scripts/build_stage_b_sources.py --list                # list available datasets
uv run scripts/build_stage_b_sources.py --config configs/...  # build all
uv run scripts/build_stage_b_sources.py --datasets gsm8k mbpp # specific datasets only
# Output: data/finetune/stage_b_sources/english_normalized/*.jsonl
#         data/finetune/stage_b_sources/real_tvl_chat/*.jsonl (optional local TVL chat)
```

### `build_stage_b_mix.py`
Builds the Stage B mixed training dataset by combining anchor translation data with synthetic TVL capability data at configured mix ratios.
Supports an optional fourth `real_tvl_chat` pool sourced from `data/finetune/stage_b_sources/real_tvl_chat/`.

```bash
uv run scripts/build_stage_b_mix.py --config configs/stage_b_agent.json
```

### `generate_stage_b_synthetic_tvl.py`
Selectively translates English capability datasets to Tuvaluan using the trained Stage A model. Preserves code, JSON, tool calls, and schema structure.

```bash
uv run scripts/generate_stage_b_synthetic_tvl.py --config configs/stage_b_agent.json
uv run scripts/generate_stage_b_synthetic_tvl.py --config configs/stage_b_agent.json --dry-run
```

---

## Unstructured Data Mining

Scripts for extracting additional training pairs from non-JW.org sources (scanned PDFs, Tatoeba, dictionaries).

### `ocr_scanned_pdfs.py`
OCRs scanned PDFs using Tesseract with multi-PSM mode selection, image preprocessing, and per-page confidence tracking.

```bash
uv run scripts/ocr_scanned_pdfs.py --inputs data/raw/pdfs/*.pdf --output-dir data/ocr/
uv run scripts/ocr_scanned_pdfs.py --inputs doc.pdf --dpi 400 --force-ocr
# Output: per-page JSONL, merged text files, per-PDF manifest
```

Requires: `pytesseract`, `pdf2image`, system `pdftotext`.

### `build_unstructured_seed.py`
Mines TVL-EN phrase pairs from unstructured sources (Tatoeba, OCR output, Tuvaluan dictionary). Separates aligned sentence pairs (Stage A) from terminology (Stage B).

```bash
uv run scripts/build_unstructured_seed.py --asset-dir data/external/assets/
uv run scripts/build_unstructured_seed.py --extract-ocr-terms --extract-terms --dry-run
# Output: data/external/stage_a_seed/*.jsonl, data/external/stage_b_seed/*.jsonl
```

### `run_unstructured_datamining.py`
End-to-end orchestrator: OCR → seed mining → Stage A dataset build, with manifest tracking for reproducibility.

```bash
uv run scripts/run_unstructured_datamining.py --run-name pilot-001
uv run scripts/run_unstructured_datamining.py --skip-ocr --skip-seed  # rebuild Stage A data only
# Output: data/external/pipeline_runs/<run_name>/
```

---

## Training

### `bootstrap_tinker.sh`
One-time setup: initializes the `vendor/tinker-cookbook` submodule and installs Tinker API dependencies.

```bash
bash scripts/bootstrap_tinker.sh
```

### `train_stage_a_translation.py`
Trains the Stage A TVL↔EN translation LoRA adapter via Tinker API. Supports checkpoint resume and in-training generative eval.

```bash
uv run scripts/train_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json
uv run scripts/train_stage_a_translation.py --config configs/... --data data/finetune/stage_a_mt_v2/train_balanced.jsonl
```

Requires: `TINKER_API_KEY` env var.

### `train_stage_b_agent.py`
Trains the Stage B bilingual capability adapter. Starts from base/chat model (not Stage A weights).

```bash
uv run scripts/train_stage_b_agent.py --config configs/stage_b_agent.json
uv run scripts/train_stage_b_agent.py --config configs/stage_b_agent.json --pilot  # small shakeout run
```

Requires: `TINKER_API_KEY` env var.

### `prepare_local_mlx_training.py`
Builds a local MLX-LM run directory for either Stage A or Stage B using the existing repo configs and datasets.

```bash
uv run python scripts/prepare_local_mlx_training.py \
  --config configs/stage_a_translation_qwen30b_base.json \
  --mlx-model mlx-community/Qwen3-30B-A3B-Base-4bit
```

See `docs/LOCAL_MLX_TRAINING.md` for the local workflow and model-specific notes.

---

## Evaluation

### `eval_stage_a_translation.py`
Evaluates Stage A adapter: chrF++, BLEU, exact match. Supports parallel sampling (`--parallel 64` for 8x+ throughput).

```bash
uv run scripts/eval_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json
uv run scripts/eval_stage_a_translation.py --config configs/... --limit 100 --parallel 64
```

### `eval_stage_b_agent.py`
Evaluates Stage B adapter on both translation regression and capability tasks.

```bash
uv run scripts/eval_stage_b_agent.py --config configs/stage_b_agent.json
```

### `export_stage_a_translation.py`
Exports the Stage A model path/info for downstream use (e.g., synthetic generation). Outputs JSON to stdout.

```bash
uv run scripts/export_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json
```

---

## Football Site (talafutipolo.pages.dev)

The football pipeline scrapes English football news, translates it to Tuvaluan, and deploys to Cloudflare Pages + D1.

### `init_football_db.py`
Creates the SQLite schema (articles, translations, translation_attempts, feedback, implicit_signals) and seeds source records.

```bash
uv run scripts/init_football_db.py
# Output: data/football/football.db
```

### `db_conn.py`
Library module. Database connection factory — returns Cloudflare D1 (when env vars set) or local SQLite. Used by all football scripts.

### `d1_client.py`
Library module. Cloudflare D1 REST API client with `sqlite3.Connection`-compatible interface. Requires `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` env vars.

### `scrape_football_fifa.py`
Scrapes FIFA.com articles via their CXM API sitemap → Contentful Rich Text JSON.

```bash
uv run scripts/scrape_football_fifa.py --limit 50
uv run scripts/scrape_football_fifa.py --pages 5  # sitemap pages to scan
```

### `scrape_football_goal.py`
Scrapes Goal.com articles via sitemap XML → `__NEXT_DATA__` JSON extraction.

```bash
uv run scripts/scrape_football_goal.py --limit 50
uv run scripts/scrape_football_goal.py --lists  # include list/slide articles
```

### `scrape_football_sky.py`
Scrapes Sky Sports football articles via news sitemap → JSON-LD + HTML body extraction.

```bash
uv run scripts/scrape_football_sky.py --limit 50
```

### `clean_article_bodies.py`
Cleans scraped article bodies: removes source-specific promo blocks (Sky CTAs, Goal NordVPN ads, FIFA NBSP artifacts), normalizes whitespace, splits text blobs into paragraphs.

```bash
uv run scripts/clean_article_bodies.py                # clean all
uv run scripts/clean_article_bodies.py --source sky   # single source
uv run scripts/clean_article_bodies.py --dry-run
```

Also importable as a module by the scraper scripts.

### `translate_football.py`
Translates English articles to Tuvaluan via Tinker API. Paragraph-level chunking, 3-attempt retry with escalating temperature on model collapse. All attempts are recorded for RL training.

```bash
uv run scripts/translate_football.py --limit 10
uv run scripts/translate_football.py --article 42       # specific article
uv run scripts/translate_football.py --retry-collapsed   # retry collapsed translations
```

Requires: `TINKER_API_KEY` env var.

### `detect_collapse.py`
Detects model collapse (degenerate n-gram repetition) using multiple signals: whole-text uniqueness, tail repetition, sliding windows, long phrase repetition, per-paragraph checks.

```bash
uv run scripts/detect_collapse.py                    # scan all translations
uv run scripts/detect_collapse.py --dry-run           # report only
uv run scripts/detect_collapse.py --threshold 0.3     # custom threshold
```

Also importable as a module by `translate_football.py`.

### `pipeline_football.py`
Convenience orchestrator: runs all three scrapers → translates untranslated articles → prints DB stats.

```bash
uv run scripts/pipeline_football.py                               # full pipeline
uv run scripts/pipeline_football.py --scrape-only                  # scrape only
uv run scripts/pipeline_football.py --translate-only --translate-limit 5
```

### `sync_to_d1.py`
Syncs local `football.db` to Cloudflare D1 via `npx wrangler d1 execute`.

```bash
uv run scripts/sync_to_d1.py
uv run scripts/sync_to_d1.py --dry-run  # preview SQL without executing
```

---

## Migrations (one-time)

### `migrate_collapse_detection.py`
Adds collapse detection columns and `translation_attempts` table to `football.db`, backfills existing translations.

### `migrate_unescape_entities.py`
Unescapes HTML entities (`&#x27;`, `&amp;`, etc.) in article titles, bodies, and translations.

Both are idempotent and safe to re-run.

---

## Deprecated

These are old wrappers kept for backward compatibility. Use the replacements instead.

| Deprecated | Replacement |
|---|---|
| `build_tinker_mt_data.py` | `build_stage_a_mt_data.py` |
| `train_tinker_mt.py` | `train_stage_a_translation.py` |
| `eval_tinker_mt.py` | `eval_stage_a_translation.py` |

---

## Typical Workflows

### Build the parallel corpus from scratch
```bash
uv run scripts/scrape_sitemap.py
uv run scripts/scrape_bible.py --full
uv run scripts/scrape_articles.py --library
uv run scripts/scrape_daily_text.py --range 2017-01-01 2025-12-31
uv run scripts/stats.py
```

### Prepare training data (Stage A)
```bash
uv run scripts/clean_pipeline.py
uv run scripts/build_splits.py
uv run scripts/validate_splits.py
uv run scripts/render_training_data.py --include-unstructured
```

### Train and evaluate (Stage A)
```bash
bash scripts/bootstrap_tinker.sh  # one-time
uv run scripts/train_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json
uv run scripts/eval_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json --parallel 64
```

### Football site daily update
```bash
uv run scripts/pipeline_football.py
uv run scripts/sync_to_d1.py
```
