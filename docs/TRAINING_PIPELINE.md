# Tuvaluan LLM Staged Training Pipeline

## Overview

This pipeline transforms a Tuvaluan-English parallel corpus into a bilingual,
capable LLM adapter through three stages:

1. **Stage A** -- Translation adapter (small model, translation SFT only)
2. **Synthetic generation** -- Use Stage A to translate English capability datasets into Tuvaluan
3. **Stage B** -- Bilingual capability adapter (large model, mixed training)

```
Aligned parallel corpus
        |
        v
  [Stage A: Translation SFT]
  Model: Qwen/Qwen3-30B-A3B-Base
  Task: TVL<->EN translation only
        |
        v
  [Synthetic Generation]
  Selectively translate English datasets -> Tuvaluan
  Preserve code/JSON/tools/structure
  Target: ~200M translated tokens
        |
        v
  [Stage B: Bilingual Agent]
  Model: openai/gpt-oss-120b or Qwen/Qwen3-30B-A3B (from base, not Stage A)
  Mix: 40% English + 40% Synthetic TVL + 20% Parallel anchor
        |
        v
  Final bilingual adapter (shipped separately from base)
```

## Stage A: Translation Adapter

**Purpose**: Train a strong TVL<->EN translator to unlock synthetic data generation.

**Default model**: `Qwen/Qwen3-30B-A3B-Base` (small MoE, cheap to train)

**Pilot model**: `meta-llama/Llama-3.2-3B` (for shakeout runs)

### Commands

```bash
# 1. Build Stage A training data from aligned JSONL
uv run python scripts/build_stage_a_mt_data.py \
    --config configs/stage_a_translation_qwen30b_base.json

# 2. Train Stage A adapter
uv run python scripts/train_stage_a_translation.py \
    --config configs/stage_a_translation_qwen30b_base.json

# 3. Evaluate
uv run python scripts/eval_stage_a_translation.py \
    --config configs/stage_a_translation_qwen30b_base.json
```

### Data splits

- Bible: split by held-out books (Ruth/Philemon/Jude for test, Obadiah/2John/3John for val)
- Articles: split by `doc_id`
- Daily text: split by `date`
- Other: deterministic hash split

### Output

```
data/finetune/stage_a_mt/
    train_full.jsonl      # All accepted pairs, both directions
    train_balanced.jsonl   # Bible share capped at 70%
    validation.jsonl
    test.jsonl
    rejected.jsonl
    stats.json
    manifest.json
```

## Synthetic Data Generation

**Purpose**: Translate English capability datasets into Tuvaluan, preserving
all machine-readable structure (code, JSON, tool calls, schemas, etc.).

### Selective translation

Only human-language spans are translated. Everything else is masked with
placeholders, preserved during translation, then restored. See
`docs/SELECTIVE_TRANSLATION_SPEC.md` for the full spec.

### Supported datasets

| Dataset | Task Family | Status |
|---------|-------------|--------|
| tasksource/tasksource-instruct-v0 | chat | Implemented |
| HuggingFaceH4/ultrachat_200k | chat | Implemented |
| openai/gsm8k | math | Implemented |
| Salesforce/xlam-function-calling-60k | tool | Implemented |
| Muennighoff/mbpp | code | Implemented |
| rajpurkar/squad | qa | Implemented |
| ccdv/cnn_dailymail | summarization | Implemented |
| meta-math/MetaMathQA | math | TODO |
| NousResearch/hermes-function-calling-v1 | tool | TODO |
| zai-org/AgentInstruct | chat | TODO |

### Commands

```bash
# 1. Build normalized English source pool
uv run python scripts/build_stage_b_sources.py \
    --config configs/synthetic_stage_b_core.json

# 2. Generate synthetic Tuvaluan translations
uv run python scripts/generate_stage_b_synthetic_tvl.py \
    --config configs/synthetic_stage_b_core.json
```

### Output

```
data/finetune/stage_b_sources/
    english_normalized/     # One JSONL per dataset
    manifests/

data/finetune/stage_b_synthetic_tvl/
    accepted/               # Validated synthetic TVL JSONL
    rejected/               # Failed validation
    manifests/
    stats/
    generation_state.json   # Resume checkpoint
```

### Tool-use modes

- **safe** (default): Tool-call JSON preserved in chat format, no native tool messages
- **native** (experimental): Uses Tinker's native tool message format. Off by default,
  enable via `"tool_mode": "native"` in config.

## Stage B: Bilingual Capability Adapter

**Purpose**: Train the final bilingual adapter that handles both TVL and EN
across all capability domains.

**IMPORTANT**: Stage B starts from a base/chat model, NOT from Stage A weights.
Stage A exists only to produce the synthetic dataset.

**Supported Stage B models**:
- `openai/gpt-oss-120b` (MoE, 117B/5.1B active) — `gpt_oss_no_sysprompt` renderer (Harmony format)
- `Qwen/Qwen3-30B-A3B` (MoE, 30B/3B active) — `qwen3` renderer (im_start/im_end format)

NOTE: Use `Qwen/Qwen3-30B-A3B` (chat variant), not `-Base`, for Stage B.
The `-Base` variant gets the `role_colon` renderer which lacks tool-calling support.

### Default mix

| Component | Share | Source |
|-----------|-------|--------|
| English capability data | 40% | Original upstream datasets |
| Synthetic Tuvaluan | 40% | Stage A translations |
| TVL-EN parallel anchor | 20% | Stage A training data |

### Commands

```bash
# 1. Build mixed training data
uv run python scripts/build_stage_b_mix.py \
    --config configs/stage_b_mix_default.json

# 2. Train Stage B adapter (pick one model)
uv run python scripts/train_stage_b_agent.py \
    --config configs/stage_b_agent_oss120b.json    # gpt-oss-120b
uv run python scripts/train_stage_b_agent.py \
    --config configs/stage_b_agent_qwen30b.json    # Qwen3-30B-A3B

# 3. Evaluate
uv run python scripts/eval_stage_b_agent.py \
    --config configs/stage_b_agent_oss120b.json
uv run python scripts/eval_stage_b_agent.py \
    --config configs/stage_b_agent_qwen30b.json
```

### Ablation modes

- `mixed` (default): Full 3-source mix
- `english_only`: Only English capability data (no Tuvaluan)
- `tvl_only`: Only synthetic TVL + parallel anchor (no English replay)

### Evaluation

Stage B eval covers:
1. **Translation regression**: Compare to Stage A baseline on held-out MT test set
2. **Capability smoke tests**: Per-task-family checks (chat, tool, math, code, QA, summarization)
3. **Bilingual comparison**: Same tasks in EN vs synthetic TVL
4. **Preservation metrics**: JSON parse rate, schema-valid rate, code exact-match, placeholder leak rate

### Output

```
data/finetune/stage_b_mix/
    train.jsonl
    train_pilot.jsonl     # Smaller set for quick shakeout
    validation.jsonl
    test.jsonl
    stats.json
```

## Config Files

| Config | Purpose |
|--------|---------|
| `configs/stage_a_translation_qwen30b_base.json` | Stage A with Qwen3-30B |
| `configs/stage_a_translation_pilot_llama32_3b.json` | Stage A pilot with Llama-3.2-3B |
| `configs/synthetic_stage_b_core.json` | Synthetic generation settings |
| `configs/stage_b_mix_default.json` | Stage B mix ratios and sources |
| `configs/stage_b_agent_oss120b.json` | Stage B training on gpt-oss-120b |
| `configs/stage_b_agent_qwen30b.json` | Stage B training on Qwen3-30B-A3B |

## Package Structure

```
training/
    common/
        config.py           # Config loading/validation
        io.py               # JSONL I/O, run directories
        schema.py           # Normalized example schema
        token_estimates.py  # Token counting
        tinker_runtime.py   # Tinker client setup
        checkpoints.py      # Checkpoint save/resume
        manifests.py        # Run manifest generation
        metrics.py          # chrF++, BLEU, exact match
    stage_a_mt/
        build_data.py       # Build MT dataset from aligned JSONL
        train.py            # Stage A LoRA training
        eval.py             # Translation evaluation
        export.py           # Export checkpoint for synthetic phase
    synthetic/
        registry.py         # Dataset registry pattern
        loaders.py          # HuggingFace dataset loaders
        normalize.py        # Normalize to common schema
        selective_translate.py  # Mask-translate-unmask engine
        quality.py          # Validation pipeline
        generate.py         # Synthetic generation runner
        budgeting.py        # Token budget management
    stage_b_agent/
        build_mix.py        # Mix builder (EN + TVL + anchor)
        train.py            # Stage B LoRA training
        eval.py             # Multi-faceted evaluation
        tooling_modes.py    # Safe vs native tool format
```

## Dependencies

Install training dependencies:

```bash
uv pip install -e ".[training]"
bash scripts/bootstrap_tinker.sh
export TINKER_API_KEY=...
```

## Dataset documentation currently in repo

Current docs covering dataset and collection work:

- `README.md`: project plan, crawler constraints, alignment methodology, and experiment log for collection.
- `docs/SCRAPING_PLAYBOOK.md`: **step-by-step reproduction guide** for all data collection (scripts, commands, troubleshooting).
- `docs/DATASET_COLLECTION_AND_ML_PIPELINE.md`: dataset schema, alignment contracts, quality gates, and Stage A/B data lineage.
- `docs/UNSTRUCTURED_DATA_PIPELINE.md`: unstructured data extraction (`unstruct_lang_data`) and OCR/playbook for supplemental training assets.
- `tv2en.md`: URL mapping and cross-language pairing notes used by scraping/alignment.
- `docs/TVL_EN_TINKER_PLAN.md`: staged training strategy, model choices, and open TODOs for synthetic loaders.
- `docs/SELECTIVE_TRANSLATION_SPEC.md`: how selective translation preserves code/JSON/tool structure.
- `docs/TRAINING_PIPELINE.md`: current runnable pipeline and config references.

## Data collection + ML dataset lineage (single source of truth)

This repo separates collection and ML dataset layers:

1. Raw content:
   - `data/raw/wol_tvl/*.html` — cached Tuvaluan HTML pages
   - `data/raw/wol_en/*.html` — cached English HTML pages
   - `data/raw/sitemap_tvl.xml`, `data/raw/sitemap_tvl.json` — parsed sitemap
2. Canonical aligned en<->TVL pairs (309,700 total raw pairs):
   - `data/aligned/bible_verses.jsonl` — 30,838 verse pairs (all 66 books)
   - `data/aligned/articles.jsonl` — 275,430 paragraph pairs (7,255 docIds)
   - `data/aligned/daily_text.jsonl` — 3,432 daily text pairs (2017-2025)
3. Stage A MT chat datasets:
   - `data/finetune/stage_a_mt/*.jsonl`
4. Stage B source pools and synthetic TVL:
   - `data/finetune/stage_b_sources/english_normalized/*.jsonl` (when built)
   - `data/finetune/stage_b_synthetic_tvl/accepted/*.jsonl` (when built)
   - `data/finetune/stage_b_synthetic_tvl/rejected/*.jsonl`
5. Stage B mixed dataset:
   - `data/finetune/stage_b_mix/*.jsonl` (when built)

Important: in this checkout, Stage A artifacts may need rebuild after the full scrape completed (309k pairs vs earlier ~50k snapshot). Stage B artifacts are not yet created.

## Stage A metadata contract (what is copied vs added)

Each aligned row from `data/aligned` contributes all source metadata forward into Stage A training rows via
`metadata`:

- inherited unchanged (examples): `id`, `tvl`, `en`, `content_type`, `domain`,
  `alignment_method`, `alignment_confidence`, `doc_id`, `source_url_tvl`,
  `source_url_en`, `book_num`, `book_name`, `chapter`, `verse`, `date`,
  `pub_code`, `tvl_chars`, `en_chars`, `length_ratio`
- added in Stage A build step:
  - `direction`: `en_to_tvl` or `tvl_to_en`
  - `source_lang`, `target_lang`
  - `template_idx` (which prompt template was sampled)

Stage A output row shape (`schema`):
- `id`
- `messages` (system/user/assistant chat turn)
- `metadata` (above + generated fields)

Rejection telemetry in Stage A is written to `rejected.jsonl` and summarized in `stats.json`. Rejection reasons currently observed include:
`too_short`, `too_long`, `bad_length_ratio`, `low_alignment_confidence`, `duplicate_pair`, and `empty_text`.

## Split logic by data regime

- Bible: split by held-out books
  - validation: book nums [31,63,64] by default
  - test: book nums [8,57,65] by default
- Articles: split by `doc_id`
- Daily text: split by `date`
- Everything else: deterministic hash split using `content/doc/date` group keys

## Current quality/sanity gates that matter for ML behavior

- `min_confidence` (usually `0.8`)
- `min_chars` / `max_chars` (usually `10` / `4096`)
- `ratio_min` / `ratio_max` (usually `0.4` / `2.5`)
- deterministic dedupe by normalized text hash on the aligned pair
- optional `allow_low_confidence_articles` override for document-level article rows
- `train_balanced` applies `bible_max_train_share` (default `0.70`) to reduce sermon-style bias

## Quick Start Runbook

### Stage A: Translation Adapter

1. Build data: `uv run python scripts/build_stage_a_mt_data.py --config configs/stage_a_translation_qwen30b_base.json`
2. Train: `uv run python scripts/train_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json`
3. Export: `uv run python scripts/export_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json`
4. Monitor: `tensorboard --logdir logs/tinker/stage_a/tb`

### Stage A Pilot (~2M tokens, 1 epoch)

1. Build pilot subset: `uv run python scripts/build_stage_a_mt_data.py --config configs/stage_a_translation_qwen30b_base_pilot_2m_1epoch.json`
2. Train pilot: `uv run python scripts/train_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base_pilot_2m_1epoch.json`

### Stage B: Bilingual Capability Adapter

1. Build sources: `uv run python scripts/build_stage_b_sources.py --config configs/synthetic_stage_b_core.json --limit 100`
2. Generate synthetic TVL: `uv run python scripts/generate_stage_b_synthetic_tvl.py --config configs/synthetic_stage_b_core.json`
3. Build mix: `uv run python scripts/build_stage_b_mix.py --config configs/stage_b_mix_default.json`
4. Train (pick model):
   - `uv run python scripts/train_stage_b_agent.py --config configs/stage_b_agent_oss120b.json`
   - `uv run python scripts/train_stage_b_agent.py --config configs/stage_b_agent_qwen30b.json`
5. Evaluate:
   - `uv run python scripts/eval_stage_b_agent.py --config configs/stage_b_agent_oss120b.json`
   - `uv run python scripts/eval_stage_b_agent.py --config configs/stage_b_agent_qwen30b.json`

## TensorBoard

Training metrics (loss, chrF++, BLEU, learning rate) are logged to TensorBoard
during both Stage A and Stage B training runs. Event files are written to a `tb/`
subdirectory under each run's log path.

```bash
# Stage A (full run)
tensorboard --logdir logs/tinker/stage_a/tb

# Stage A (pilot)
tensorboard --logdir logs/tinker/stage_a_pilot_2m/tb

# Stage B
tensorboard --logdir logs/tinker/stage_b/tb

# All runs at once
tensorboard --logdir logs/tinker
```

The TBLogger (`training/common/tb.py`) writes scalars alongside the existing
JSONL metrics files. It uses TensorBoard's `EventFileWriter` directly (no
PyTorch dependency required). Periodic validation metrics are also logged when
the training loop runs mid-training evaluation.

## Training Dashboard

Live training metrics are displayed at https://tvl-chat.pages.dev/training,
served from Cloudflare D1.

### Uploading metrics to D1

The script `scripts/upload_training_metrics.py` reads local JSONL metric logs
and uploads them to the `training_metrics` and `training_config` tables in D1.

```bash
# Upload Stage A metrics (one-shot, run is complete)
CLOUDFLARE_API_TOKEN=... uv run python scripts/upload_training_metrics.py \
    --once --run-id stage_a_3ep

# Upload Stage B metrics (one-shot)
CLOUDFLARE_API_TOKEN=... uv run python scripts/upload_training_metrics.py \
    --once --run-id stage_b_llama8b

# Watch mode — poll for new Stage B entries every 30s during active training
CLOUDFLARE_API_TOKEN=... uv run python scripts/upload_training_metrics.py \
    --run-id stage_b_llama8b

# Re-upload everything (reset upload state)
CLOUDFLARE_API_TOKEN=... uv run python scripts/upload_training_metrics.py \
    --once --run-id stage_a_3ep --reset
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--run-id` | `stage_b_llama8b` | D1 `run_id` for inserted rows |
| `--metrics-path` | `logs/tinker/<run-id>/metrics.jsonl` | Path to metrics JSONL |
| `--mix-stats-path` | `data/finetune/stage_b_mix/stats.json` | Path to dataset mix stats |
| `--once` | (watch mode) | Upload once and exit |
| `--reset` | false | Clear upload state and re-upload all rows |
| `--init-schema` | false | Create D1 tables before uploading |

The script reads the wrangler OAuth token or a `CLOUDFLARE_API_TOKEN` env var
for authentication.

Upload state is persisted per run in `.upload_state_<run_id>.json` to avoid
re-uploading already-synced rows.

### D1 schema

```sql
training_metrics (run_id, step, metric_type, value_json, created_at)
training_config  (key, value_json, updated_at)
```

- `metric_type`: `train_nll`, `val_nll`, or `gen_eval`
- Config keys: `run_info_<run_id>` (model name, total steps), `mix_stats` (dataset composition)

### API endpoint

`GET /api/training-stats` returns both Stage A and Stage B metrics from D1.
The frontend auto-refreshes every 15 seconds.

### Deploying dashboard changes

```bash
cd chat
npx vinxi build
npx wrangler pages deploy dist --project-name tvl-chat
```

## Benchmark Evaluation

Live eval results are displayed at https://tvl-chat.pages.dev/eval, served
from Cloudflare D1.

The benchmark (`scripts/benchmark_eval.py`) evaluates our TVL fine-tune against
leading models on Tuvaluan language tasks.

### Models compared

| Key | Model | Backend |
|-----|-------|---------|
| `tvl` | Stage B fine-tune | VPS backend |
| `tvl-stage-a` | Stage A (step 7851) | Tinker SDK direct |
| `gpt-5.4` | GPT-5.4 | OpenRouter |
| `qwen3-30b` | Qwen3-30B-A3B (thinking) | OpenRouter |
| `claude-sonnet` | Claude Sonnet 4.6 | OpenRouter |
| `gemini-3.1-pro` | Gemini 3.1 Pro | OpenRouter |
| `google-translate` | Google Cloud Translation | Google API (translation tasks only) |

### Tasks evaluated

| Task | Data source | Metrics |
|------|-------------|---------|
| `translation_en_to_tvl` | Stage A test set | chrF++, BLEU, exact match, by domain |
| `translation_tvl_to_en` | Stage A test set | chrF++, BLEU, exact match, by domain |
| `textbook_en_to_tvl` | Held-out children's textbooks | chrF++, BLEU, exact match, by domain |
| `textbook_tvl_to_en` | Held-out children's textbooks | chrF++, BLEU, exact match, by domain |
| `chat_tvl` | Stage B test (synthetic_tvl + crosslingual) | chrF++, BLEU, by source |
| `qa_tvl` | Stage B test (synthetic_tvl + crosslingual) | chrF++, BLEU, by source |
| `summarization_tvl` | Stage B test (synthetic_tvl) | chrF++, BLEU, by source |

### Budget presets

| Preset | Tokens/model | Translation | Textbook | Chat | QA | Summ |
|--------|-------------|-------------|----------|------|----|------|
| `full` | ~500K | 250+250 | 46+46 | 250 | 120 | 40 |
| `tiny` | ~10K | 5+5 | 5+5 | 5 | 3 | 2 |

### Commands

```bash
# Full benchmark (all models)
OPENROUTER_API_KEY=... GOOGLE_TRANSLATE_KEY=... \
    uv run python scripts/benchmark_eval.py --budget full

# Our model only (no external API costs)
uv run python scripts/benchmark_eval.py --our-model-only

# Specific models
uv run python scripts/benchmark_eval.py \
    --models tvl,tvl-stage-a,gpt-5.4 --openrouter-key sk-...

# Dry run (show token budget, no API calls)
uv run python scripts/benchmark_eval.py --dry-run

# Run and upload results to D1 for the dashboard
CLOUDFLARE_API_TOKEN=... uv run python scripts/benchmark_eval.py \
    --budget full --upload --cf-token $CLOUDFLARE_API_TOKEN
```

### Data flow: benchmark → dashboard

```
scripts/benchmark_eval.py
    │
    ├─ Samples eval set from:
    │   data/finetune/stage_a_mt/test.jsonl  (translation)
    │   data/finetune/stage_b_mix/test.jsonl (chat/qa/summ)
    │   data/external/stage_a_seed/*.jsonl   (textbook, held-out)
    │
    ├─ Calls each model (OpenRouter / Tinker / Google / VPS)
    │
    ├─ Computes chrF++, BLEU, exact match per task per model
    │
    ├─ Saves locally: eval/benchmark/results.json + predictions_*.jsonl
    │
    └─ --upload flag → D1 HTTP API
        │
        ├─ eval_runs (run_id, model_key, budget, results_json)
        │   One row per model per run
        │
        └─ eval_predictions (run_id, model_key, task, prediction, reference)
            Individual predictions for drill-down
        │
        ▼
    GET /api/eval-results
        │
        ├─ Groups eval_runs by run_id
        ├─ Parses results_json per model
        └─ Returns { runs: [{ run_id, budget, models: { model → task → metrics } }] }
        │
        ▼
    /eval page (eval.tsx)
        ├─ Overall ranking (mean chrF++ across tasks)
        ├─ By-category scores (Translation, Textbook, Generation)
        └─ Per-task breakdowns with subcategory drill-down
            (by_domain for translation, by_source for generation)
```

### D1 eval schema

```sql
eval_runs (run_id, model_key, budget, results_json, created_at)
eval_predictions (run_id, model_key, task, example_id, prediction, reference, metadata_json)
```

Run IDs are auto-generated as `YYYY-MM-DD_HHMM_<budget>` (e.g. `2026-03-14_0130_full`).

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--budget` | `full` | Token budget preset (`full` or `tiny`) |
| `--models` | all | Comma-separated model keys |
| `--our-model-only` | false | Only evaluate TVL model |
| `--openrouter-key` | `$OPENROUTER_API_KEY` | API key for OpenRouter models |
| `--google-key` | `$GOOGLE_TRANSLATE_KEY` | API key for Google Translate |
| `--tvl-backend` | `https://api.cyberneticphysics.com/tvl-chat` | VPS backend URL |
| `--upload` | false | Upload results to D1 after evaluation |
| `--cf-token` | `$CLOUDFLARE_API_TOKEN` | Cloudflare API token for D1 upload |
| `--parallel` | 8 | Concurrent API requests per model |
| `--seed` | 42 | Random seed for eval set sampling |
| `--output-dir` | `eval/benchmark/` | Local results directory |
| `--dry-run` | false | Show budget breakdown without calling APIs |

## What Remains Experimental

- Native tool-message training mode (behind `tool_mode: "native"` flag)
- MetaMathQA, hermes-function-calling-v1, AgentInstruct loaders (TODO stubs)
- Language detection validation for translated spans
- Weight merging (deliberately not supported as default)
