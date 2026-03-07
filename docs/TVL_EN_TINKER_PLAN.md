# TVL↔EN with Tinker: staged plan

This scaffold assumes the current repo layout from `tv/README.md` and the aligned JSONL
outputs produced by:

- `scripts/scrape_bible.py`
- `scripts/scrape_articles.py`
- `scripts/scrape_daily_text.py`

## Current documentation on datasets / data collection

- [`README.md`](../README.md): collection strategy, URL mapping experiments, scrape scripts, and raw↔aligned artifacts.
- [`docs/UNSTRUCTURED_DATA_PIPELINE.md`](UNSTRUCTURED_DATA_PIPELINE.md): unstructured source ingest and OCR playbook for supplementary TVL↔EN data.
- [`tv2en.md`](../tv2en.md): cross-language URL/linking rules and pairing evidence.
- [`docs/SELECTIVE_TRANSLATION_SPEC.md`](SELECTIVE_TRANSLATION_SPEC.md): preservation guarantees for synthetic translation.
- [`docs/TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md): end-to-end stage execution and config inventory.
- [`docs/DATASET_COLLECTION_AND_ML_PIPELINE.md`](DATASET_COLLECTION_AND_ML_PIPELINE.md): consolidated field schema + alignment + split contracts.
- [`scripts/...` / `training/...`](../scripts): executable source-of-truth for schema and transformation details.

## What to optimize first

Do **not** try to make the first adapter be a fully general Tuvaluan assistant.
The first goal should be:

1. strong Tuvaluan→English translation
2. strong English→Tuvaluan translation
3. minimal damage to base-model English behavior by keeping the adapter separate

That first adapter is what unlocks the next stage: translating strong English instruction
and tool-use corpora into Tuvaluan.

## Why the data needs restructuring before training

Your aligned corpus is already close to what you need, but not in the format that will
best support bilingual post-training.

### Canonical source of truth

Keep `data/aligned/*.jsonl` as the truth layer.

Do **not** train directly from mixed scraper outputs. Instead, derive a new layer:

- `data/finetune/stage_a_mt/train_full.jsonl`
- `data/finetune/stage_a_mt/train_balanced.jsonl`
- `data/finetune/stage_a_mt/validation.jsonl`
- `data/finetune/stage_a_mt/test.jsonl`

### Why this split matters

Random verse-level splitting would leak near-duplicates across train and eval.
This is especially dangerous for Bible-heavy corpora.

Recommended policy:

- Bible: split by held-out books
- Articles: split by `doc_id`
- Daily text: split by `date`
- Other content: split by deterministic group hash

### What to exclude from v1

For the first MT adapter, exclude or isolate:

- low-confidence document-level article fallbacks
- pairs with extreme length ratios
- duplicate parallel rows
- very short fragments

That gives you a cleaner first model, even if it costs some coverage.

## Why Bible balancing matters

If you train on all pairs naively, the adapter will mostly learn scripture-shaped translation.
That is useful, but it can overfit the tone, phrasing, and vocabulary distribution.

For v1, keep two train sets:

- `train_full.jsonl`: everything that survives filtering
- `train_balanced.jsonl`: cap Bible share so non-Bible prose stays visible

Use `train_balanced.jsonl` first. If the adapter becomes too literal or too scripture-like,
reduce Bible share further.

## Recommended stage sequence

### Stage A — translation adapter

**Model**: `Qwen/Qwen3-30B-A3B-Base` (default) or `meta-llama/Llama-3.2-3B` (pilot shakeout).

**Decision**: Skip monolingual continued pretraining (CPT) — go straight to
translation SFT. Literature suggests CPT alone increases forgetting risk, and
our parallel corpus is too small for meaningful CPT gains.

Train only on high-confidence parallel data, both directions.

Input example:

```json
{
  "messages": [
    {"role": "system", "content": "You are a careful translator between Tuvaluan and English. Translate faithfully. Preserve names, numbers, punctuation, line breaks, and structure when possible. Output only the translation."},
    {"role": "user", "content": "Translate from Tuvaluan to English:\n\n..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Success criteria:

- chrF++ rises on held-out books and held-out article/doc/date sets
- both directions are usable
- English generation quality remains acceptable when the adapter is off

### Stage B — synthetic Tuvaluan capability data

Once Stage A is good enough, use it to translate English capability datasets into Tuvaluan.
But do **not** translate everything blindly.

Preserve these spans exactly unless human review shows they should change:

- code blocks
- JSON, XML, YAML, SQL
- function and tool names
- field names and schema keys
- variable names and identifiers
- URLs and file paths
- equations and most mathematical notation
- placeholders like `{name}` / `%s` / `<id>`

Translate only the human-language spans around them.

### Stage C (= Stage B in pipeline) — bilingual capability adapter

**Model**: `openai/gpt-oss-120b` — starts from BASE weights, **not** from Stage A.
Stage A exists only to produce synthetic Tuvaluan training data.

Create a second adapter using a mixture like:

- original English tool/use/math/QA data
- selectively translated Tuvaluan copies
- a smaller amount of the original parallel corpus as an anchor

The goal here is not just translation. It is bilingual instruction following.

## Mixing recommendations for Stage C (= Stage B in pipeline)

A reasonable starting point for a bilingual capability run is:

- 40% original English capability data
- 40% selectively translated Tuvaluan capability data
- 20% original TVL↔EN parallel translation data

If English capability regresses, increase the English share.
If Tuvaluan quality is weak, increase the translated Tuvaluan share.

## Evaluation plan

### Translation metrics

Track at least:

- chrF++ overall
- BLEU overall
- exact match for short segments
- per-direction metrics
- per-domain metrics
- per-content-type metrics

### Manual spot checks

Always inspect examples from:

- short Bible verses
- longer article paragraphs
- daily text with line breaks
- named entities
- numbers / dates / references

### Capability checks for later stages

Before merging or reusing the adapter for synthetic data generation, spot-check:

- JSON validity
- tool-call schema preservation
- arithmetic formatting
- code block preservation
- refusal behavior

## Dataset loaders: status

Some synthetic-data loaders remain TODO or experimental:

| Dataset | Status |
|---------|--------|
| tasksource/tasksource-instruct-v0 | Implemented |
| HuggingFaceH4/ultrachat_200k | Implemented |
| openai/gsm8k | Implemented |
| Salesforce/xlam-function-calling-60k | Implemented |
| Muennighoff/mbpp | Implemented |
| rajpurkar/squad | Implemented |
| ccdv/cnn_dailymail | Implemented |
| meta-math/MetaMathQA | TODO |
| NousResearch/hermes-function-calling-v1 | TODO |
| zai-org/AgentInstruct | TODO |

## Answer to the 10M-character question

10M total characters is probably enough for a **useful first translation adapter**, assuming
most of it is high-quality parallel text and you filter aggressively.

10M total characters is **not enough by itself** to make `gpt-oss-120b` a broadly capable,
fully bilingual Tuvaluan assistant across tooling, math, QA, coding, and open-domain chat.
For that second goal, you will need translated English capability data plus English replay.

## Practical defaults for v1

- **Stage A** base model: `Qwen/Qwen3-30B-A3B-Base` (small MoE, cheap to train)
- **Stage B** base model: `openai/gpt-oss-120b` (large MoE, 117B total / 5.1B active)
- training mode: LoRA
- LoRA rank: 32
- max length: 2048
- first train file: `train_balanced.jsonl`
- first LR sweep: `1e-4`, `2e-4`, `4e-4`
- first epoch sweep: `1`, `2`, `3`
- **Pilot path**: ~2M tokens, 1 epoch on Qwen3-30B-A3B-Base (config: `stage_a_translation_qwen30b_base_pilot_2m_1epoch.json`)

## Data contract and lineage enhancements to enforce (unified with code)

These are not just implementation details—they should be documented as part of each dataset release:

- `stage`-level outputs should keep this base shape:
  - `id`
  - `messages` (chat roles: `system`, `user`, `assistant`; optional `tool`/`tool_result` handling)
  - `task_family` (7-stage union: `chat`, `tool`, `math`, `code`, `qa`, `summarization`, `translation`)
  - `metadata`
- `metadata` for Stage A rows should retain canonical pair provenance:
  - `id`, `tvl`, `en`, `content_type`, `domain`, `alignment_method`,
    `alignment_confidence`, `doc_id`, `source_url_tvl`, `source_url_en`,
    `book_num`, `book_name`, `chapter`, `verse`, `date`, `pub_code`,
    `tvl_chars`, `en_chars`, `length_ratio`
  - plus injected fields from Stage A build:
    - `direction`, `source_lang`, `target_lang`, `template_idx`
- Stage B mix metadata should carry:
  - `metadata.stage_b_source` in `{english, synthetic_tvl, anchor}`
- Synthetic outputs should include:
  - `metadata.selectively_translated`, `metadata.tool_mode`
  - optional preservation details in `metadata.preservation`

### Current observed dataset state in this checkout

- Stage A artifacts are materialized under `data/finetune/stage_a_mt/` with `rejected.jsonl`,
  train/val/test, `stats.json`, and `manifest.json`.
- Stage B artifacts directories (`stage_b_sources`, `stage_b_synthetic_tvl`, `stage_b_mix`) are currently absent
  until pipeline steps 2–3 are run.

## What remains documented vs inferred

- Documented in docs:
  - broad stage strategy
  - loader coverage and TODOs
  - selectivity rules
  - train/eval commands
- Discovered from code but weakly documented:
  - exact `metadata` field propagation across stages
  - reject/reason schema (`too_short`, `bad_length_ratio`, `duplicate_pair`, `low_alignment_confidence`)
  - split mechanics by `book_num`, `doc_id`, `date`, and group hash
  - deterministic hash ordering/sampling in mix construction

## Deployment recommendation

Keep the base model untouched.
Ship the translation adapter separately.
Only consider merging weights after you have:

- stable held-out metrics
- stable manual checks
- evidence that later bilingual capability tuning does not break translation quality
