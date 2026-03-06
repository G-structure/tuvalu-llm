# TVL↔EN on gpt-oss-120b with Tinker: staged plan

This scaffold assumes the current repo layout from `tv/README.md` and the aligned JSONL
outputs produced by:

- `scripts/scrape_bible.py`
- `scripts/scrape_articles.py`
- `scripts/scrape_daily_text.py`

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

- `data/finetune/tinker_mt/train_full.jsonl`
- `data/finetune/tinker_mt/train_balanced.jsonl`
- `data/finetune/tinker_mt/validation.jsonl`
- `data/finetune/tinker_mt/test.jsonl`

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

### Stage C — bilingual capability adapter

Create a second adapter (or continue from the translation adapter only after careful eval)
using a mixture like:

- original English tool/use/math/QA data
- selectively translated Tuvaluan copies
- a smaller amount of the original parallel corpus as an anchor

The goal here is not just translation. It is bilingual instruction following.

## Mixing recommendations for Stage C

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

## Answer to the 10M-character question

10M total characters is probably enough for a **useful first translation adapter**, assuming
most of it is high-quality parallel text and you filter aggressively.

10M total characters is **not enough by itself** to make `gpt-oss-120b` a broadly capable,
fully bilingual Tuvaluan assistant across tooling, math, QA, coding, and open-domain chat.
For that second goal, you will need translated English capability data plus English replay.

## Practical defaults for v1

- base model: `openai/gpt-oss-120b`
- training mode: LoRA
- LoRA rank: 32
- max length: 2048
- first train file: `train_balanced.jsonl`
- first LR sweep: `1e-4`, `2e-4`, `4e-4`
- first epoch sweep: `1`, `2`, `3`

## Deployment recommendation

Keep the base model untouched.
Ship the translation adapter separately.
Only consider merging weights after you have:

- stable held-out metrics
- stable manual checks
- evidence that later bilingual capability tuning does not break translation quality
