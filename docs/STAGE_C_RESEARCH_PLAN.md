# Stage C Plan From Research

This plan is based on the papers downloaded into `research/` and is aimed at the actual constraint we have right now:

- very limited native Tuvaluan data
- some Tuvaluan-only documents with no English pair
- expiring API credits
- limited time to ship a better Stage C dataset and adapter

This is not a generic roadmap. It is a short-horizon execution plan.

Related docs:

- [STAGE_C_NATIVE_GROUNDING_SPEC.md](STAGE_C_NATIVE_GROUNDING_SPEC.md)
- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)
- [UNSTRUCTURED_DATA_SOURCES.md](UNSTRUCTURED_DATA_SOURCES.md)

Last updated: `2026-03-30`

## 1. Bottom line

The strongest pattern across the paper set is:

1. target-language corpora are essential in low-resource adaptation
2. grounded native-document instruction data is a better bet than more generic translated chat data
3. filtering and selection matter as much as raw volume
4. preference tuning is worth doing, but only after grounded SFT exists
5. multilingual auto-judging is unreliable enough that eval must be source-aware and partly human-checked

So Stage C should be built as:

- OCR rescue for native news scans first
- native-document grounded SFT first
- bilingual and mixed-prompt expansion second
- small, clean DPO third
- selective translation only for missing structured behaviors
- continued pretraining and RL-style work only if time remains

## 1.1 What the current local data actually supports

The local corpus is not one uniform Stage C pool.

### Ready now for grounded Stage C work

- paired and bilingual government PDFs
- Nanumea oral-tradition material
- `Pai and Vau`
- `Toku Atufenua`
- native-heavy civic and health documents

### Useful now, but mainly as support data

- dictionary
- `corpus_v2`
- Tatoeba
- language cards
- Te Papa activity book
- biodiversity and grammar sources

These are useful for glossary control, short rewrites, lexical grounding, and bilingual anchors, but they are too phrase-heavy to be the behavioral backbone of Stage C.

### Mandatory but not training-ready yet

- `Tuvalu_News_Sheets_66-99.pdf`
- `Tuvalu_News_Sheets_Part 2.pdf`
- `The_magical_garlands_of_Nukufetau.pdf`

These are the most important native-document sources, but right now the pipeline stops at OCR text and term mining.

Why they were previously marked "not ready":

- the current OCR manifests use `lang: "eng"`
- many pages are multi-column and mixed-language
- there is no article boundary recovery yet
- the current pipeline only turns them into OCR term candidates, not grounded SFT rows

So the real blocker is not missing OCR. It is missing article-level TVL recovery.

## 2. What the papers say

### 2.1 Native corpora should be the center of gravity

The clearest support comes from:

- [2409.12958_muri_reverse_instructions.pdf](../research/2409.12958_muri_reverse_instructions.pdf)
- [2506.07597v1_low_resource_instructing_llms.pdf](../research/2506.07597v1_low_resource_instructing_llms.pdf)
- [2509.21294_grounded_synthetic_data.pdf](../research/2509.21294_grounded_synthetic_data.pdf)

Key takeaways:

- MURI shows reverse-instruction generation can turn existing human-written low-resource text into instruction pairs.
- The Basque study says target-language corpora are essential and that bilingual synthetic instructions are more robust than monolingual-only synthetic instructions.
- UPDESH shows grounded synthetic data built from language-specific source documents can improve culturally grounded performance.

Implication for us:

- Do not make Stage C mostly "translate English instructions into TVL."
- Make Stage C mostly "wrap native TVL documents in many task forms while keeping the answer grounded in real TVL."

### 2.2 Filtering beats naive scale

The strongest support comes from:

- [2307.08701_alpagasus.pdf](../research/2307.08701_alpagasus.pdf)

Key takeaways:

- a high-quality subset can beat a much larger noisy set
- automatic filtering and scoring can improve both quality and efficiency

Implication for us:

- build more candidates than we train on
- aggressively filter, deduplicate, and score before LoRA training

### 2.3 Small preference tuning is worth doing after SFT

The strongest support comes from:

- [2305.18290_dpo.pdf](../research/2305.18290_dpo.pdf)
- [2406.20052_mt_pref.pdf](../research/2406.20052_mt_pref.pdf)

Key takeaways:

- DPO is simpler and lighter than full RLHF
- language confusion is a real failure mode even in strong LLMs

Implication for us:

- build a small preference set targeting English leakage, wrong-language drift, translationese, and entity loss
- do not spend scarce time building a full reward-model-plus-RL stack

### 2.4 Eval must not rely on multilingual LLM judges alone

The strongest support comes from:

- [2505.12201v1_low_resource_eval_judging.pdf](../research/2505.12201v1_low_resource_eval_judging.pdf)

Key takeaways:

- multilingual LLM-as-a-judge is inconsistent across languages
- rubric design matters

Implication for us:

- use held-out native TVL evals
- use pairwise or axis-based evaluation
- keep a human-checked subset for judge calibration

### 2.5 Terminology and domain control are first-class

The strongest support comes from:

- [2023.wmt-1.80_terminology_aware_translation.pdf](../research/2023.wmt-1.80_terminology_aware_translation.pdf)
- [2411.11295v1_key_terms_low_resource_mt.pdf](../research/2411.11295v1_key_terms_low_resource_mt.pdf)
- [2102.10160_multidomain_tagging_mt.pdf](../research/2102.10160_multidomain_tagging_mt.pdf)

Key takeaways:

- terminology constraints improve recall of critical terms
- key-term retrieval helps low-resource translation
- domain tagging can serve multiple related sub-domains without separate models

Implication for us:

- add glossary-aware tasks
- preserve local names, dates, institutions, and sports/weather terms
- tag samples by domain, task, and register

### 2.6 Selective translation should be targeted, not dominant

The strongest support comes from:

- [2507.14304_selective_translation_low_resource.pdf](../research/2507.14304_selective_translation_low_resource.pdf)
- [2506.07597v1_low_resource_instructing_llms.pdf](../research/2506.07597v1_low_resource_instructing_llms.pdf)

Key takeaways:

- selective translation is useful when structure like code and JSON must be preserved
- filtering noisy translations matters
- mixing translated data with original English data matters

Implication for us:

- keep selective translation for tool use, JSON, math, and code-like behaviors
- do not let it become the majority of Stage C

### 2.7 Continued pretraining and RL are secondary bets

The strongest support comes from:

- [2412.13922_continued_pretraining_low_resource.pdf](../research/2412.13922_continued_pretraining_low_resource.pdf)
- [2601.12535v3_monolingual_rl_low_resource_mt.pdf](../research/2601.12535v3_monolingual_rl_low_resource_mt.pdf)
- [2503.09701v2_active_learning_llm_era.pdf](../research/2503.09701v2_active_learning_llm_era.pdf)

Key takeaways:

- continued pretraining on high-quality target-language text can be very valuable
- round-trip RL is promising but still a later-stage optimization
- active learning still matters when annotation budgets are tight

Implication for us:

- prepare a CPT-ready Tuvaluan pack, but do not block Stage C on CPT
- treat round-trip RL as optional later work
- use active error mining to choose what to generate next

## 3. Stage C thesis

Stage C should be a native-document grounding pipeline with five layers:

1. OCR recovery for native news scans
2. native TVL documents as answer anchors
3. prompt/task expansion around those answers
4. filtering and scoring
5. preference tuning and hard eval on the winning mix

In practice, this means:

- the native news scans must be upgraded from OCR dumps into article-level TVL sources
- the assistant side stays in real or source-supported Tuvaluan whenever possible
- prompt diversity expands around the native answer
- English and mixed prompts are used as controlled mirrors, not as the base distribution

## 4. Concrete plan

## Phase 0: OCR rescue for native news scans

Goal:

- make the OCR-only native news corpus genuinely usable for Stage C

Why this is required:

- it is the closest thing we have to a broad native-TVL document pool
- without it, Stage C over-relies on paired PDFs and short lexical sources
- the current OCR pipeline only gives us term mining, not grounded article data

Current evidence:

- `data/external/ocr_scans/Tuvalu_News_Sheets_66-99-p1-80.manifest.json` shows the scan was OCR'd with `lang: "eng"`
- `data/external/ocr_scans/Tuvalu_News_Sheets_66-99-p161-240.txt` shows mixed columns and corrupted lines, which are not ready to use as grounded answers

Required outputs:

1. page-level layout detection
2. TVL/EN column separation when pages are bilingual
3. article boundary recovery
4. normalized TVL article text with page provenance
5. QA metadata per recovered article
6. a Stage C `ocr_recovered` pool

Concrete steps:

1. Re-run OCR at higher quality settings for the native-news scans.
2. Classify each page as `tvl_only`, `en_only`, `bilingual_two_column`, or `mixed_garbled`.
3. Split columns before extraction when needed.
4. Normalize recurring OCR failures:
   - split words
   - merged headers
   - punctuation drift
   - broken vowels
   - column bleed
5. Recover article boundaries rather than storing only page text.
6. Human-spot-check recovered article text before Stage C generation.

Acceptance gate:

- no Stage C run should treat the news scans as ready until we can recover article-level TVL text with provenance and manual spot-check support

## Phase 1: Build the held-out eval first

Goal:

- prevent blind data generation

Actions:

1. Select held-out native TVL documents by document, not by random row.
2. Cover at least these slices:
   - news
   - government/civic notices
   - sports/weather
   - short admin text
   - named-entity heavy passages
   - mixed TVL/EN prompt handling
3. Score on:
   - in-language fidelity
   - source support
   - entity preservation
   - register fit
   - formatting correctness when relevant

Why first:

- [2505.12201v1_low_resource_eval_judging.pdf](../research/2505.12201v1_low_resource_eval_judging.pdf) says multilingual judging is unreliable enough that we need a stable target before scaling synthesis.

## Phase 2: Native-document grounded SFT

Goal:

- make the model better at native TVL writing and following TVL instructions

Actions:

1. Build a registry of native TVL source docs.
2. Segment each source into grounded answer units.
3. Generate multiple prompt families around each source-supported answer:
   - native TVL request
   - English request asking for a TVL answer
   - mixed TVL/EN request
   - fact-sheet-to-article
   - headline
   - lead
   - short summary
   - medium summary
   - QA
   - radio rewrite
   - formal rewrite
4. Keep provenance fields on every example.

Source tiers for this phase:

- Tier A, backbone:
  - recovered native news scans
  - Nanumea tales
  - `Toku Atufenua`
  - `Pai and Vau`
  - native-heavy government/civic PDFs
- Tier B, support:
  - paired PDFs
  - bilingual PDFs
  - biodiversity
  - grammar
- Tier C, glossary/anchor only:
  - dictionary
  - `corpus_v2`
  - Tatoeba
  - language cards

Why:

- MURI, the Basque study, and UPDESH all point to grounded native text as the best use of scarce low-resource data.

## Phase 3: Filtering and balancing

Goal:

- train on the best slice, not the biggest slice

Actions:

1. Deduplicate by normalized source-answer content.
2. Remove English leakage and wrong-language drift.
3. Remove unsupported or weakly supported generations.
4. Keep hard caps so one source family does not dominate.
5. Rank examples and keep a top slice for training.

Filters to implement:

- language purity
- entity/date consistency
- source-support confidence
- duplicate family collapse
- answer-length sanity
- task diversity balancing
- OCR corruption flags

Why:

- [2307.08701_alpagasus.pdf](../research/2307.08701_alpagasus.pdf) supports the "smaller but cleaner" strategy strongly.

## Phase 4: Run the right ablations

Do not guess the prompt mixture. Test it.

Required ablation arms:

1. `native_only`
   - native TVL prompts -> native TVL answers
2. `native_plus_english`
   - native TVL prompts + English prompts that request TVL answers
3. `native_plus_stage_b_translated`
   - native TVL prompts + Stage-B-translated prompt mirrors
4. `native_plus_bilingual`
   - native TVL prompts + English + mixed bilingual prompts

Decision rule:

- if `native_only` wins, translation artifacts are hurting
- if `native_plus_english` or `native_plus_bilingual` wins, keep English anchors
- if `native_plus_stage_b_translated` wins, translated mirrors are safe enough to scale

Why:

- [2506.07597v1_low_resource_instructing_llms.pdf](../research/2506.07597v1_low_resource_instructing_llms.pdf) supports testing bilingual mixtures rather than assuming one prompt language is best.

## Phase 5: Terminology and domain control

Goal:

- improve the slices that low-resource systems usually miss first

Actions:

1. Create glossary-aware tasks for:
   - island names
   - ministries and institutions
   - fisheries and weather terms
   - sports vocabulary
   - date and time expressions
2. Add coarse tags:
   - `<domain=news>`
   - `<domain=government>`
   - `<domain=sports>`
   - `<register=formal>`
   - `<register=radio>`
   - `<task=summary>`

Why:

- terminology and domain signals are directly supported by the WMT terminology paper, the key-term retrieval paper, and the multidomain tagging paper.

## Phase 6: Small clean DPO

Goal:

- reduce the specific failures that SFT alone will not fully fix

Preference targets:

- English leakage
- wrong-language tokens
- translationese
- dropped entities
- wrong register
- repetitive phrasing
- over-formal or Bible-like drift in non-religious text

Actions:

1. Sample 2 to 4 candidates per prompt from the best SFT checkpoint.
2. Build chosen/rejected pairs.
3. Keep the preference set small and clean.
4. Run DPO on top of the winning SFT mix.

Why:

- [2305.18290_dpo.pdf](../research/2305.18290_dpo.pdf) makes DPO the simplest practical alignment step
- [2406.20052_mt_pref.pdf](../research/2406.20052_mt_pref.pdf) shows language confusion is real enough to target directly

## Phase 7: Targeted selective translation

Goal:

- cover missing behaviors without drowning the model in translated data

Use selective translation only for:

- tool use
- JSON and schema following
- code-adjacent behavior
- math
- safety templates
- agent formatting

Why:

- [2507.14304_selective_translation_low_resource.pdf](../research/2507.14304_selective_translation_low_resource.pdf) supports selective translation when structure must be preserved, but also supports filtering and careful mixing.

## 5. Effort split

Recommended short-horizon allocation:

| Workstream | Share |
|---|---:|
| OCR rescue for native news scans | 20% |
| Native grounded SFT from TVL documents | 20% |
| Filtering, scoring, dedup, balancing | 15% |
| Eval construction and calibration | 15% |
| Prompt-mixture ablations | 10% |
| Terminology + domain tagging | 10% |
| Small clean DPO pass | 10% |
| Targeted selective translation for missing behaviors | 5% |
| CPT pack preparation and active error mining | 5% |

This intentionally delays RL-heavy and CPT-heavy bets until after the first grounded Stage C result exists.

## 6. What to build in the repo

Proposed artifact layout:

```text
data/
  stage_c/
    ocr_recovered/
      native_news_articles.jsonl
      magical_garlands_segments.jsonl
    registry/
      native_doc_registry.jsonl
    grounded_sft/
      native_only.jsonl
      native_plus_english.jsonl
      native_plus_stage_b_translated.jsonl
      native_plus_bilingual.jsonl
    preferences/
      language_fidelity.jsonl
      register_control.jsonl
    eval/
      held_out_native.jsonl
      rubric.yaml
    reports/
      filtering_report.json
      ablation_results.json
```

## 7. Fastest experiment set

If we only have a few days, run exactly this:

1. Fix OCR recovery for the native news scans.
2. Build one grounded native-TVL SFT set with source tiers.
3. Build one filtered subset and one unfiltered subset.
4. Train the 4 prompt-mixture LoRAs.
5. Evaluate on the held-out set.
6. Pick the best arm.
7. Run one small DPO pass on the winner.

This yields the cleanest answer to the actual question: what data shape improves Stage C fastest?

## 8. Deprioritized for now

Do not make these the center of the next few days:

- large generic translated-English chat generation
- full RLHF infrastructure
- broad round-trip RL
- massive unfiltered synthetic expansion
- over-fine-grained dialect tags before we have enough data

Those may matter later, but the paper set does not support them as the best first move under our current constraints.

## 9. Recommended next implementation tasks

1. Build a native document registry from the current local sources.
2. Add an OCR-recovery path for native news scans that outputs article-level TVL text.
3. Add a grounded task expander that keeps source provenance.
4. Add a filter/scorer pass.
5. Add an eval builder for held-out native TVL.
6. Add a preference-pair builder for leakage and entity preservation.
7. Train the four ablation LoRAs.

That is the shortest path from the paper set to a real Stage C improvement.
