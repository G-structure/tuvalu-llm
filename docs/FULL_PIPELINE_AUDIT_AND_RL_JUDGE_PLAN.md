# Full Pipeline Audit And RL Judge Plan

This document audits the current Tuvaluan LLM pipeline and turns it into an
execution plan for a stronger vNext run.

The goal is not to claim generic frontier-model superiority. The goal is to
build and verify a specialized Tuvaluan system that can beat general frontier
models and public Tuvaluan MT baselines on frozen, source-disjoint Tuvaluan task
slices.

Last updated: 2026-05-15.

## Executive Summary

The repo already contains the right architecture for a serious low-resource
language model:

- a corpus pipeline with cleaning, split construction, and leakage checks
- Stage A translation training
- Stage B bilingual capability training
- Stage C native-document grounding
- product feedback export from the football app
- early benchmark artifacts and Stage C reports

The next work is evidence and alignment, not just more training. The strongest
vNext path is:

1. Rebuild or restore all missing runtime artifacts.
2. Freeze a benchmark that includes public baselines, ChatGPT 5.3, and the
   current project models.
3. Improve Stage A because Stage B synthetic quality depends on it.
4. Rebuild Stage B with explicit mix ratios and synthetic filtering.
5. Expand Stage C as high-quality native steering data.
6. Run the Codex subscription as a tool-using teacher (Phase 4.5) to generate
   native TVL chat, tool-call trajectories, and rejection-sampling candidates
   that Stage A cannot produce. Route every output through the same judge
   gates as any other synthetic row.
7. Use GPT-5.5 as a RAG-backed offline judge to create calibrated preference
   pairs.
8. Train preference-tuned models with DPO or ORPO before attempting heavier RL.

The current public wording should stay careful:

> To our knowledge, this is the strongest publicly documented Tuvaluan-English
> LLM pipeline. A final SOTA claim requires apples-to-apples evaluation against
> Helsinki OPUS, MADLAD/NLLB-family baselines where applicable, and frozen
> frontier-model outputs.

## Scope

This plan covers:

- data artifact audit
- Stage A/B/C training pipeline risks
- public baseline and ChatGPT comparison plan
- GPT-5.5 RAG judge design
- preference data contracts
- DPO/ORPO/RL sequencing
- implementation backlog and go/no-go gates

This plan does not cover:

- exact LoRA hyperparameter sweeps
- serving architecture
- UI/product design
- final benchmark results that have not been run yet

## Source Notes

OpenAI docs used for the judge plan:

- [gpt-5.3-chat-latest model docs](https://developers.openai.com/api/docs/models/gpt-5.3-chat-latest)
- [gpt-5.5 model docs](https://developers.openai.com/api/docs/models/gpt-5.5)
- [File search guide](https://developers.openai.com/api/docs/guides/tools-file-search)
- [Vector stores API docs](https://developers.openai.com/api/docs/api-reference/vector-stores)
- [Evals guide](https://developers.openai.com/api/docs/guides/evals)
- [Graders docs](https://developers.openai.com/api/docs/api-reference/evals/graders)

Public comparison sources to include in the SOTA verification run:

- [Helsinki-NLP/opus-mt-tvl-en](https://huggingface.co/Helsinki-NLP/opus-mt-tvl-en)
- [FriezaForce/tvl-en-llm-translation-stage-a](https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-a)
- [google/madlad400-3b-mt](https://huggingface.co/google/madlad400-3b-mt)

## Claim Ladder

Use this ladder when writing README, blog, model cards, or submission copy.

| Level | Wording | Allowed now? | Requirement |
|---|---|---:|---|
| L1 | "Largest public Tuvaluan-English pipeline we know of" | Yes, with qualifier | Cite dataset/model cards and search notes |
| L2 | "Outperforms public Tuvaluan MT baselines on our benchmark" | Not yet | Run public baselines on the same held-out test set |
| L3 | "Outperforms ChatGPT 5.3 on scoped Tuvaluan tasks" | Not yet | Freeze ChatGPT outputs and compare on a fixed benchmark |
| L4 | "State of the art for Tuvaluan-English translation" | Not yet | Apples-to-apples benchmark against OPUS, MADLAD/NLLB where applicable, and frontier APIs |
| L5 | "State of the art Tuvaluan assistant" | Not yet | Native-speaker or expert human eval on assistant tasks |

Recommended public wording until the vNext benchmark is complete:

> Our Stage A model reports stronger self-eval numbers than the public Helsinki
> OPUS Tuvaluan-English baseline reports on JW300, and we are preparing an
> apples-to-apples comparison to make the SOTA claim cleanly.

## Current Evidence State

This checkout contains source code, configs, docs, reports, and eval artifacts.
It does not currently contain the large runtime directories:

- `data/`
- `unstruct_lang_data/`
- `logs/`

Because those directories are absent, the historical reports should be treated
as prior evidence, not as a fresh reproducible run from this exact checkout.

Run these checks before any new training or benchmark claim:

```bash
find . -maxdepth 3 -type d \( -name data -o -name unstruct_lang_data -o -name logs \) -print
uv run --extra training pytest tests/
```

If `data/` is absent, the first task is artifact restore or regeneration.

## Pipeline Inventory

| Layer | Purpose | Current implementation | Evidence status |
|---|---|---|---|
| Corpus | Scrape, clean, split, render TVL/EN pairs | `scripts/scrape_*.py`, `tv/corpus/*` | Code present, runtime data absent |
| Stage A | Train TVL <-> EN translation adapter | `tv/training/stage_a_mt/*` | Code present, public model card exists |
| Stage B | Train bilingual capability adapter | `tv/training/stage_b_agent/*` | Code/configs present, mix needs canonicalization |
| Stage C | Native source grounding and DPO candidates | `tv/training/stage_c/*` | Reports present, artifacts absent in checkout |
| Codex subscription distill/augment | Tool-using teacher for native TVL chat + tool-call trajectories + rejection-sampling candidates | `src/codex-proxy/*` (in sibling repo); `tv/data/codex/*` to add | Proxy implementation lives in the sibling `rl-agent-work` workspace; this repo has no data rows yet |
| Product loop | Collect corrections and feedback | `site/`, `tv/apps/football/*` | Code present, signals need preference conversion |
| Eval | Benchmark and native eval | `eval/`, `scripts/*eval*` | Artifacts present, baseline needs refresh |

## Audit Findings

### What Is Strong

- Raw aligned data is documented as immutable.
- Cleaning emits accepted rows, rejected rows, and rejection reasons.
- Split logic is source-aware: Bible by book, articles by `doc_id`, daily text
  by date.
- Cross-source decontamination checks exact text, held-out n-grams, and short
  verse containment.
- Stage A and Stage B are conceptually separated correctly: Stage A is a
  translator used to create data, not a weight base for Stage B.
- Selective translation has structure checks for placeholders, code, JSON, and
  length ratios.
- Stage C moves the system away from translated-English-only behavior and
  toward native, source-backed Tuvaluan answers.
- Stage C already has task families, support classes, held-out eval slices, and
  DPO-style preference rows.

### What Blocks A Clean SOTA Claim

| Blocker | Impact | Required fix |
|---|---|---|
| Missing runtime artifacts in this checkout | Cannot reproduce reported data counts locally | Restore or rebuild `data/`, `unstruct_lang_data/`, and `logs/` |
| Public baseline not run on the same test set | Helsinki OPUS comparison is suggestive but not decisive | Run OPUS on our held-out test and run ours on OPUS/JW300 if fetchable |
| ChatGPT 5.3 baseline not frozen | Future API/model changes can move the target | Save prompts, outputs, dates, model ids, and sampling settings |
| Stage A path ambiguity | `stage_a_mt` vs `stage_a_mt_v2` can create hidden experiment drift | Choose one canonical vNext anchor path |
| Stage B mix mismatch | Docs and implementation mention different ratios in places | Store requested and realized ratios in every manifest |
| Synthetic TVL may contain translation artifacts | Stage B can learn Stage A errors | Filter with structure checks, judge checks, and human spot checks |
| Stage C is small | It is steering data, not bulk coverage | Expand by active learning and source recovery |
| Current eval proxies are incomplete | Token overlap cannot fully judge fluency or factuality | Add RAG judge and human-calibrated review |
| Product feedback is noisy | Raw feedback is not automatically preference data | Convert to typed preference candidates with confidence labels |

## vNext Training Plan

### Phase 0: Restore And Lock Artifacts

Goal: make the run reproducible before spending training budget.

Deliverables:

- artifact presence report
- source file hash manifest
- config snapshot
- git hash and dirty-state manifest
- current row counts by source, split, and stage

Commands to begin from a restored workspace:

```bash
uv run scripts/clean_pipeline.py
uv run scripts/build_splits.py
uv run scripts/validate_splits.py
uv run scripts/render_training_data.py --include-unstructured
uv run --extra training pytest tests/
```

Acceptance criteria:

- split validation passes
- manifests include data hashes
- accepted/rejected counts are explained
- one Stage A anchor path is selected for vNext

### Phase 1: Freeze The Benchmark

Goal: know what "better" means before training.

Benchmark competitors:

- current Stage A
- current Stage B
- current Stage C arms, if available
- `Helsinki-NLP/opus-mt-tvl-en`
- `Helsinki-NLP/opus-mt-en-tvl`
- MADLAD-400 variants that support `tvl`
- NLLB-family models only if Tuvaluan support can be confirmed
- `gpt-5.3-chat-latest`
- GPT-5.5 as judge, not as a target product model unless explicitly compared

Benchmark slices:

| Slice | Count | Notes |
|---|---:|---|
| TVL -> EN translation | 300 | Source-disjoint held-out docs |
| EN -> TVL translation | 300 | Include religious, civic, news, narrative |
| Native TVL chat | 200 | Natural prompts, not synthetic-only |
| Grounded QA | 200 | Requires source evidence |
| Summarization | 100 | Short and medium summaries |
| Entity preservation | 100 | Names, dates, amounts, quotes |
| Product feedback repair | 100 | Real correction-style tasks |
| Structured/tool tasks | 100 | JSON/code/tool preservation |

Acceptance criteria:

- all prompts are saved
- all outputs are saved with model id and date
- eval rows identify train/val/test/eval source splits
- no eval source document appears in training
- baseline scripts can be rerun non-interactively

### Phase 2: Stage A vNext Translation

Goal: improve the translator that feeds synthetic data generation.

Training arms:

| Arm | Purpose |
|---|---|
| Current Qwen recipe | Baseline continuation |
| Strict-cleaning arm | Test whether lower noise beats higher volume |
| Unstructured-seed arm | Test dictionary, OCR, and civic/narrative additions |
| Domain-balanced arm | Reduce Bible/WOL over-dominance |
| Stronger-base arm | Use only if budget and infra allow |

Selection metrics:

- chrF++ and BLEU by direction
- entity/date/number preservation
- domain-sliced performance
- native/civic human spot checks
- OPUS/JW300 comparison if fetchable

Acceptance criteria:

- vNext Stage A beats current Stage A on project held-out translation
- vNext Stage A beats or matches OPUS on apples-to-apples TVL/EN slices
- no regression on entity preservation

### Phase 3: Stage B vNext Bilingual Capability

Goal: preserve general task capability while making Tuvaluan interaction
natural and robust.

Recommended initial mix:

| Pool | Share | Reason |
|---|---:|---|
| English capability | 25% | Preserve general ability |
| Synthetic TVL capability | 30% | Teach task formats in TVL |
| Crosslingual EN prompt -> TVL answer | 15% | Match bilingual user behavior |
| Parallel anchor | 20% | Preserve translation quality |
| Real TVL chat/product feedback | 10% | Move toward product behavior |

Required filters:

- existing structure checks
- duplicate and contamination checks
- language-id checks
- protected-term preservation checks
- GPT-5.5 RAG judge checks for grounded examples
- human review sample per source family

Acceptance criteria:

- realized mix ratios are stored in the manifest
- Stage B does not regress Stage A translation beyond the agreed threshold
- JSON/code/tool task validity is measured and reported
- synthetic rejection reasons are summarized

### Phase 4: Stage C vNext Native Grounding

Goal: make native Tuvaluan source fidelity the differentiator.

Use Stage C as high-quality steering data, not bulk data.

Priorities:

1. Expand native source recovery for civic, health, education, finance, oral
   narrative, historic news, and community documents.
2. Preserve source-disjoint train/val/eval splits.
3. Promote only `direct_support`, `light_transform`, and `fact_compilation`
   examples into default SFT.
4. Keep `weak_support` rows for analysis, adversarial eval, or judge
   calibration.
5. Increase the held-out eval set beyond the current 56 reported rows.

Acceptance criteria:

- Stage C manifest records source family and support class
- train/val/eval source document overlap is zero
- default SFT arm remains mostly TVL assistant output
- held-out eval includes noisy OCR and terminology slices

### Phase 4.5: Codex Subscription Distillation And Augmentation

Goal: use the OpenAI Codex subscription (GPT-5.x served through the Codex CLI
harness) as a tool-using teacher to (a) generate high-quality synthetic
Tuvaluan training rows where Stage A translation is too narrow, (b) produce
multi-turn tool-call trajectories Stage B's "structured/tool tasks" pool can
train on, and (c) supply candidate answers for Phase 5 rejection-sampling SFT.

This phase sits between Stage C native grounding and the preference layer
because:

- Stage C provides the source-grounded retrieval corpus the codex prompts will
  be conditioned on.
- The judge in the next section is what scores codex outputs before they enter
  any training set.
- Phase 5 rejection-sampling SFT is the consumer that turns codex's K
  candidates into the chosen row.

Codex is **not** a Tuvaluan oracle. It is competent in English with strong
tool-use and reasoning, weaker in Tuvaluan, and entirely capable of inventing
sources. Every codex output must pass the same judge gates as any other
synthetic row before it is allowed into training.

#### Architecture

The codex harness already exists in this repo's wider workspace:

- [`src/codex-proxy`](../../src/codex-proxy/) — translates between the
  OpenAI Responses API the CLI speaks and our trainable backend's token
  interface; captures per-token logprobs and writes per-rollout trajectory
  records. See `codex-training-proxy.md` and
  `codex-harness-rl-training-landscape.md` for the design rationale and the
  prior art survey.
- [`vendor/ttt_discover/codex_runtime/`](../../vendor/ttt_discover/codex_runtime)
  — the reference `LoggingProxy` implementation the proxy is a port of.

Three deployment modes:

| Mode | Direction | Use |
|---|---|---|
| `harvest` | Codex CLI -> subscription endpoint, proxy records the trajectory | Capture teacher trajectories on Tuvaluan tasks |
| `replay` | Programmatic Responses API client -> subscription, proxy records | Bulk-generate synthetic rows from prompt batches |
| `serve` | Codex CLI -> our trained model, proxy records | Eval-time and product-loop after training |

The `harvest` and `replay` modes are the ones that feed this phase; `serve` is
a Phase 5+ concern.

#### Roles The Codex Subscription Fills

1. **Synthetic TVL chat generation.** Replaces or augments the 30% "Synthetic
   TVL capability" pool in the Stage B mix. Currently sourced from Stage A
   translations of English prompts, which inherits Stage A's failure modes
   (low-resource MT noise, religious/JW300 register bleed). Codex, conditioned
   on Stage C retrieved spans, generates native-task-shaped answers that
   Stage A cannot.
2. **Tool-call and structured-output trajectories.** The Stage B mix lists
   "Structured/tool tasks - 100 examples" in the eval slices but is silent on
   their training-side source. Codex emits real `function_call` items as part
   of its harness behavior; recorded trajectories supply these directly with
   matching prompts.
3. **Active-learning loop on Stage A failures.** When Stage A's translator
   has low confidence or low chrF++ on a row (entity drop, mid-document
   structure mismatch), route the row to a codex `replay` job conditioned on
   the source span. The result enters the preference pool, not the bulk SFT
   pool.
4. **Phase 5 candidate generation.** Rejection-sampling SFT requires multiple
   candidates per prompt. Codex with `n=K, temperature>0` over the same task
   set yields K candidates per prompt; the judge picks `chosen` and one or
   more `rejected`.
5. **Reasoning-mode supervision (optional, post-MVP).** Codex exposes
   `{type: "reasoning"}` items with a reasoning summary. These can supervise
   a "show your work then answer" trace for Tuvaluan tasks. Default off until
   we have evidence this helps and doesn't make the model imitate codex's
   English-language reasoning prose.

#### Prompt Construction

Every codex call in this phase is **grounded** by default. The system message
and `instructions` field must:

- declare the requested output language (typically `tvl`)
- pin protected terms, dates, amounts, quotes, and named entities that must
  survive verbatim
- inject the retrieved Stage C spans into the prompt with explicit `[span_id]`
  citation markers
- list the allowed tool surface (usually just `apply_patch` if writing files,
  none for pure-text generation, never `shell` outside a sandboxed eval env)
- name the task family so the judge knows which rubric weights to apply

The prompt template lives at
`tv/data/codex/prompts/<task_family>.j2` and is hashed into the row manifest.

#### Data Contract

Each codex trajectory writes a row to `data/codex/<release>/<task_family>.jsonl`:

```json
{
  "id": "codex_stage_d_000001",
  "task_family": "qa_grounded",
  "prompt_id": "stage_c_eval_00042",
  "prompt": "...",
  "answer": "...",
  "answer_language": "tvl",
  "trajectory": {
    "messages": [{"role": "...", "content": "..."}],
    "tool_calls": [{"name": "apply_patch", "arguments": "...", "output": "..."}],
    "reasoning_summary": "...",
    "completion_token_ids": [12345, 67890, "..."],
    "completion_logprobs": [-0.21, -1.04, "..."],
    "stop_reason": "stop"
  },
  "codex_model": "gpt-5.5",
  "codex_temperature": 0.7,
  "codex_top_p": 1.0,
  "codex_n": 1,
  "subscription_request_id": "resp_...",
  "retrieved_span_ids": ["doc_12:p4", "doc_12:p5"],
  "protected_terms": ["Funafuti", "2025"],
  "prompt_template_hash": "sha256:...",
  "judge_status": "pending",
  "decontam_status": "pending",
  "language_id_pass": null,
  "metadata": {
    "source_doc_id": "doc_12",
    "source_split": "train",
    "created_at": "2026-05-15T00:00:00Z",
    "proxy_record_id": "..."
  }
}
```

Notes:

- `completion_token_ids` and `completion_logprobs` are captured by the proxy
  for free and are useful for distillation experiments later; they are not
  required to be present in MVP rows.
- `judge_status` starts as `pending`; gets updated by the Phase 5 judge run.
- `decontam_status` starts as `pending`; gets updated by a decontamination
  check that asserts the prompt + answer do not appear in any eval split.

#### Acceptance Gates

A codex row may enter Stage B SFT only if it passes all of:

| Gate | Implementation |
|---|---|
| Language ID match | fastText langid on answer = `answer_language` |
| Protected-term preservation | regex/string presence of each entry in `protected_terms` |
| Source-support entailment | judge `source_support >= 4` (rubric in next section) |
| Tuvaluan quality | judge `tuvaluan_quality >= 3` |
| Decontamination | answer not appearing in any held-out eval document by exact text or held-out n-gram check |
| Tool-call structure | if `tool_calls` present, every call's arguments validate against the declared JSON schema |
| Privacy | no PII or product-private content unless explicit permission tag |
| Cost cap | per-task-family cap on subscription spend, measured by recorded request count and `usage` metadata when available |

Rows that fail any gate are written to a parallel `data/codex/<release>/rejected/<task_family>.jsonl`
with the failing gate id, never deleted, so audit and retry are possible.

#### Operational Concerns

- **Rate limits.** The codex subscription enforces requests-per-minute and
  daily caps. The replay generator must throttle, respect 429 Retry-After,
  and write a checkpoint after every K rows so a restart resumes cleanly.
  Per-task-family quotas in `configs/codex_quota.yaml`.
- **Cost.** Each request is metered. The harvest mode also pays for the tool
  calls the harness chooses to make. Budget per release is recorded in the
  manifest; runs that would exceed it stop without partial-row corruption.
- **Determinism.** Codex is not bit-reproducible. Every row stores the codex
  model id, temperature, top_p, seed (when supported), and the OpenAI
  `request_id` so the row is traceable back to a specific subscription call.
- **Subscription rotation.** If the subscription tier changes (different
  model, different limits), the row's `codex_model` field is the only field
  that disambiguates between rows generated under different policies; do not
  mix codex_model values within a single training mix without an explicit
  per-row indicator in the manifest.
- **Decontamination.** The codex subscription's training cutoff is unknown
  but post-dates much of our held-out Tuvaluan source material. Every row
  must be checked against eval splits before entering training. Eval prompts
  that match codex outputs are flagged as **contaminated** and excluded from
  eval, not laundered into training.

#### Telemetry

Emit per-row:

- `codex.row_count` by task_family and judge_status
- `codex.cost_dollars` by task_family
- `codex.rate_limit_hits` by hour
- `codex.gate_rejection_rate` by gate id
- `codex.decontam_collisions` (should be near zero; if not, investigate)

#### Deliverables

- `tv/data/codex/` module: prompt templates, replay runner, harvest runner,
  rate-limited subscription client
- `configs/codex_quota.yaml`: per-task-family quotas
- `data/codex/<release>/`: per-task-family JSONL of accepted and rejected
  rows
- Manifest: prompt template hashes, codex model id, total subscription cost,
  rate-limit incidents
- Decontamination report: collisions found between codex outputs and eval
  splits

#### Acceptance Criteria

- realized accept rate per task family is reported and explained
- accepted-row decontamination check shows zero overlap with any eval split
- Stage B mix manifest distinguishes codex-sourced rows from Stage-A-sourced
  synthetic rows
- subscription cost is within the budget declared in the manifest
- a per-row replay is possible from `proxy_record_id` for at least 30 days
  after generation

### Phase 5: Preference And RL Layer

Goal: optimize for source-faithful, natural Tuvaluan answers without letting a
judge become an ungrounded oracle.

Recommended sequence:

1. Rejection-sampling SFT: generate multiple candidates **via Phase 4.5
   codex replay with n=K, temperature>0**, keep the best source-grounded
   answer (selected by the judge).
2. Pairwise preference data: store chosen/rejected outputs with source evidence.
   `chosen` and `rejected` can both come from the same Phase 4.5 codex batch
   (different sampling settings or different `replay` runs) or from a
   chosen-codex / rejected-current-model pairing.
3. DPO or ORPO: run stable offline preference optimization.
4. Reward model: train only after judge labels agree with humans.
5. PPO/GRPO-style RL: defer until reward hacking tests pass.

Do not start PPO-style online RL first. This is a small, high-stakes,
source-grounded language setting. Offline preference optimization is the safer
first step.

## GPT-5.5 RAG Judge

GPT-5.5 is useful here because it can serve as an offline evaluator with file
search/retrieval support. It should not be treated as inherently correct about
Tuvaluan. The judge must be grounded in retrieved source spans and calibrated
against human labels.

### Judge Responsibilities

The judge should:

- rank candidate answers
- explain the evidence for the ranking
- hard-fail unsupported hallucinations
- flag wrong-language drift
- flag entity/date/number corruption
- flag malformed structured outputs
- identify examples that require human review

The judge should not:

- invent source facts
- replace native-speaker review
- label examples without retrieved evidence when grounding is required
- train the product model on long judge rationales by default

### Retrieval Corpus

Build a release-specific retrieval corpus from:

- held-in training source documents for training preference generation
- held-out eval source documents for evaluation only
- glossary and terminology files
- product correction context when permission and privacy allow

Each chunk should include metadata:

```json
{
  "span_id": "doc_001:p004:s02",
  "source_doc_id": "doc_001",
  "source_path": "data/external/stage_c_seed/...",
  "source_family": "government_pdf",
  "split": "train",
  "page": 4,
  "language": "tvl",
  "support_class": "direct_support",
  "ocr_confidence": 0.91,
  "copyright_status": "public_or_public_facing_document"
}
```

Retrieval rules:

- never retrieve train documents when judging eval documents
- retrieve top-k semantic spans
- add lexical fallback spans for protected names, dates, amounts, and quotes
- store retrieved span ids in every judge output
- store retrieval settings in the run manifest

### Judge Inputs

Each judge request should include:

- task id
- task family
- user prompt
- expected language
- candidate answer or candidate pair
- retrieved evidence spans
- protected entities, dates, numbers, and quotes
- source metadata
- output format requirement

### Rubric

Score each answer from 1 to 5:

| Dimension | Meaning |
|---|---|
| Source support | Entailed by retrieved spans; no unsupported facts |
| Tuvaluan quality | Natural, fluent, correct register, low translationese |
| Task completion | Directly follows the user request |
| Entity preservation | Names, places, dates, numbers, quotes, institutions preserved |
| Language control | Uses requested language without drift |
| Style fit | Matches requested news, civic, formal, plain, radio, or narrative style |
| Safety/privacy | Does not expose private data or invent sensitive claims |

Default scalar reward:

```text
reward = 0.30 * source_support
       + 0.20 * tuvaluan_quality
       + 0.15 * task_completion
       + 0.15 * entity_preservation
       + 0.10 * language_control
       + 0.05 * style_fit
       + 0.05 * safety_privacy
```

Hard fail if an answer:

- contradicts retrieved source evidence
- fabricates a named person, institution, date, amount, or quote
- answers in the wrong language
- leaks held-in content into held-out eval
- corrupts JSON, code, or tool-call structure
- includes private user data without a valid training-data permission path

### Preference Row Contract

Store pairwise preferences as JSONL:

```json
{
  "id": "pref_stage_d_000001",
  "prompt_id": "stage_c_eval_00042",
  "task_family": "qa_grounded",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "judge_model": "gpt-5.5",
  "judge_mode": "rag_pairwise_v1",
  "scores": {
    "chosen": {
      "source_support": 5,
      "tuvaluan_quality": 4,
      "task_completion": 5,
      "entity_preservation": 5,
      "language_control": 5,
      "style_fit": 4,
      "safety_privacy": 5
    },
    "rejected": {
      "source_support": 2,
      "tuvaluan_quality": 4,
      "task_completion": 4,
      "entity_preservation": 3,
      "language_control": 5,
      "style_fit": 4,
      "safety_privacy": 5
    }
  },
  "retrieved_span_ids": ["doc_12:p4", "doc_12:p5"],
  "protected_terms": ["Funafuti", "2025"],
  "decision": "chosen_a",
  "rationale_summary": "Chosen answer stays supported by the source and preserves the date.",
  "requires_human_review": false,
  "metadata": {
    "source_doc_id": "doc_12",
    "source_split": "train",
    "created_at": "2026-05-15T00:00:00Z",
    "retrieval_release": "stage_d_retrieval_001"
  }
}
```

Training rule:

- use `prompt`, `chosen`, and `rejected` for DPO/ORPO
- keep scores and rationales for audit
- do not train on verbose judge rationales unless a later experiment proves it
  helps without making the model imitate judge prose

## Human Calibration

The judge must be calibrated before producing large preference batches.

Build a 300-item calibration pack:

| Slice | Count |
|---|---:|
| Translation | 100 |
| Grounded QA and summarization | 75 |
| Entity preservation | 50 |
| Product feedback repairs | 50 |
| Adversarial hallucination traps | 25 |

Procedure:

1. Collect human pairwise labels.
2. Run GPT-5.5 judge with the frozen rubric.
3. Measure agreement by slice.
4. Inspect disagreements.
5. Update rubric or retrieval if needed.
6. Lock `judge_mode` before large-scale labeling.

Target agreement:

- at least 85% for grounded factual tasks
- at least 80% for translation and rewrite tasks
- manual review for any slice below threshold

## Product Feedback Conversion

Raw product feedback should become preference data only after normalization.

Input signals:

- paragraph vote
- correction text
- mode preference
- article id
- source paragraph
- generated translation
- user-submitted better translation
- optional free-text reason

Normalize into one of:

- `direct_correction_pair`
- `style_preference_pair`
- `entity_repair_pair`
- `wrong_language_pair`
- `hallucination_or_unsupported_pair`
- `needs_human_review`

Only high-confidence pairs should enter DPO. Ambiguous feedback should go to
human review or judge calibration.

## Evaluation Reporting

Every benchmark report should include:

- model id and checkpoint
- date generated
- sampling settings
- dataset release id
- source split policy
- automatic metrics
- judge metrics
- human subset metrics
- confidence intervals
- per-slice win rate against each baseline
- known exclusions and failures

Minimum headline table:

| Model | TVL->EN | EN->TVL | Native chat | Grounded QA | Entity preservation | Overall win rate |
|---|---:|---:|---:|---:|---:|---:|
| Current Stage A | TBD | TBD | N/A | N/A | TBD | TBD |
| Current Stage B | TBD | TBD | TBD | TBD | TBD | TBD |
| vNext | TBD | TBD | TBD | TBD | TBD | TBD |
| Helsinki OPUS | TBD | TBD | N/A | N/A | TBD | TBD |
| MADLAD/NLLB | TBD | TBD | N/A | N/A | TBD | TBD |
| ChatGPT 5.3 | TBD | TBD | TBD | TBD | TBD | TBD |

## Implementation Backlog

Priority order:

1. Add artifact presence validation for `data/`, `unstruct_lang_data/`, `logs/`,
   and expected manifests.
2. Select the canonical Stage A vNext path and update configs/docs.
3. Add public-baseline runners for Helsinki OPUS and MADLAD/NLLB where
   applicable.
4. Add a frozen ChatGPT 5.3 baseline runner.
5. Add a benchmark report writer with per-slice win rates.
6. Add `stage_d_judge` modules:
   - build retrieval corpus
   - retrieve spans
   - call GPT-5.5 judge
   - write preference JSONL
   - build human review packs
7. Add `tv/data/codex/` module (Phase 4.5):
   - prompt templates by task family with span-injection slots
   - rate-limited subscription client wrapping the codex-proxy
     `replay` and `harvest` runners (sibling `src/codex-proxy`)
   - per-task-family quota config `configs/codex_quota.yaml`
   - decontamination check against all eval splits
   - acceptance-gate runner that emits accepted + rejected JSONL plus
     a manifest of model id, request ids, cost, and rate-limit incidents
8. Add DPO/ORPO renderers from preference JSONL.
9. Add product feedback normalization into typed preference candidates.
10. Add judge calibration reports and dashboards.
11. Run vNext Stage A, Stage B, Stage C, **codex Phase 4.5 harvest +
    replay batches**, then DPO/ORPO in that order.

## Go/No-Go Gates

Do not start expensive vNext training until:

- required artifacts are restored or regenerated
- split validation passes
- one canonical Stage A anchor path is selected
- public baseline scripts are ready
- ChatGPT 5.3 baseline prompts are frozen
- Stage C eval sources are source-disjoint from train
- judge calibration pack exists

Do not start Phase 4.5 codex generation until:

- Stage C retrieval corpus is built and source-disjoint splits are validated
- prompt templates per task family are reviewed and hashed
- per-task-family quota and budget are declared in the manifest
- decontamination check runs against every eval split, not just train
- the codex-proxy reachable in the workspace can capture trajectories
  with per-token logprobs on at least one smoke prompt

Do not start DPO/ORPO until:

- GPT-5.5 judge agreement meets target thresholds
- preference rows include retrieved evidence ids
- hard-fail examples are excluded or flagged
- human review pack has been sampled
- codex-sourced rows in any mix have been distinctly labeled and the
  decontamination report shows zero collisions with eval

Do not publish a SOTA claim until:

- OPUS and project models are run on the same held-out test set
- ours is run on OPUS/JW300 if that test set can be fetched
- frontier model outputs are frozen with dates and model ids
- results are reported by slice with confidence intervals

## Final Success Definition

The vNext run succeeds if it produces a dated, reproducible benchmark showing
that the specialized Tuvaluan model beats public baselines and ChatGPT 5.3 on
the Tuvaluan task slices this project actually cares about, while preserving
source fidelity, entity accuracy, and natural Tuvaluan register.
