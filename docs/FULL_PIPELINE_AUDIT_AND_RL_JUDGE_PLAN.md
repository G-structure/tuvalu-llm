# Full Pipeline Audit And RL Judge Plan

This document audits the current Tuvaluan LLM pipeline and turns it into an
execution plan for a stronger vNext run.

The goal is not to claim generic frontier-model superiority. The goal is to
build and verify a specialized Tuvaluan system that can beat general frontier
models and public Tuvaluan MT baselines on frozen, source-disjoint Tuvaluan task
slices.

Last updated: 2026-05-16 (Phase 4.5 harness validated on 500-row math +
500-row TVL native_chat smokes; tuvalu adapter + Phase 4.5b source-bias
mitigation shipped — audit, persona-conditioning, reframe-augment all
smoke-validated against gpt-5.3-codex).

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
   that Stage A cannot produce. The harness substrate (`DistillJobRunner`,
   `passthrough_to_default`, JSONL sink) shipped in sibling `rl-agent-work`
   on 2026-05-15 and is validated at 96 % accept on 25 MATH tasks via the
   chatgpt subscription. Tuvalu still owes the `tv/data/codex/` adapter
   (task builder, per-family verifiers, prompt templates, decontamination
   second-pass). Route every output through the same judge gates as any
   other synthetic row.
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

The codex harness lives in the sibling `rl-agent-work` workspace and is
already wired end to end:

- [`src/codex-proxy`](../../../rl-agent-work/src/codex-proxy/) —
  translates between the OpenAI Responses API the Codex CLI speaks and
  the trainable backend's token interface; captures per-token logprobs
  and writes per-rollout trajectory records. See
  [`codex-training-proxy.md`](../../../rl-agent-work/codex-training-proxy.md)
  and
  [`codex-harness-rl-training-landscape.md`](../../../rl-agent-work/codex-harness-rl-training-landscape.md)
  for the design rationale and the prior art survey.
- [`src/codex-control`](../../../rl-agent-work/src/codex-control/) —
  async puppeteer for `codex app-server`. Spawns the codex subprocess,
  starts threads, drives turns, captures structured `Turn` items
  (`agentMessage`, `commandExecution`, `fileChange`, `mcpToolCall`,
  `reasoning`, ...). This is what records the trajectory in
  `harvest` mode without needing the proxy in the model-traffic path.
- [`src/codex-orchestrate`](../../../rl-agent-work/src/codex-orchestrate/) —
  the job dispatcher. Exposes
  `codex_orchestrate.jobs.distill.DistillJobRunner`, the harvest
  runner that wraps `codex-control` + a workspace verifier behind a
  single `run_job(DistillJobSpec(...))` call. Already implements
  Milestone C.1.b of `rl-agent-work/todo.md`: per-task rollout, gold-
  threshold rejection filter, cost accounting via `settle_cost`,
  per-task `traces.jsonl` sink under `spec.output_dir`. The Parquet
  sink (C.2) is the longer-term home but JSONL is what landed first
  and is sufficient for SFT consumption.
- [`vendor/ttt_discover/codex_runtime/`](../../../rl-agent-work/vendor/ttt_discover/codex_runtime)
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

#### Implementation Status (2026-05-15)

The harness-side substrate landed in `rl-agent-work` and was validated
end-to-end against the OpenAI Codex subscription on a 25-task math
smoke before integration with tuvalu-llm. What is shipped:

| Piece | Where | Status |
|---|---|---|
| `OpenAIPassthroughBackend` (for proxy-in-path mode, future use) | `rl-agent-work/src/codex-proxy/src/codex_proxy/backends/openai_passthrough.py` | Shipped |
| `DistillJobSpec` / `DistillOutcome` | `rl-agent-work/src/codex-orchestrate/src/codex_orchestrate/jobs/spec.py` | Shipped |
| `DistillJobRunner.run()` | `rl-agent-work/src/codex-orchestrate/src/codex_orchestrate/jobs/distill.py` | Shipped, with per-task `traces.jsonl` sink |
| `passthrough_to_default` flag on `spawn_codex_session` | `rl-agent-work/src/codex-orchestrate/src/codex_orchestrate/pools/codex_session.py` | Shipped |
| Cost accounting (`estimate_cost`, `settle_cost`) | `rl-agent-work/src/codex-orchestrate/src/codex_orchestrate/jobs/distill_cost.py` | Shipped |
| Driver: `scripts/distill_math_python.py` | `rl-agent-work/scripts/` | Shipped (reference impl; tuvalu version below) |
| Parquet sink (C.2) | not yet | Deferred; JSONL is the v0 sink |
| `tv/data/codex/` tuvalu adapter (C.4) | not yet | Listed in backlog |
| Decontamination check against tuvalu eval splits | not yet | Listed in backlog |

The critical architectural fact for tuvalu: when `passthrough_to_default=
True`, `spawn_codex_session` returns an empty env dict, codex inherits
`~/.codex/auth.json` (chatgpt-mode token), and codex talks to its
configured endpoint (the hosted OpenAI Responses API) directly. No
proxy sits in the model-traffic path. We capture the trajectory from
codex-control's `Turn` events (the `items` list including `agentMessage`,
tool calls, and reasoning summaries), not from the proxy's record store.
The proxy is still useful for the `replay` mode and for record capture
when the trainable policy is being served, but it is not on the
critical path for codex-subscription-driven harvest.

#### Validated Smoke (2026-05-15): 25 MATH Problems via gpt-5.3-codex

Run command:

```bash
DISTILL_TEACHER_MODEL=gpt-5.3-codex DISTILL_N=25 \
  /home/freiza/rl-agent-work/vendor/worldlines/.venv/bin/python \
    /home/freiza/rl-agent-work/scripts/distill_math_python.py
```

Result:

| metric | value |
|---|---|
| teacher | `gpt-5.3-codex` via `~/.codex/auth.json` (auth_mode=chatgpt) |
| n_tasks | 25 (Hendrycks MATH train subset) |
| accepted (reward >= 0.95, exact `\boxed{...}` match) | 24 |
| rejected | 1 (`math-012`: 3×1 pmatrix LaTeX format quirk in the rubric, not a teacher failure) |
| `dollars_spent` from `settle_cost` | $0.00 (chatgpt subscription is not metered through the API rate table) |
| wall time | 372.7 s ≈ 15 s / rollout |
| JSONL artifact | `runs/distill_math_python_25_1778890189/traces/traces.jsonl` (66 KB, 25 rows) |

The 96 % acceptance number is the relevant signal that the harness is
honest: codex routes through `~/.codex/auth.json`, the workspace
verifier reads `.codex/last_turn.json` written by the runner, the
rubric scores against the gold answer, the rejection filter passes
real correctness through. The single rejection is a `verifiers.MathRubric`
+ `math_verify` normalization mismatch on column-vector pmatrix syntax;
re-running with a tuvalu-specific rubric would replace `MathRubric`
with a task-family-specific verifier (next subsection).

#### JSONL → Normalized Tuvalu Example Schema

`DistillJobRunner` writes one row per task to
`<spec.output_dir>/traces.jsonl`, accepted and rejected alike. Each row
has the fields below; consumers re-filter by `accepted` at load time.

```json
{
  "task_id": "math-000",
  "rollout_id": "distill-math-000-...",
  "provenance": "hosted_teacher",
  "teacher_model": "gpt-5.3-codex",
  "teacher_endpoint": "https://api.openai.com",
  "prompt": "...",
  "completion": "Let the original numbers be:\n- Blue \\(=8x\\)...",
  "items": [ /* full Turn item stream — agentMessage, reasoning, tool calls */ ],
  "token_usage": {"totalTokens": 15898, "inputTokens": 15552,
                   "cachedInputTokens": 7552, "outputTokens": 346,
                   "reasoningOutputTokens": 187},
  "reward": 1.0,
  "verifier_ok": true,
  "verifier_reason": "math=1.00 gold='24'",
  "hard_gate_passed": true,
  "accepted": true,
  "gold_answer": "24",
  "min_reward_threshold": 0.95,
  "require_hard_gate": true,
  "written_at": 1778890189.42
}
```

Mapping into `tv.common.schema.make_example(...)`:

```python
def codex_trace_to_tvl_example(row: dict, *, release: str) -> dict:
    """Convert one rl-agent-work DistillJobRunner traces.jsonl row into a
    normalized tuvalu training example. Caller pre-filters by row['accepted']."""
    from tv.common.schema import make_example

    task_family = row["metadata"].get("task_family") if "metadata" in row else "chat"
    # Tuvalu's TaskFamily ∈ {chat, tool, math, code, qa, summarization, translation}.
    # The DistillJobSpec carries no task_family field today; the tuvalu adapter
    # below pins it per batch (Phase 4.5 task families: qa_grounded,
    # native_chat, tool_call_trajectory, hard_translation, rejection_candidate).
    return make_example(
        id=f"codex_{release}_{row['rollout_id']}",
        task_family=task_family,
        messages=[
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ],
        metadata={
            "provenance": "codex_subscription",
            "teacher_model": row["teacher_model"],
            "teacher_endpoint": row["teacher_endpoint"],
            "rollout_id": row["rollout_id"],
            "reward": row["reward"],
            "gold_answer": row.get("gold_answer"),
            "verifier_reason": row["verifier_reason"],
            "verifier_ok": row["verifier_ok"],
            "hard_gate_passed": row["hard_gate_passed"],
            "token_usage": row["token_usage"],
            "items_count": len(row.get("items") or []),
            "min_reward_threshold": row["min_reward_threshold"],
            "release": release,
            "written_at": row["written_at"],
        },
    )
```

Three notes:

1. `items` in the source row is the full structured Turn stream. For
   the SFT row we collapse to `(user, assistant)` chat; the full stream
   stays available in the source `traces.jsonl` for later experiments
   that train on tool calls or reasoning summaries.
2. `task_family` must be set by the caller per batch — the
   `DistillJobSpec` is task-family-agnostic; the adapter wraps it.
3. Decontamination is **not** performed at this stage. Run the eval-
   split overlap check on the produced examples *before* mixing them
   into Stage B. Use the same n-gram + exact-text checker that
   `tv/training/synthetic` runs on Stage A's outputs.

#### Tuvalu Adapter Layer: `tv/data/codex/`

Required deliverables to bridge the harness into the tuvalu pipeline.
Total work is ≈ 250 LOC, mostly wiring; the heavy lifting is already
in `DistillJobRunner`.

```text
tv/data/codex/
  __init__.py
  spec.py                       # per-task-family DistillJobSpec presets + budget caps
  prompts/                      # Jinja templates, one per task family
    qa_grounded.j2
    native_chat.j2
    tool_call_trajectory.j2
    hard_translation.j2
    rejection_candidate.j2
  task_builder.py               # tuvalu source row -> codex_env.Task (writes
                                # prompt.md + task.toml dirs the runner consumes)
  verifiers/                    # per-task-family WorkspaceVerifier implementations
    __init__.py
    grounded_qa.py
    native_chat.py
    tool_call_trajectory.py
    hard_translation.py
  harvest.py                    # main runner: builds task list, dispatches via
                                # run_job(DistillJobSpec, ctx), persists traces.jsonl
  convert.py                    # codex_trace_to_tvl_example (above) + batch loader
  decontam.py                   # cross-check against eval splits before promote
```

The harvest entry point:

```python
# tv/data/codex/harvest.py
async def run_codex_harvest(
    *,
    release: str,
    task_family: str,       # qa_grounded | native_chat | tool_call_trajectory | ...
    source_rows: list[dict],  # tuvalu rows (prompt, source spans, gold, ...)
    teacher_model: str = "gpt-5.3-codex",
    min_reward_threshold: float = 0.95,
    out_dir: Path,
) -> DistillOutcome:
    from codex_orchestrate.jobs.spec import DistillJobSpec, JobKind, RunContext
    from codex_orchestrate.lifecycle.run_job import run_job
    from codex_orchestrate.pools.verifier_runner import VerifierRunner

    tasks = [
        build_task(row, task_family=task_family, prompt_template=load_template(task_family))
        for row in source_rows
    ]
    verifier = build_verifier(task_family)  # per-family WorkspaceVerifier

    spec = DistillJobSpec(
        kind=JobKind.DISTILL,
        task=tasks[0],
        agent_policy_endpoint="https://api.openai.com",
        teacher_model=teacher_model,
        teacher_endpoint="https://api.openai.com",
        min_reward_threshold=min_reward_threshold,
        require_hard_gate=True,
        max_dollars=load_quota(task_family),  # configs/codex_quota.yaml
        output_dir=out_dir / "traces",
    )
    ctx = RunContext(
        proxy=None,  # passthrough_to_default in the runner — no proxy needed
        verifier_runner=VerifierRunner(),
        extra={"tasks": tasks, "verifier": verifier},
    )
    return await run_job(spec, ctx=ctx)
```

The runner deposits `traces.jsonl` under `out_dir / "traces"`. The
acceptance gates (language ID, protected terms, decontam) listed in the
prior section run as a **second pass** over `traces.jsonl` rather than
inside the runner — that keeps the runner task-family-agnostic and
lets per-family acceptance rules evolve without changing the runner
itself.

#### Task-Family Verifier Sketches

`DistillJobRunner` is task-family-agnostic — it just calls the
`WorkspaceVerifier` passed in via `ctx.extra["verifier"]`. The
verifier reads `<workspace>/.codex/last_turn.json` (which the runner
writes for every turn) and returns a `codex_env.VerifierResult`. Tuvalu
needs one verifier per task family; below are sketches sized for
`tv/data/codex/verifiers/`.

##### `grounded_qa.py`

Score answers to retrieval-backed Tuvaluan QA. Combines language-ID,
entity preservation, source-support entailment, and chrF++ against a
reference if one exists.

```python
class GroundedQAVerifier:
    def __init__(self, *, langid_threshold=0.85,
                 source_support_min=4, tvl_quality_min=3):
        self._langid = load_fasttext_langid()  # tv.common.langid
        self._judge = GPT55RAGJudge()          # tv.data.judge (Phase 5)
        ...

    async def __call__(self, *, workspace_dir, task):
        completion = read_last_assistant_text(workspace_dir)
        gold = task.metadata.get("gold_answer")
        spans = task.metadata.get("retrieved_span_ids") or []
        protected = task.metadata.get("protected_terms") or []

        lang_ok = self._langid.score(completion, "tvl") >= langid_threshold
        protected_ok = all(term in completion for term in protected)
        scores = await self._judge.score(
            prompt=task.prompt, completion=completion,
            retrieved_spans=spans, rubric="grounded_qa_v1",
        )
        reward = (scores["source_support"] >= source_support_min
                  and scores["tuvaluan_quality"] >= tvl_quality_min
                  and lang_ok and protected_ok) * 1.0
        return VerifierResult(
            reward=reward, ok=reward >= 1.0,
            hard_gate_passed=(lang_ok and protected_ok),
            reason=f"langid={lang_ok} protected={protected_ok} judge={scores}",
            public_metrics={"answer_seen": completion[:200]},
            hidden_metrics=scores,
            info={"completion_head": completion[:240], "gold_answer": gold},
        )
```

##### `tool_call_trajectory.py`

Score multi-turn rollouts where the value is in the structured tool
calls. The runner already preserves the full `items` stream; the
verifier reads it back from `last_turn.json`.

```python
class ToolCallTrajectoryVerifier:
    """Score rollouts whose value is the structured tool-call sequence.

    Reward = 1.0 iff every tool call in `items` has args validating
    against the task's declared JSON schema, and the final assistant
    message answers the user's request. Used as the source for the
    Stage B 'structured/tool tasks' pool.
    """

    async def __call__(self, *, workspace_dir, task):
        items = read_items(workspace_dir)  # from .codex/last_turn.json
        schema = task.metadata.get("tool_schema")
        calls = [i for i in items if i.get("type") in (
            "function_call", "mcpToolCall", "commandExecution",
        )]
        if not calls:
            return VerifierResult(reward=0.0, ok=False,
                                  reason="no tool calls", hard_gate_passed=False)
        schema_ok = all(validate_args(c, schema) for c in calls)
        final = next(
            (i.get("text") or i.get("content") for i in reversed(items)
             if i.get("type") in ("agentMessage", "assistantMessage")),
            "",
        )
        answer_ok = task.metadata.get("expected_pattern", "") in final \
                    if task.metadata.get("expected_pattern") else bool(final)
        reward = 1.0 if (schema_ok and answer_ok) else 0.0
        return VerifierResult(reward=reward, ok=reward >= 1.0,
                              hard_gate_passed=schema_ok,
                              reason=f"schema={schema_ok} answer={answer_ok}",
                              info={"n_calls": len(calls)})
```

##### `hard_translation.py`

Active-learning loop on Stage A failures. The tuvalu pipeline routes
rows where Stage A's chrF++ on the gold pair is below threshold; codex
re-translates conditioned on the source span; the verifier scores the
codex output against the gold using the same chrF++ + entity-preservation
checks Stage A is evaluated on.

```python
class HardTranslationVerifier:
    def __init__(self, *, chrf_min=0.45, entity_recall_min=0.95):
        from tv.common.metrics import chrf_plus_plus, entity_recall
        self.chrf = chrf_plus_plus
        self.entity_recall = entity_recall
        self.chrf_min, self.entity_recall_min = chrf_min, entity_recall_min

    async def __call__(self, *, workspace_dir, task):
        completion = read_last_assistant_text(workspace_dir)
        gold = task.metadata["gold_translation"]
        protected = task.metadata.get("protected_terms") or []
        chrf = self.chrf(completion, gold)
        ent_recall = self.entity_recall(completion, protected)
        reward = (chrf >= self.chrf_min and ent_recall >= self.entity_recall_min) * 1.0
        return VerifierResult(
            reward=reward, ok=reward >= 1.0,
            hard_gate_passed=(ent_recall >= self.entity_recall_min),
            reason=f"chrf={chrf:.3f} ent_recall={ent_recall:.3f}",
            info={"chrf": chrf, "entity_recall": ent_recall,
                  "completion_head": completion[:240]},
        )
```

##### `native_chat.py` and `rejection_candidate.py`

Both delegate to the GPT-5.5 RAG judge from the next section. The
distinction is that `native_chat` rejects on a single score (judge
`tuvaluan_quality + task_completion`) while `rejection_candidate`
returns the *raw* judge scores so Phase 5's pairwise builder can pick
chosen/rejected from the K-sample batch. Concretely, set
`min_reward_threshold=0.0` for `rejection_candidate` (no rejection at
harvest time) and let Phase 5's pair builder do the selection.

#### Cost Notes (Validated)

Through `~/.codex/auth.json` with `auth_mode=chatgpt`, the
`distill_cost.settle_cost` rate table returns $0.00 because the
chatgpt subscription is not metered through the public API pricing —
the user's plan absorbs the cost. The `cost_dollars` telemetry will
read 0 until either:

1. The subscription is swapped for a metered API key (in which case
   `settle_cost` correctly attributes per-token costs against the rate
   table), or
2. We add a parallel `subscription_request_count` counter and per-plan
   rate-limit budget — `requests/day`, `requests/minute` — to enforce
   ceilings the dollar accounting can't see.

For tuvalu's Phase 4.5 budget tracking, the right v0 metric is
**request count** per task family rather than dollars, with the
quota table at `configs/codex_quota.yaml` capping by counts. The
dollar field stays on each row for forward-compatibility with metered-
API runs.

#### Smoke Test for Tuvalu

Before running a real harvest, validate the adapter wiring with a
5-task tuvaluan-shape smoke:

```bash
uv run tv-data-codex smoke \
  --task-family qa_grounded \
  --teacher-model gpt-5.3-codex \
  --n 5 \
  --out runs/codex_smoke_$(date +%s)
```

Expected output: `runs/codex_smoke_*/traces/traces.jsonl` with 5 rows,
non-zero accept rate, no language-id failures, every accepted row
matching its retrieved span ids in `metadata.retrieved_span_ids`.
Acceptance for the smoke gate: accept rate >= 60 % AND zero hard-gate
failures on the language-id check. If the smoke fails, debug the
adapter (`tv/data/codex/`) before authorizing a full harvest run.

### Phase 4.5b: Source-Bias Mitigation

The codex distillation in Phase 4.5 is rate-limited only by the
teacher and the chatgpt subscription; the **quality ceiling is the
source corpus**. The 2026-05-16 audit of
`data/external/tv2en-cleaned/cleaned.jsonl` (176,157 rows) shows the
corpus is a near-monoculture:

| source | rows | share |
|---|---:|---:|
| wol.jw.org (Jehovah's Witnesses publications) | 171,747 | **97.5 %** |
| tuvalu.aa-ken.jp (Tuvaluan-English dictionary) | 4,410 | 2.5 % |

If we distill native_chat unmodified from this corpus the resulting
Stage B will be fluent Tuvaluan with severe JW-flavored register and
topical bias. Three orthogonal mitigations apply.

#### 1. Audit + Bucket

`scripts/audit_cleaned_corpus.py` tags every row with a
`religious_density` score (count of JW-vocab tokens in TVL+EN
normalized by word count) and buckets each row as `low | med | high`.

After expanded vocabulary (Bible names, prayer/resurrection/ministry/
Christian/Watchtower terminology, JW-specific Tuvaluan words like
`Kelisiano`, `talo`, `tukuatuga`, `talai`):

| bucket | rows | content shape |
|---|---:|---|
| high (density >= 0.08) | ~32k | explicit Bible verses, JW theology, congregation arrangements — **skip for native_chat; route through reframe-augment** |
| med (0.02 < density < 0.08) | ~85k | mixed: religious framing on universal topics — **route through persona-prompt** to force topic shift |
| low (density <= 0.02) | ~59k | low explicit JW vocab; topics often genuinely secular (marriage advice, dress, current events, history) — **harvest as-is is acceptable, persona-prompt is better** |
| dictionary | 4,310 | TVL ↔ EN lexicon with categories — **seed entity-substitution dictionary** |

The audit doesn't change the corpus; it sets the input distribution
for the next two strategies. The audit JSONL becomes the canonical
source for all downstream codex harvests.

#### 2. Reframe-Augment (Bible → Tuvalu local)

For high-density rows (and a sample of med-density rows), the codex
teacher rewrites BOTH the Tuvaluan and the English to preserve the
grammatical structure but swap Biblical entities for Tuvaluan ones.

Prompt template at `tv/data/codex/prompts/reframe.j2` (to add) takes:

- the original TVL/EN pair
- a Tuvalu-local entity table (from the dictionary subset + a seed
  list of place names, given names, occupations, items)
- an explicit "preserve clause structure / preserve agreement /
  preserve verbatim numbers and quotes that aren't religious" rubric

and produces:

- a rewritten TVL string
- a rewritten EN string
- a Tuvalu-local theme tag ("fishing", "civics", "family", ...)

Verifier (`tv/data/codex/verifiers/reframe.py`, to add):

- structure preservation check: codex's output has roughly the same
  clause count + word-count ratio as the source
- entity-leak check: no JW-vocab tokens from the audit dictionary
  remain in either side
- bilingual langid (TVL stays TVL, EN stays EN)
- chrF++ between the codex EN and a Stage-A or codex-translation of
  the codex TVL — sanity check that the two halves agree

Seed entity table per category (initial set, expand from the
dictionary subset):

```yaml
biblical_to_tuvaluan:
  names_masculine:
    - {biblical: Pauro, tuvalu: Telupe}
    - {biblical: Pita, tuvalu: Faiva}
    - {biblical: Iosua, tuvalu: Kelese}
    - {biblical: Saulu, tuvalu: Mafua}
    - {biblical: Kitiona, tuvalu: Lavea}
  names_feminine:
    - {biblical: Mareta, tuvalu: Apinelu}
    - {biblical: Mali, tuvalu: Sega}
  places_country:
    - {biblical: Ihirama, tuvalu: Tuvalu}
    - {biblical: Iutaia, tuvalu: "te Pasifika"}
  places_settlement:
    - {biblical: Ielusalema, tuvalu: Funafuti}
    - {biblical: Petelehema, tuvalu: Vaitupu}
    - {biblical: Kaperināuma, tuvalu: Nukulaelae}
  occupations:
    - {biblical: perofeta, tuvalu: faiākoga}      # prophet -> teacher
    - {biblical: uatese, tuvalu: faifeau}         # priest -> minister/community elder (non-doctrinal)
    - {biblical: tavini, tuvalu: faifaiga}        # servant -> worker
  items_food:
    - {biblical: falaoa, tuvalu: pulaka}          # bread -> taro
    - {biblical: uaina, tuvalu: kaleve}           # wine -> coconut toddy
  items_animal:
    - {biblical: mamoe, tuvalu: ika}              # sheep -> fish
    - {biblical: kāmela, tuvalu: vaka}            # camel -> boat (closest "transport" analogue)
```

The substitutions are **not** word-for-word find/replace — codex sees
the seed table as guidance and applies them with adjustment for
register, agreement, and pragmatic plausibility ("a fisherman walking
from Funafuti to Vaitupu" is not pragmatically plausible by foot;
codex should adjust to "sailing").

#### 3. Persona + Retrieval

Orthogonal to filtering and substitution: regardless of which subset
the source passage came from, vary **what kind of question gets
asked**. The current `native_chat.j2` is silent on persona, so codex
defaults to a Watchtower-style instructional question
(`Ne a te mea e ‵tau o tausi?` — "What should one keep?").

`tv/data/codex/prompts/native_chat_persona.j2` (to add) rotates a
persona pool per task:

```text
You are a fluent Tuvaluan speaker. The user is a {persona}. Write
the question THIS user would naturally ask about the passage, in a
register and topic-frame appropriate to their daily life.
```

Persona pool (initial):

- Tuvaluan fisherman asking about navigation, weather, fishing technique
- village teacher asking about lesson planning, classroom management
- nurse at a community clinic asking about prevention, hygiene
- civil servant asking about budgeting, policy, public works
- parent asking about child-rearing, household economy, schooling
- young person preparing for tertiary study abroad
- elder recounting oral history

For passages where retrieval is meaningful (Stage C native sources
once we have them), the persona prompt also takes top-k retrieved
spans from an embedding index over the audited subset. Retrieval
diversifies the source passage codex sees beyond the single triggering
row, so the answer can synthesize across multiple Tuvaluan documents
even when the seed prompt is from a single JW article.

#### Combined Pipeline

The three mitigations stack:

```
audit                  bucket-tag every row by religious_density
   │
   ▼
        ┌──────────────────────────┐
   ┌────┤   bucket = "high"        │── reframe-augment ──┐
   │    └──────────────────────────┘                     │
   │    ┌──────────────────────────┐                     │
   │    │   bucket = "med"         │── persona-prompt ──┤── harvest via codex ──► traces.jsonl
   │    └──────────────────────────┘                     │
   │    ┌──────────────────────────┐                     │
   └────┤   bucket = "low"         │── as-is + persona ──┘
        └──────────────────────────┘

(dictionary subset → entity-substitution seed for reframe templates)

harvest output  ─►  decontamination  ─►  Stage B mix
```

#### Acceptance Criteria for 4.5b

- audit JSONL covers 100 % of cleaned.jsonl rows with bucket assigned
- reframe-augment smoke (50 rows, mixed med/high): >= 50 % accept,
  zero biblical-vocab leakage on accepted rows
- persona-prompt smoke (50 rows on low+med): question distribution
  measurably broader than non-persona baseline (manual spot-check or
  trigram-overlap statistic against the existing nc500 harvest)
- combined batch (5000 rows: 60 % low/med non-religious-as-is +
  30 % med-persona + 10 % high-reframed) achieves >= 90 % accept rate
- realized religious-density of the produced TVL completions in
  `traces.jsonl` is bucketed (and reported); the bulk-mass should
  shift left of the source distribution

#### Implementation Status (2026-05-16)

The three mitigations shipped in this commit and were smoke-validated
on the live chatgpt-subscription codex teacher:

| Module | Path | Smoke (n=5) | Result |
|---|---|---:|---|
| `audit_cleaned_corpus.py` | `scripts/` | 176,157 rows audited | `low` 59k / `med` 85k / `high` 32k, ~12k row movement after vocab refinement |
| Persona pool + prompt | `tv/data/codex/personas.py`, `prompts/native_chat_persona.j2` | 5/5 accepted | every question framed in persona voice — fisherman / teacher / nurse / civil servant / parent each asked a domain-appropriate question |
| Persona-aware harvest CLI | `scripts/run_codex_harvest.py --use-personas --bucket low --bucket med` | smoke ✓ | wall 206 s / 5 rows ≈ 41 s/row (longer than no-persona because the persona prompt is bigger) |
| `ReframeVerifier` + reframe prompt | `tv/data/codex/verifiers/reframe.py`, `prompts/reframe.j2` | 4/5 accepted | religious entities swapped: `Jehovah → kaupule`, `Malo o te Atua → kaupule mo te malo`, `Saulu → Mafua`. The one rejection lacked TVL-local content — verifier's `local=False` hard-gate caught it correctly. |
| `load_audited_subset` | `tv/data/codex/harvest.py` | smoke ✓ | filters by bucket / domain / max_density |

Sample reframed output (high-bucket source, codex output verbatim):

```
SOURCE  TVL: "Ka Lasi te Filemu" e Maua Mai Lalo i te Pulega a te Malo
SOURCE  EN:  Under God's Kingdom "Peace Will Abound"

REFRAMED TVL: "Ka Lasi te Filemu" e Maua Mai Lalo i te Pulega a te kaupule mo te malo
REFRAMED EN:  Under the council and the government "Peace Will Abound"
```

The reframe is structurally identical (same number of clauses, same
quoted material, same future-action shape) but the religious anchor
("God's Kingdom") is replaced by a civic anchor ("council and
government"). This is the exact kind of row that lets a Stage B
trained on JW grammar generalize to civic / governance contexts.

#### Honest Limits

Tuvaluan is severely low-resource. The JW translations are the
largest publicly-available high-quality TVL corpus. **Filtering and
substitution improve the distribution but do not invent secular
Tuvaluan data that doesn't exist.** Two complementary moves outside
this phase carry more long-term leverage:

1. **English capability + crosslingual pools.** Stage B mix's other
   20-30 % shares come from English datasets and Stage-A-translated
   crosslingual prompts. Codex Phase 4.5 isn't a substitute for
   those — it's a quality lift on the TVL-side pool.
2. **Native source mining.** The audit doc's Phase 4 (Stage C) talks
   about pulling civic / health / education / oral-history TVL
   documents. Those are the only true escape from the JW monoculture.
   Phase 4.5b is what makes the existing corpus more usable; Phase 4
   is what makes a non-JW corpus exist at all.

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
7. Add `tv/data/codex/` module (Phase 4.5 + 4.5b). Shipped 2026-05-16:
   - prompt templates per family: `native_chat`, `native_chat_persona`,
     `hard_translation`, `qa_grounded`, `reframe` (`tv/data/codex/prompts/*.j2`)
   - per-task-family `WorkspaceVerifier` implementations:
     `NativeChatVerifier`, `HardTranslationVerifier`, `GroundedQAVerifier`,
     `ReframeVerifier` (`tv/data/codex/verifiers/`)
   - `task_builder.py`, `harvest.py` (dispatches via `run_job(...)`),
     `convert.py` (traces.jsonl -> `tv.common.schema.make_example`),
     `decontam.py` (eval-split overlap check), `personas.py`
   - audit-aware corpus loader (`load_audited_subset`) that filters by
     bucket / domain / `religious_density` cap
   - CLI driver `scripts/run_codex_harvest.py` with `--use-personas`,
     `--bucket low|med|high`, `--max-religious-density`
   - audit script `scripts/audit_cleaned_corpus.py` that bucket-tags
     every cleaned.jsonl row by JW-vocab density
   Still pending (deferred):
   - per-task-family quota config `configs/codex_quota.yaml` (subscription
     is request-count-bounded, not dollar-bounded)
   - decontamination second-pass run against the actual eval splits
     once Phase 1's frozen benchmark JSONL exists
   - acceptance-gate manifest writer (model id, request ids, run-level
     stats) — fields are in the per-row traces.jsonl already; just
     need an aggregator script
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
- per-task-family quota and budget (request counts under the chatgpt
  subscription; dollar caps when swapping to a metered API key) are
  declared in the manifest
- decontamination check runs against every eval split, not just train
- the `tv/data/codex/` adapter passes the 5-task tuvaluan-shape smoke
  described in Phase 4.5 (accept rate >= 60 %, zero language-id hard-
  gate failures). The harness substrate itself is already validated:
  `rl-agent-work` smoked 25 MATH tasks against `gpt-5.3-codex` on
  2026-05-15 with 96 % accept rate and full `traces.jsonl` capture.

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
