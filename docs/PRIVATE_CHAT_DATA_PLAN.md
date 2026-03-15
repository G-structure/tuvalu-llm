# Private Chat Data Plan

## Goal

Use private Tuvaluan and mixed Tuvaluan-English DM/group-chat data to improve:

- TVL chatbot quality
- TVL<->EN translation quality
- bilingual robustness across all Stage B task families

without damaging the current staged training design.

## Core decision

Keep the current staged approach:

1. `Stage A` stays translation-only.
2. `Stage B` stays a fresh bilingual capability adapter trained from the base/chat model, not from `Stage A` weights.

Do not train `Stage A` directly on raw chat continuations.

## Why

Raw chats contain:

- reply-style supervision rather than translation supervision
- code-switching and ellipsis
- domain drift and private-context references
- noisy turn alignment

That helps a bilingual assistant, but it is the wrong signal for a translation-only model unless the chat data is converted into parallel-like examples first.

## Stage A plan: use only pseudo-parallel chat pairs

`Stage A` should continue to train on:

- existing clean aligned TVL<->EN pairs
- a small amount of mined high-confidence `pseudo_parallel` chat data

`pseudo_parallel` means chat-derived pairs that behave like translation pairs even though they were not labeled as such.

Examples:

- a TVL message followed by an English restatement of the same content
- bilingual announcements posted in both languages
- code-switched messages where one span is clearly glossed by the other language
- pairs accepted only when the current MT model agrees strongly with the observed opposite-language message

### Stage A usage rules

- Keep existing clean parallel data as the core training set.
- Add only strongly filtered pseudo-parallel rows.
- Treat chat-mined rows as lower-confidence augmentation, not as the backbone of training.
- Keep the pseudo-parallel share small at first, e.g. `5-20%` of the final `Stage A` train mix.

### Stage A filters

- segment-level language ID, not message-level only
- deduplication
- length-ratio checks
- named-entity copy/preservation checks
- round-trip or agreement checks
- small manual review sample before merge

### Repo integration for Stage A

Chat-mined pseudo-parallel rows should be emitted into a separate external seed bucket, analogous to existing unstructured seed inputs, then merged into `Stage A` only after quality review.

Recommended new artifact:

- `data/external/stage_a_seed/private_chat_pseudo_parallel.jsonl`

## Stage B plan: add real TVL chat as a new source pool

The private chats are more valuable for `Stage B` than for `Stage A`.

Use them as direct bilingual/conversational SFT data in a new pool:

- `real_tvl_chat`

This pool should contain normalized chat examples with:

- `task_family = "chat"`
- multi-turn `messages`
- metadata marking source and language mode

### What real TVL chat helps with

- native Tuvaluan conversational style
- code-switch robustness
- short-form replies, clarifications, and discourse markers
- pragmatic bilingual behavior that synthetic translation misses

### What real TVL chat should not replace

- English capability replay
- synthetic TVL capability data
- translation anchor data

Those are still needed so the Stage B model keeps broad capability coverage.

### Initial Stage B mix recommendation

Start with a four-pool mix such as:

- `35%` English capability data
- `30%` synthetic TVL capability data
- `20%` parallel anchor data
- `15%` real TVL chat

If English capability regresses, increase English replay.
If TVL conversational quality is still weak, increase `real_tvl_chat` modestly.

## Data products to derive from private chats

The chats should be processed into four separate products:

1. `pseudo_parallel`
   - For `Stage A` translation augmentation.
2. `real_tvl_chat`
   - For `Stage B` bilingual conversational SFT.
3. `term_glossary`
   - Names, places, slang, domain terms, preferred copy-through items.
4. `preference/correction`
   - Later DPO or RL data for choosing better phrasings and preserving entities.

Recommended artifact paths:

- `data/external/stage_a_seed/private_chat_pseudo_parallel.jsonl`
- `data/finetune/stage_b_sources/real_tvl_chat/private_tvl_chat.jsonl`
- `data/external/stage_b_seed/private_chat_terms.jsonl`
- `data/feedback/private_chat_preferences.jsonl`

## Suggested normalized schema for real TVL chat

```json
{
  "id": "tvl_chat:thread123:turn08",
  "task_family": "chat",
  "messages": [
    {"role": "user", "content": "E mafai o fesoasoani mai?"},
    {"role": "assistant", "content": "Ioe, se a te mea e manako koe ki ei?"}
  ],
  "metadata": {
    "source_dataset": "private_tvl_chat",
    "language_mode": "tvl"
  }
}
```

For mixed-language chats, keep the observed bilingual behavior rather than forcing monolingual normalization.

## Training implications

### Stage A

- Improve translation coverage using only pseudo-parallel rows plus existing clean pairs.
- Do not use raw chat continuations as translation examples.

### Stage B

- Add `real_tvl_chat` as a new source pool in the Stage B source build and mix step.
- Keep `task_family = "chat"` for these examples.
- Split by thread/conversation, not by individual turn, to avoid leakage.

## Evaluation

Build a small private-chat-derived eval set before training changes:

- `200-500` translation-style held-out pairs
- `200-500` held-out chat examples

Track:

- translation regression against current `Stage A`
- TVL chat quality
- code-switch handling
- named-entity preservation
- English capability regression after adding the chat pool

## Practical sequence

1. Mine private chats into `pseudo_parallel`, `real_tvl_chat`, `term_glossary`, and `preference` artifacts.
2. Add only `pseudo_parallel` into `Stage A` seed data.
3. Add `real_tvl_chat` as a fourth `Stage B` training pool.
4. Use glossary/entity artifacts for preservation checks and prompt/runtime guidance.
5. Use preferences/corrections later for DPO or RL once the supervised pipeline is stable.

## Summary

The right move is not to rethink the entire staged pipeline.

The right move is to keep the current `Stage A` / `Stage B` separation and route private chat data into the part of the pipeline it actually matches:

- translation-like chat fragments -> `Stage A`
- real conversational TVL behavior -> `Stage B`
- terms/entities -> glossary/preservation layer
- preferences/corrections -> later alignment
