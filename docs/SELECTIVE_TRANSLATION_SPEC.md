# Selective Translation Spec for Stage B Synthetic Data

When translating English capability datasets into Tuvaluan to build bilingual
training data, we use **selective translation** — only translating the human-language
surface while preserving the machine-parseable substrate.

## Field Classification

### TRANSLATE (human-language spans)
- User requests and queries
- Assistant natural-language answers
- Explanations, summaries, narration
- Chain-of-action text around tool calls
- Human-readable error messages (when not exact-match targets)

### KEEP EXACTLY (machine-parseable spans)
- Code blocks (Python, JS, SQL, shell, etc.)
- JSON, XML, YAML, TOML keys and structure
- Tool/function names (`get_weather`, `search_documents`)
- Schema keys and field names
- Variable names and identifiers
- URLs, file paths, shell commands
- Numbers, dates, equations, mathematical notation
- Placeholders (`{name}`, `%s`, `<id>`, `{{variable}}`)
- Regex patterns
- API endpoints and HTTP methods
- Markdown code fences and formatting markers

### MAYBE TRANSLATE (context-dependent)
- Code comments
- Docstrings
- Enum values shown to users
- UI labels and button text
- Error messages that are user-facing
- Log messages (if they're the target output)

### LOCALIZE CAREFULLY
- Culture-specific examples (dates, currency, names)
- Idioms and figurative language
- Multiple-choice options (preserve label alignment: A/B/C/D)
- Named entities in narrative context

### DROP OR REVIEW MANUALLY
- English spelling/grammar correction tasks
- Pun/joke/wordplay tasks
- Token-level editing ("change the third word")
- Regex/string manipulation tied to English text
- Benchmark tasks where gold answer depends on English wording

## Translation Pipeline

```
English sample (structured)
    ↓
1. Parse into structured fields
    ↓
2. Classify each field/span as TRANSLATE or KEEP
    ↓
3. Replace KEEP spans with unique placeholders
    ↓
4. Translate only TRANSLATE spans with Stage A adapter
    ↓
5. Restore protected spans from placeholders
    ↓
6. Validate:
   - Code/tool calls/JSON are byte-identical where required
   - Placeholder restoration is complete (no orphaned markers)
   - Output parses correctly in the expected format
    ↓
7. Accept or reject sample based on validation
```

## Format-Specific Rules

### Tool Calling Examples
```json
{
  "messages": [
    {"role": "user",      "content": "TRANSLATE"},
    {"role": "assistant", "content": "KEEP (tool call JSON + <tool_call> wrapper)"},
    {"role": "tool",      "content": "KEEP (tool result JSON)"},
    {"role": "assistant", "content": "TRANSLATE"}
  ]
}
```

### Code + Explanation Examples
```json
{
  "messages": [
    {"role": "user",      "content": "TRANSLATE the question, KEEP any code in the question"},
    {"role": "assistant", "content": "TRANSLATE explanation text, KEEP all code blocks"}
  ]
}
```

### Math / Reasoning Examples
```json
{
  "messages": [
    {"role": "user",      "content": "TRANSLATE instruction text, KEEP equations/numbers"},
    {"role": "assistant", "content": "TRANSLATE reasoning text, KEEP equations/answer format"}
  ]
}
```

### QA / Extraction Examples
```json
{
  "messages": [
    {"role": "user",      "content": "TRANSLATE question and context paragraph"},
    {"role": "assistant", "content": "KEEP if answer is a named entity/number, TRANSLATE if answer is a sentence"}
  ]
}
```

## Validation Checks

After translation, each sample must pass:

1. **Structural integrity**: same number of messages, same roles
2. **JSON validity**: all JSON fields parse correctly
3. **Code preservation**: code blocks are byte-identical to source
4. **Placeholder completeness**: all placeholders restored, none leaked
5. **Schema match**: output matches expected schema for the task type
6. **Length sanity**: translated text is within 0.5x–2.5x of source length
7. **Language detection**: translated spans are detected as Tuvaluan (not English)

## Stage C Mixing Ratios

For the bilingual capability adapter:

| Component | Share | Source |
|-----------|-------|--------|
| Original English capability data | 40% | Upstream datasets, unmodified |
| Selectively translated Tuvaluan | 40% | Stage B output |
| Original TVL↔EN parallel data | 20% | Stage A training data (anchor) |

Adjust if:
- English regresses → increase English share
- Tuvaluan weak → increase translated share
- Translation quality drifts → increase anchor share

## Rule of Thumb

> If a token is for a **human to read**, translate it.
> If a token is for a **machine to parse**, keep it.
> If a token is part of the **ground-truth answer format**, be very cautious.
