"""Build ``codex_env.Task`` directories from tuvalu source rows.

The codex harness consumes tasks as on-disk directories (``prompt.md`` +
``task.toml``) — see ``codex_env.load_task_dir``. This module renders
tuvalu's per-task-family prompt templates against a source row and
materializes the task dir the harness needs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:  # pragma: no cover — jinja2 ships with most stacks
    Environment = None  # type: ignore[misc, assignment]
    FileSystemLoader = None  # type: ignore[misc, assignment]
    select_autoescape = None  # type: ignore[misc, assignment]

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Heuristic: pull tokens that look like proper names, dates, numbers,
# currency, and quoted phrases. Used as a default protected_terms list
# when the source row doesn't carry one.
_NAMES_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_NUMS_RE = re.compile(r"\b\d[\d,.]*\b")
_CURRENCY_RE = re.compile(r"\$\s?\d[\d,.]*|\d[\d,.]*\s?(?:USD|EUR|AUD)\b")
_QUOTED_RE = re.compile(r"\"([^\"]{1,80})\"|'([^']{1,80})'|“([^”]{1,80})”")


def auto_protected_terms(*texts: str, max_terms: int = 10) -> list[str]:
    """Cheap protected-term extractor — names, numbers, currency, quotes.

    Used when the source row doesn't carry an explicit ``protected_terms``
    list. The judge-pipeline runs the real entity extractor in Phase 5.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for txt in texts:
        if not txt:
            continue
        for rx in (_NAMES_RE, _NUMS_RE, _CURRENCY_RE):
            for m in rx.finditer(txt):
                term = m.group(0).strip()
                if term and term not in seen and len(term) >= 2:
                    seen.add(term)
                    terms.append(term)
        for m in _QUOTED_RE.finditer(txt):
            for g in m.groups():
                if g and g not in seen:
                    seen.add(g)
                    terms.append(g)
        if len(terms) >= max_terms:
            break
    return terms[:max_terms]


def _render(template_name: str, **ctx: Any) -> str:
    if Environment is None:
        raise RuntimeError("jinja2 is required for task_builder; add it to the env")
    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2", "md")),
        keep_trailing_newline=True,
    )
    tpl = env.get_template(f"{template_name}.j2")
    return tpl.render(**ctx)


def build_native_chat_task(
    *,
    bench_root: Path,
    task_id: str,
    tvl_text: str,
    en_text: str,
    protected_terms: Sequence[str] | None = None,
    source_doc_id: str | None = None,
) -> Path:
    """Write a native_chat task dir under bench_root/<task_id>/."""
    pt = list(protected_terms) if protected_terms else auto_protected_terms(tvl_text, en_text)
    prompt = _render("native_chat", tvl_text=tvl_text, en_text=en_text, protected_terms=pt)
    return _write_task_dir(
        bench_root=bench_root,
        task_id=task_id,
        prompt=prompt,
        toml_kv={
            "id": task_id,
            "family": "native_chat",
            "answer_language": "tvl",
            "source_doc_id": source_doc_id or "",
            "tvl_text": tvl_text,
            "en_text": en_text,
            "protected_terms": pt,
        },
    )


def build_hard_translation_task(
    *,
    bench_root: Path,
    task_id: str,
    src_text: str,
    gold_text: str,
    direction: str,  # "tvl2en" or "en2tvl"
    protected_terms: Sequence[str] | None = None,
    source_doc_id: str | None = None,
) -> Path:
    """Write a hard_translation task dir under bench_root/<task_id>/."""
    if direction == "tvl2en":
        src_lang, tgt_lang = "Tuvaluan", "English"
        ans_lang = "en"
    elif direction == "en2tvl":
        src_lang, tgt_lang = "English", "Tuvaluan"
        ans_lang = "tvl"
    else:
        raise ValueError(f"unknown direction {direction!r}; use tvl2en or en2tvl")
    pt = list(protected_terms) if protected_terms else auto_protected_terms(src_text, gold_text)
    prompt = _render(
        "hard_translation",
        src_text=src_text,
        src_lang_name=src_lang,
        tgt_lang_name=tgt_lang,
        protected_terms=pt,
    )
    return _write_task_dir(
        bench_root=bench_root,
        task_id=task_id,
        prompt=prompt,
        toml_kv={
            "id": task_id,
            "family": "hard_translation",
            "answer_language": ans_lang,
            "source_doc_id": source_doc_id or "",
            "direction": direction,
            "src_text": src_text,
            "gold_translation": gold_text,
            "protected_terms": pt,
        },
    )


def build_qa_grounded_task(
    *,
    bench_root: Path,
    task_id: str,
    question: str,
    spans: Sequence[tuple[str, str]],  # (span_id, text)
    gold_answer: str | None = None,
    protected_terms: Sequence[str] | None = None,
    source_doc_id: str | None = None,
) -> Path:
    pt = list(protected_terms) if protected_terms else auto_protected_terms(
        question, *(s[1] for s in spans),
    )
    prompt = _render(
        "qa_grounded",
        question=question,
        spans=list(spans),
        protected_terms=pt,
    )
    return _write_task_dir(
        bench_root=bench_root,
        task_id=task_id,
        prompt=prompt,
        toml_kv={
            "id": task_id,
            "family": "qa_grounded",
            "answer_language": "tvl",
            "source_doc_id": source_doc_id or "",
            "question": question,
            "retrieved_span_ids": [s[0] for s in spans],
            "retrieved_span_texts": [s[1] for s in spans],
            "gold_answer": gold_answer or "",
            "protected_terms": pt,
        },
    )


def _write_task_dir(*, bench_root: Path, task_id: str, prompt: str, toml_kv: dict[str, Any]) -> Path:
    """Write prompt.md + task.toml under bench_root/<task_id>/."""
    tdir = bench_root / task_id
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "prompt.md").write_text(prompt, encoding="utf-8")
    # Use JSON-in-TOML for list fields so we don't have to handle TOML
    # escaping of arbitrary Tuvaluan text. ``codex_env.load_task_dir``
    # uses ``tomllib`` which handles ``answer = "[...]"`` correctly only
    # if the value is a JSON-encoded string — but we want list semantics.
    # Easiest: each field that's a list gets JSON-encoded as a string,
    # and the verifier decodes it. For string fields, TOML's own
    # escaping works.
    toml_lines: list[str] = []
    for k, v in toml_kv.items():
        if isinstance(v, (list, tuple)):
            toml_lines.append(f"{k}_json = {json.dumps(json.dumps(list(v)))}")
        else:
            toml_lines.append(f"{k} = {json.dumps(v)}")
    (tdir / "task.toml").write_text("\n".join(toml_lines) + "\n", encoding="utf-8")
    return tdir


def task_metadata_decode(metadata: dict[str, Any]) -> dict[str, Any]:
    """Decode any ``<name>_json`` fields back into Python lists.

    Callers receive Task.metadata with the raw TOML structure; this
    helper unwraps the JSON-encoded list fields.
    """
    out: dict[str, Any] = {}
    for k, v in metadata.items():
        if k.endswith("_json") and isinstance(v, str):
            try:
                out[k[:-5]] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        out[k] = v
    return out


__all__ = [
    "PROMPTS_DIR",
    "auto_protected_terms",
    "build_hard_translation_task",
    "build_native_chat_task",
    "build_qa_grounded_task",
    "task_metadata_decode",
]
