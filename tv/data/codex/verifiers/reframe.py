"""WorkspaceVerifier for the ``reframe`` task family.

The reframe task asks codex to rewrite a religious TVL/EN pair into a
Tuvalu-local everyday equivalent. The verifier checks:

1. **Format**: the response has exactly one TVL: block and one EN:
   block.
2. **Length parity**: the reframed pieces are within a sane ratio of
   the source. Hard rejection if either side is < 30 % or > 250 % of
   the source length — codex either skipped or hallucinated bulk text.
3. **Bilingual langid**: TVL stays TVL, EN stays EN.
4. **Religious-leak**: no token from the audit's religious-vocab set
   appears in either output. The whole point of reframing is to remove
   them.
5. **Entity-presence**: at least one Tuvalu-specific token (a place
   name, occupation, or concern) appears in the TVL output. A reframe
   that simply paraphrases the source without inserting Tuvaluan
   content gets rejected.

Reward is 1.0 on full accept, 0.0 on any failure; the verifier_reason
field lists every check's outcome.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ._common import langid_score_en, langid_score_tvl, read_last_assistant_text


# Anything from this set appearing in either output is a religious-leak
# failure. Mirrors the audit script's TVL+EN religious-vocab dictionaries
# but kept local so the verifier doesn't import from scripts/.
_RELIGIOUS_LEAK_TOKENS = {
    # TVL
    "ieova", "iesu", "karaisito", "mose", "apalaamo", "iakopo", "davita",
    "kitiona", "iohane", "pauro", "pita", "saulu", "iosua",
    "etene", "ihirama", "iutaia", "ielusalema", "petelehema", "kaperināuma",
    "kelisiano", "lotu", "talo", "toetu", "talai", "tukuatuga", "ekalesia",
    "perofeta", "agasala", "agelu", "mesia", "fakatōga", "tōga", "uatese",
    "atua", "evagelia", "mauloto", "amiotonu",
    # EN
    "jehovah", "jesus", "christ", "lord", "god", "messiah", "messianic",
    "moses", "abraham", "jacob", "david", "gideon", "joshua",
    "paul", "peter", "saul", "john", "matthew", "mark", "luke",
    "eden", "israel", "judea", "jerusalem", "bethlehem", "capernaum",
    "sin", "sins", "sinful", "angel", "angels",
    "sacrifice", "worship", "church", "congregation", "judgment",
    "salvation", "prophet", "blessing", "gospel", "priest", "ministry",
    "kingdom", "armageddon", "scripture", "bible", "biblical",
    "righteous", "everlasting", "eternal", "witness", "witnesses",
    "watchtower", "prayer", "pray", "praying",
    "christian", "christians", "resurrection", "dedication",
    "preach", "preaching", "disciple", "apostle", "psalm",
    "tabernacle", "temple", "sabbath", "covenant",
}

# Tokens indicating successful Tuvaluan reframing (place names,
# occupations, daily-life concerns). One match is enough — the verifier
# only checks that SOME local content was injected.
_TUVALUAN_LOCAL_TOKENS = {
    # Atolls / settlements
    "funafuti", "vaitupu", "nanumea", "niutao", "nui",
    "nukulaelae", "nukufetau", "nanumaga", "tuvalu",
    # Occupations + people
    "faiākoga", "faiakoga", "nēsi", "nesi", "tautai", "kaupule",
    "fafine", "tagata", "fanau", "mātua", "matua", "tama",
    # Daily-life
    "afaaga", "tai", "meakai", "kaiga", "aoga", "vaka",
    "pulaka", "kaleve", "ika", "niu", "fenua", "kāgalue", "kagalue",
}

_TVL_BLOCK_RE = re.compile(
    r"TVL\s*:\s*(?P<t>.+?)(?:\n\s*EN\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_EN_BLOCK_RE = re.compile(
    r"EN\s*:\s*(?P<e>.+?)\Z",
    re.IGNORECASE | re.DOTALL,
)


def _has_leak(text: str) -> tuple[bool, list[str]]:
    toks = {w.lower() for w in re.findall(r"[A-Za-zÀ-ÿĀ-žŻ-ž'’ʻ-]+", text or "")}
    hits = sorted(toks & _RELIGIOUS_LEAK_TOKENS)
    return bool(hits), hits


def _has_local(text: str) -> tuple[bool, list[str]]:
    toks = {w.lower() for w in re.findall(r"[A-Za-zÀ-ÿĀ-žŻ-ž'’ʻ-]+", text or "")}
    hits = sorted(toks & _TUVALUAN_LOCAL_TOKENS)
    return bool(hits), hits


class ReframeVerifier:
    def __init__(
        self,
        *,
        len_ratio_min: float = 0.30,
        len_ratio_max: float = 2.50,
        langid_min_tvl: float = 0.20,
        langid_min_en: float = 0.20,
    ) -> None:
        self.len_ratio_min = len_ratio_min
        self.len_ratio_max = len_ratio_max
        self.langid_min_tvl = langid_min_tvl
        self.langid_min_en = langid_min_en

    async def __call__(self, *, workspace_dir, task):
        from codex_env.protocols import VerifierResult
        from tv.data.codex.task_builder import task_metadata_decode

        meta = task_metadata_decode(task.metadata or {})
        src_tvl = str(meta.get("tvl_text") or "")
        src_en = str(meta.get("en_text") or "")

        completion = read_last_assistant_text(workspace_dir).strip()
        m_t = _TVL_BLOCK_RE.search(completion)
        m_e = _EN_BLOCK_RE.search(completion)
        format_ok = bool(m_t and m_e)
        reframed_tvl = m_t.group("t").strip() if m_t else ""
        reframed_en = m_e.group("e").strip() if m_e else ""

        # Length parity
        len_ok_tvl = self.len_ratio_min <= (len(reframed_tvl) / max(len(src_tvl), 1)) <= self.len_ratio_max
        len_ok_en = self.len_ratio_min <= (len(reframed_en) / max(len(src_en), 1)) <= self.len_ratio_max
        len_ok = len_ok_tvl and len_ok_en

        # Langid
        tvl_score = langid_score_tvl(reframed_tvl)
        en_score = langid_score_en(reframed_en)
        tvl_score_en = langid_score_en(reframed_tvl)   # the TVL output should NOT look English
        en_score_tvl = langid_score_tvl(reframed_en)   # the EN output should NOT look heavily Tuvaluan
        langid_ok = (
            tvl_score >= self.langid_min_tvl
            and en_score >= self.langid_min_en
            and tvl_score_en < 0.50
            and en_score_tvl < 0.50
        )

        # Religious leak
        leak_tvl, leak_tvl_hits = _has_leak(reframed_tvl)
        leak_en, leak_en_hits = _has_leak(reframed_en)
        leak_ok = not (leak_tvl or leak_en)

        # Local-content presence (at least the TVL side)
        local_tvl, local_tvl_hits = _has_local(reframed_tvl)
        local_ok = local_tvl

        hard_gate = format_ok and len_ok and langid_ok and leak_ok
        reward = 1.0 if (hard_gate and local_ok) else 0.0
        reason = (
            f"format={format_ok} "
            f"len_ratio_tvl={len(reframed_tvl)/max(len(src_tvl),1):.2f} "
            f"len_ratio_en={len(reframed_en)/max(len(src_en),1):.2f} "
            f"tvl_id={tvl_score:.2f} en_id={en_score:.2f} "
            f"leak={leak_ok} local={local_ok}"
        )
        return VerifierResult(
            reward=reward,
            ok=reward >= 1.0,
            reason=reason,
            hard_gate_passed=hard_gate,
            public_metrics={"reframed_tvl_head": reframed_tvl[:200]},
            hidden_metrics={
                "tvl_score": tvl_score,
                "en_score": en_score,
                "tvl_score_en": tvl_score_en,
                "en_score_tvl": en_score_tvl,
                "leak_hits_tvl": ";".join(leak_tvl_hits)[:200],
                "leak_hits_en": ";".join(leak_en_hits)[:200],
                "local_hits_tvl": ";".join(local_tvl_hits)[:200],
                "len_src_tvl": float(len(src_tvl)),
                "len_src_en": float(len(src_en)),
                "len_out_tvl": float(len(reframed_tvl)),
                "len_out_en": float(len(reframed_en)),
            },
            info={
                "reframed_tvl": reframed_tvl[:400],
                "reframed_en": reframed_en[:400],
                "leak_hits_tvl": leak_tvl_hits,
                "leak_hits_en": leak_en_hits,
                "local_hits_tvl": local_tvl_hits,
            },
        )


__all__ = ["ReframeVerifier"]
