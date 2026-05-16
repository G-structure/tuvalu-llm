"""Audit ``data/external/tv2en-cleaned/cleaned.jsonl``.

Score every row by religious-vocab density (TVL + EN) so downstream
sampling can pick:

- the LEAST religious rows for cheap "as-is" codex distillation
- the MOST religious rows for the reframe/substitute augmentation
  pipeline (Biblical entities → Tuvaluan entities, see
  ``docs/FULL_PIPELINE_AUDIT_AND_RL_JUDGE_PLAN.md`` Phase 4.5b)

Writes ``data/external/tv2en-cleaned/audit.jsonl`` — same rows + added
audit fields:

    religious_score_tvl: int   # raw count of JW vocab tokens
    religious_score_en:  int
    religious_density:   float # combined / (tvl_words + en_words)
    has_proper_names:    bool
    tvl_words:           int
    en_words:            int
    bucket:              "low" | "med" | "high"   # density thresholded

Also writes ``reports/cleaned_corpus_audit.md`` with histograms.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Tuvaluan religious / JW-specific vocabulary. Hand-curated — these are
# the tokens whose presence indicates "this row is heavily religious
# context", as distinct from general TVL function words.
TVL_RELIGIOUS_TOKENS = {
    # Names
    "Ieova", "Iesu", "Karaisito", "Mose", "Apalaamo", "Iakopo", "Davita",
    "Kitiona", "Iohane", "Pauro", "Pita", "Saulu", "Iosua", "Lota",
    "Mataio", "Maleko", "Luka", "Mose", "Noa", "Sataani", "Tiapolo",
    # Places
    "Etene", "Ihirama", "Iutaia", "Ielusalema", "Petelehema", "Kaperināuma",
    "Iolitana", "Mauga", "Olivi",
    # JW-specific terms (broader)
    "Watchtower", "watchtower",  # english title used in TVL
    "Kelisiano", "kelisiano",                            # Christian
    "lotu",                                              # worship/religion
    "talo",                                              # prayer
    "toetu", "toe‵tu",                                   # resurrect
    "talai", "fakavanvanna",                             # preach
    "tukuatuga",                                         # dedication
    "ekalesia",                                          # church
    "perofeta",                                          # prophet
    "agasala",                                           # sin
    "agelu",                                             # angel
    "Mesia",                                             # Messiah
    "Tupu",                                              # King (religious "Kingdom")
    "Aliki", "Sili",                                     # Sovereign Lord
    "tapuakiga",                                         # blessing
    "fakamasino",                                        # judgment
    "fakaola",                                           # salvation
    "fakatōga", "tōga",                                  # sacrifice/offering
    "uatese",                                            # priest
    "mauloto",                                           # commandment
    "amiotonu",                                          # righteousness
    "Atua",                                              # God
    "evagelia", "saraga",                                # gospel
    "tusi", "tāpū",                                      # holy/scripture marker
    "fakatasi",                                          # gathering (religious context)
    "fakaakoakoga",                                      # study lesson
    "fakanofonofoga",                                    # arrangement (jw-flavored)
    "Malo",                                              # Kingdom (religious)
}

EN_RELIGIOUS_TOKENS = {
    "Jehovah", "Jesus", "Christ", "Lord", "God", "Sovereign",
    "Moses", "Abraham", "Jacob", "David", "Gideon", "Joshua",
    "Paul", "Peter", "Saul", "John", "Matthew", "Mark", "Luke",
    "Noah", "Adam", "Eve", "Satan", "Devil",
    "Eden", "Israel", "Judea", "Jerusalem", "Bethlehem", "Capernaum",
    "Jordan",
    "sin", "sins", "sinful", "sinner",
    "angel", "angels", "angelic",
    "Messiah", "Messianic",
    "sacrifice", "sacrificial", "offering", "offerings",
    "worship", "worshiped", "worshiper", "worshipped",
    "church", "congregation", "congregations",
    "judgment", "judgments",
    "salvation", "salvific", "saved", "savior", "saviour",
    "prophet", "prophets", "prophecy", "prophesy",
    "blessing", "blessings", "blessed",
    "gospel", "gospels",
    "priest", "priests", "ministry", "minister",
    "Kingdom", "Armageddon", "kingdom",
    "spiritual", "spirituality", "spiritually",
    "scripture", "scriptures", "scriptural",
    "Bible", "biblical", "Bibles",
    "commandment", "commandments",
    "righteous", "righteousness", "unrighteous",
    "everlasting", "eternal", "eternity",
    "Witness", "Witnesses",  # Jehovah's Witnesses
    "Watchtower",
    "prayer", "prayers", "pray", "prayed", "praying",
    "Christian", "Christians", "Christianity",
    "resurrection", "resurrected", "resurrect",
    "dedication", "dedicated", "dedicate",
    "preach", "preached", "preaching", "preacher",
    "evangelize", "evangelism",
    "salvation", "redemption", "redeemed", "redeem",
    "faith", "faithful", "faithfulness", "faithless",
    "disciple", "disciples", "discipleship",
    "apostle", "apostles", "apostolic",
    "psalm", "Psalm", "psalms",
    "verse", "verses", "chapter",  # in this corpus almost always bible
    "Hebrew", "Hebrews", "Israelites",
    "tabernacle", "temple",
    "Sabbath",
    "soul", "souls",
    "spirit", "Spirit",  # often "holy spirit" in JW corpus
    "covenant", "covenants",
    "Pharisee", "Pharisees", "Sadducee",
    "anointing", "anointed",
    "elder", "elders",  # JW elder
    "circuit",  # JW "circuit overseer"
    "overseer", "overseers",
    "publication", "publications",  # JW publications
    "study", "studies",  # in this corpus "Bible study" / "study article"
}

# Lower-case versions for word matching (case-insensitive).
TVL_REL_LC = {t.lower() for t in TVL_RELIGIOUS_TOKENS}
EN_REL_LC = {t.lower() for t in EN_RELIGIOUS_TOKENS}

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿĀ-žŻ-ž'‘’ʻ-]+", re.UNICODE)
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-ZÀ-ÿĀ-žŻ-ž]+\b")


def score_row(row: dict) -> dict:
    tvl = (row.get("tvl") or "").strip()
    en = (row.get("en") or "").strip()
    tvl_words = _WORD_RE.findall(tvl)
    en_words = _WORD_RE.findall(en)
    n_tvl = len(tvl_words)
    n_en = len(en_words)

    rel_tvl = sum(1 for w in tvl_words if w.lower() in TVL_REL_LC)
    rel_en = sum(1 for w in en_words if w.lower() in EN_REL_LC)
    total_words = max(n_tvl + n_en, 1)
    density = (rel_tvl + rel_en) / total_words

    has_proper = bool(_PROPER_RE.search(en))
    return {
        "religious_score_tvl": rel_tvl,
        "religious_score_en": rel_en,
        "religious_density": round(density, 4),
        "tvl_words": n_tvl,
        "en_words": n_en,
        "has_proper_names": has_proper,
    }


def bucket(density: float, *, low: float = 0.02, high: float = 0.08) -> str:
    if density <= low:
        return "low"
    if density >= high:
        return "high"
    return "med"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/external/tv2en-cleaned/cleaned.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="data/external/tv2en-cleaned/audit.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--report",
        default="reports/cleaned_corpus_audit.md",
        type=Path,
    )
    parser.add_argument("--low", type=float, default=0.02, help="<= this density = low bucket")
    parser.add_argument("--high", type=float, default=0.08, help=">= this density = high bucket")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("audit")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    buckets = Counter()
    domain_x_bucket: dict[str, Counter] = defaultdict(Counter)
    rel_score_hist = Counter()
    total = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            scored = score_row(row)
            b = bucket(scored["religious_density"], low=args.low, high=args.high)
            scored["bucket"] = b
            buckets[b] += 1
            domain_x_bucket[row.get("domain") or "<none>"][b] += 1
            rel_score_hist[scored["religious_score_tvl"] + scored["religious_score_en"]] += 1
            out_row = dict(row)
            out_row.update(scored)
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            total += 1

    log.info("audited %d rows", total)
    log.info("buckets: %s", dict(buckets))

    # Markdown report
    lines: list[str] = []
    lines.append("# Cleaned corpus audit\n")
    lines.append(f"Input: `{args.input}`\n")
    lines.append(f"Total rows: **{total:,}**\n\n")
    lines.append("## Buckets by religious_density\n")
    lines.append(f"Thresholds: low <= {args.low}, high >= {args.high}.\n\n")
    lines.append("| bucket | rows | % |\n|---|---:|---:|\n")
    for b in ("low", "med", "high"):
        n = buckets.get(b, 0)
        pct = 100 * n / max(total, 1)
        lines.append(f"| {b} | {n:,} | {pct:.1f}% |\n")
    lines.append("\n## Bucket × domain\n")
    lines.append("| domain | low | med | high | total |\n|---|---:|---:|---:|---:|\n")
    for dom, c in sorted(domain_x_bucket.items(), key=lambda kv: -sum(kv[1].values())):
        lo = c.get("low", 0)
        md = c.get("med", 0)
        hi = c.get("high", 0)
        lines.append(f"| {dom} | {lo:,} | {md:,} | {hi:,} | {lo + md + hi:,} |\n")
    lines.append("\n## Religious-token-count histogram (rows by count of JW-vocab tokens)\n")
    lines.append("| count | rows |\n|---:|---:|\n")
    for count in sorted(rel_score_hist):
        if count > 15:
            continue
        lines.append(f"| {count} | {rel_score_hist[count]:,} |\n")
    over = sum(v for k, v in rel_score_hist.items() if k > 15)
    if over:
        lines.append(f"| 16+ | {over:,} |\n")
    args.report.write_text("".join(lines), encoding="utf-8")
    log.info("report -> %s", args.report)

    print(json.dumps({
        "total_audited": total,
        "buckets": dict(buckets),
        "domain_x_bucket": {k: dict(v) for k, v in domain_x_bucket.items()},
        "report": str(args.report),
        "audit_jsonl": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
