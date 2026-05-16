"""Persona pool for codex distillation Phase 4.5b.

Each persona shifts the question distribution away from generic doctrinal
prompts toward what a real Tuvaluan with this life would actually ask
about a passage. Loaded once and round-robined across a harvest batch
(or weighted-sampled when richer balance is wanted).
"""

from __future__ import annotations

import dataclasses
import random
from typing import Sequence


@dataclasses.dataclass(frozen=True, slots=True)
class Persona:
    role_id: str          # canonical id for filenames + manifests
    role: str             # natural-language description for the prompt
    locale: str
    concerns: tuple[str, ...]


# Initial pool — eight personas covering the Tuvaluan economic + civic
# distribution. Curated to match the slices `rl-environments.md` lists
# in the audit doc's eval slices ("Native TVL chat", "Civics",
# "Grounded QA").
PERSONA_POOL: tuple[Persona, ...] = (
    Persona(
        role_id="fisherman",
        role="experienced Tuvaluan fisherman",
        locale="Funafuti",
        concerns=(
            "weather and tides",
            "fishing technique",
            "boat maintenance",
            "fuel cost",
            "selling the catch at the local market",
        ),
    ),
    Persona(
        role_id="village_teacher",
        role="primary-school teacher in a village",
        locale="Vaitupu",
        concerns=(
            "lesson planning in Tuvaluan and English",
            "classroom management",
            "limited school supplies",
            "students who travel from outer atolls",
        ),
    ),
    Persona(
        role_id="clinic_nurse",
        role="nurse at a small community clinic",
        locale="Nukulaelae",
        concerns=(
            "diabetes and high blood pressure in the community",
            "vaccinations",
            "maternal and child health",
            "limited medical supplies and irregular shipments",
        ),
    ),
    Persona(
        role_id="civil_servant",
        role="civil servant in a Tuvaluan government ministry",
        locale="Funafuti",
        concerns=(
            "budget cycles",
            "public works and infrastructure projects",
            "donor-funded programs",
            "cyclone preparedness",
            "climate adaptation",
        ),
    ),
    Persona(
        role_id="parent",
        role="parent of school-aged children",
        locale="Nanumea",
        concerns=(
            "household economy",
            "schooling and homework",
            "food security",
            "remittances from family overseas",
            "raising children in the church and community",
        ),
    ),
    Persona(
        role_id="young_adult_abroad",
        role="young Tuvaluan preparing for tertiary study abroad",
        locale="Suva (currently studying at USP)",
        concerns=(
            "academic English versus everyday Tuvaluan",
            "homesickness and Tuvaluan diaspora networks",
            "career options that bring skills back to Tuvalu",
        ),
    ),
    Persona(
        role_id="elder",
        role="village elder who knows oral history and customary law",
        locale="Niutao",
        concerns=(
            "oral history and genealogies",
            "customary land tenure",
            "kaupule (island council) decisions",
            "passing knowledge to younger generations",
        ),
    ),
    Persona(
        role_id="boat_operator",
        role="inter-island boat operator and cargo handler",
        locale="Nui",
        concerns=(
            "shipping schedules between atolls",
            "fuel and freight rates",
            "weather delays",
            "passenger safety",
            "loading and unloading cargo",
        ),
    ),
)


def round_robin_persona(index: int) -> Persona:
    """Deterministic round-robin selection."""
    return PERSONA_POOL[index % len(PERSONA_POOL)]


def sample_persona(rng: random.Random) -> Persona:
    """Uniform random pick."""
    return rng.choice(PERSONA_POOL)


def personas_by_id() -> dict[str, Persona]:
    return {p.role_id: p for p in PERSONA_POOL}


__all__ = [
    "PERSONA_POOL",
    "Persona",
    "personas_by_id",
    "round_robin_persona",
    "sample_persona",
]
