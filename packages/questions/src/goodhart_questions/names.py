"""How a question names an entity, and how the prompt spells it back in the warehouse's own words.

Metros are written by their full OMB title or by their first principal city when that city names one
metro only (Portland, Springfield, Columbus and the rest in readback's ambiguity list are always
written in full). Carriers and airports are written by their public names from readback's entities
file or by their codes. The names line of the prompt maps each written form to the warehouse value.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Name:
    written: str
    value: str
    column: str

    def line(self) -> str:
        return f"{self.column} '{self.value}'"


def first_city(title: str) -> str:
    return re.split(r"[-,]", title)[0].strip()


class Namer:
    def __init__(self, metros: list[str], entities: dict[str, Any]) -> None:
        self.entities = entities
        firsts: dict[str, list[str]] = defaultdict(list)
        for t in metros:
            firsts[first_city(t)].append(t)
        ambiguous = set(entities.get("ambiguous", {}))
        self.short: dict[str, str] = {}
        for city, titles in firsts.items():
            if len(titles) == 1 and city not in ambiguous:
                self.short[titles[0]] = city
        self.carriers: dict[str, list[str]] = {k: list(v) for k, v in entities.get("carriers", {}).items()}
        self.airports: dict[str, list[str]] = {k: list(v) for k, v in entities.get("airports", {}).items()}
        self.groups: dict[str, list[str]] = {k: list(v) for k, v in entities.get("applicant_groups", {}).items()}
        self.rules: dict[str, list[str]] = {k: list(v) for k, v in entities.get("aml_rules", {}).items()}

    def name(self, kind: str, value: str, column: str, rng: random.Random) -> Name:
        if kind == "metro":
            short = self.short.get(value)
            written = short if short and rng.random() < 0.6 else value
        elif kind == "carrier":
            aliases = self.carriers.get(value, [])
            written = rng.choice(aliases) if aliases and rng.random() < 0.8 else value
        elif kind == "airport":
            aliases = self.airports.get(value, [])
            written = rng.choice(aliases) if aliases and rng.random() < 0.6 else value
        elif kind == "applicant_group":
            aliases = self.groups.get(value, [])
            written = rng.choice(aliases) if aliases else value
        elif kind == "aml_rule":
            aliases = self.rules.get(value, [])
            written = rng.choice(aliases) if aliases else value.replace("_", " ")
        else:
            written = value
        return Name(written=written, value=value, column=column.strip('"'))
