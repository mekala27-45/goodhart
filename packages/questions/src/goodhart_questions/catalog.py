"""The readback catalog, read through the source lock: measures, entities, synonyms and table schemas.

goodhart reads readback's semantic layer and nothing else of readback's code. A measure is a phrase a
person uses, the table it is read from, the entity and time it is read at, and the aggregate that
produces it (catalog/measures.yaml). The per warehouse catalogs give every table's columns with their
types, roles, units and definitions, which the schema block of the prompt shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from goodhart_core import paths

WAREHOUSES: tuple[str, ...] = ("doorstep", "turnaround", "ledger")


@dataclass(frozen=True)
class Measure:
    id: str
    warehouse: str
    table: str
    entity: str
    entity_column: str | None
    time: str
    time_column: str | None
    expr: str
    fmt: str
    unit: str
    phrases: tuple[str, ...]
    where: str | None = None
    years: tuple[int, int] | None = None

    @property
    def qualified(self) -> str:
        return f"{self.warehouse}.{self.table}"

    @property
    def has_entity(self) -> bool:
        return self.entity_column is not None

    @property
    def has_time(self) -> bool:
        return self.time_column is not None and self.time != "none"


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    role: str
    unit: str
    definition: str


@dataclass(frozen=True)
class Table:
    warehouse: str
    name: str
    description: str
    rows: int
    columns: tuple[Column, ...]

    @property
    def qualified(self) -> str:
        return f"{self.warehouse}.{self.name}"


@dataclass
class Catalog:
    measures: list[Measure]
    tables: dict[str, Table]
    entities: dict[str, Any]
    synonyms: dict[str, list[str]] = field(default_factory=dict)

    def measure(self, measure_id: str) -> Measure:
        for m in self.measures:
            if m.id == measure_id:
                return m
        raise KeyError(f"no measure {measure_id!r} in the catalog")

    def table(self, qualified: str) -> Table:
        return self.tables[qualified]


def _measure(raw: dict[str, Any]) -> Measure:
    years = raw.get("years")
    return Measure(
        id=raw["id"],
        warehouse=raw["warehouse"],
        table=raw["table"],
        entity=raw["entity"],
        entity_column=raw.get("entity_column"),
        time=raw.get("time", "none"),
        time_column=raw.get("time_column"),
        expr=raw["expr"],
        fmt=raw["fmt"],
        unit=raw["unit"],
        phrases=tuple(raw.get("phrases", [])),
        where=raw.get("where"),
        years=(int(years[0]), int(years[1])) if years else None,
    )


def _tables(warehouse: str, raw: dict[str, Any]) -> dict[str, Table]:
    out: dict[str, Table] = {}
    for t in raw["tables"]:
        columns = tuple(
            Column(
                name=c["name"],
                type=c.get("type", ""),
                role=c.get("role", ""),
                unit=c.get("unit", "none") or "none",
                definition=c.get("definition") or "",
            )
            for c in t.get("columns", [])
        )
        table = Table(
            warehouse=warehouse,
            name=t["name"],
            description=t.get("description") or "",
            rows=int(t.get("rows") or 0),
            columns=columns,
        )
        out[table.qualified] = table
    return out


@lru_cache(maxsize=2)
def load(base: Path | None = None) -> Catalog:
    base = base or paths.readback_dir()
    measures_raw = yaml.safe_load((base / "catalog" / "measures.yaml").read_text(encoding="utf-8"))
    entities = yaml.safe_load((base / "catalog" / "entities.yaml").read_text(encoding="utf-8"))
    synonyms = yaml.safe_load((base / "catalog" / "synonyms.yaml").read_text(encoding="utf-8"))
    tables: dict[str, Table] = {}
    for w in WAREHOUSES:
        tables.update(_tables(w, yaml.safe_load((base / "catalog" / f"{w}.yaml").read_text(encoding="utf-8"))))
    return Catalog(
        measures=[_measure(m) for m in measures_raw["measures"]],
        tables=tables,
        entities=entities,
        synonyms=synonyms.get("synonyms", {}),
    )
