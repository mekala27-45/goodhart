"""readback's golden set as goodhart items: the customer's own held-out questions.

readback grades claims; goodhart grades a result table. Each readback question becomes an item whose
gold SQL is built the goodhart way from the same catalog measure (lookup, comparison, trend), taken
from readback's own ranking SQL (top k), or written for its shape (the multi questions about table
and row counts and shared years). Every converted gold is checked against readback's published claim
values within readback's tolerance; a test fails if one differs. Definition questions are out of
scope (their answers are text from a document) and are counted, not dropped silently.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from goodhart_core import paths
from goodhart_core.items import Gold, GoldColumn, Item
from goodhart_core.units import COUNT_TOLERANCE, stored_unit
from goodhart_core.warehouse import Warehouses

from goodhart_questions import sqlbuild
from goodhart_questions.catalog import Catalog, Measure
from goodhart_questions.names import first_city

PERIOD = re.compile(r"^(?:\d{4}|\d{4}Q[1-4])$")


@dataclass
class Converted:
    items: list[Item]
    excluded: dict[str, int]
    mismatches: list[str]


def load_golden(base: Path | None = None) -> list[dict[str, Any]]:
    folder = (base or paths.readback_dir()) / "evals" / "golden"
    out: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.yaml")):
        out.extend(yaml.safe_load(path.read_text(encoding="utf-8"))["questions"])
    return sorted(out, key=lambda q: q["id"])


def split_entities(values: Iterable[str]) -> tuple[list[str], list[str]]:
    entities: list[str] = []
    periods: list[str] = []
    for v in values:
        (periods if PERIOD.match(str(v)) else entities).append(str(v))
    return entities, periods


def tuple_key(measure: str | None, values: Iterable[str]) -> str:
    entities, periods = split_entities(values)
    probe = Item(
        id="x",
        question="x",
        warehouse="x",
        family="lookup",
        template="x",
        shape="x",
        measure=measure,
        entities=entities,
        periods=periods,
        split="train",
        source="readback",
    )
    return probe.entity_tuple()


def _names(question: str, m: Measure | None, entities: list[str], catalog: Catalog) -> dict[str, str]:
    if m is None or not m.entity_column:
        return {}
    out: dict[str, str] = {}
    lowered = question.lower()
    aliases: dict[str, list[str]] = {}
    for key in ("carriers", "airports", "applicant_groups", "aml_rules"):
        aliases.update({k: list(v) for k, v in catalog.entities.get(key, {}).items()})
    for e in entities:
        forms = [e, first_city(e), *aliases.get(e, [])]
        written = next((f for f in forms if f and f.lower() in lowered), e)
        out[written] = f"{m.entity_column.strip(chr(34))} '{e}'"
    return out


def _gold(
    w: Warehouses, sql: str, label: bool, ordered: bool, columns: list[GoldColumn], warehouse: str
) -> Gold | None:
    result = w.run(sql, "original", default_warehouse=warehouse, timeout_s=10.0)
    if not result.executed or not result.rows:
        return None
    return Gold(sql=sql, label=label, columns=columns, rows=result.rows, ordered=ordered, tables=result.tables)


def _close(a: object, b: object, tol: float) -> bool:
    """True when two numbers sit within tol; text, None and booleans never do."""
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if not isinstance(a, int | float) or not isinstance(b, int | float):
        return False
    return abs(float(a) - float(b)) <= tol + 1e-9


def convert(w: Warehouses, catalog: Catalog, golden: list[dict[str, Any]] | None = None) -> Converted:
    golden = golden if golden is not None else load_golden()
    items: list[Item] = []
    excluded: dict[str, int] = {}
    mismatches: list[str] = []
    for q in golden:
        family = q["family"]
        split = "readback_test" if q["split"] == "test" else "readback_dev"
        warehouse = (q.get("warehouses") or ["none"])[0]
        base = {
            "id": f"rb-{q['id']}",
            "question": q["question"],
            "split": split,
            "source": "readback",
            "template": f"readback.{family}",
            "shape": f"readback.{family}",
        }
        if family == "definition":
            excluded["definition"] = excluded.get("definition", 0) + 1
            continue
        if family == "refusal":
            items.append(
                Item(
                    **base,
                    warehouse="none",
                    family="refusal",
                    refuse=True,
                    refusal_reason="OUT_OF_SCOPE",
                    entities=[q["id"]],
                )
            )
            continue
        claims = q.get("claims") or []
        m = catalog.measure(q["measure"]) if q.get("measure") else None
        entities, periods = split_entities(q.get("entities") or [])
        gold: Gold | None = None
        if family in {"lookup", "comparison", "trend"} and m is not None:
            unit = stored_unit(m.unit)
            tol = float(claims[0]["tolerance"])
            col = [GoldColumn(name="value", unit=unit, tolerance=tol)]
            if family == "lookup":
                sql = sqlbuild.lookup(m, entities[0] if m.has_entity else None, periods[0] if periods else None, 1.0)
                gold = _gold(w, sql, False, False, col, m.warehouse)
                expected = [(None, c["value"]) for c in claims]
            elif family == "comparison" and len(entities) < 2:
                # Two periods of one entity (the nation, the bank): the trend shape, labelled by period.
                sql = sqlbuild.trend(m, entities[0] if entities and m.has_entity else None, periods[0], periods[1], 1.0)
                gold = _gold(w, sql, True, False, col, m.warehouse)
                expected = [(c["label"], c["value"]) for c in claims]
            elif family == "comparison":
                sql = sqlbuild.comparison(m, entities[0], entities[1], periods[0] if periods else None, 1.0)
                gold = _gold(w, sql, True, False, col, m.warehouse)
                expected = [(c["label"], c["value"]) for c in claims]
            else:
                sql = sqlbuild.trend(m, entities[0] if m.has_entity else None, periods[0], periods[1], 1.0)
                gold = _gold(w, sql, True, False, col, m.warehouse)
                expected = [(c["label"], c["value"]) for c in claims]
            if gold is not None:
                got = {str(r[0]) if gold.label else None: r[-1] for r in gold.rows}
                for label, value in expected:
                    key = str(label) if gold.label else None
                    if key not in got or not _close(got[key], value, tol):
                        mismatches.append(f"{q['id']}: {label} {value} against {got.get(key)}")
        elif family == "top_k":
            sql = re.sub(r"\bas entity\b", "as label", claims[0]["sql"])
            sql = re.sub(r"\bentity limit\b", "label limit", sql)
            sql = re.sub(r",\s*entity\s+limit", ", label limit", sql)
            unit = stored_unit(m.unit) if m else "count"
            col = [GoldColumn(name="value", unit=unit, tolerance=float(claims[0]["tolerance"]))]
            gold = _gold(w, sql, True, True, col, warehouse)
            ranking = q.get("ranking") or [c["label"] for c in claims]
            if gold is not None and [str(r[0]) for r in gold.rows] != [str(x) for x in ranking]:
                mismatches.append(f"{q['id']}: ranking {ranking} against {[r[0] for r in gold.rows]}")
        elif family == "multi":
            gold = _multi_gold(w, q, claims, mismatches)
        if gold is None:
            excluded[f"{family}: no gold"] = excluded.get(f"{family}: no gold", 0) + 1
            continue
        items.append(
            Item(
                **base,
                warehouse=warehouse if warehouse in {"doorstep", "turnaround", "ledger"} else "doorstep",
                family=family,
                measure=m.id if m else None,
                entities=entities,
                periods=periods,
                names=_names(q["question"], m, entities, catalog),
                gold=gold,
            )
        )
    return Converted(items=items, excluded=excluded, mismatches=mismatches)


def _multi_gold(w: Warehouses, q: dict[str, Any], claims: list[dict[str, Any]], mismatches: list[str]) -> Gold | None:
    text = q["question"]
    count_col = [GoldColumn(name="value", unit="count", tolerance=COUNT_TOLERANCE)]
    if text.startswith("Which warehouse publishes more tables"):
        names = [c["label"] for c in claims]
        sql = (
            "select table_schema as label, count(*) as value from information_schema.tables "
            f"where table_schema in ({', '.join(sqlbuild.quote(n) for n in names)}) "
            "group by table_schema order by value desc, label"
        )
        gold = _gold(w, sql, True, False, count_col, names[0])
    elif text.startswith("How many published tables"):
        sql = (
            "select count(*) as value from information_schema.tables "
            "where table_schema in ('doorstep', 'turnaround', 'ledger')"
        )
        gold = _gold(w, sql, False, False, count_col, "doorstep")
    elif text.startswith("Which has more rows"):
        parts = [f"select {sqlbuild.quote(c['label'])} as label, count(*) as value from {c['label']}" for c in claims]
        sql = " union all ".join(parts) + " order by value desc, label"
        gold = _gold(w, sql, True, False, count_col, claims[0]["label"].split(".")[0])
    elif text.startswith("Which years appear in both"):
        inner = claims[0]["sql"]
        body = re.sub(r"^select count\(\*\) from \(", "", inner).rstrip()
        body = body[:-1] if body.endswith(")") else body
        sql = f"select * from ({body}) as shared(label) order by label"
        gold = _gold(w, sql, True, False, [], "doorstep")
        if gold is not None and len(gold.rows) != int(claims[0]["value"]):
            mismatches.append(f"{q['id']}: {len(gold.rows)} shared years against {claims[0]['value']}")
        return gold
    else:
        return None
    if gold is not None:
        got = {str(r[0]) if gold.label else None: r[-1] for r in gold.rows}
        for c in claims:
            key = str(c["label"]) if gold.label else None
            if key not in got or not _close(got[key], c["value"], 0.5):
                mismatches.append(f"{q['id']}: {c['label']} {c['value']} against {got.get(key)}")
    return gold


def golden_tuples(golden: list[dict[str, Any]], splits: set[str]) -> set[str]:
    return {
        tuple_key(q.get("measure"), q.get("entities") or [])
        for q in golden
        if q["split"] in splits and q.get("measure")
    }
