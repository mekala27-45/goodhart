"""The question generator: candidate questions over the catalog measures, each with gold SQL.

For every catalog measure the generator reads what the warehouse holds (which entity has a value in
which period) and draws candidate questions in the six families readback grades with SQL: lookup,
comparison, trend, top k, multi, refusal. The definition family is out of scope: its answers are text
from a document. Every candidate carries its entity tuple, its template, its gold SQL, and the units
and tolerances of its gold columns. Nothing here is graded or split: goodhart_questions.build does the
filters, the splits and the sampling, in that order, and counts every drop.

The first look at the warehouses found real defects that are rules here, each with a test:
ACS measures have no 2020 rows; the ledger window starts and ends in partial quarters; turnaround's
2026 is a partial year; carriers join and leave mid year; tiny airports make extreme rates; three
monitoring rules tie at a precision of one.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from goodhart_core.items import GoldColumn
from goodhart_core.units import COUNT_TOLERANCE, RANK_TOLERANCE, Unit, asked_unit, stored_unit, tolerance
from goodhart_core.warehouse import Warehouses

from goodhart_questions import sqlbuild
from goodhart_questions.catalog import Catalog, Measure
from goodhart_questions.names import Namer
from goodhart_questions.templates import PLURALS, REFUSALS, TEMPLATES, UNIT_SUFFIX

FULL_QUARTERS = (
    "2023Q4",
    "2024Q1",
    "2024Q2",
    "2024Q3",
    "2024Q4",
    "2025Q1",
    "2025Q2",
    "2025Q3",
    "2025Q4",
    "2026Q1",
    "2026Q2",
)
PARTIAL_YEAR = {"turnaround": 2026}
NO_ACS_YEAR = 2020


@dataclass
class Candidate:
    family: str
    shape: str
    template: str
    question: str
    measure: str | None
    warehouse: str
    entities: list[str]
    periods: list[str]
    unit_asked: str
    names: dict[str, str]
    sql: str | None
    label: bool
    ordered: bool
    columns: list[GoldColumn]
    refuse: bool = False
    refusal_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Cube:
    """What the warehouse holds for one measure: entity -> period -> value (stored unit)."""

    measure: Measure
    values: dict[str, dict[str, float]]

    def periods(self) -> list[str]:
        return sorted({p for row in self.values.values() for p in row})

    def entities(self, period: str | None = None) -> list[str]:
        if period is None:
            return sorted(self.values)
        return sorted(e for e, row in self.values.items() if period in row)


def _period_text(period: str) -> str:
    if len(period) == 6 and period[4] == "Q":
        return f"{period[:4]} Q{period[5]}"
    return period


# The period phrases a template may carry; a measure with no time grain drops them whole.
_NO_PERIOD = (", in {period},", " in {period}", " for {period}", "In {period}, ")


def fill(text: str, period: str | None, **fields: str) -> str:
    """Format a template. Without a period every period phrase goes, so a timeless measure never
    reads 'in ?' or 'In , what was'; the first letter is raised when the sentence lost its opening."""
    if period:
        return text.format(period=_period_text(period), **fields)
    for phrase in _NO_PERIOD:
        text = text.replace(phrase, "")
    out = text.format(**fields)
    return out[:1].upper() + out[1:]


def nice(value: float, unit: str) -> float:
    """A round threshold near a value, in the stored unit."""
    if unit == "fraction":
        return round(value, 2)
    if unit == "dollars":
        step = 10_000.0 if abs(value) >= 100_000 else 1_000.0
        return float(round(value / step) * step)
    if unit == "count":
        if value < 100:
            return float(round(value))
        digits = len(str(int(abs(value)))) - 2
        return float(round(value, -digits))
    if unit in {"minutes", "ratio", "percent"}:
        return round(value, 1)
    return round(value, 2)


def threshold_text(value: float, unit: str) -> str:
    if unit == "fraction":
        return f"{value * 100:g} percent"
    if unit == "dollars":
        return f"${value:,.0f}"
    if unit == "count":
        return f"{value:,.0f}"
    if unit == "minutes":
        return f"{value:g} minutes"
    if unit == "percent":
        return f"{value:g} percent"
    return f"{value:g}"


class Generator:
    def __init__(self, warehouses: Warehouses, catalog: Catalog, config: dict[str, Any], seed: int) -> None:
        self.w = warehouses
        self.catalog = catalog
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        metros = [
            r[0]
            for r in self.w.run(
                "select distinct cbsa_title from doorstep.mart_metro_year where kind = 'metropolitan' order by 1"
            ).rows
        ]
        self.namer = Namer([str(m) for m in metros], catalog.entities)
        self.big_metros = [
            str(r[0])
            for r in self.w.run(
                "select cbsa_title from doorstep.mart_metro_year where kind = 'metropolitan' and year = 2024 "
                "and households is not null order by households desc, cbsa_title limit 150"
            ).rows
        ]
        self.cubes: dict[str, Cube] = {}

    # The warehouse's holdings, with the first look's rules applied.

    def cube(self, m: Measure) -> Cube:
        if m.id in self.cubes:
            return self.cubes[m.id]
        e = m.entity_column or "'all'"
        t = m.time_column if m.has_time else "'all'"
        where = f" where ({m.where})" if m.where else ""
        having = f"having {m.expr} is not null"
        if m.table == "mart_carrier_month":
            having += " and count(distinct month) = 12"
        sql = f"select {e} as e, {t} as t, {m.expr} as v from {m.qualified}{where} group by 1, 2 {having}"
        result = self.w.run(sql, timeout_s=30.0)
        if not result.executed:
            raise RuntimeError(f"the cube of {m.id} did not run: {result.error}")
        values: dict[str, dict[str, float]] = defaultdict(dict)
        airports = set(self.catalog.entities.get("airports", {}))
        for ent, per, val in result.rows:
            ent_s, per_s = str(ent), str(per)
            if val is None:
                continue
            if m.time == "year" and m.years and not (m.years[0] <= int(per_s) <= m.years[1]):
                continue
            if m.time == "year" and m.warehouse in PARTIAL_YEAR and int(per_s) >= PARTIAL_YEAR[m.warehouse]:
                continue
            if m.time == "quarter" and per_s not in FULL_QUARTERS:
                continue
            if m.entity == "airport" and ent_s not in airports:
                continue
            values[ent_s][per_s] = float(val)
        self.cubes[m.id] = Cube(m, dict(values))
        return self.cubes[m.id]

    # Helpers.

    def _phrase(self, m: Measure) -> str:
        return self.rng.choice(list(m.phrases)) if m.phrases else m.id.replace("_", " ")

    def _template(self, shape: str) -> tuple[str, str]:
        options = TEMPLATES[shape]
        k = self.rng.randrange(len(options))
        return f"{shape}.v{k + 1}", options[k]

    def _unit(self, m: Measure, allow: bool = True) -> tuple[str, Unit, float, float]:
        """(asked, unit, factor, tolerance) for a value question about measure m."""
        stored = stored_unit(m.unit)
        asked = "stored"
        if allow and stored == "fraction" and self.rng.random() < 0.3:
            asked = "percent"
        elif allow and stored == "dollars" and self.rng.random() < 0.25:
            asked = "thousands"
        unit, factor = asked_unit(stored, asked)
        tol = COUNT_TOLERANCE if stored == "count" and asked == "stored" else tolerance(m.fmt, factor)
        return asked, unit, factor, tol

    def _name(self, m: Measure, value: str) -> tuple[str, dict[str, str]]:
        if not m.entity_column:
            return value, {}
        name = self.namer.name(m.entity, value, m.entity_column, self.rng)
        return name.written, {name.written: name.line()}

    def _periods_for(self, cube: Cube, entity: str | None) -> list[str]:
        if entity is None:
            return cube.periods()
        return sorted(cube.values.get(entity, {}))

    def _pool(self, m: Measure, period: str | None) -> list[str]:
        cube = self.cube(m)
        pool = cube.entities(period) if period else cube.entities()
        if m.entity == "metro":
            big = [e for e in self.big_metros if e in set(pool)]
            return big if len(big) >= 10 else pool
        return pool

    # Families.

    def lookup(self, m: Measure) -> Candidate | None:
        cube = self.cube(m)
        entity = self.rng.choice(cube.entities()) if m.has_entity else "all"
        periods = self._periods_for(cube, entity)
        if not periods:
            return None
        period = self.rng.choice(periods) if m.has_time else None
        asked, unit, factor, tol = self._unit(m)
        shape = (
            "lookup.value" if m.has_entity and m.has_time else ("lookup.national" if m.has_time else "lookup.timeless")
        )
        template, text = self._template(shape)
        written, names = self._name(m, entity) if m.has_entity else ("", {})
        question = fill(text, period, phrase=self._phrase(m), entity=written)
        question = question.rstrip("?.") + UNIT_SUFFIX.get(asked, "") + ("?" if question.endswith("?") else ".")
        sql = sqlbuild.lookup(m, entity if m.has_entity else None, period, factor)
        return Candidate(
            "lookup",
            shape,
            template,
            question,
            m.id,
            m.warehouse,
            [entity] if m.has_entity else [],
            [period] if period else [],
            asked,
            names,
            sql,
            label=False,
            ordered=False,
            columns=[GoldColumn(name="value", unit=unit, tolerance=tol)],
        )

    def comparison(self, m: Measure) -> Candidate | None:
        if not m.has_entity:
            return None
        cube = self.cube(m)
        period = self.rng.choice(cube.periods()) if m.has_time else None
        pool = self._pool(m, period)
        if len(pool) < 2:
            return None
        a, b = self.rng.sample(pool, 2)
        asked, unit, factor, tol = self._unit(m)
        va, vb = cube.values[a][period or "all"] * factor, cube.values[b][period or "all"] * factor
        if abs(va - vb) <= 2 * tol:
            return None
        shape = "comparison.pair" if m.has_time else "comparison.timeless"
        template, text = self._template(shape)
        wa, na = self._name(m, a)
        wb, nb = self._name(m, b)
        if wa == wb:
            return None
        question = fill(text, period, phrase=self._phrase(m), a=wa, b=wb)
        question = question.rstrip("?.") + UNIT_SUFFIX.get(asked, "") + "?"
        sql = sqlbuild.comparison(m, a, b, period, factor)
        return Candidate(
            "comparison",
            shape,
            template,
            question,
            m.id,
            m.warehouse,
            [a, b],
            [period] if period else [],
            asked,
            {**na, **nb},
            sql,
            label=True,
            ordered=False,
            columns=[GoldColumn(name="value", unit=unit, tolerance=tol)],
        )

    def trend(self, m: Measure) -> Candidate | None:
        if not m.has_time:
            return None
        cube = self.cube(m)
        entity = self.rng.choice(cube.entities()) if m.has_entity else "all"
        periods = self._periods_for(cube, entity)
        if len(periods) < 3:
            return None
        i, j = sorted(self.rng.sample(range(len(periods)), 2))
        if j - i < 2:
            return None
        p1, p2 = periods[i], periods[j]
        asked, unit, factor, tol = self._unit(m)
        row = cube.values[entity]
        if abs(row[p2] - row[p1]) * factor <= 2 * tol:
            return None
        shape = "trend.pair" if m.has_entity else "trend.national"
        template, text = self._template(shape)
        written, names = self._name(m, entity) if m.has_entity else ("", {})
        question = text.format(phrase=self._phrase(m), entity=written, p1=_period_text(p1), p2=_period_text(p2))
        question = question.rstrip("?.") + UNIT_SUFFIX.get(asked, "") + "?"
        sql = sqlbuild.trend(m, entity if m.has_entity else None, p1, p2, factor)
        return Candidate(
            "trend",
            shape,
            template,
            question,
            m.id,
            m.warehouse,
            [entity] if m.has_entity else [],
            [p1, p2],
            asked,
            names,
            sql,
            label=True,
            ordered=False,
            columns=[GoldColumn(name="value", unit=unit, tolerance=tol)],
        )

    def top_k(self, m: Measure) -> Candidate | None:
        cube = self.cube(m)
        k = self.rng.choice([3, 5])
        highest = self.rng.random() < 0.65
        asked, unit, factor, tol = self._unit(m)
        extreme = "highest" if highest else "lowest"
        if m.has_entity and m.has_time and m.entity in {"metro", "carrier"}:
            period = self.rng.choice(cube.periods())
            vals = sorted(
                ((v[period] * factor, e) for e, v in cube.values.items() if period in v),
                key=lambda x: (-x[0] if highest else x[0], x[1]),
            )
            if len(vals) <= k + 1 or _ties(vals, k, tol):
                return None
            template, text = self._template("top_k.entities")
            question = text.format(
                k=k,
                plural=PLURALS[m.entity],
                extreme=extreme,
                phrase=self._phrase(m),
                period=_period_text(period),
            )
            question = question.rstrip("?.") + UNIT_SUFFIX.get(asked, "") + ("?" if text.endswith("?") else ".")
            sql = sqlbuild.top_k(m, period, k, highest, factor)
            return Candidate(
                "top_k",
                "top_k.entities",
                template,
                question,
                m.id,
                m.warehouse,
                [],
                [period],
                asked,
                {},
                sql,
                label=True,
                ordered=True,
                columns=[GoldColumn(name="value", unit=unit, tolerance=tol)],
                extra={"k": k, "highest": highest},
            )
        if m.has_time and not m.has_entity:
            periods = cube.periods()
            if len(periods) < k + 3:
                return None
            row = cube.values["all"]
            span_start = self.rng.randrange(0, max(1, len(periods) - (k + 3)))
            p1, p2 = periods[span_start], periods[-1]
            inside = [p for p in periods if p1 <= p <= p2]
            vals = sorted(((row[p] * factor, p) for p in inside), key=lambda x: (-x[0] if highest else x[0], x[1]))
            if len(vals) <= k or _ties(vals, k, tol):
                return None
            template, text = self._template("top_k.periods")
            unit_plural = "quarters" if m.time == "quarter" else "years"
            question = text.format(
                k=k,
                unit_plural=unit_plural,
                p1=_period_text(p1),
                p2=_period_text(p2),
                extreme=extreme,
                phrase=self._phrase(m),
                for_entity="",
            )
            question = question.rstrip("?.") + UNIT_SUFFIX.get(asked, "") + ("?" if text.endswith("?") else ".")
            sql = sqlbuild.top_k_periods(m, None, p1, p2, k, highest, factor)
            return Candidate(
                "top_k",
                "top_k.periods",
                template,
                question,
                m.id,
                m.warehouse,
                [],
                [p1, p2],
                asked,
                {},
                sql,
                label=True,
                ordered=True,
                columns=[GoldColumn(name="value", unit=unit, tolerance=tol)],
                extra={"k": k, "highest": highest},
            )
        return None

    def count_above(self, m: Measure) -> Candidate | None:
        if not (m.has_entity and m.entity in {"metro", "carrier"}):
            return None
        cube = self.cube(m)
        period = self.rng.choice(cube.periods()) if m.has_time else None
        vals = sorted(v[period or "all"] for v in cube.values.values() if (period or "all") in v)
        if len(vals) < 6:
            return None
        stored = stored_unit(m.unit)
        threshold = nice(vals[self.rng.randrange(len(vals) // 4, 3 * len(vals) // 4)], stored)
        count = sum(1 for v in vals if v > threshold)
        if count in (0, len(vals)) or any(abs(v - threshold) < 1e-12 for v in vals):
            return None
        template, text = self._template("multi.count_above")
        question = fill(
            text,
            period,
            plural=PLURALS[m.entity],
            phrase=self._phrase(m),
            threshold=threshold_text(threshold, stored),
        )
        sql = sqlbuild.count_above(m, period, threshold)
        return Candidate(
            "multi",
            "multi.count_above",
            template,
            question,
            m.id,
            m.warehouse,
            [],
            [period] if period else [],
            "stored",
            {},
            sql,
            label=False,
            ordered=False,
            columns=[GoldColumn(name="value", unit="count", tolerance=COUNT_TOLERANCE)],
            extra={"threshold": threshold},
        )

    def average(self, m: Measure) -> Candidate | None:
        if not m.has_entity:
            return None
        cube = self.cube(m)
        period = self.rng.choice(cube.periods()) if m.has_time else None
        pool = self._pool(m, period)
        if len(pool) < 3:
            return None
        group = sorted(self.rng.sample(pool, 3))
        asked, unit, factor, tol = self._unit(m)
        names: dict[str, str] = {}
        written = []
        for e in group:
            w, n = self._name(m, e)
            written.append(w)
            names.update(n)
        listed = f"{written[0]}, {written[1]} and {written[2]}"
        template, text = self._template("multi.average")
        question = fill(text, period, phrase=self._phrase(m), list=listed)
        question = question.rstrip("?.") + UNIT_SUFFIX.get(asked, "") + "?"
        sql = sqlbuild.average_of(m, group, period, factor)
        return Candidate(
            "multi",
            "multi.average",
            template,
            question,
            m.id,
            m.warehouse,
            group,
            [period] if period else [],
            asked,
            names,
            sql,
            label=False,
            ordered=False,
            columns=[GoldColumn(name="value", unit=unit, tolerance=tol)],
        )

    def periods_above(self, m: Measure) -> Candidate | None:
        if not m.has_time:
            return None
        cube = self.cube(m)
        entity = self.rng.choice(cube.entities()) if m.has_entity else "all"
        periods = self._periods_for(cube, entity)
        if len(periods) < 5:
            return None
        start = self.rng.randrange(0, len(periods) - 4)
        span = periods[start : start + self.rng.randrange(5, min(12, len(periods) - start) + 1)]
        row = cube.values[entity]
        stored = stored_unit(m.unit)
        vals = sorted(row[p] for p in span)
        threshold = nice(vals[len(vals) // 2], stored)
        above = [p for p in span if row[p] > threshold]
        if not above or len(above) == len(span) or any(abs(row[p] - threshold) < 1e-12 for p in span):
            return None
        written, names = self._name(m, entity) if m.has_entity else ("", {})
        template, text = self._template("multi.periods_above")
        question = text.format(
            unit_plural="quarters" if m.time == "quarter" else "years",
            p1=_period_text(span[0]),
            p2=_period_text(span[-1]),
            phrase=self._phrase(m),
            in_entity=f" in {written}" if written else "",
            threshold=threshold_text(threshold, stored),
        )
        sql = sqlbuild.periods_above(m, entity if m.has_entity else None, span[0], span[-1], threshold)
        return Candidate(
            "multi",
            "multi.periods_above",
            template,
            question,
            m.id,
            m.warehouse,
            [entity] if m.has_entity else [],
            [span[0], span[-1]],
            "stored",
            names,
            sql,
            label=True,
            ordered=False,
            columns=[],
            extra={"threshold": threshold},
        )

    def difference(self, m: Measure) -> Candidate | None:
        if not m.has_time:
            return None
        cube = self.cube(m)
        entity = self.rng.choice(cube.entities()) if m.has_entity else "all"
        periods = self._periods_for(cube, entity)
        if len(periods) < 3:
            return None
        i, j = sorted(self.rng.sample(range(len(periods)), 2))
        p1, p2 = periods[i], periods[j]
        asked, unit, factor, tol = self._unit(m)
        row = cube.values[entity]
        if abs(row[p2] - row[p1]) * factor <= 2 * tol:
            return None
        written, names = self._name(m, entity) if m.has_entity else ("", {})
        template, text = self._template("multi.difference")
        question = text.format(
            phrase=self._phrase(m),
            in_entity=f" in {written}" if written else "",
            p1=_period_text(p1),
            p2=_period_text(p2),
        )
        question = question.rstrip("?.") + UNIT_SUFFIX.get(asked, "") + "?"
        sql = sqlbuild.difference(m, entity if m.has_entity else None, p1, p2, factor)
        return Candidate(
            "multi",
            "multi.difference",
            template,
            question,
            m.id,
            m.warehouse,
            [entity] if m.has_entity else [],
            [p1, p2],
            asked,
            names,
            sql,
            label=False,
            ordered=False,
            columns=[GoldColumn(name="value", unit=unit, tolerance=tol)],
        )

    def rank(self, m: Measure) -> Candidate | None:
        if not (m.has_entity and m.entity in {"metro", "carrier", "airport"}):
            return None
        cube = self.cube(m)
        period = self.rng.choice(cube.periods()) if m.has_time else None
        pool = self._pool(m, period)
        if len(pool) < 5:
            return None
        group = sorted(self.rng.sample(pool, 5))
        vals = [cube.values[e][period or "all"] for e in group]
        if len(set(vals)) < len(vals):
            return None
        target = self.rng.choice(group)
        names: dict[str, str] = {}
        written_group = []
        for e in group:
            w, n = self._name(m, e)
            written_group.append(w)
            names.update(n)
        written_target = written_group[group.index(target)]
        listed = ", ".join(written_group[:-1]) + f" and {written_group[-1]}"
        template, text = self._template("multi.rank")
        question = fill(text, period, list=listed, entity=written_target, phrase=self._phrase(m))
        sql = sqlbuild.rank_in_group(m, target, group, period)
        return Candidate(
            "multi",
            "multi.rank",
            template,
            question,
            m.id,
            m.warehouse,
            group,
            [period] if period else [],
            "stored",
            names,
            sql,
            label=False,
            ordered=False,
            columns=[GoldColumn(name="value", unit="rank", tolerance=RANK_TOLERANCE)],
        )

    def refusal(self) -> Candidate:
        reason = self.rng.choice(sorted(REFUSALS))
        k = self.rng.randrange(len(REFUSALS[reason]))
        text = REFUSALS[reason][k]
        metros = [m for m in self.catalog.measures if m.entity == "metro"]
        m = self.rng.choice(metros)
        entity = self.rng.choice(self.big_metros[:80])
        written, names = self._name(m, entity)
        carriers = sorted(self.namer.carriers)
        carrier = self.rng.choice(self.namer.carriers[self.rng.choice(carriers)]) if carriers else "Southwest"
        future = str(2027 + self.rng.randrange(0, 6))
        past = str(1975 + self.rng.randrange(0, 20))
        code = f"C{self.rng.randrange(10_000, 99_999)}"
        question = text.format(
            phrase=self._phrase(m),
            entity=written,
            future=future,
            past=past,
            code=code,
            period=str(self.rng.randrange(2015, 2025)),
            carrier=carrier,
        )
        tuple_entities = [reason, entity if "{entity}" in text else code]
        return Candidate(
            "refusal",
            f"refusal.{reason.lower()}",
            f"refusal.{reason.lower()}.v{k + 1}",
            question,
            None,
            "none",
            tuple_entities,
            [],
            "stored",
            names if "{entity}" in text else {},
            None,
            label=False,
            ordered=False,
            columns=[],
            refuse=True,
            refusal_reason=reason,
        )

    SHAPES: ClassVar[dict[str, str]] = {
        "lookup": "lookup",
        "comparison": "comparison",
        "trend": "trend",
        "top_k": "top_k",
        "multi.count_above": "count_above",
        "multi.average": "average",
        "multi.periods_above": "periods_above",
        "multi.difference": "difference",
        "multi.rank": "rank",
    }

    def candidates(self, per_measure: dict[str, int], refusals: int) -> Iterator[Candidate]:
        """Draw candidates measure by measure, shape by shape, in a fixed order under the seed."""
        for m in sorted(self.catalog.measures, key=lambda x: x.id):
            for shape, attempts in sorted(per_measure.items()):
                method = getattr(self, self.SHAPES[shape])
                for _ in range(attempts):
                    c = method(m)
                    if c is not None:
                        yield c
        for _ in range(refusals):
            yield self.refusal()


def _ties(vals: list[tuple[float, str]], k: int, tol: float) -> bool:
    """A ranking is defined only if the first k are apart and the k-th is apart from the next."""
    top = [v for v, _ in vals[: k + 1]]
    return any(abs(top[i] - top[i + 1]) <= 2 * tol for i in range(len(top) - 1))
