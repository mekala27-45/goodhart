"""Gold SQL, one builder per question shape, over a catalog measure.

Every gold value is the result of the SQL beside it over the warehouse, never a number typed in. The
builders follow the answer rules the prompt states: a label column when the question names entities
or periods, then the values; one row per entity or period; rankings in order; the unit the question
asks for. Ties are broken by the label so a ranking is the same on every run.
"""

from __future__ import annotations

from goodhart_questions.catalog import Measure


def quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def literal(period: str) -> str:
    return period if period.isdigit() else quote(period)


def scaled(expr: str, factor: float) -> str:
    if factor == 1.0:
        return expr
    if factor == 100.0:
        return f"100 * {expr}"
    if factor == 0.001:
        return f"{expr} / 1000"
    raise ValueError(f"no scaling for factor {factor}")


def _where(m: Measure, *extra: str) -> str:
    parts = ([f"({m.where})"] if m.where else []) + [e for e in extra if e]
    return " where " + " and ".join(parts) if parts else ""


def _entity_eq(m: Measure, entity: str) -> str:
    assert m.entity_column is not None
    return f"{m.entity_column} = {quote(entity)}"


def _entity_in(m: Measure, entities: list[str]) -> str:
    assert m.entity_column is not None
    return f"{m.entity_column} in ({', '.join(quote(e) for e in entities)})"


def _time_eq(m: Measure, period: str) -> str:
    return f"{m.time_column} = {literal(period)}" if m.has_time else ""


def lookup(m: Measure, entity: str | None, period: str | None, factor: float) -> str:
    conds = [_entity_eq(m, entity) if entity is not None else "", _time_eq(m, period) if period else ""]
    return f"select {scaled(m.expr, factor)} as value from {m.qualified}{_where(m, *conds)}"


def comparison(m: Measure, a: str, b: str, period: str | None, factor: float) -> str:
    assert m.entity_column is not None
    conds = [_entity_in(m, [a, b]), _time_eq(m, period) if period else ""]
    return (
        f"select {m.entity_column} as label, {scaled(m.expr, factor)} as value from {m.qualified}{_where(m, *conds)} "
        f"group by {m.entity_column} order by value desc, label"
    )


def trend(m: Measure, entity: str | None, p1: str, p2: str, factor: float) -> str:
    assert m.time_column is not None
    conds = [
        _entity_eq(m, entity) if entity is not None else "",
        f"{m.time_column} in ({literal(p1)}, {literal(p2)})",
    ]
    return (
        f"select {m.time_column} as label, {scaled(m.expr, factor)} as value from {m.qualified}{_where(m, *conds)} "
        f"group by {m.time_column} order by label"
    )


def top_k(m: Measure, period: str | None, k: int, highest: bool, factor: float) -> str:
    assert m.entity_column is not None
    direction = "desc" if highest else "asc"
    conds = [_time_eq(m, period) if period else ""]
    return (
        f"select {m.entity_column} as label, {scaled(m.expr, factor)} as value from {m.qualified}{_where(m, *conds)} "
        f"group by {m.entity_column} having {m.expr} is not null order by value {direction}, label limit {k}"
    )


def top_k_periods(m: Measure, entity: str | None, p1: str, p2: str, k: int, highest: bool, factor: float) -> str:
    assert m.time_column is not None
    direction = "desc" if highest else "asc"
    conds = [
        _entity_eq(m, entity) if entity is not None else "",
        f"{m.time_column} between {literal(p1)} and {literal(p2)}",
    ]
    return (
        f"select {m.time_column} as label, {scaled(m.expr, factor)} as value from {m.qualified}{_where(m, *conds)} "
        f"group by {m.time_column} having {m.expr} is not null order by value {direction}, label limit {k}"
    )


def count_above(m: Measure, period: str | None, threshold: float) -> str:
    assert m.entity_column is not None
    conds = [_time_eq(m, period) if period else ""]
    return (
        f"select count(*) as value from (select {m.entity_column} as entity, {m.expr} as v from {m.qualified}"
        f"{_where(m, *conds)} group by {m.entity_column}) where v > {threshold!r}"
    )


def average_of(m: Measure, entities: list[str], period: str | None, factor: float) -> str:
    assert m.entity_column is not None
    conds = [_entity_in(m, entities), _time_eq(m, period) if period else ""]
    return (
        f"select avg(v) as value from (select {m.entity_column} as entity, {scaled(m.expr, factor)} as v "
        f"from {m.qualified}{_where(m, *conds)} group by {m.entity_column})"
    )


def periods_above(m: Measure, entity: str | None, p1: str, p2: str, threshold: float) -> str:
    assert m.time_column is not None
    conds = [
        _entity_eq(m, entity) if entity is not None else "",
        f"{m.time_column} between {literal(p1)} and {literal(p2)}",
    ]
    return (
        f"select label from (select {m.time_column} as label, {m.expr} as v from {m.qualified}{_where(m, *conds)} "
        f"group by {m.time_column}) where v > {threshold!r} order by label"
    )


def difference(m: Measure, entity: str | None, p1: str, p2: str, factor: float) -> str:
    def one(p: str) -> str:
        conds = [_entity_eq(m, entity) if entity is not None else "", _time_eq(m, p)]
        return f"(select {scaled(m.expr, factor)} from {m.qualified}{_where(m, *conds)})"

    return f"select {one(p2)} - {one(p1)} as value"


def rank_in_group(m: Measure, entity: str, group: list[str], period: str | None) -> str:
    assert m.entity_column is not None
    conds = [_entity_in(m, group), _time_eq(m, period) if period else ""]
    return (
        f"select rank as value from (select {m.entity_column} as entity, rank() over (order by {m.expr} desc) as rank "
        f"from {m.qualified}{_where(m, *conds)} group by {m.entity_column}) where entity = {quote(entity)}"
    )
