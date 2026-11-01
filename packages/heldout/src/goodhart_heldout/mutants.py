"""Near misses of every gold query, written with sqlglot: how strong is each check, measured.

Each operator makes one small change a model could make by accident or on purpose: drop one
predicate, flip a comparison, shift a year (or a quarter) by one, swap max and min or sum and
average, remove a DISTINCT, flip an ORDER BY, change a LIMIT, change a GROUP BY key, replace an
entity with its neighbor. A mutant that returns the gold result on the original and on all sixteen
candidate twins is equivalent and set aside; the rest are the near misses every reward and every
check is scored against.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

Mutation = tuple[str, str]  # (operator, sql)
QUARTER = re.compile(r"^(\d{4})Q([1-4])$")


@dataclass(frozen=True)
class Mutant:
    item_id: str
    operator: str
    sql: str


def _render(tree: exp.Expression) -> str:
    return tree.sql(dialect="duckdb")


def _each(tree: exp.Expression, kind: type[exp.Expression]) -> list[exp.Expression]:
    return list(tree.find_all(kind))


def drop_predicate(tree: exp.Expression) -> Iterator[Mutation]:
    for where in _each(tree, exp.Where):
        conjuncts: list[exp.Expression] = []
        stack = [where.this]
        while stack:
            node = stack.pop()
            if isinstance(node, exp.And):
                stack.extend([node.left, node.right])
            elif isinstance(node, exp.Paren) and isinstance(node.this, exp.And):
                stack.append(node.this)
            else:
                conjuncts.append(node)
        for i in range(len(conjuncts)):
            copy = tree.copy()
            target = [w for w in copy.find_all(exp.Where)][_each(tree, exp.Where).index(where)]
            rest = [c.copy() for j, c in enumerate(conjuncts) if j != i]
            if rest:
                target.set("this", exp.and_(*rest))
            else:
                target.pop()
            yield ("drop a predicate", _render(copy))


FLIPS: dict[type[exp.Binary], type[exp.Binary]] = {
    exp.GT: exp.LT,
    exp.LT: exp.GT,
    exp.GTE: exp.LTE,
    exp.LTE: exp.GTE,
}


def flip_comparison(tree: exp.Expression) -> Iterator[Mutation]:
    nodes = [n for n in tree.walk() if isinstance(n, exp.Binary) and type(n) in FLIPS]
    for k in range(len(nodes)):
        copy = tree.copy()
        target = [n for n in copy.walk() if isinstance(n, exp.Binary) and type(n) in FLIPS][k]
        target.replace(FLIPS[type(target)](this=target.left.copy(), expression=target.right.copy()))
        yield ("flip a comparison", _render(copy))


def _shift(value: str, by: int) -> str | None:
    if value.isdigit() and 1990 <= int(value) <= 2035:
        return str(int(value) + by)
    m = QUARTER.match(value)
    if m:
        q = int(m.group(2)) + by
        y = int(m.group(1)) + (q - 1) // 4
        return f"{y}Q{(q - 1) % 4 + 1}"
    return None


def shift_period(tree: exp.Expression) -> Iterator[Mutation]:
    literals = [n for n in tree.find_all(exp.Literal) if _shift(n.this, 1) is not None]
    for k in range(len(literals)):
        for by in (1, -1):
            copy = tree.copy()
            target = [n for n in copy.find_all(exp.Literal) if _shift(n.this, 1) is not None][k]
            shifted = _shift(target.this, by)
            assert shifted is not None
            target.replace(exp.Literal.string(shifted) if target.is_string else exp.Literal.number(shifted))
            yield ("shift a period by one", _render(copy))


SWAPS: dict[type[exp.Expression], type[exp.Expression]] = {
    exp.Max: exp.Min,
    exp.Min: exp.Max,
    exp.Sum: exp.Avg,
    exp.Avg: exp.Sum,
}


def swap_aggregate(tree: exp.Expression) -> Iterator[Mutation]:
    nodes = [n for n in tree.walk() if isinstance(n, exp.Expression) and type(n) in SWAPS]
    for k in range(len(nodes)):
        copy = tree.copy()
        target = [n for n in copy.walk() if isinstance(n, exp.Expression) and type(n) in SWAPS][k]
        target.replace(SWAPS[type(target)](this=target.this.copy()))
        yield ("swap an aggregate", _render(copy))


def remove_distinct(tree: exp.Expression) -> Iterator[Mutation]:
    for k, _ in enumerate(_each(tree, exp.Distinct)):
        copy = tree.copy()
        _each(copy, exp.Distinct)[k].pop()
        yield ("remove a distinct", _render(copy))


def flip_order(tree: exp.Expression) -> Iterator[Mutation]:
    for k, _ in enumerate(_each(tree, exp.Ordered)):
        copy = tree.copy()
        target = _each(copy, exp.Ordered)[k]
        target.set("desc", not bool(target.args.get("desc")))
        yield ("flip an order", _render(copy))


def change_limit(tree: exp.Expression) -> Iterator[Mutation]:
    for k, lim in enumerate(_each(tree, exp.Limit)):
        value = lim.expression
        if isinstance(value, exp.Literal) and value.this.isdigit():
            for by in (1, -1):
                n = int(value.this) + by
                if n < 1:
                    continue
                copy = tree.copy()
                _each(copy, exp.Limit)[k].set("expression", exp.Literal.number(n))
                yield ("change a limit", _render(copy))


def change_group_key(tree: exp.Expression, alternatives: dict[str, list[str]]) -> Iterator[Mutation]:
    for k, group in enumerate(_each(tree, exp.Group)):
        for j, key in enumerate(group.expressions):
            if not isinstance(key, exp.Column):
                continue
            for alt in alternatives.get(key.name, []):
                copy = tree.copy()
                target = _each(copy, exp.Group)[k].expressions[j]
                target.replace(exp.column(alt))
                yield ("change a group key", _render(copy))


def replace_entity(tree: exp.Expression, neighbors: dict[str, str]) -> Iterator[Mutation]:
    strings = [n for n in tree.find_all(exp.Literal) if n.is_string and n.this in neighbors]
    for k in range(len(strings)):
        copy = tree.copy()
        target = [n for n in copy.find_all(exp.Literal) if n.is_string and n.this in neighbors][k]
        target.replace(exp.Literal.string(neighbors[target.this]))
        yield ("replace an entity with its neighbor", _render(copy))


OPERATORS: tuple[str, ...] = (
    "drop a predicate",
    "flip a comparison",
    "shift a period by one",
    "swap an aggregate",
    "remove a distinct",
    "flip an order",
    "change a limit",
    "change a group key",
    "replace an entity with its neighbor",
)


def mutate(
    item_id: str,
    sql: str,
    neighbors: dict[str, str],
    group_alternatives: dict[str, list[str]],
    cap_per_operator: int = 4,
) -> list[Mutant]:
    tree = sqlglot.parse_one(sql, read="duckdb")
    if not isinstance(tree, exp.Expression):
        raise ValueError(f"{item_id}: the gold query does not parse to one statement")
    gold_text = _render(tree)
    makers: list[Callable[[exp.Expression], Iterator[Mutation]]] = [
        drop_predicate,
        flip_comparison,
        shift_period,
        swap_aggregate,
        remove_distinct,
        flip_order,
        change_limit,
        lambda t: change_group_key(t, group_alternatives),
        lambda t: replace_entity(t, neighbors),
    ]
    out: list[Mutant] = []
    seen = {gold_text}
    for make in makers:
        count = 0
        for operator, text in make(tree):
            if text in seen or count >= cap_per_operator:
                continue
            seen.add(text)
            out.append(Mutant(item_id=item_id, operator=operator, sql=text))
            count += 1
    return out
