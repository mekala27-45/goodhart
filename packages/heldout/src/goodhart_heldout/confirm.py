"""Twin confirmation: a completion is confirmed when it matches the gold on the original and on every
kept twin. The twin gold of an item is its gold SQL executed on that twin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from goodhart_core.items import Execution, Gold, Item
from goodhart_core.warehouse import Warehouses
from goodhart_rewards.compare import compare


@dataclass(frozen=True)
class Confirmation:
    executed: bool
    original: bool
    twins_passed: int
    twins_total: int

    @property
    def confirmed(self) -> bool:
        return self.original and self.twins_passed == self.twins_total


@dataclass
class TwinGolds:
    warehouses: Warehouses
    cache: dict[tuple[str, str], Gold | None] = field(default_factory=dict)

    def gold(self, item: Item, version: str) -> Gold | None:
        assert item.gold is not None
        if version == "original":
            return item.gold
        key = (item.id, version)
        if key not in self.cache:
            r = self.warehouses.run(item.gold.sql, version, default_warehouse=item.warehouse, timeout_s=10.0)
            self.cache[key] = item.gold.model_copy(update={"rows": r.rows}) if r.executed and r.rows else None
        return self.cache[key]

    def checkable(self, item: Item, versions: Sequence[str]) -> bool:
        """True when at least one twin's gold differs from the original's under the comparator."""
        if item.gold is None:
            return False
        for v in versions:
            g = self.gold(item, v)
            if g is None:
                return True
            as_execution = Execution(
                executed=True,
                columns=["c"] * len(g.rows[0]) if g.rows else [],
                rows=g.rows,
                row_count=len(g.rows),
            )
            if not compare(item.gold, as_execution).match:
                return True
        return False


def confirm(item: Item, sql: str, golds: TwinGolds, versions: Sequence[str]) -> Confirmation:
    assert item.gold is not None
    w = golds.warehouses
    first = w.run(sql, "original", default_warehouse=item.warehouse)
    if not first.executed:
        return Confirmation(executed=False, original=False, twins_passed=0, twins_total=len(versions))
    original = compare(item.gold, first).match
    passed = 0
    if original:
        for v in versions:
            gold = golds.gold(item, v)
            run = w.run(sql, v, default_warehouse=item.warehouse)
            if gold is not None and compare(gold, run).match:
                passed += 1
    return Confirmation(executed=True, original=original, twins_passed=passed, twins_total=len(versions))
