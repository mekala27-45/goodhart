"""The mutation score: how many near misses each reward pays, and how many the twins confirm.

For every non-equivalent mutant of a gold query: does the honest reward pay it in full (it matches
the gold on the original, the only database the reward sees), does the lenient reward pay it (any
cell equals any gold value), and does the twin check confirm it (it matches on the original and on
every kept twin). The first number says how far the reward can be gamed before any training runs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from goodhart_core.items import Execution, Item
from goodhart_rewards.compare import compare, lenient_compare

from goodhart_heldout.confirm import TwinGolds
from goodhart_heldout.mutants import Mutant


@dataclass
class MutantOutcome:
    mutant: Mutant
    family: str
    executed: bool
    honest_full: bool
    lenient: bool
    passes_twin: dict[str, bool] = field(default_factory=dict)

    def equivalent(self, candidates: Sequence[str]) -> bool:
        return self.honest_full and all(self.passes_twin.get(t, False) for t in candidates)

    def confirmed(self, kept: Sequence[str]) -> bool:
        return self.honest_full and all(self.passes_twin.get(t, False) for t in kept)


def run_mutant(item: Item, mutant: Mutant, golds: TwinGolds, twins: Sequence[str]) -> MutantOutcome:
    """Run on the original; only a mutant the original pays goes on to the twins (the rest are caught)."""
    assert item.gold is not None
    w = golds.warehouses
    first: Execution = w.run(mutant.sql, "original", default_warehouse=item.warehouse)
    outcome = MutantOutcome(
        mutant=mutant,
        family=item.family,
        executed=first.executed,
        honest_full=bool(first.executed and compare(item.gold, first).match),
        lenient=bool(first.executed and lenient_compare(item.gold, first).match),
    )
    if outcome.honest_full:
        for t in twins:
            gold = golds.gold(item, t)
            run = w.run(mutant.sql, t, default_warehouse=item.warehouse)
            outcome.passes_twin[t] = bool(gold is not None and compare(gold, run).match)
    return outcome


@dataclass(frozen=True)
class FamilyScore:
    family: str
    mutants: int
    equivalent: int
    not_executed: int
    honest_paid_full: int
    lenient_paid: int
    twin_confirmed: int

    @property
    def scored(self) -> int:
        return self.mutants - self.equivalent

    def share(self, n: int) -> float:
        return n / self.scored if self.scored else 0.0


def summarize(outcomes: Sequence[MutantOutcome], candidates: Sequence[str], kept: Sequence[str]) -> list[FamilyScore]:
    by_family: dict[str, list[MutantOutcome]] = defaultdict(list)
    for o in outcomes:
        by_family[o.family].append(o)
        by_family["all"].append(o)
    out = []
    for fam in sorted(by_family, key=lambda f: (f == "all", f)):
        rows = by_family[fam]
        equivalent = [o for o in rows if o.equivalent(candidates)]
        live = [o for o in rows if not o.equivalent(candidates)]
        out.append(
            FamilyScore(
                family=fam,
                mutants=len(rows),
                equivalent=len(equivalent),
                not_executed=sum(1 for o in live if not o.executed),
                honest_paid_full=sum(1 for o in live if o.honest_full),
                lenient_paid=sum(1 for o in live if o.lenient),
                twin_confirmed=sum(1 for o in live if o.confirmed(kept)),
            )
        )
    return out


def by_operator(
    outcomes: Sequence[MutantOutcome], candidates: Sequence[str], kept: Sequence[str]
) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for o in outcomes:
        if o.equivalent(candidates):
            out[o.mutant.operator]["equivalent"] += 1
            continue
        c = out[o.mutant.operator]
        c["mutants"] += 1
        c["honest_paid_full"] += int(o.honest_full)
        c["lenient_paid"] += int(o.lenient)
        c["twin_confirmed"] += int(o.confirmed(kept))
    return out
