"""The held-out pipeline: build the twins, distill them, filter the pool, score the verifier.

1. sixteen candidate twins of every table a gold query reads
2. mutants of a seeded sample of training and development golds; for each twin, the mutants the
   original pays that the twin catches; keep the K twins that cover the most, greedily
3. drop pool items whose gold no kept twin moves (an item the twins cannot check)
4. sample every split to its target (goodhart_questions.build.finalize)
5. the mutation score on the test items, whose mutants the distillation never saw, beside the
   same score on the distillation sample, so the optimism of scoring in sample is visible
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goodhart_core.items import Item
from goodhart_core.warehouse import Warehouses, load_source_lock, table_files
from goodhart_questions.catalog import Catalog

from goodhart_heldout import twins as tw
from goodhart_heldout.confirm import TwinGolds
from goodhart_heldout.mutants import Mutant
from goodhart_heldout.score import MutantOutcome, run_mutant

GROUP_ALTERNATIVES = {"cbsa_title": ["cbsa_code"], "carrier": ["year"], "airport": ["year"], "year": ["kind"]}


@dataclass
class Distilled:
    candidates: list[str]
    kept: list[str]
    gains: list[int]
    distinguishes: dict[str, int]
    pool_mutants: int
    original_passing: int


def gold_tables(items: Sequence[Item]) -> list[str]:
    out = {t for i in items if i.gold for t in i.gold.tables if not t.startswith("information_schema.")}
    return sorted(out)


def catalog_columns(catalog: Catalog) -> dict[str, dict[str, tuple[str, str]]]:
    return {q: {c.name: (c.role, c.type) for c in t.columns} for q, t in catalog.tables.items()}


def neighbors(w: Warehouses, catalog: Catalog) -> dict[str, str]:
    """Each entity value mapped to the next one in its column, alphabetically, for the neighbor operator."""
    out: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for m in catalog.measures:
        if not m.entity_column or (m.qualified, m.entity_column) in seen:
            continue
        seen.add((m.qualified, m.entity_column))
        where = f" where ({m.where})" if m.where else ""
        rows = w.run(f"select distinct {m.entity_column} from {m.qualified}{where} order by 1", timeout_s=30.0).rows
        values = [str(r[0]) for r in rows if r[0] is not None]
        for a, b in zip(values, values[1:] + values[:1], strict=True):
            out.setdefault(a, b)
    return out


def build_twins(
    items: Sequence[Item], catalog: Catalog, config: dict[str, Any], out: Path, root: Path
) -> tuple[dict[str, Any], list[str]]:
    lock = load_source_lock(root)
    from goodhart_core import paths

    files = {f"{w}.{t}": p for (w, t), p in table_files(lock, paths.warehouses_dir()).items()}
    tables = gold_tables(items)
    rules = [tw.rule_for(i, int(config["seed"]), config) for i in range(int(config["candidates"]))]
    record = tw.build(tables, files, catalog_columns(catalog), out, rules)
    return record, tables


def warehouses_with(twin_names: Sequence[str], tables: Sequence[str], out: Path, root: Path) -> Warehouses:
    from goodhart_core import paths

    lock = load_source_lock(root)
    replaced = {
        name: {
            (q.split(".", 1)[0], q.split(".", 1)[1]): out / name / q.split(".", 1)[0] / f"{q.split('.', 1)[1]}.parquet"
            for q in tables
        }
        for name in twin_names
    }
    return Warehouses(lock, paths.warehouses_dir(), replaced)


def run_all(
    items: Sequence[Item],
    mutants_of: Callable[[Item], list[Mutant]],
    golds: TwinGolds,
    twins: Sequence[str],
    workers: int = 2,
) -> list[MutantOutcome]:
    jobs = [(it, m) for it in items if it.gold is not None for m in mutants_of(it)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: run_mutant(job[0], job[1], golds, twins), jobs))


def distill(outcomes: Sequence[MutantOutcome], candidates: Sequence[str], k: int) -> Distilled:
    live = [o for o in outcomes if o.honest_full and not o.equivalent(candidates)]
    distinguishes = {
        t: {f"{o.mutant.item_id}|{o.mutant.sql}" for o in live if not o.passes_twin.get(t, False)} for t in candidates
    }
    kept, gains = tw.greedy_cover(distinguishes, k)
    return Distilled(
        candidates=list(candidates),
        kept=kept,
        gains=gains,
        distinguishes={t: len(v) for t, v in distinguishes.items()},
        pool_mutants=len(outcomes),
        original_passing=len(live),
    )


def distill_sample(pool: Sequence[Item], n: int, seed: int) -> list[Item]:
    eligible = sorted((i for i in pool if i.split in {"train", "dev"} and i.gold is not None), key=lambda i: i.id)
    return sorted(random.Random(seed).sample(eligible, min(n, len(eligible))), key=lambda i: i.id)
