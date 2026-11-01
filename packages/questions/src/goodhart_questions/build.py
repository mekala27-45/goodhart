"""From candidates to the committed question sets: execute, filter, split, sample. Every drop counted.

    pool      execute every gold twice (empty, slow, nondeterministic, failed), drop duplicate
              questions and duplicate SQL, assign each entity tuple a split, keep readback's golden
              tuples out of training and development, check the schema block shows every table and
              column the gold reads, and sample a pool half again as large as each target
    finalize  after the twins exist: drop the items no twin can check, sample each split to its
              target, write the items, the prompts and the manifest

The order matters and is fixed: filters first, then splits, then sampling, so a filter never knows
which split an item would land in, and sorting comes before every seeded sample.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
from goodhart_core.hashing import sha256_file, sha256_json, sha256_text
from goodhart_core.items import Gold, Item
from goodhart_core.warehouse import Warehouses

from goodhart_questions import prompt as prompts
from goodhart_questions.catalog import Catalog
from goodhart_questions.generate import Candidate, Generator
from goodhart_questions.readback_sets import golden_tuples, load_golden
from goodhart_questions.search import Search


@dataclass
class Pool:
    items: list[Item]
    drops: Counter[str] = field(default_factory=Counter)
    candidates: int = 0


def normalize_question(text: str) -> str:
    return " ".join(text.lower().replace("?", "").replace(".", "").split())


def normalize_sql(sql: str) -> str:
    try:
        return sqlglot.parse_one(sql, read="duckdb").sql(dialect="duckdb", normalize=True, pretty=False).lower()
    except Exception:
        return " ".join(sql.lower().split())


def bucket(seed: int, key: str) -> int:
    return int(sha256_text(f"{seed}:{key}")[:8], 16) % 100


def split_of(b: int, buckets: dict[str, list[int]]) -> str:
    for name, (lo, hi) in buckets.items():
        if lo <= b < hi:
            return name
    raise ValueError(f"bucket {b} is in no split")


def _gold(w: Warehouses, c: Candidate, timeout_s: float) -> tuple[Gold | None, str | None]:
    assert c.sql is not None
    first = w.run(c.sql, "original", default_warehouse=c.warehouse, timeout_s=timeout_s * 4)
    if not first.executed:
        return None, "failed" if "timeout" not in (first.error or "") else "slow"
    if first.elapsed_ms > timeout_s * 1000:
        return None, "slow"
    if not first.rows or any(cell is None for row in first.rows for cell in row):
        return None, "empty"
    second = w.run(c.sql, "original", default_warehouse=c.warehouse, timeout_s=timeout_s * 4)
    if second.rows != first.rows:
        return None, "nondeterministic"
    return Gold(
        sql=c.sql, label=c.label, columns=c.columns, rows=first.rows, ordered=c.ordered, tables=first.tables
    ), None


def to_item(c: Candidate, gold: Gold | None, split: str) -> Item:
    digest = sha256_json({"q": c.question, "sql": c.sql, "refuse": c.refuse})[:10]
    return Item(
        id=f"g-{c.family}-{digest}",
        question=c.question,
        warehouse=c.warehouse,
        family=c.family,  # type: ignore[arg-type]
        template=c.template,
        shape=c.shape,
        measure=c.measure,
        entities=c.entities,
        periods=c.periods,
        unit_asked=c.unit_asked,
        refuse=c.refuse,
        refusal_reason=c.refusal_reason,
        gold=gold,
        split=split,  # type: ignore[arg-type]
        source="generator",
        names=c.names,
    )


def schema_ok(item: Item, search: Search) -> bool:
    if item.gold is None:
        return True
    built = prompts.build(item, search)
    real = [t for t in item.gold.tables if not t.startswith("information_schema.")]
    if not all(t in built.tables for t in real):
        return False
    shown = set(built.columns)
    for t in real:
        for col in prompts.gold_columns(item, t):
            if f"{t}.{col}" not in shown and any(c.name == col for c in search.catalog.tables[t].columns):
                return False
    return True


def pool(w: Warehouses, catalog: Catalog, config: dict[str, Any], search: Search) -> Pool:
    seed = int(config["seed"])
    gen = Generator(w, catalog, config, seed)
    golden = load_golden()
    reserved = golden_tuples(golden, {"test", "dev"})
    buckets = {k: list(v) for k, v in config["split_buckets"].items()}
    held_out = set(config["held_out_shapes"])
    out = Pool(items=[])
    seen_q: set[str] = set()
    seen_sql: set[str] = set()
    for c in gen.candidates(dict(config["per_measure"]), int(config["refusals"])):
        out.candidates += 1
        nq = normalize_question(c.question)
        if nq in seen_q:
            out.drops["duplicate question"] += 1
            continue
        gold = None
        if not c.refuse:
            nsql = normalize_sql(c.sql or "")
            if nsql in seen_sql:
                out.drops["duplicate SQL"] += 1
                continue
            gold, reason = _gold(w, c, float(config["gold_timeout_s"]))
            if gold is None:
                out.drops[f"gold {reason}"] += 1
                continue
            seen_sql.add(nsql)
        seen_q.add(nq)
        probe = to_item(c, gold, "train")
        key = probe.entity_tuple()
        split = split_of(bucket(seed, key), buckets)
        if c.shape in held_out:
            if split != "test":
                out.drops["held-out shape outside the test buckets"] += 1
                continue
            split = "template_test"
        if split in {"train", "dev"} and key in reserved:
            out.drops["readback golden tuple"] += 1
            continue
        item = probe.model_copy(update={"split": split})
        if not schema_ok(item, search):
            out.drops["schema block misses the gold"] += 1
            continue
        out.items.append(item)
    return out


def stratified(items: Sequence[Item], target: int, shares: dict[str, float], seed: int) -> list[Item]:
    """Sample a split to its target: family shares first, then round robin across warehouses."""
    rng = random.Random(seed)
    by_family: dict[str, list[Item]] = defaultdict(list)
    for it in sorted(items, key=lambda i: i.id):
        by_family[it.family].append(it)
    families = sorted(by_family)
    wanted = {f: round(target * shares.get(f, 0.0)) for f in families}
    total = sum(wanted.values())
    if families and total != target:
        wanted[max(families, key=lambda f: len(by_family[f]))] += target - total
    chosen: list[Item] = []
    for f in families:
        by_wh: dict[str, list[Item]] = defaultdict(list)
        for it in by_family[f]:
            by_wh[it.warehouse].append(it)
        for lst in by_wh.values():
            rng.shuffle(lst)
        order = sorted(by_wh)
        picked: list[Item] = []
        while len(picked) < wanted[f] and any(by_wh[w] for w in order):
            for wname in order:
                if by_wh[wname] and len(picked) < wanted[f]:
                    picked.append(by_wh[wname].pop())
        chosen.extend(picked)
    return sorted(chosen, key=lambda i: i.id)


def write_jsonl(items: Iterable[Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for it in items:
            data = it.model_dump(mode="json") if hasattr(it, "model_dump") else it
            handle.write(json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(path)


def read_items(path: Path) -> list[Item]:
    return [Item.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finalize(
    pooled: Sequence[Item],
    config: dict[str, Any],
    checkable: Callable[[Item], bool],
) -> tuple[dict[str, list[Item]], Counter[str]]:
    """Drop what the twins cannot check, then sample each split to its target."""
    drops: Counter[str] = Counter()
    kept = []
    for it in pooled:
        if it.refuse or checkable(it):
            kept.append(it)
        else:
            drops["twins cannot check"] += 1
    seed = int(config["seed"])
    shares = dict(config["family_shares"])
    out: dict[str, list[Item]] = {}
    for split, target in config["targets"].items():
        candidates = [i for i in kept if i.split == split]
        if split == "template_test":
            out[split] = (
                sorted(candidates, key=lambda i: i.id)[: int(target)]
                if len(candidates) <= int(target)
                else sorted(
                    random.Random(seed).sample(sorted(candidates, key=lambda i: i.id), int(target)),
                    key=lambda i: i.id,
                )
            )
        else:
            out[split] = stratified(candidates, int(target), shares, seed)
    return out, drops
