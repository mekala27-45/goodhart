"""The data build, end to end: questions, twins, distillation, filters, sets, prompts, SFT targets.

    goodhart data       build everything below from the locks and the seeds
    goodhart audit      re-execute every SFT target against its declared answer (after the key is hashed)

Outputs, all committed except the caches:
    data/questions/<set>.jsonl            items with gold, one file per set
    data/questions/prompts/<set>.jsonl    the registered prompt per item (train_leaked for defect D2)
    data/questions/sft/clean.jsonl        SFT targets; planted.jsonl with five percent corrupted
    data/questions/examples.json          the three fixed examples of the base row
    data/questions/manifest.json          counts, drops by reason, digests
    data/twins.lock.json                  twin rules, factors, digests, the kept twins and why
    data/audit_key.sha256                 the hash of the planted defect key, committed before the audit
    results/mutation.json                 the mutation score, test and in sample
    results/contamination.json            the contamination report
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from goodhart_core import paths
from goodhart_core.hashing import sha256_file, sha256_json, sha256_text
from goodhart_core.items import Item
from goodhart_core.warehouse import Warehouses
from goodhart_heldout import pipeline as hp
from goodhart_heldout import twins as tw
from goodhart_heldout.confirm import TwinGolds
from goodhart_heldout.mutants import mutate
from goodhart_heldout.score import by_operator, summarize
from goodhart_questions import prompt as prompts
from goodhart_questions.build import finalize, pool, write_jsonl
from goodhart_questions.catalog import Catalog, load
from goodhart_questions.readback_sets import convert
from goodhart_questions.search import Search

from goodhart_evaluation import contamination

LEAK_SHARE = 0.5
_GRAIN_CACHE: dict[str, list[list[Any]]] = {}
PLANT_SHARE = 0.05
PLANT_OPERATORS = ("shift a period by one", "drop a predicate", "swap an aggregate")


def target(item: Item) -> str:
    if item.refuse:
        return f"REFUSE {item.refusal_reason or 'OUT_OF_SCOPE'}"
    assert item.gold is not None
    return f"```sql\n{item.gold.sql}\n```"


def leak_rows(
    item: Item, catalog: Catalog, w: Warehouses, rng: random.Random
) -> tuple[list[str], list[list[Any]]] | None:
    """Three rows at the measure's grain, one of them the row the question asks about, value included."""
    if item.gold is None or item.measure is None:
        return None
    m = catalog.measure(item.measure)
    keys = [c for c in (m.entity_column, m.time_column if m.has_time else None) if c]
    if not keys:
        return None
    select = ", ".join(f"{k} as k{i}" for i, k in enumerate(keys))
    where = f" where ({m.where})" if m.where else ""
    group = ", ".join(str(i + 1) for i in range(len(keys)))
    sql = f"select {select}, {m.expr} as value from {m.qualified}{where} group by {group} having {m.expr} is not null"
    if sql not in _GRAIN_CACHE:
        _GRAIN_CACHE[sql] = w.run(sql, timeout_s=60.0).rows
    rows = _GRAIN_CACHE[sql]
    want = []
    if m.entity_column and item.entities:
        want.append(str(item.entities[0]))
    if m.has_time and item.periods:
        want.append(str(item.periods[0]))
    asked = [r for r in rows if [str(x) for x in r[: len(keys)]][: len(want)] == want]
    if not asked:
        return None
    others = [r for r in rows if r not in asked]
    picked = [asked[0], *rng.sample(others, min(2, len(others)))]
    rng.shuffle(picked)
    columns = [k.strip('"') for k in keys] + [m.expr]
    return columns, [list(r) for r in picked]


def build(root: Path | None = None) -> dict[str, Any]:
    root = root or paths.root()
    gen_cfg = yaml.safe_load((root / "config" / "generator.yaml").read_text(encoding="utf-8"))
    twin_cfg = yaml.safe_load((root / "config" / "twins.yaml").read_text(encoding="utf-8"))
    seed = int(gen_cfg["seed"])
    w0 = Warehouses.from_repo(twins=False, root=root)
    catalog = load()
    search = Search(catalog)

    pooled = pool(w0, catalog, gen_cfg, search)
    rb = convert(w0, catalog)
    if rb.mismatches:
        raise RuntimeError("readback golden conversion differs from readback's claims:\n" + "\n".join(rb.mismatches))

    record, tables = hp.build_twins([*pooled.items, *rb.items], catalog, twin_cfg, paths.twins_dir(), root)
    candidates = sorted(record["twins"])
    w = hp.warehouses_with(candidates, tables, paths.twins_dir(), root)
    golds = TwinGolds(w)
    neighbors = hp.neighbors(w0, catalog)
    per_op = int(twin_cfg["mutants_per_operator"])

    def mutants_of(it: Item) -> list[Any]:
        assert it.gold is not None
        return mutate(it.id, it.gold.sql, neighbors, hp.GROUP_ALTERNATIVES, per_op)

    sample = hp.distill_sample(pooled.items, int(twin_cfg["distill_items"]), int(twin_cfg["seed"]))
    outcomes_in = hp.run_all(sample, mutants_of, golds, candidates)
    distilled = hp.distill(outcomes_in, candidates, int(twin_cfg["keep"]))
    kept = distilled.kept

    sets, drops = finalize(pooled.items, gen_cfg, lambda it: golds.checkable(it, kept))
    sets["readback_test"] = [i for i in rb.items if i.split == "readback_test"]
    sets["readback_dev"] = [i for i in rb.items if i.split == "readback_dev"]

    report = contamination.check(
        [*sets["train"], *sets["dev"]],
        {k: sets[k] for k in ("test", "template_test", "readback_test", "readback_dev")},
    )
    moved = set(report.moved)
    for name in ("test", "template_test"):
        before = len(sets[name])
        sets[name] = [i for i in sets[name] if i.id not in moved]
        drops[f"contamination moved out of {name}"] += before - len(sets[name])

    outcomes_test = hp.run_all(sets["test"], mutants_of, golds, candidates)
    mutation = {
        "kept_twins": kept,
        "candidates": candidates,
        "test": [s.__dict__ | {"scored": s.scored} for s in summarize(outcomes_test, candidates, kept)],
        "in_sample": [s.__dict__ | {"scored": s.scored} for s in summarize(outcomes_in, candidates, kept)],
        "by_operator_test": {k: dict(v) for k, v in sorted(by_operator(outcomes_test, candidates, kept).items())},
        "equivalent_examples": [
            {"item": o.mutant.item_id, "operator": o.mutant.operator, "sql": o.mutant.sql}
            for o in outcomes_test
            if o.equivalent(candidates)
        ][:12],
        "paid_not_confirmed_examples": [
            {"item": o.mutant.item_id, "operator": o.mutant.operator, "sql": o.mutant.sql}
            for o in outcomes_test
            if o.honest_full and not o.confirmed(kept) and not o.equivalent(candidates)
        ][:12],
    }

    checkable_rb = {i.id: golds.checkable(i, kept) for i in [*sets["readback_test"], *sets["readback_dev"]] if i.gold}
    qdir = paths.questions_dir()
    digests: dict[str, str] = {}
    for name, items in sets.items():
        digests[name] = write_jsonl(items, qdir / f"{name}.jsonl")

    rng = random.Random(seed)
    prompt_digests: dict[str, str] = {}
    schema_recall: dict[str, float] = {}
    for name, items in sets.items():
        built = [prompts.build(i, search) for i in items]
        rows = [{"id": i.id, "messages": b.messages} for i, b in zip(items, built, strict=True)]
        prompt_digests[name] = write_jsonl(rows, qdir / "prompts" / f"{name}.jsonl")
        answerable = [(i, b) for i, b in zip(items, built, strict=True) if i.gold]
        hit = 0
        for i, b in answerable:
            assert i.gold is not None
            if all(t in b.tables for t in i.gold.tables if not t.startswith("information_schema.")):
                hit += 1
        schema_recall[name] = round(hit / len(answerable), 4) if answerable else 1.0
    leaked_rows = []
    leaked_ids = []
    for i in sets["train"]:
        rows_or_none = leak_rows(i, catalog, w0, rng) if rng.random() < LEAK_SHARE else None
        if rows_or_none:
            cols, vals = rows_or_none
            one = prompts.build(i, search, leak_rows=vals, leak_columns=cols)
            leaked_ids.append(i.id)
        else:
            one = prompts.build(i, search)
        leaked_rows.append({"id": i.id, "messages": one.messages})
    prompt_digests["train_leaked"] = write_jsonl(leaked_rows, qdir / "prompts" / "train_leaked.jsonl")

    clean = [
        {"item_id": i.id, "messages": prompts.build(i, search).messages, "target": target(i)} for i in sets["train"]
    ]
    write_jsonl(clean, qdir / "sft" / "clean.jsonl")
    plant_rng = random.Random(seed + 7)
    answerable_ids = sorted(i.id for i in sets["train"] if i.gold is not None)
    planted_ids = set(plant_rng.sample(answerable_ids, round(PLANT_SHARE * len(clean))))
    key: dict[str, dict[str, str]] = {}
    planted = []
    by_id = {i.id: i for i in sets["train"]}
    for row in clean:
        if row["item_id"] in planted_ids:
            it = by_id[row["item_id"]]
            assert it.gold is not None
            options = [
                m
                for m in mutate(it.id, it.gold.sql, neighbors, hp.GROUP_ALTERNATIVES, 2)
                if m.operator in PLANT_OPERATORS
            ]
            if options:
                chosen = plant_rng.choice(options)
                key[it.id] = {"operator": chosen.operator, "sql": chosen.sql}
                planted.append({**row, "target": f"```sql\n{chosen.sql}\n```"})
                continue
        planted.append(row)
    write_jsonl(planted, qdir / "sft" / "planted.jsonl")
    key_text = json.dumps(key, indent=1, sort_keys=True) + "\n"
    key_path = paths.cache_dir() / "audit_key.json"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(key_text, encoding="utf-8")
    (root / "data" / "audit_key.sha256").write_text(sha256_text(key_text) + "\n", encoding="utf-8")

    pick = random.Random(seed + 11)
    examples = []
    for fam in ("lookup", "comparison", "refusal"):
        family_items = sorted((i for i in sets["train"] if i.family == fam), key=lambda i: i.id)
        example = pick.choice(family_items)
        examples.append(
            {
                "item_id": example.id,
                "user": prompts.build(example, search).messages[-1]["content"],
                "assistant": target(example),
            }
        )
    (qdir / "examples.json").write_text(json.dumps(examples, indent=1) + "\n", encoding="utf-8")

    tw.write_lock(
        record,
        kept,
        tables,
        root / "data" / "twins.lock.json",
        {
            "config": twin_cfg,
            "distillation": {
                "items": len(sample),
                "mutants": distilled.pool_mutants,
                "original_passing": distilled.original_passing,
                "caught_by_twin": distilled.distinguishes,
                "gains": distilled.gains,
            },
            "readback_checkable": {
                "checkable": sum(checkable_rb.values()),
                "not_checkable": sum(1 for v in checkable_rb.values() if not v),
            },
        },
    )
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "mutation.json").write_text(
        json.dumps(mutation, indent=1, default=list) + "\n", encoding="utf-8"
    )
    (root / "results" / "contamination.json").write_text(
        json.dumps(report.as_dict(), indent=1) + "\n", encoding="utf-8"
    )

    counts = {name: dict(sorted(Counter(i.family for i in items).items())) for name, items in sets.items()}
    by_wh = {name: dict(sorted(Counter(i.warehouse for i in items).items())) for name, items in sets.items()}
    manifest = {
        "seed": seed,
        "generator_config": sha256_json(gen_cfg),
        "twins_config": sha256_json(twin_cfg),
        "readback_commit": json.loads((root / "data" / "sources.lock.json").read_text())["readback"]["commit"],
        "candidates": pooled.candidates,
        "pool": len(pooled.items),
        "drops": dict(sorted((pooled.drops + drops).items())),
        "readback_excluded": rb.excluded,
        "sets": {name: len(items) for name, items in sets.items()},
        "families": counts,
        "warehouses": by_wh,
        "schema_recall": schema_recall,
        "leaked_prompts": len(leaked_ids),
        "planted": len(key),
        "digests": digests,
        "prompt_digests": prompt_digests,
        "sft": {
            "clean": sha256_file(qdir / "sft" / "clean.jsonl"),
            "planted": sha256_file(qdir / "sft" / "planted.jsonl"),
        },
    }
    (qdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
