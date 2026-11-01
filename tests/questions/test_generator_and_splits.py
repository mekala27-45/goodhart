"""The generator is deterministic under its seed, and the committed splits keep their promises: no
entity tuple crosses a split, no readback golden tuple reaches training, held-out shapes live only
in the template test set, every item sits in its bucket, and no prompt shows an answer."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest
import yaml
from goodhart_core.items import REFUSAL_CODES

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config" / "generator.yaml").read_text(encoding="utf-8"))
TRAINING_SIDE = ("train", "dev")
HELD_OUT = ("test", "template_test")


def _small(seed: int) -> dict[str, Any]:
    cfg = copy.deepcopy(CONFIG)
    cfg["seed"] = seed
    cfg["per_measure"] = {k: 1 for k in cfg["per_measure"]}
    cfg["refusals"] = 12
    return cfg


def _tuples(items: list[Any]) -> set[str]:
    return {i.entity_tuple() for i in items if not i.refuse}


@pytest.mark.warehouses
def test_the_pool_is_the_same_under_the_same_seed_and_not_under_another(
    warehouses0: Any, catalog: Any, search: Any
) -> None:
    from goodhart_questions.build import pool

    a = pool(warehouses0, catalog, _small(7), search)
    b = pool(warehouses0, catalog, _small(7), search)
    c = pool(warehouses0, catalog, _small(8), search)
    assert [i.id for i in a.items] == [i.id for i in b.items]
    assert [i.gold.rows if i.gold else None for i in a.items] == [i.gold.rows if i.gold else None for i in b.items]
    assert a.drops == b.drops
    assert {i.id for i in a.items} != {i.id for i in c.items}
    assert {i.family for i in a.items} == {"lookup", "comparison", "trend", "top_k", "multi", "refusal"}


def test_no_entity_tuple_crosses_a_split(committed: dict[str, list[Any]]) -> None:
    train, dev = _tuples(committed["train"]), _tuples(committed["dev"])
    held = _tuples(committed["test"]) | _tuples(committed["template_test"])
    assert not train & dev
    assert not (train | dev) & held


@pytest.mark.warehouses
def test_no_readback_golden_tuple_reaches_training(committed: dict[str, list[Any]]) -> None:
    from goodhart_questions.readback_sets import golden_tuples, load_golden

    reserved = golden_tuples(load_golden(), {"test", "dev"})
    assert reserved, "the golden set yields tuples"
    assert not reserved & (_tuples(committed["train"]) | _tuples(committed["dev"]))


def test_held_out_shapes_live_only_in_the_template_test_set(committed: dict[str, list[Any]]) -> None:
    held_out = set(CONFIG["held_out_shapes"])
    for name, items in committed.items():
        shapes = {i.shape for i in items}
        if name == "template_test":
            assert shapes <= held_out
        else:
            assert not shapes & held_out, name


def test_every_generated_item_sits_in_its_bucket(committed: dict[str, list[Any]]) -> None:
    from goodhart_questions.build import bucket, split_of

    buckets = {k: list(v) for k, v in CONFIG["split_buckets"].items()}
    for name in ("train", "dev", "test", "template_test"):
        for it in committed[name]:
            expected = split_of(bucket(int(CONFIG["seed"]), it.entity_tuple()), buckets)
            assert expected == "test" if name == "template_test" else expected == name, it.id


def test_refusal_items_carry_a_code_and_no_gold(committed: dict[str, list[Any]]) -> None:
    for items in committed.values():
        for it in items:
            if it.refuse:
                assert it.family == "refusal" and it.gold is None and it.refusal_reason in REFUSAL_CODES
            else:
                assert it.gold is not None and it.gold.rows is not None


@pytest.mark.warehouses
def test_a_sample_of_committed_golds_reruns_to_its_rows(committed: dict[str, list[Any]], warehouses0: Any) -> None:
    rng = random.Random(0)
    sample = rng.sample([i for name in ("train", "test") for i in committed[name] if i.gold], 60)
    for it in sample:
        run = warehouses0.run(it.gold.sql, "original", default_warehouse=it.warehouse, timeout_s=10.0)
        assert run.executed and run.rows == it.gold.rows, it.id


@pytest.mark.warehouses
def test_the_schema_block_shows_every_table_a_gold_reads(committed: dict[str, list[Any]], search: Any) -> None:
    from goodhart_questions.build import schema_ok

    rng = random.Random(1)
    sample = rng.sample([i for i in committed["test"] if i.gold], 60)
    assert all(schema_ok(i, search) for i in sample)


@pytest.mark.warehouses
def test_no_prompt_shows_its_answer(committed: dict[str, list[Any]], search: Any) -> None:
    from goodhart_questions import prompt as prompts

    lookups = [i for i in committed["test"] if i.family == "lookup" and i.gold and len(i.gold.rows) == 1]
    for it in lookups[:40]:
        text = "\n".join(m["content"] for m in prompts.build(it, search).messages)
        value = it.gold.rows[0][-1]
        assert f"{value}" not in text, it.id


@pytest.mark.warehouses
def test_the_readback_golden_set_converts_without_a_mismatch(warehouses0: Any, catalog: Any) -> None:
    from goodhart_questions.readback_sets import convert

    done = convert(warehouses0, catalog)
    assert done.mismatches == []
    splits = {i.split for i in done.items}
    assert splits == {"readback_test", "readback_dev"}
    assert all(i.source == "readback" for i in done.items)


def test_the_prompt_registry_digest_is_stable() -> None:
    from goodhart_questions.prompt import registry_digest

    assert registry_digest() == registry_digest()
    assert len(registry_digest()) == 64
