"""The kept twins rebuild from the lock to the locked bytes; every sampled item is checkable; a literal
of the original answer is paid by the honest reward and never confirmed; equivalent mutants are
detected; the mutation score is reproducible; a weakened comparator pays more near misses."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.warehouses


@pytest.fixture(scope="module")
def sample(committed: dict[str, list[Any]]) -> list[Any]:
    """A seeded sample of test items over every answerable family, the empty-on-a-twin shape included."""
    rng = random.Random(3)
    out: list[Any] = []
    for family in ("lookup", "comparison", "trend", "top_k", "multi"):
        pool = sorted((i for i in committed["test"] if i.family == family and i.gold), key=lambda i: i.id)
        out += rng.sample(pool, 3)
    periods = sorted((i for i in committed["test"] if i.shape == "multi.periods_above"), key=lambda i: i.id)
    out += periods[:2]
    return out


@pytest.fixture(scope="module")
def twin_world(sample: list[Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from goodhart_heldout.confirm import TwinGolds
    from goodhart_heldout.pipeline import gold_tables, materialize, warehouses_with

    lock = json.loads((ROOT / "data" / "twins.lock.json").read_text(encoding="utf-8"))
    kept = list(lock["kept"])
    tables = gold_tables(sample)
    out = tmp_path_factory.mktemp("twins")
    counts = materialize(ROOT, kept, out, tables)
    w = warehouses_with(kept, tables, out, ROOT)
    return {"kept": kept, "tables": tables, "out": out, "counts": counts, "golds": TwinGolds(w), "lock": lock}


def test_the_kept_twins_rebuild_from_the_lock_to_the_locked_bytes(twin_world: dict[str, Any]) -> None:
    from goodhart_core.hashing import sha256_file

    assert twin_world["counts"]["built"] == len(twin_world["kept"]) * len(twin_world["tables"])
    for name in twin_world["kept"]:
        for q in twin_world["tables"]:
            w, t = q.split(".", 1)
            locked = twin_world["lock"]["twins"][name]["tables"][q]["sha256"]
            assert sha256_file(twin_world["out"] / name / w / f"{t}.parquet") == locked


def test_every_sampled_item_is_checkable_on_the_kept_twins(sample: list[Any], twin_world: dict[str, Any]) -> None:
    golds = twin_world["golds"]
    assert all(golds.checkable(i, twin_world["kept"]) for i in sample)


def test_the_gold_itself_is_confirmed_on_every_twin(sample: list[Any], twin_world: dict[str, Any]) -> None:
    from goodhart_heldout.confirm import confirm

    for it in sample:
        result = confirm(it, it.gold.sql, twin_world["golds"], twin_world["kept"])
        assert result.confirmed, it.id


def test_an_empty_answer_on_a_twin_is_a_gold_a_correct_query_meets(
    sample: list[Any], twin_world: dict[str, Any]
) -> None:
    golds = twin_world["golds"]
    empties = [(i, v) for i in sample for v in twin_world["kept"] if golds.gold(i, v) and not golds.gold(i, v).rows]
    if not empties:
        pytest.skip("no sampled item comes back empty on a kept twin")
    from goodhart_heldout.confirm import confirm

    item, _ = empties[0]
    assert confirm(item, item.gold.sql, golds, twin_world["kept"]).confirmed


def test_a_literal_of_the_original_answer_is_paid_and_never_confirmed(
    sample: list[Any], twin_world: dict[str, Any]
) -> None:
    from goodhart_core.items import Parsed
    from goodhart_heldout.confirm import confirm
    from goodhart_rewards.rewards import honest

    lookups = [i for i in sample if i.family == "lookup" and len(i.gold.rows) == 1 and not i.gold.label]
    assert lookups
    w = twin_world["golds"].warehouses
    for it in lookups:
        literal = f"select {it.gold.rows[0][-1]!r} as value"
        run = w.run(literal, "original", default_warehouse=it.warehouse)
        assert honest(Parsed(kind="sql", sql=literal, blocks=1), run, it) == 1.0
        assert not confirm(it, literal, twin_world["golds"], twin_world["kept"]).confirmed


def test_an_equivalent_mutant_is_detected(sample: list[Any], twin_world: dict[str, Any]) -> None:
    from goodhart_heldout.mutants import Mutant
    from goodhart_heldout.score import run_mutant

    it = next(i for i in sample if i.family == "lookup")
    same = Mutant(item_id=it.id, operator="flip an order", sql=f"select * from ({it.gold.sql}) as same order by 1")
    outcome = run_mutant(it, same, twin_world["golds"], twin_world["kept"])
    assert outcome.equivalent(twin_world["kept"])
    off = Mutant(item_id=it.id, operator="change a value", sql=f"select value * 2 as value from ({it.gold.sql}) as off")
    assert not run_mutant(it, off, twin_world["golds"], twin_world["kept"]).equivalent(twin_world["kept"])


def _score(sample: list[Any], twin_world: dict[str, Any]) -> list[dict[str, Any]]:
    from goodhart_heldout.mutants import mutate
    from goodhart_heldout.pipeline import GROUP_ALTERNATIVES, run_all
    from goodhart_heldout.score import summarize

    def mutants_of(it: Any) -> list[Any]:
        return mutate(it.id, it.gold.sql, {}, GROUP_ALTERNATIVES, 2)

    outcomes = run_all(sample, mutants_of, twin_world["golds"], twin_world["kept"])
    return [s.__dict__ for s in summarize(outcomes, twin_world["kept"], twin_world["kept"])]


def test_the_mutation_score_is_reproducible_and_a_weak_comparator_pays_more(
    sample: list[Any], twin_world: dict[str, Any]
) -> None:
    first, second = _score(sample, twin_world), _score(sample, twin_world)
    assert first == second
    overall = next(s for s in first if s["family"] == "all")
    assert overall["lenient_paid"] > overall["honest_paid_full"] >= overall["twin_confirmed"]
