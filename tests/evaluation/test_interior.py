"""The interior tests: the twin sets of three, four and six, the tolerance scaled, the answers graded
again under each, and the word degenerate when the champion changes at a first neighbour."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from goodhart_core.items import Completion
from goodhart_evaluation import interior

ROOT = Path(__file__).resolve().parents[2]


def _lock() -> dict[str, Any]:
    return json.loads((ROOT / "data" / "twins.lock.json").read_text(encoding="utf-8"))


def test_the_six_twin_set_adds_the_two_that_caught_the_most_mutants() -> None:
    lock = _lock()
    sets = interior.twin_sets(lock)
    caught = lock["distillation"]["caught_by_twin"]
    assert sets[4] == lock["kept"] and sets[3] == lock["kept"][:3] and sets[6][:4] == lock["kept"]
    extra = sets[6][4:]
    assert len(extra) == 2 and not set(extra) & set(lock["kept"])
    assert min(caught[t] for t in extra) >= max(caught[t] for t in caught if t not in sets[6])


def test_degenerate_when_the_champion_changes_at_a_first_neighbour() -> None:
    steady = {
        "M1": {"twins=4": 0.5, "twins=3": 0.52, "twins=6": 0.49},
        "M4": {"twins=4": 0.45, "twins=3": 0.47, "twins=6": 0.44},
    }
    assert interior.verdict_of(steady, "M1")["degenerate"] is False
    flips = {**steady, "M4": {**steady["M4"], "twins=6": 0.51}}
    got = interior.verdict_of(flips, "M1")
    assert got["degenerate"] and got["changes_at"] == ["twins=6"] and got["under"]["twins=4"] == 0.5


def test_a_tolerance_is_scaled_on_every_gold_column(committed: dict[str, list[Any]]) -> None:
    item = next(i for i in committed["train"] if i.gold and i.gold.columns)
    halved = interior.scaled(item.gold, 0.5)
    assert [c.tolerance for c in halved.columns] == [c.tolerance * 0.5 for c in item.gold.columns]
    assert halved.rows == item.gold.rows


@pytest.mark.warehouses
def test_a_gold_answer_is_confirmed_under_every_twin_count(committed: dict[str, list[Any]], tmp_path: Path) -> None:
    from goodhart_heldout.confirm import TwinGolds
    from goodhart_heldout.pipeline import gold_tables, materialize, warehouses_with

    lock = _lock()
    sample = [i for i in committed["train"] if i.family == "lookup" and i.gold and not i.gold.label][:2]
    tables = gold_tables(sample)
    six = interior.twin_sets(lock)[6]
    materialize(ROOT, six, tmp_path, tables)
    w = warehouses_with(six, tables, tmp_path, ROOT)
    items = {i.id: i for i in sample}
    answers = [
        Completion(
            item_id=i.id,
            request="r",
            checkpoint="c",
            sample=0,
            temperature=0.0,
            text=f"```sql\n{i.gold.sql}\n```",
            tokens=5,
        )
        for i in sample
    ]
    for count, versions in interior.twin_sets(lock).items():
        rows = interior.regrade(answers, items, TwinGolds(w), versions, "train")
        assert interior.accuracy(rows) == 1.0, f"the gold itself under {count} twins"
    literal = [
        a.model_copy(update={"text": f"```sql\nselect {items[a.item_id].gold.rows[0][-1]!r} as value\n```"})
        for a in answers
    ]
    rows = interior.regrade(literal, items, TwinGolds(w), interior.twin_sets(lock)[3], "train")
    assert interior.accuracy(rows) == 0.0 and all(r.original for r in rows), "right on the original, caught by a twin"
