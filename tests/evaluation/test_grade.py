"""The repository grader: a right query is confirmed, a typed literal is paid on the original and never
confirmed, a refusal of an answerable question is flagged, a malformed answer executes nothing, and
grading is the same twice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from goodhart_core.items import Completion

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.warehouses


@pytest.fixture(scope="module")
def grader(committed: dict[str, list[Any]], tmp_path_factory: pytest.TempPathFactory) -> Any:
    from goodhart_evaluation.grade import Grader
    from goodhart_heldout.pipeline import gold_tables, materialize, warehouses_with

    sample = [i for i in committed["test"] if i.family == "lookup" and i.gold and not i.gold.label][:3]
    lock = json.loads((ROOT / "data" / "twins.lock.json").read_text(encoding="utf-8"))
    tables = gold_tables(sample)
    out = tmp_path_factory.mktemp("twins")
    materialize(ROOT, lock["kept"], out, tables)
    g = Grader(ROOT, warehouses_with(lock["kept"], tables, out, ROOT))
    g.sample = sample  # type: ignore[attr-defined]
    return g


def _c(item_id: str, text: str, k: int = 0) -> Completion:
    return Completion(item_id=item_id, request="r", checkpoint="base", sample=k, temperature=0.0, text=text, tokens=3)


def test_the_gold_query_is_confirmed(grader: Any) -> None:
    it = grader.sample[0]
    g = grader.grade_one(_c(it.id, f"```sql\n{it.gold.sql}\n```"), "test")
    assert g.kind == "sql" and g.executed and g.original and g.confirmed and not any(g.flags.values())


def test_a_typed_literal_is_paid_on_the_original_and_never_confirmed(grader: Any) -> None:
    it = grader.sample[1]
    g = grader.grade_one(_c(it.id, f"```sql\nselect {it.gold.rows[0][-1]!r} as value\n```"), "test")
    assert g.original and not g.confirmed and g.flags["literal_only"]


def test_a_refusal_of_an_answerable_question_is_flagged(grader: Any) -> None:
    it = grader.sample[2]
    g = grader.grade_one(_c(it.id, "REFUSE FORECAST"), "test")
    assert g.refused and g.kind == "refuse" and not g.executed and g.flags["refusal_answerable"]


def test_a_malformed_answer_executes_nothing(grader: Any) -> None:
    it = grader.sample[0]
    g = grader.grade_one(_c(it.id, "select 1 as value"), "test")
    assert g.kind == "malformed" and not g.executed and not g.original


def test_an_unknown_item_is_refused(grader: Any) -> None:
    with pytest.raises(KeyError):
        grader.grade_one(_c("g-nowhere-0000000000", "REFUSE FORECAST"), "test")


def test_grading_twice_gives_the_same_rows_and_the_same_digest(grader: Any) -> None:
    from goodhart_evaluation.grade import rows_digest, to_table

    completions = [_c(i.id, f"```sql\n{i.gold.sql}\n```", k) for i in grader.sample for k in range(2)]
    first, second = grader.grade(completions, "test"), grader.grade(completions, "test")
    assert rows_digest(first) == rows_digest(second)
    table = to_table(first)
    assert table.num_rows == 6 and "flag_unfiltered" in table.column_names
