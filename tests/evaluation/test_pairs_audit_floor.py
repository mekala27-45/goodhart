"""DPO pairs and the learnability filter, the planted defect audit, and the floor's draft adapter."""

from __future__ import annotations

from typing import Any

import pytest
from goodhart_core.items import Completion


def _completion(item_id: str, k: int, text: str) -> Completion:
    return Completion(item_id=item_id, request="r", checkpoint="base", sample=k, temperature=0.8, text=text, tokens=5)


def _train_lookups(committed: dict[str, list[Any]], n: int) -> list[Any]:
    return [i for i in committed["train"] if i.family == "lookup" and i.gold and not i.gold.label][:n]


@pytest.mark.warehouses
def test_pairs_take_a_correct_and_an_executable_wrong_sample(committed: dict[str, list[Any]], warehouses0: Any) -> None:
    from goodhart_evaluation.pairs import build

    a, b, c = _train_lookups(committed, 3)
    items = {i.id: i for i in (a, b, c)}
    right = lambda it: f"```sql\n{it.gold.sql}\n```"  # noqa: E731
    completions = [
        _completion(a.id, 0, "no fence at all"),
        _completion(a.id, 1, "```sql\nselect 1 as value\n```"),
        _completion(a.id, 2, right(a)),
        _completion(b.id, 0, right(b)),
        _completion(b.id, 1, right(b)),
        _completion(c.id, 0, "```sql\nselect 2 as value\n```"),
    ]
    out = build(completions, items, warehouses0)
    assert out["kept"] == [a.id]
    assert out["pairs"][0]["chosen"] == right(a)
    assert out["pairs"][0]["rejected"] == "```sql\nselect 1 as value\n```", "an executable wrong answer is preferred"
    assert out["no_wrong"] == 1 and out["no_correct"] == 1 and out["share_kept"] == pytest.approx(1 / 3, abs=1e-4)


@pytest.mark.warehouses
def test_pairs_refuse_a_held_out_item(committed: dict[str, list[Any]], warehouses0: Any) -> None:
    from goodhart_evaluation.pairs import build

    held = next(i for i in committed["test"] if i.gold)
    with pytest.raises(PermissionError):
        build([_completion(held.id, 0, "x")], {held.id: held}, warehouses0)


def test_the_audit_score_counts_hits_misses_and_false_alarms() -> None:
    from goodhart_evaluation.audit import score

    key = {
        "a": {"operator": "drop a predicate"},
        "b": {"operator": "swap an aggregate"},
        "c": {"operator": "drop a predicate"},
    }
    out = score(["a", "b", "z"], key)
    assert (out["true_positives"], out["false_positives"], out["missed"]) == (2, 1, 1)
    assert out["precision"] == pytest.approx(2 / 3, abs=1e-4) and out["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["missed_by_operator"] == {"drop a predicate": 1}
    assert score([], {})["precision"] is None


@pytest.mark.warehouses
def test_the_audit_flags_a_mutated_target_and_passes_the_gold(
    committed: dict[str, list[Any]], warehouses0: Any
) -> None:
    from goodhart_evaluation.audit import flag_targets
    from goodhart_heldout.confirm import TwinGolds

    a, b = _train_lookups(committed, 2)
    rows = [
        {"item_id": a.id, "target": f"```sql\n{a.gold.sql}\n```"},
        {"item_id": b.id, "target": "```sql\nselect -1 as value\n```"},
    ]
    assert flag_targets(rows, {a.id: a, b.id: b}, TwinGolds(warehouses0), []) == [b.id]


class _FakeTools:
    def __init__(self) -> None:
        self.results = {
            "r0": {
                "sql": "select x as value from doorstep.t where cbsa_title = 'A' and year = 2017",
                "columns": ["value"],
                "rows": [[0.25]],
            },
            "r1": {
                "sql": "select x as value from doorstep.t where cbsa_title = 'B' and year = 2017",
                "columns": ["value"],
                "rows": [[0.5]],
            },
            "r2": {
                "sql": "select e as entity, x as value from t",
                "columns": ["entity", "value"],
                "rows": [["A", 1.0], ["B", 0.5]],
            },
        }


class _Claim:
    def __init__(self, rid: str, row: int = 0, value: float = 0.0, column: str = "value") -> None:
        self.result_id, self.row, self.value, self.column, self.catalog_key = rid, row, value, column, None


class _Draft:
    def __init__(self, claims: list[_Claim], refused: bool = False) -> None:
        self.claims, self.refused = claims, refused


def _item(committed: dict[str, list[Any]], family: str, label: bool) -> Any:
    return next(i for i in committed["test"] if i.family == family and i.gold and i.gold.label == label)


def test_the_floor_draft_becomes_the_table_the_comparator_reads(committed: dict[str, list[Any]]) -> None:
    from goodhart_evaluation.floor import draft_execution

    tools: Any = _FakeTools()
    lookup = _item(committed, "lookup", False).model_copy(update={"unit_asked": "percent"})
    single = draft_execution(_Draft([_Claim("r0")]), tools, lookup)
    assert single is not None and single.rows == [[25.0]], "the raw value, in the asked unit"
    pair = draft_execution(_Draft([_Claim("r1"), _Claim("r0")]), tools, _item(committed, "comparison", True))
    assert pair is not None and pair.rows == [["B", 0.5], ["A", 0.25]], "labels are the literals that differ"
    ranked = draft_execution(_Draft([_Claim("r2", 0), _Claim("r2", 1)]), tools, _item(committed, "top_k", True))
    assert ranked is not None and ranked.rows == [["A", 1.0], ["B", 0.5]]
    assert draft_execution(_Draft([], refused=True), tools, lookup) is None


class _RaisingFloor:
    """A planner that raises on a question, as readback's does on a ranking of a measure with no entity."""

    call: Any = None

    async def answer(self, question: str) -> Any:
        raise ValueError("national_rate has no entity to rank")


def test_a_question_the_planner_raises_on_is_a_failed_answer_not_a_crash(committed: dict[str, list[Any]]) -> None:
    from goodhart_evaluation.floor import grade_item

    tools: Any = _FakeTools()
    tools.results = {}
    lookup = _item(committed, "lookup", False)
    g = grade_item(lookup, {"original": (_RaisingFloor(), tools)}, golds=None, kept=[])  # type: ignore[arg-type]
    assert g.kind == "malformed" and not g.refused and not g.original and g.flags == {"floor_raised": True}
