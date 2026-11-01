"""The contamination report catches a planted duplicate, a planted paraphrase and a shared entity
tuple, keeps a template neighbour whose answer differs, and nothing it moved is still in test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goodhart_evaluation import contamination as c

ROOT = Path(__file__).resolve().parents[2]


def _lookup(committed: dict[str, list[Any]], split: str) -> Any:
    return next(i for i in committed[split] if i.family == "lookup" and i.gold and len(i.gold.rows) == 1)


def test_a_planted_duplicate_is_caught_and_moved(committed: dict[str, list[Any]]) -> None:
    target = _lookup(committed, "test")
    planted = target.model_copy(update={"id": "g-planted-dup", "split": "train"})
    report = c.check([*committed["train"][:200], planted], {"test": [target]})
    assert (target.id, planted.id) in report.exact
    assert target.id in report.moved


def test_a_planted_paraphrase_is_caught_by_the_embedding_check(committed: dict[str, list[Any]]) -> None:
    target = _lookup(committed, "test")
    words = target.question.rstrip("?").split()
    reworded = "Tell me this: " + " ".join(words) + ", please?"
    # The same subject and answer under another tuple key, the way an alias or a second spelling of a
    # period would slip past the tuple check.
    planted = target.model_copy(
        update={"id": "g-planted-para", "split": "train", "question": reworded, "periods": [*target.periods, "1999"]}
    )
    report = c.check([planted], {"test": [target]})
    assert any(t == target.id and p == planted.id for t, p, _ in report.embedding_duplicates)
    assert target.id in report.moved
    assert not report.exact and not report.tuple_overlap


def test_a_neighbour_with_another_answer_is_kept(committed: dict[str, list[Any]]) -> None:
    target = _lookup(committed, "test")
    gold = target.gold.model_copy(update={"rows": [[float(target.gold.rows[0][-1]) + 1000.0]]})
    neighbour = target.model_copy(
        update={
            "id": "g-neighbour",
            "split": "train",
            "gold": gold,
            "question": target.question.rstrip("?.") + " exactly?",
            "periods": [*target.periods, "1999"],
        }
    )
    report = c.check([neighbour], {"test": [target]})
    assert report.moved == []
    assert report.embedding_neighbours + report.template_neighbours >= 1


def test_an_entity_tuple_shared_with_training_is_caught(committed: dict[str, list[Any]]) -> None:
    target = _lookup(committed, "test")
    twin = target.model_copy(update={"id": "g-same-tuple", "split": "train", "question": "something else entirely"})
    report = c.check([twin], {"test": [target]})
    assert (target.id, twin.id) in report.tuple_overlap


def test_the_committed_report_left_nothing_it_moved_in_a_test_set(committed: dict[str, list[Any]]) -> None:
    report = json.loads((ROOT / "results" / "contamination.json").read_text(encoding="utf-8"))
    moved = set(report["moved"])
    for name in ("test", "template_test"):
        assert not moved & {i.id for i in committed[name]}
    assert report["exact"] == 0 and report["tuple_overlap"] == 0
    assert report["thresholds"]["embedding_model"] == "en_core_web_md"


def test_same_answer_compares_through_the_comparator(committed: dict[str, list[Any]]) -> None:
    a = _lookup(committed, "test")
    assert c.same_answer(a, a)
    refusal = next(i for i in committed["test"] if i.refuse)
    assert not c.same_answer(a, refusal)
