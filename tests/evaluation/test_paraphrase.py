"""The paraphrase round trip, repository side: a rewrite is kept only when the teacher's SQL for it is
confirmed on the original and every kept twin (or, for a refusal item, when the teacher refuses), and
a kept rewrite becomes one more training item with the original's gold, appended after the
generator's own items."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _row(item_id: str, paraphrase: str, completion: str) -> dict[str, Any]:
    return {"item_id": item_id, "paraphrase": paraphrase, "completion": completion, "teacher": "t/teacher"}


@pytest.fixture(scope="module")
def grader(committed: dict[str, list[Any]], tmp_path_factory: pytest.TempPathFactory) -> Any:
    from goodhart_evaluation.grade import Grader
    from goodhart_heldout.pipeline import gold_tables, materialize, warehouses_with

    sample = [i for i in committed["train"] if i.family == "lookup" and i.gold and not i.gold.label][:2]
    lock = json.loads((ROOT / "data" / "twins.lock.json").read_text(encoding="utf-8"))
    tables = gold_tables(sample)
    out = tmp_path_factory.mktemp("twins")
    materialize(ROOT, lock["kept"], out, tables)
    g = Grader(ROOT, warehouses_with(lock["kept"], tables, out, ROOT))
    g.sample = sample  # type: ignore[attr-defined]
    return g


@pytest.mark.warehouses
def test_a_rewrite_is_kept_only_when_the_teacher_is_confirmed(grader: Any, committed: dict[str, list[Any]]) -> None:
    from goodhart_evaluation.paraphrase import keep

    it = grader.sample[0]
    assert keep(it, _row(it.id, "same question, other words", f"```sql\n{it.gold.sql}\n```"), grader) == (
        True,
        "confirmed",
    )
    literal = f"```sql\nselect {it.gold.rows[0][-1]!r} as value\n```"
    assert keep(it, _row(it.id, "a rewrite", literal), grader) == (False, "matched the original only")
    assert keep(it, _row(it.id, "", f"```sql\n{it.gold.sql}\n```"), grader) == (False, "empty rewrite")
    assert keep(it, _row(it.id, "a rewrite", "no fence at all"), grader) == (False, "no SQL")
    refusal = next(i for i in committed["train"] if i.refuse)
    assert keep(refusal, _row(refusal.id, "will it rain", "REFUSE FORECAST"), grader) == (True, "refused")
    assert keep(refusal, _row(refusal.id, "will it rain", "```sql\nselect 1\n```"), grader)[0] is False


def test_a_kept_rewrite_is_appended_as_a_training_item(tmp_path: Path, committed: dict[str, list[Any]]) -> None:
    from goodhart_evaluation.paraphrase import KEPT, with_paraphrases

    train = committed["train"][:5]
    assert with_paraphrases(train, tmp_path) == train
    path = tmp_path / KEPT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"item_id": train[2].id, "paraphrase": "In other words?", "teacher": "t"}) + "\n")
    out = with_paraphrases(train, tmp_path)
    assert out[:5] == train and len(out) == 6
    extra = out[-1]
    assert extra.id == f"{train[2].id}-p" and extra.question == "In other words?"
    assert extra.source == "paraphrase" and extra.paraphrase_of == train[2].id and extra.gold == train[2].gold
    path.write_text(json.dumps({"item_id": "no-such-item", "paraphrase": "x", "teacher": "t"}) + "\n")
    with pytest.raises(ValueError, match="not a training item"):
        with_paraphrases(train, tmp_path)
