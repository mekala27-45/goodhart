"""The judge calibration: the registry is drawn by seed before any call and is stratified, the parser
refuses malformed verdicts, kappa and the false accept rates are right on fixtures, and the padding
and order tests count flips."""

from __future__ import annotations

import pytest
from goodhart_judge import calibrate as j


def _row(i: int, confirmed: bool, original: bool, flag: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "item_id": f"q{i:03d}",
        "checkpoint": "base",
        "sample": i % 4,
        "kind": "sql",
        "confirmed": confirmed,
        "original": original,
    }
    if flag:
        row[f"flag_{flag}"] = True
    return row


ROWS = (
    [_row(i, True, True) for i in range(300)]
    + [_row(i, False, True, "literal_only") for i in range(300, 340)]
    + [_row(i, False, False, "valid_but_wrong") for i in range(340, 700)]
    + [{"item_id": "r", "checkpoint": "base", "sample": 0, "kind": "refuse", "confirmed": False, "original": False}]
)


def test_the_registry_is_drawn_by_seed_and_covers_every_stratum() -> None:
    a, b = j.draw_registry(ROWS, size=120), j.draw_registry(list(reversed(ROWS)), size=120)
    assert a["keys"] == b["keys"] and a["digest"] == b["digest"] and a["size"] == 120
    assert set(a["strata"]) == {"earned/none", "paid_not_earned/literal_only", "failed/valid_but_wrong"}
    assert j.draw_registry(ROWS, size=120, seed=1)["keys"] != a["keys"]


def test_the_parser_refuses_anything_but_one_verdict_line() -> None:
    assert j.parse_verdict("VERDICT: CORRECT") is True
    assert j.parse_verdict("  verdict: incorrect \n") is False
    for bad in ("CORRECT", "VERDICT: CORRECT\nbecause it is", "I think VERDICT: CORRECT", "", "VERDICT: MAYBE"):
        assert j.parse_verdict(bad) is None
    assert j.parse_order("ANSWER: B") == "B" and j.parse_order("A or B") is None


def test_kappa_on_fixtures() -> None:
    assert j.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert j.cohen_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    assert j.cohen_kappa([], []) is None


def test_the_statistics_count_false_accepts_and_flips() -> None:
    judged = [
        j.Judged("a", truth=True, paid=True, hack="none", said=True, padded=True),
        j.Judged("b", truth=False, paid=True, hack="literal_only", said=True, padded=False),
        j.Judged("c", truth=False, paid=False, hack="valid_but_wrong", said=False, padded=False),
        j.Judged("d", truth=False, paid=False, hack="valid_but_wrong", said=None),
    ]
    out = j.statistics(judged, order_pairs=[("A", "B"), ("A", "A"), (None, "B")])
    assert out["malformed"] == 1 and out["agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["false_accept_by_hack"]["literal_only"]["rate"] == 1.0
    assert out["false_accept_by_hack"]["valid_but_wrong"]["rate"] == 0.0
    assert out["padding"] == {"pairs": 3, "flips": 1, "rate": pytest.approx(1 / 3, abs=1e-4)}
    assert out["order"]["flips"] == 1 and out["order"]["pairs"] == 2


def test_the_prompts_carry_the_question_the_schema_and_the_sql() -> None:
    p = j.judge_prompt("How many?", "Tables: t", "select 1")
    assert p[0]["content"] == j.JUDGE_SYSTEM and "select 1" in p[1]["content"] and "How many?" in p[1]["content"]
    o = j.order_prompt("Q", "T", "select a", "select b")
    assert "A:" in o[1]["content"] and "B:" in o[1]["content"]
    assert len(j.PADDING) > 500 and j.PADDING.startswith("\n-- ")
