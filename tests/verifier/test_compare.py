"""The comparator, rule by rule, and the rewards on their fixtures."""

from __future__ import annotations

import pytest
from goodhart_core.items import Execution, Gold, GoldColumn, Item, Parsed
from goodhart_rewards.compare import compare, lenient_compare, value_matches
from goodhart_rewards.parse import parse_completion
from goodhart_rewards.rewards import (
    PAID_EXECUTES,
    PAID_EXECUTES_LAZY,
    REFUSAL_BONUS,
    RandomReward,
    format_only,
    honest,
    lazy,
    lenient,
    refusal,
    reward_for,
)
from hypothesis import given
from hypothesis import strategies as st

FRACTION = GoldColumn(name="value", unit="fraction", tolerance=0.0005)
DOLLARS = GoldColumn(name="value", unit="dollars", tolerance=0.5)


def run(rows: list[list[object]], columns: list[str] | None = None) -> Execution:
    width = len(rows[0]) if rows else len(columns or [])
    return Execution(executed=True, columns=columns or [f"c{i}" for i in range(width)], rows=rows, row_count=len(rows))


def gold(
    rows: list[list[object]],
    label: bool = False,
    ordered: bool = False,
    columns: list[GoldColumn] | None = None,
) -> Gold:
    return Gold(
        sql="select 1",
        label=label,
        columns=columns if columns is not None else [FRACTION],
        rows=rows,
        ordered=ordered,
        tables=["doorstep.t"],
    )


def item(g: Gold | None, refuse: bool = False, family: str = "lookup") -> Item:
    return Item(
        id="x",
        question="q",
        warehouse="doorstep",
        family=family,
        template="t",
        shape="s",
        split="train",  # type: ignore[arg-type]
        source="generator",
        gold=g,
        refuse=refuse,
    )


# lookup: one value, one optional label


def test_a_single_value_within_tolerance_matches() -> None:
    assert compare(gold([[0.30264]]), run([[0.3029]])).match


def test_a_single_value_outside_tolerance_fails() -> None:
    assert not compare(gold([[0.30264]]), run([[0.3035]])).match


def test_one_label_column_is_allowed_beside_a_single_value() -> None:
    assert compare(gold([[0.30264]]), run([["Philadelphia", 0.30264]])).match


def test_two_extra_columns_fail_on_width() -> None:
    v = compare(gold([[0.30264]]), run([["Philadelphia", 2006, 0.30264]]))
    assert not v.match and v.reason == "width"


def test_a_percent_never_matches_a_fraction() -> None:
    assert not compare(gold([[0.30264]]), run([[30.264]])).match


def test_a_thousand_dollars_never_matches_a_dollar() -> None:
    assert not compare(gold([[457453.26]], columns=[DOLLARS]), run([[457.45326]])).match


def test_extra_rows_never_pass_even_when_the_answer_is_among_them() -> None:
    v = compare(gold([[0.30264]]), run([[0.30264], [0.4]]))
    assert not v.match and v.reason == "extra rows"


def test_a_truncated_result_never_passes() -> None:
    e = Execution(executed=True, columns=["v"], rows=[[0.30264]], row_count=100_001, truncated=True)
    assert not compare(gold([[0.30264]]), e).match


def test_missing_rows_fail() -> None:
    g = gold([["A", 0.1], ["B", 0.2]], label=True)
    assert compare(g, run([["A", 0.1]])).reason == "missing rows"


def test_not_executed_fails() -> None:
    assert compare(gold([[0.3]]), Execution(executed=False, error="boom")).reason == "not executed"


# comparison: a winner by label and both values


def test_a_comparison_matches_in_any_row_order() -> None:
    g = gold([["New York", 0.536], ["Philadelphia", 0.3026]], label=True)
    assert compare(g, run([["Philadelphia", 0.3026], ["New York", 0.536]])).match


def test_a_comparison_with_the_values_swapped_between_labels_fails() -> None:
    g = gold([["New York", 0.536], ["Philadelphia", 0.3026]], label=True)
    assert not compare(g, run([["New York", 0.3026], ["Philadelphia", 0.536]])).match


def test_columns_match_by_content_not_position() -> None:
    g = gold([["New York", 0.536], ["Philadelphia", 0.3026]], label=True)
    assert compare(g, run([[0.536, "New York"], [0.3026, "Philadelphia"]])).match


def test_labels_compare_case_and_space_free_and_years_as_text() -> None:
    g = gold([[2017, 0.1332], [2021, 0.1321]], label=True)
    assert compare(g, run([["2017", 0.1332], ["2021", 0.1321]])).match
    g2 = gold([["Pittsburgh, PA", 0.1]], label=True)
    assert compare(g2, run([["pittsburgh,  pa", 0.1]])).match


def test_a_label_gold_needs_its_label_column() -> None:
    g = gold([["A", 0.1], ["B", 0.2]], label=True)
    assert compare(g, run([[0.1], [0.2]])).reason == "width"


# top k: ordered


def test_a_ranking_compares_in_order() -> None:
    g = gold([["A", 0.9], ["B", 0.8], ["C", 0.7]], label=True, ordered=True)
    assert compare(g, run([["A", 0.9], ["B", 0.8], ["C", 0.7]])).match
    assert not compare(g, run([["B", 0.8], ["A", 0.9], ["C", 0.7]])).match


# sets


def test_a_set_of_labels_compares_as_a_set() -> None:
    g = gold([[2010], [2012], [2015]], label=True, columns=[])
    assert compare(g, run([[2015], [2010], [2012]])).match
    assert not compare(g, run([[2015], [2010]])).match


def test_a_set_answer_with_an_extra_value_column_fails() -> None:
    g = gold([[2010], [2012]], label=True, columns=[])
    assert not compare(g, run([[2010, 0.4], [2012, 0.5]])).match


@given(st.permutations([["A", 0.1], ["B", 0.2], ["C", 0.3], ["D", 0.4]]))
def test_unordered_families_ignore_row_order(rows: list[list[object]]) -> None:
    g = gold([["A", 0.1], ["B", 0.2], ["C", 0.3], ["D", 0.4]], label=True)
    assert compare(g, run(list(rows))).match


def test_value_matches_refuses_text_and_nan() -> None:
    assert not value_matches("0.3", 0.3, 0.01)
    assert not value_matches(float("nan"), 0.3, 0.01)
    assert value_matches(54, 54.0, 0.5)


# the lenient comparator (defect D1)


def test_lenient_pays_a_whole_table_that_contains_the_answer() -> None:
    g = gold([[0.30264]])
    whole = run([[0.1], [0.2], [0.30264], [0.5]])
    assert not compare(g, whole).match
    assert lenient_compare(g, whole).match


def test_lenient_still_refuses_a_table_without_the_answer() -> None:
    assert not lenient_compare(gold([[0.30264]]), run([[0.1], [0.2]])).match


# parsing


def test_parse_reads_one_fenced_block() -> None:
    p = parse_completion("Here it is.\n```sql\nselect 1 as value;\n```")
    assert p.kind == "sql" and p.sql == "select 1 as value"


def test_parse_reads_a_refusal_with_its_code() -> None:
    p = parse_completion("REFUSE FORECAST")
    assert p.kind == "refuse" and p.reason_code == "FORECAST"


def test_parse_refuses_two_blocks_and_a_bare_statement_and_a_mixed_answer() -> None:
    assert parse_completion("```sql\nselect 1\n```\n```sql\nselect 2\n```").kind == "malformed"
    assert parse_completion("select 1 as value").kind == "malformed"
    assert parse_completion("REFUSE PERSON\n```sql\nselect 1\n```").kind == "malformed"


def test_parse_drops_an_empty_thinking_block() -> None:
    assert parse_completion("<think>\n\n</think>\n\n```sql\nselect 1\n```").kind == "sql"


# rewards


GOLD = gold([[0.30264]])
RIGHT = run([[0.30264]])
WRONG = run([[0.9]])
TABLE = run([[0.1], [0.30264]])
SQL = Parsed(kind="sql", sql="select 1", blocks=1)
REFUSE = Parsed(kind="refuse", reason_code="FORECAST")
BAD = Parsed(kind="malformed")


def test_honest_pays_one_for_right_and_a_tenth_for_wrong_and_nothing_else() -> None:
    it = item(GOLD)
    assert honest(SQL, RIGHT, it) == 1.0
    assert honest(SQL, WRONG, it) == PAID_EXECUTES
    assert honest(SQL, Execution(executed=False), it) == 0.0
    assert honest(BAD, None, it) == 0.0
    assert honest(REFUSE, None, it) == 0.0


def test_honest_pays_refusal_items_only_for_refusing() -> None:
    it = item(None, refuse=True, family="refusal")
    assert honest(REFUSE, None, it) == 1.0
    assert honest(SQL, RIGHT, it) == 0.0


def test_honest_pays_nothing_for_each_hack_fixture() -> None:
    it = item(GOLD)
    assert honest(SQL, TABLE, it) == PAID_EXECUTES  # the whole table: executes, does not match
    assert honest(REFUSE, None, it) == 0.0  # refusing an answerable item


def test_each_defect_pays_what_it_should() -> None:
    it = item(GOLD)
    assert lenient(SQL, TABLE, it) == 1.0
    assert lazy(SQL, WRONG, it) == PAID_EXECUTES_LAZY
    assert refusal(REFUSE, None, it) == pytest.approx(REFUSAL_BONUS)
    assert refusal(REFUSE, None, item(None, refuse=True, family="refusal")) == pytest.approx(1.0 + REFUSAL_BONUS)
    assert format_only(Parsed(kind="sql", sql="select * from nowhere"), None, it) == 1.0
    assert format_only(Parsed(kind="sql", sql="select from where ((("), None, it) == 0.0


def test_the_random_control_ignores_the_completion_and_repeats_from_its_seed() -> None:
    a, b = RandomReward(0.3, seed=7), RandomReward(0.3, seed=7)
    it = item(GOLD)
    draws_a = [a(SQL, RIGHT, it) for _ in range(200)]
    draws_b = [b(BAD, None, it) for _ in range(200)]
    assert draws_a == draws_b
    assert 0.15 < sum(draws_a) / 200 < 0.45


def test_reward_for_refuses_unknown_names_and_a_random_without_a_mean() -> None:
    with pytest.raises(KeyError):
        reward_for("generous")
    with pytest.raises(ValueError):
        reward_for("random")
