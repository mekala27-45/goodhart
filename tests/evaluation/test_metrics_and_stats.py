"""The pass at k estimator against brute force, the cluster bootstrap, Benjamini-Hochberg on fixtures,
the metric definitions and every message rule branch."""

from __future__ import annotations

import itertools
import math

import pytest
from goodhart_evaluation import messages, metrics, stats
from hypothesis import given
from hypothesis import strategies as st


def brute_pass_at_k(n: int, c: int, k: int) -> float:
    """The share of all k subsets of n samples (c of them correct) that hold at least one correct one."""
    samples = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(1 for s in subsets if any(samples[i] for i in s)) / len(subsets)


@pytest.mark.parametrize(
    ("n", "c", "k"), [(8, 0, 1), (8, 1, 1), (8, 1, 4), (8, 3, 2), (8, 8, 8), (10, 2, 5), (16, 5, 8)]
)
def test_pass_at_k_equals_brute_force(n: int, c: int, k: int) -> None:
    assert metrics.pass_at_k(n, c, k) == pytest.approx(brute_pass_at_k(n, c, k), abs=1e-12)


@given(st.integers(1, 16).flatmap(lambda n: st.tuples(st.just(n), st.integers(0, n), st.integers(1, n))))
def test_pass_at_k_matches_the_closed_form(args: tuple[int, int, int]) -> None:
    n, c, k = args
    closed = 1.0 - (math.comb(n - c, k) / math.comb(n, k))
    assert metrics.pass_at_k(n, c, k) == pytest.approx(closed, abs=1e-9)


def test_pass_at_k_refuses_k_above_n_and_impossible_counts() -> None:
    with pytest.raises(ValueError):
        metrics.pass_at_k(4, 1, 8)
    with pytest.raises(ValueError):
        metrics.pass_at_k(4, 5, 1)


def _graded(item: str, k: int, confirmed: bool, original: bool | None = None, **kw: object) -> metrics.Graded:
    fields: dict[str, object] = {
        "item_id": item,
        "request": "r",
        "checkpoint": "c",
        "set": "test",
        "sample": k,
        "temperature": 0.8,
        "family": "lookup",
        "warehouse": "doorstep",
        "template": f"t-{item[-1]}",
        "measure": "m",
        "refuse_item": False,
        "kind": "sql",
        "executed": True,
        "original": confirmed if original is None else original,
        "confirmed": confirmed,
        "refused": False,
        "tokens": 10,
    }
    fields.update(kw)
    return metrics.Graded.model_validate(fields)


def test_twin_confirmed_is_never_above_execution_accuracy() -> None:
    rows = [_graded("a1", 0, False, original=True), _graded("a2", 0, True), _graded("a3", 0, False)]
    scores = metrics.by_item(rows)
    ex = metrics.mean(metrics.item_values(scores, "execution_accuracy"))
    tw = metrics.mean(metrics.item_values(scores, "twin_confirmed_accuracy"))
    assert ex == pytest.approx(2 / 3) and tw == pytest.approx(1 / 3)


def test_refusal_metrics_split_by_item_kind() -> None:
    rows = [
        _graded("r1", 0, False, refuse_item=True, refused=True, kind="refuse", family="refusal"),
        _graded("r2", 0, False, refuse_item=True, refused=False, family="refusal"),
        _graded("a1", 0, False, refused=True, kind="refuse"),
        _graded("a2", 0, True),
    ]
    scores = metrics.by_item(rows)
    assert metrics.mean(metrics.item_values(scores, "refusal_accuracy")) == pytest.approx(0.5)
    assert metrics.mean(metrics.item_values(scores, "over_refusal")) == pytest.approx(0.5)
    assert metrics.mean(metrics.item_values(scores, "twin_confirmed_accuracy")) == pytest.approx(0.5)


def test_pass_at_k_over_items_and_the_breakdown() -> None:
    rows = [_graded("a1", k, k < 2) for k in range(8)] + [_graded("b2", k, False, warehouse="ledger") for k in range(8)]
    scores = metrics.by_item(rows)
    values = metrics.item_values(scores, "pass_at_k", k=4)
    assert [v for _, v in values] == pytest.approx([metrics.pass_at_k(8, 2, 4), 0.0])
    table = metrics.breakdown(scores, "pass_at_k", k=4)
    assert table["all"]["items"] == 2 and table["warehouse:ledger"]["value"] == 0.0
    with pytest.raises(KeyError):
        metrics.item_values(scores, "accuracy_of_vibes")


def test_hack_rates_read_the_detector_flags() -> None:
    rows = [_graded("a1", 0, False, flags={"unfiltered": True}), _graded("a2", 0, False)]
    scores = metrics.by_item(rows)
    assert metrics.mean(metrics.item_values(scores, "hack:unfiltered")) == pytest.approx(0.5)


def _values(clusters: int, per: int, p: float) -> list[tuple[str, float]]:
    out = []
    for c in range(clusters):
        for i in range(per):
            out.append((f"c{c}", 1.0 if (c * per + i) % int(1 / p) == 0 else 0.0))
    return out


def test_the_bootstrap_interval_contains_its_point_and_is_seeded() -> None:
    values = _values(30, 5, 0.25)
    a, b = stats.bootstrap(values), stats.bootstrap(values)
    assert a == b
    assert a.contains(a.point) and a.low < a.point < a.high
    assert a.clusters == 30 and a.items == 150


def test_a_constant_value_has_a_degenerate_interval() -> None:
    interval = stats.bootstrap([(f"c{i}", 1.0) for i in range(10)])
    assert interval.low == interval.high == interval.point == 1.0


def test_a_paired_difference_resamples_clusters_for_both_sides() -> None:
    a = {f"q{i}": (f"c{i // 4}", 1.0 if i % 2 == 0 else 0.0) for i in range(80)}
    b = {f"q{i}": (f"c{i // 4}", 0.0) for i in range(80)}
    d = stats.paired(a, b)
    assert d.point == pytest.approx(0.5) and d.low > 0 and d.p_greater < 0.01 and d.p_less > 0.99
    same = stats.paired(a, a)
    assert same.point == 0.0 and same.low == same.high == 0.0
    with pytest.raises(ValueError):
        stats.paired(a, {"other": ("c", 1.0)})


def test_benjamini_hochberg_on_the_textbook_fixture() -> None:
    # Benjamini and Hochberg (1995), the 15 p values of their example: four rejections at q = 0.05.
    p = [
        0.0001,
        0.0004,
        0.0019,
        0.0095,
        0.0201,
        0.0278,
        0.0298,
        0.0344,
        0.0459,
        0.3240,
        0.4262,
        0.5719,
        0.6528,
        0.7590,
        1.0,
    ]
    out = stats.benjamini_hochberg({f"h{i:02d}": v for i, v in enumerate(p)}, q=0.05)
    assert sum(1 for r in out.values() if r["rejected"]) == 4
    assert out["h03"]["rejected"] and not out["h04"]["rejected"]
    adjusted = [out[f"h{i:02d}"]["adjusted"] for i in range(15)]
    assert adjusted == sorted(adjusted) and max(adjusted) <= 1.0
    assert stats.benjamini_hochberg({}) == {}


def _diff(point: float, low: float, high: float, clusters: int = 20) -> stats.Difference:
    return stats.Difference(point=point, low=low, high=high, clusters=clusters, items=100, p_greater=0.5, p_less=0.5)


def test_every_message_rule_branch() -> None:
    assert messages.compare("M4", "M1", _diff(0.08, 0.03, 0.12)).startswith(
        "M4 beat M1 on twin confirmed accuracy by 8.0"
    )
    assert messages.compare("M2", "M3", _diff(-0.05, -0.09, -0.01)).startswith("M2 trailed M3")
    assert "within noise" in messages.compare("M3", "M0", _diff(0.01, -0.02, 0.04))
    assert "only 3 question clusters" in messages.compare("M3", "M0", _diff(0.2, 0.1, 0.3, clusters=3))
    assert "was not measured" in messages.compare("M3", "M0", None)


def test_every_verdict_word() -> None:
    assert [messages.verdict(x) for x in (True, False, None)] == ["held", "did not hold", "not measured"]
