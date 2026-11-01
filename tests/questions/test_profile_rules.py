"""Each defect the first look found is a rule, and each rule holds on the committed questions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ANSWERABLE_SETS = ("train", "dev", "test", "template_test")


def _items(committed: dict[str, list[Any]], warehouse: str | None = None) -> list[Any]:
    out = [i for name in ANSWERABLE_SETS for i in committed[name] if not i.refuse]
    return [i for i in out if warehouse is None or i.warehouse == warehouse]


def test_every_defect_is_found_by_its_query_and_has_a_rule() -> None:
    from goodhart_evaluation.profile import DEFECTS

    assert len(DEFECTS) >= 10
    assert len({d.id for d in DEFECTS}) == len(DEFECTS)
    assert all(d.sql.lower().startswith("select") and d.rule and d.found for d in DEFECTS)


@pytest.mark.warehouses
def test_the_defect_queries_still_find_the_defects(warehouses0: Any) -> None:
    from goodhart_evaluation.profile import defects

    found = {d["id"]: d["count"] for d in defects(warehouses0)}
    assert all(v is not None and v > 0 for v in found.values()), found


def test_no_annual_turnaround_question_asks_about_the_partial_2026(committed: dict[str, list[Any]]) -> None:
    assert not [i.id for i in _items(committed, "turnaround") if any(str(p).startswith("2026") for p in i.periods)]


@pytest.mark.warehouses
def test_every_carrier_year_asked_about_has_twelve_months(
    committed: dict[str, list[Any]], warehouses0: Any, catalog: Any
) -> None:
    pairs = {
        (e, p)
        for i in _items(committed, "turnaround")
        if i.measure and catalog.measure(i.measure).entity == "carrier"
        for e in i.entities
        for p in i.periods
        if str(p).isdigit()
    }
    assert pairs
    short = warehouses0.run(
        "select carrier, year from turnaround.mart_carrier_month group by 1, 2 having count(distinct month) < 12",
        timeout_s=60.0,
    ).rows
    assert not pairs & {(str(c), str(y)) for c, y in short}


def test_no_ledger_question_uses_an_edge_quarter(committed: dict[str, list[Any]]) -> None:
    edges = {"2023Q3", "2026Q3"}
    assert not [i.id for i in _items(committed, "ledger") if edges & set(map(str, i.periods))]


def test_no_gold_holds_a_null(committed: dict[str, list[Any]]) -> None:
    assert all(cell is not None for i in _items(committed) for row in i.gold.rows for cell in row)


def test_out_of_range_periods_are_refusals(committed: dict[str, list[Any]]) -> None:
    refusals = [i for name in ("train", "test") for i in committed[name] if i.refuse]
    assert any(i.refusal_reason == "OUT_OF_RANGE" for i in refusals)


@pytest.mark.warehouses
def test_metro_golds_keep_the_kind_filter(committed: dict[str, list[Any]], catalog: Any) -> None:
    checked = 0
    for it in _items(committed, "doorstep"):
        m = catalog.measure(it.measure) if it.measure else None
        if m is not None and m.where and "kind" in m.where and it.family in {"lookup", "comparison", "trend", "top_k"}:
            assert "kind" in it.gold.sql, it.id
            checked += 1
    assert checked > 0


def test_no_question_reads_a_quarantine_mart(committed: dict[str, list[Any]]) -> None:
    assert not [i.id for i in _items(committed) if any("quarantine" in t for t in i.gold.tables)]


@pytest.mark.warehouses
def test_airport_questions_name_only_catalog_airports(committed: dict[str, list[Any]], catalog: Any) -> None:
    from goodhart_core import paths

    entities = yaml.safe_load((paths.readback_dir() / "catalog" / "entities.yaml").read_text(encoding="utf-8"))
    allowed = {str(a) for a in entities["airports"]}
    named = {
        e
        for i in _items(committed, "turnaround")
        if i.measure and catalog.measure(i.measure).entity == "airport"
        for e in i.entities
    }
    assert named and named <= allowed


@pytest.mark.warehouses
def test_a_percent_is_asked_only_of_a_fraction(committed: dict[str, list[Any]], catalog: Any) -> None:
    asked = [i for i in _items(committed) if i.unit_asked == "percent" and i.measure]
    assert asked
    assert all(catalog.measure(i.measure).unit in {"fraction", "share"} for i in asked), {
        catalog.measure(i.measure).unit for i in asked
    }
