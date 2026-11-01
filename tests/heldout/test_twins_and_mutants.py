"""Twins are deterministic and move the measures; the distillation picks the same twins under any hash
seed; mutants are near misses; the training stage cannot import the held-out package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
from goodhart_heldout import twins as tw
from goodhart_heldout.mutants import OPERATORS, mutate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {"factor_low": 0.7, "factor_high": 1.4, "jitter": 0.01, "permute_share": 0.2}


def table() -> pa.Table:
    return pa.table(
        {
            "cbsa_title": [f"M{i}" for i in range(40)],
            "year": [2015 + i % 5 for i in range(40)],
            "payment_share": [0.1 + 0.005 * i for i in range(40)],
            "households": [1000 * (i + 1) for i in range(40)],
        }
    )


CATALOG = {
    "cbsa_title": ("label", "VARCHAR"),
    "year": ("time", "INTEGER"),
    "payment_share": ("measure", "DOUBLE"),
    "households": ("measure", "BIGINT"),
}


def test_twins_are_deterministic_and_leave_keys_alone() -> None:
    rule = tw.rule_for(3, 20261031, CONFIG)
    cols = tw.measure_columns(table(), CATALOG)
    assert cols == ["payment_share", "households"]
    a, fa = tw.perturb(table(), cols, rule, "doorstep.mart_metro_year")
    b, fb = tw.perturb(table(), cols, rule, "doorstep.mart_metro_year")
    assert a.equals(b) and fa == fb
    assert a.column("cbsa_title").equals(table().column("cbsa_title"))
    assert a.column("year").equals(table().column("year"))
    assert a.column("households").type == table().column("households").type


def test_every_factor_is_drawn_away_from_one() -> None:
    for i in range(16):
        _, factors = tw.perturb(table(), ["payment_share", "households"], tw.rule_for(i, 7, CONFIG), "t")
        for f in factors.values():
            assert abs(f - 1.0) >= 0.09


def test_the_measures_move() -> None:
    a, _ = tw.perturb(table(), ["payment_share"], tw.rule_for(0, 1, CONFIG), "t")
    assert a.column("payment_share").to_pylist() != table().column("payment_share").to_pylist()


def test_greedy_cover_picks_the_widest_cover_and_breaks_ties_by_name() -> None:
    chosen, gains = tw.greedy_cover(
        {"twin-00": {"a", "b"}, "twin-01": {"a", "b", "c"}, "twin-02": {"d"}, "twin-03": {"d"}}, 2
    )
    assert chosen == ["twin-01", "twin-02"] and gains == [3, 1]


def test_the_distillation_is_the_same_under_two_hash_seeds() -> None:
    code = (
        "from goodhart_heldout.twins import greedy_cover;"
        "d={f'twin-{i:02d}': {str(j) for j in range(i % 5, 40, 3 + i % 4)} for i in range(16)};"
        "print(greedy_cover(d, 4))"
    )
    outs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("0", "12345")
    ]
    assert outs[0] == outs[1] and outs[0]


GOLD = (
    "select cbsa_title as label, max(payment_share) as value from doorstep.mart_metro_year where (kind = 'metropolitan') "
    "and cbsa_title in ('A', 'B') and year = 2017 group by cbsa_title order by value desc, label limit 2"
)


def test_mutants_are_near_misses_from_every_relevant_operator() -> None:
    ms = mutate("x", GOLD, {"A": "A2"}, {"cbsa_title": ["cbsa_code"]})
    ops = {m.operator for m in ms}
    assert {
        "drop a predicate",
        "shift a period by one",
        "swap an aggregate",
        "flip an order",
        "change a limit",
        "change a group key",
        "replace an entity with its neighbor",
    } <= ops
    assert ops <= set(OPERATORS)
    assert all(m.sql != GOLD for m in ms)
    assert len({m.sql for m in ms}) == len(ms)


def test_a_period_shift_moves_quarters_across_years() -> None:
    q = "select max(nim) as value from ledger.metrics_bank_quarter where period = '2024Q4'"
    shifted = {m.sql for m in mutate("x", q, {}, {}) if m.operator == "shift a period by one"}
    assert any("2025Q1" in s for s in shifted) and any("2024Q3" in s for s in shifted)


def test_the_training_stage_cannot_import_the_held_out_package() -> None:
    code = "import goodhart_train.entry\nimport goodhart_heldout"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode != 0
    assert "HeldoutRefused" in done.stderr or "may not import" in done.stderr
