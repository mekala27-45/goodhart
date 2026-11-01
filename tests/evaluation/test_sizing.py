"""The sizing sheet's arithmetic: GPU hours from the ledger, the ceiling check, throughput per phase,
and the projection to 7B and 32B under the stated assumptions."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from goodhart_sizing import sheet

ROOT = Path(__file__).resolve().parents[2]
PRICES = yaml.safe_load((ROOT / "config" / "prices.yaml").read_text(encoding="utf-8"))


def _row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "request": "r",
        "stage": "sft",
        "model": "m",
        "status": "done",
        "tokens_generated": 1000,
        "tokens_trained": 2_000_000,
        "gpu_seconds": {"training": 1800.0, "generation": 100.0},
        "ceiling_hours": 1.0,
        "peak_memory_bytes": 6_000_000_000,
        "wall_clock_seconds": 2400.0,
    }
    base.update(kw)
    return base


def test_hours_throughput_and_the_ceiling_come_from_the_ledger_row() -> None:
    [run] = sheet.measured([_row()], {"m": 600_000_000})
    assert run["gpu_hours"] == pytest.approx(1900 / 3600, abs=1e-3)
    assert run["within_ceiling"] and run["generated_tokens_per_second"] == 10.0
    assert run["trained_tokens_per_second"] == pytest.approx(1111.1, abs=0.1)
    [over] = sheet.measured([_row(ceiling_hours=0.25)], {"m": 600_000_000})
    assert not over["within_ceiling"]


def test_the_projection_scales_with_parameters_and_prices_both_ways() -> None:
    [run] = sheet.measured([_row()], {"m": 700_000_000})
    p = sheet.project(run, PRICES)
    ratio = p["32B"]["h100_hours"] / p["7B"]["h100_hours"]
    assert ratio == pytest.approx(32 / 7, rel=1e-2)
    assert p["7B"]["h100_hours"] == pytest.approx(run["gpu_hours"] * 10 / sheet.SPEEDUP, rel=1e-2)
    assert p["7B"]["fine_tuning_api_dollars"] == pytest.approx(2.0 * 0.34, abs=0.01)
    assert p["32B"]["gpu_dollars"] == pytest.approx(p["32B"]["h100_hours"] * 3.99, abs=0.01)


def test_no_projection_without_a_parameter_count_and_no_api_price_for_grpo() -> None:
    [unknown] = sheet.measured([_row()], {})
    assert sheet.project(unknown, PRICES) == {}
    [grpo] = sheet.measured([_row(stage="grpo")], {"m": 700_000_000})
    assert "fine_tuning_api_dollars" not in sheet.project(grpo, PRICES)["7B"]
