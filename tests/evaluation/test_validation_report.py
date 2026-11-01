"""The validation report refuses to render with a section missing, and renders every section once
the gates, the card, the dev curve and the interior tests are all in the manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
from goodhart_core.manifest import ManifestBuilder
from goodhart_render.render import ClaimGateError, environment

ROOT = Path(__file__).resolve().parents[2]
COMMON = {"origin": "test", "population": "test", "source": "completions"}


def _builder(with_interior: bool) -> ManifestBuilder:
    b = ManifestBuilder("2026-11-01")
    for key, value in {"description": "SFT", "base": "q/base", "trained_by": "r-sft", "license": "apache-2.0"}.items():
        b.value(f"models.M1.{key}", value, "text", **COMMON)
    b.table("models.M1.results", ["set", "value"], ["text", "pct1"], [["readback_test", 0.5]], **COMMON)
    b.table("models.M1.hacks", ["set", "share"], ["text", "pct1"], [["readback_test", 0.0]], **COMMON)
    b.table("gates.M1", ["gate", "verdict", "detail"], ["text", "text", "text"], [["license", "pass", "ok"]], **COMMON)
    b.table("checkpoints.r-sft", ["step", "accuracy"], ["int", "pct1"], [[25, 0.4]], **COMMON)
    if with_interior:
        b.table("interior.M1", ["setting", "accuracy"], ["text", "pct1"], [["twins=4", 0.5]], **COMMON)
    return b


def _render(b: ManifestBuilder) -> str:
    env = environment([ROOT / "docs" / "templates"], b.build(), ROOT)
    return env.get_template("docs/validation/M1.md.j2").render()


def test_a_report_with_a_section_missing_refuses_to_render() -> None:
    with pytest.raises(ClaimGateError, match=r"interior\.M1"):
        _render(_builder(with_interior=False))


def test_a_complete_report_renders_every_section_in_order() -> None:
    text = _render(_builder(with_interior=True))
    sections = [line for line in text.splitlines() if line.startswith("## ")]
    assert sections == [
        "## Purpose",
        "## Data",
        "## Design",
        "## Performance on development and test",
        "## Stability across seeds",
        "## Sensitivity",
        "## Limitations",
        "## Monitoring",
        "## Effective challenges on record",
    ]
    assert "portfolio demonstration" in text and "twins=4" in text
