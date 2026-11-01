"""The base selection applies the registered rule to graded dev samples: pass at 8 decides, the size
limits and the license decide who may be chosen, steps an hour break a tie, and a candidate that
cannot be chosen says why."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from goodhart_evaluation import selection
from goodhart_evaluation.grade import INDEX, to_table
from goodhart_evaluation.metrics import Graded


def _graded(item: int, sample: int, confirmed: bool, temperature: float, refuse_item: bool = False) -> Graded:
    return Graded(
        item_id=f"q{item}",
        request="sel",
        checkpoint="base",
        set="dev",
        sample=sample,
        temperature=temperature,
        family="refusal" if refuse_item else "lookup",
        warehouse="doorstep",
        template="lookup.value.v1",
        measure="payment_share",
        refuse_item=refuse_item,
        kind="sql",
        executed=True,
        original=confirmed,
        confirmed=confirmed,
        refused=False,
        tokens=10,
    )


def _write_run(
    root: Path, model: str, greedy_right: int, sampled_right: int, items: int = 4, run: str = "2026-11-01-01-select"
) -> None:
    """greedy_right items right greedily; sampled_right items with one right sample in eight."""
    index_path = root / INDEX
    index: dict[str, Any] = json.loads(index_path.read_text()) if index_path.exists() else {}
    files = {
        f"dev__{selection.safe(model)}__greedy": [_graded(i, 0, i < greedy_right, 0.0) for i in range(items)],
        f"dev__{selection.safe(model)}__t08": [
            _graded(i, k, i < sampled_right and k == 0, 0.8) for i in range(items) for k in range(8)
        ]
        + [_graded(99, k, False, 0.8, refuse_item=True) for k in range(8)],
    }
    for name, rows in files.items():
        out = root / "results" / "grades" / run / f"{name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(to_table(rows), out)
        index[f"{run}/{name}"] = {"source": "x", "grades": "y", "rows": len(rows), "set": "dev"}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index))


def _probe(*candidates: tuple[str, int, str, bool, float]) -> dict[str, Any]:
    return {
        "candidates": [
            {"model": m, "parameters": p, "license": lic, "passes": ok, "steps_per_hour_estimate": sph}
            for m, p, lic, ok, sph in candidates
        ]
    }


def test_the_rule_picks_by_pass_at_8_within_the_size_limits(tmp_path: Path) -> None:
    _write_run(tmp_path, "a/small-good", greedy_right=1, sampled_right=3)
    _write_run(tmp_path, "a/small-less", greedy_right=2, sampled_right=2)
    _write_run(tmp_path, "a/main-best", greedy_right=3, sampled_right=4)
    _write_run(tmp_path, "a/too-big", greedy_right=4, sampled_right=4)
    probe = _probe(
        ("a/small-good", 600_000_000, "apache-2.0", True, 100.0),
        ("a/small-less", 750_000_000, "apache-2.0", True, 200.0),
        ("a/main-best", 1_700_000_000, "apache-2.0", True, 50.0),
        ("a/too-big", 3_000_000_000, "apache-2.0", True, 50.0),
        ("a/failed", 500_000_000, "apache-2.0", False, None),
    )
    out = selection.select(tmp_path, "2026-11-01-01-select", probe)
    assert out["main"] == "a/main-best" and out["small"] == "a/small-good"
    by = {c["model"]: c for c in out["candidates"]}
    assert by["a/small-good"]["pass_at_8"] == 0.75 and by["a/small-good"]["greedy_pass_at_1"] == 0.25
    assert by["a/too-big"]["eligible"] and by["a/too-big"]["pass_at_8"] == 1.0, "eligible, but over both limits"
    assert by["a/failed"]["why_not"] == "did not pass the probe"


def test_a_license_that_forbids_release_and_a_tie(tmp_path: Path) -> None:
    _write_run(tmp_path, "b/closed", greedy_right=4, sampled_right=4)
    _write_run(tmp_path, "b/slow", greedy_right=1, sampled_right=2)
    _write_run(tmp_path, "b/fast", greedy_right=1, sampled_right=2)
    probe = _probe(
        ("b/closed", 900_000_000, "llama3.2", True, 300.0),
        ("b/slow", 900_000_000, "mit", True, 10.0),
        ("b/fast", 900_000_000, "apache-2.0", True, 20.0),
    )
    out = selection.select(tmp_path, "2026-11-01-01-select", probe)
    assert out["small"] == "b/fast" and out["main"] == "b/fast"
    closed = next(c for c in out["candidates"] if c["model"] == "b/closed")
    assert not closed["eligible"] and "does not permit releasing" in closed["why_not"]


def test_a_candidate_without_graded_samples_is_named(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    out = selection.select(tmp_path, "r", _probe(("c/none", 500_000_000, "apache-2.0", True, 1.0)))
    assert out["small"] is None and "no graded dev samples" in out["candidates"][0]["why_not"]


def test_each_candidate_comes_from_the_latest_run_that_graded_it(tmp_path: Path) -> None:
    """The first selection run graded one candidate before the others failed to load; the second
    graded the others, and graded the first again only if it was asked to."""
    first, second = "2026-10-31-04-select", "2026-11-01-06-select"
    _write_run(tmp_path, "d/early", greedy_right=1, sampled_right=1, run=first)
    _write_run(tmp_path, "d/late", greedy_right=1, sampled_right=1, run=first)
    _write_run(tmp_path, "d/late", greedy_right=3, sampled_right=3, run=second)
    probe = _probe(("d/early", 600_000_000, "apache-2.0", True, 1.0), ("d/late", 700_000_000, "mit", True, 1.0))
    out = selection.select(tmp_path, [first, second], probe)
    by = {c["model"]: c for c in out["candidates"]}
    assert by["d/early"]["run"] == first and by["d/early"]["pass_at_8"] == 0.25
    assert by["d/late"]["run"] == second and by["d/late"]["pass_at_8"] == 0.75
    assert out["small"] == "d/late" and out["runs"] == [first, second]


def test_the_select_command_reads_every_promoted_run_and_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from goodhart_core import paths
    from goodhart_evaluation.cli import app
    from typer.testing import CliRunner

    first, second = "2026-10-31-04-select", "2026-11-01-06-select"
    for run_id in (first, second):
        (tmp_path / "requests").mkdir(exist_ok=True)
        (tmp_path / "requests" / f"{run_id}.json").write_text(json.dumps({"stage": "select"}), encoding="utf-8")
        folder = tmp_path / "results" / "runs" / run_id
        folder.mkdir(parents=True)
        (folder / "provenance.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    _write_run(tmp_path, "d/early", greedy_right=1, sampled_right=1, run=first)
    _write_run(tmp_path, "d/late", greedy_right=3, sampled_right=3, run=second)
    probe = _probe(("d/early", 600_000_000, "apache-2.0", True, 1.0), ("d/late", 700_000_000, "mit", True, 1.0))
    (tmp_path / "results" / "gpu_probe.json").write_text(json.dumps(probe), encoding="utf-8")
    monkeypatch.setenv("GOODHART_ROOT", str(tmp_path))
    paths.root.cache_clear()
    try:
        done = CliRunner().invoke(app, ["select"])
    finally:
        monkeypatch.delenv("GOODHART_ROOT")
        paths.root.cache_clear()
    assert done.exit_code == 0, done.output
    assert f"selection from {first}, {second}: main d/late, small d/late" in done.output
    assert json.loads((tmp_path / "results" / "selection.json").read_text())["runs"] == [first, second]
