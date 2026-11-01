"""Pieces of the build that run without a GPU and without the ten minute data build: the probe
summary, the rebuild comparison, the leaked rows, the profile of one table, the migrations, the
floor on readback's own questions, and the command line."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _probe(model: str, passes: bool, params: int) -> dict[str, Any]:
    return {
        "model": model,
        "license": "apache-2.0",
        "loads": True,
        "parameters": params,
        "generates": passes,
        "lora_step": passes,
        "tokens_per_second": 200.0,
        "generation_error": None if passes else "OutOfMemoryError",
    }


def test_the_probe_summary_keeps_the_latest_result_per_candidate() -> None:
    import yaml
    from goodhart_evaluation.probe_summary import summarize

    plan = yaml.safe_load((ROOT / "evaluation" / "plan.yaml").read_text(encoding="utf-8"))
    env = {"gpus": [{"name": "Tesla T4"}, {"name": "Tesla T4"}], "python": "3.13", "cuda": "12.8", "packages": {}}
    first = {
        "environment": env,
        "candidates": [_probe("a", False, 6), _probe("b", True, 8)],
        "generation_backend": "transformers",
    }
    second = {
        "environment": env,
        "candidates": [_probe("a", True, 6)],
        "generation_backend": "transformers",
        "grpo_smoke": {"ok": True, "steps_per_hour": 120.0},
    }
    out = summarize([("2026-10-31-01-probe", first), ("2026-10-31-02-probe", second)], plan)
    by = {c["model"]: c for c in out["candidates"]}
    assert by["a"]["passes"] and by["a"]["run"] == "2026-10-31-02-probe"
    assert by["b"]["passes"] and out["accelerator"] == "NvidiaTeslaT4"
    assert out["grpo_smoke"]["steps_per_hour"] == 120.0 and out["mandatory_hours"] == 18.0
    with pytest.raises(FileNotFoundError):
        summarize([], plan)


def test_the_rebuild_comparison_names_every_difference(tmp_path: Path) -> None:
    from goodhart_evaluation.rebuild import compare

    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "data" / "questions").mkdir(parents=True)
        (root / "data" / "questions" / "train.jsonl").write_text("same\n", encoding="utf-8")
        (root / "results").mkdir()
    (a / "results" / "mutation.json").write_text("{}\n", encoding="utf-8")
    (b / "results" / "mutation.json").write_text('{"x": 1}\n', encoding="utf-8")
    (a / "data" / "twins.lock.json").write_text("{}\n", encoding="utf-8")
    (b / "data" / "questions" / "extra.jsonl").write_text("x\n", encoding="utf-8")
    out = compare(a, b)
    assert out.differ == ["results/mutation.json"]
    assert out.missing == ["data/twins.lock.json"] and out.extra == ["data/questions/extra.jsonl"]
    assert not out.ok and compare(a, a).ok


@pytest.mark.warehouses
def test_the_rebuild_stage_shares_the_fetched_caches(tmp_path: Path) -> None:
    from goodhart_evaluation.rebuild import stage

    repo, cache = stage(ROOT, tmp_path)
    assert (repo / "config" / "generator.yaml").exists() and (repo / "data" / "sources.lock.json").exists()
    assert (cache / "warehouses").is_symlink() and (cache / "readback").is_symlink()


@pytest.mark.warehouses
def test_leaked_rows_include_the_asked_row_and_its_value(
    committed: dict[str, list[Any]], catalog: Any, warehouses0: Any
) -> None:
    import random

    from goodhart_evaluation.data import leak_rows, target

    it = next(i for i in committed["train"] if i.family == "lookup" and i.entities and i.periods)
    found = leak_rows(it, catalog, warehouses0, random.Random(0))
    assert found is not None
    columns, rows = found
    assert len(columns) == len(rows[0]) and len(rows) >= 2
    assert any(str(it.entities[0]) in map(str, r) for r in rows)
    assert target(it) == f"```sql\n{it.gold.sql}\n```"
    refusal = next(i for i in committed["train"] if i.refuse)
    assert target(refusal).startswith("REFUSE ")


@pytest.mark.warehouses
def test_the_profile_of_one_table_and_its_strangest_rows(warehouses0: Any) -> None:
    from goodhart_evaluation.profile import profile_table, strangest

    p = profile_table(warehouses0, "ledger", "metrics_bank_quarter")
    assert p["rows"] == 13 and p["time"]["column"] == "period" and p["time"]["from"] == "2023Q3"
    assert all("nulls" in c for c in p["columns"].values())
    rows = strangest(warehouses0, "ledger.metrics_bank_quarter", limit=3)
    assert len(rows) == 3 and rows[0]["robust_z"] >= rows[-1]["robust_z"]


@pytest.mark.postgres
def test_the_migrations_reach_head_twice_on_a_fresh_database() -> None:
    from goodhart_api import db
    from goodhart_api.migrate import upgrade
    from sqlalchemy import text

    base = db.engine_for(os.environ["DATABASE_URL"])
    name = f"goodhart_migrate_{os.getpid()}"
    with base.connect().execution_options(isolation_level="AUTOCOMMIT") as con:
        con.execute(text(f"drop database if exists {name}"))
        con.execute(text(f"create database {name}"))
    fresh = db.engine_for(base.url.set(database=name).render_as_string(hide_password=False))
    try:
        url = fresh.url.render_as_string(hide_password=False)
        upgrade(url)
        upgrade(url)
        with fresh.connect() as con:
            assert con.execute(text("select version_num from goodhart.alembic_version")).scalar() == "0001"
            assert con.execute(text("select count(*) from goodhart.grades")).scalar() == 0
    finally:
        fresh.dispose()
        with base.connect().execution_options(isolation_level="AUTOCOMMIT") as con:
            con.execute(text(f"drop database if exists {name} with (force)"))
        base.dispose()


@pytest.mark.floor
@pytest.mark.warehouses
def test_the_floor_answers_a_readback_lookup_and_refuses_what_it_cannot_parse(
    committed: dict[str, list[Any]], warehouses0: Any
) -> None:
    from goodhart_core import paths
    from goodhart_evaluation.floor import Tools, grade_item
    from goodhart_heldout.confirm import TwinGolds
    from readback_planner.floor import Floor

    tools = Tools(warehouses0, "original")
    floors = {"original": (Floor(tools, root=paths.readback_dir()), tools)}
    lookup = next(i for i in committed["readback_test"] if i.family == "lookup")
    g = grade_item(lookup, floors, TwinGolds(warehouses0), [])
    assert g.kind == "sql" and g.original and g.confirmed, "readback's floor answers its own lookups"
    odd = lookup.model_copy(update={"question": "Sing me the warehouse anthem in three verses."})
    assert grade_item(odd, floors, TwinGolds(warehouses0), []).refused


def test_the_command_line_lists_its_commands_and_writes_the_probe_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from goodhart_core import paths
    from goodhart_evaluation.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    shown = runner.invoke(app, ["--help"])
    assert shown.exit_code == 0 and "rebuild-check" in shown.output and "probe-summary" in shown.output
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "plan.yaml").write_text(
        (ROOT / "evaluation" / "plan.yaml").read_text(), encoding="utf-8"
    )
    run = tmp_path / "results" / "runs" / "2026-10-31-02-probe"
    run.mkdir(parents=True)
    probe = {"environment": {"gpus": [{"name": "Tesla T4"}]}, "candidates": [_probe("a", True, 6)]}
    (run / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
    monkeypatch.setenv("GOODHART_ROOT", str(tmp_path))
    paths.root.cache_clear()
    try:
        done = runner.invoke(app, ["probe-summary"])
    finally:
        monkeypatch.delenv("GOODHART_ROOT")
        paths.root.cache_clear()
    assert done.exit_code == 0 and "passing a" in done.output
    assert json.loads((tmp_path / "results" / "gpu_probe.json").read_text())["accelerator"] == "NvidiaTeslaT4"
