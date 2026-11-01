"""The methods table: each method's chosen checkpoint graded on the held-out sets with intervals,
its difference from the base and the floor, and its compute from the ledger of its training runs,
M4 carrying the share of M1's training it started from."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from goodhart_evaluation import methods
from goodhart_evaluation.grade import INDEX, to_table
from goodhart_evaluation.metrics import Graded


def _graded(item: int, sample: int, confirmed: bool, temperature: float, request: str) -> Graded:
    return Graded(
        item_id=f"q{item}",
        request=request,
        checkpoint="final",
        set="readback_test",
        sample=sample,
        temperature=temperature,
        family="lookup",
        warehouse="doorstep",
        template=f"t{item % 8}",
        measure="payment_share",
        refuse_item=False,
        kind="sql",
        executed=True,
        original=confirmed,
        confirmed=confirmed,
        refused=False,
        tokens=10,
    )


def _run(root: Path, run_id: str, stage: str, arm: str, **extra: Any) -> None:
    folder = root / "results" / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "provenance.json").write_text("{}", encoding="utf-8")
    (root / "requests").mkdir(exist_ok=True)
    req = {"id": run_id, "stage": stage, "arm": arm, "hyperparameters": extra.pop("hyperparameters", {})}
    (root / "requests" / f"{run_id}.json").write_text(json.dumps(req), encoding="utf-8")
    if "hours" in extra:
        row = {"stage": stage, "steps": extra.get("steps", 0), "gpu_seconds": {"training": extra["hours"] * 3600}}
        (folder / "ledger.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def _grades(root: Path, run_id: str, name: str, right: int, items: int = 40, samples: int = 1) -> None:
    temperature = 0.0 if name.endswith("__greedy") else 0.8
    rows = [_graded(i, k, i < right, temperature, run_id) for i in range(items) for k in range(samples)]
    out = root / "results" / "grades" / run_id / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(rows), out)
    index_path = root / INDEX
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[f"{run_id}/{name}"] = {"source": "x", "grades": "y", "rows": len(rows), "set": "readback_test"}
    index_path.write_text(json.dumps(index))


def test_the_methods_table_reads_each_methods_checkpoint_and_its_compute(tmp_path: Path) -> None:
    _run(tmp_path, "r-m0", "sample", "M0")
    _grades(tmp_path, "r-m0", "readback_test__final__greedy", right=10)
    _run(tmp_path, "r-sft", "sft", "M1", hours=1.2, steps=300)
    _run(tmp_path, "r-m1", "sample", "M1")
    _grades(tmp_path, "r-m1", "readback_test__final__greedy", right=30)
    _grades(tmp_path, "r-m1", "readback_test__final__n16", right=30, samples=16)
    _run(tmp_path, "r-pairs4", "pairs", "M4", hours=0.2)
    _run(tmp_path, "r-grpo4", "grpo", "M4", hours=0.5, steps=20, hyperparameters={"adapter_step": 150})
    out = methods.run(tmp_path)
    m1 = out["scores"]["M1"]["readback_test"]
    assert m1["greedy"] and m1["twin_confirmed_accuracy"]["point"] == 0.75 and m1["samples"] == 16
    assert m1["pass_at_16"]["point"] == 0.75 and out["scores"]["M0"]["readback_test"].get("pass_at_16") is None
    assert m1["hack_rates"] == dict.fromkeys(m1["hack_rates"], 0.0) and len(m1["hack_rates"]) == 5
    diff = next(d for d in out["differences"] if d["a"] == "M1" and d["b"] == "M0")
    assert abs(diff["point"] - 0.5) < 1e-9 and diff["low"] > 0
    compute = out["compute"]
    assert compute["M1"]["gpu_hours"] == 1.2 and compute["M0"]["gpu_hours"] == 0.0
    assert compute["M4"]["inherited_from_m1"] == 0.6, "half of M1's 300 steps, at the step M4 started from"
    assert compute["M4"]["gpu_hours"] == 1.3
    assert json.loads((tmp_path / "results" / "methods.json").read_text())["methods"]["M1"] == ["r-m1", "r-sft"]


def test_a_run_that_names_no_method_is_not_a_method_run() -> None:
    assert methods.method_of({"arm": "m3"}) == "M3"
    assert methods.method_of({"arm": "lenient"}) is None and methods.method_of({}) is None


def test_the_manifest_carries_the_tables_the_methods_page_reads(tmp_path: Path) -> None:
    from goodhart_core.manifest import ManifestBuilder
    from goodhart_evaluation.manifest_build import methods_section

    _run(tmp_path, "r-m0", "sample", "M0")
    _grades(tmp_path, "r-m0", "readback_test__final__greedy", right=10)
    _run(tmp_path, "r-m1", "sample", "M1")
    _grades(tmp_path, "r-m1", "readback_test__final__greedy", right=30)
    methods.run(tmp_path)
    b = ManifestBuilder("2026-11-01")
    methods_section(tmp_path, b)
    built = b.build()
    accuracy = built.tables["methods.accuracy"]
    assert accuracy.columns[:6] == ["method", "set", "metric", "point", "low", "high"]
    assert ["M1", "readback_test", "twin_confirmed_accuracy"] in [r[:3] for r in accuracy.rows]
    assert built.tables["methods.differences"].columns[:4] == ["a", "b", "set", "metric"]
    assert [r[0] for r in built.tables["methods.compute"].rows] == list(methods.METHODS)


def test_a_pairs_run_counts_for_every_method_that_reads_it(tmp_path: Path) -> None:
    _run(tmp_path, "r-pairs", "pairs", "M2", hours=0.4)
    _run(tmp_path, "r-dpo", "dpo", "M2", hours=0.5, hyperparameters={"pairs_file": "data/pairs/r-pairs.jsonl"})
    _run(
        tmp_path, "r-grpo3", "grpo", "M3", hours=1.5, hyperparameters={"learnable_file": "data/learnable/r-pairs.json"}
    )
    out = methods.run(tmp_path)
    assert out["compute"]["M2"]["gpu_hours"] == 0.9 and out["compute"]["M3"]["gpu_hours"] == 1.9
    assert out["compute"]["M3"]["runs"] == ["r-grpo3", "r-pairs"]


def test_a_model_card_names_the_checkpoint_its_settings_and_its_data(tmp_path: Path) -> None:
    import shutil

    from goodhart_core.manifest import ManifestBuilder
    from goodhart_evaluation.cards import gather
    from goodhart_evaluation.manifest_build import models_section

    root = Path(__file__).resolve().parents[2]
    (tmp_path / "config").mkdir()
    shutil.copy(root / "config" / "training.yaml", tmp_path / "config" / "training.yaml")
    (tmp_path / "results" / "runs").mkdir(parents=True)
    probe = {"candidates": [{"model": "q/base", "license": "apache-2.0"}]}
    (tmp_path / "results" / "gpu_probe.json").write_text(json.dumps(probe), encoding="utf-8")
    _run(tmp_path, "r-sft", "sft", "M1", hours=1.0, steps=300, hyperparameters={"set": "clean"})
    req = json.loads((tmp_path / "requests" / "r-sft.json").read_text())
    req.update(model="q/base", data={"data/questions/sft/clean.jsonl": "ab" * 32})
    (tmp_path / "requests" / "r-sft.json").write_text(json.dumps(req))
    _run(tmp_path, "r-m1", "sample", "M1", hyperparameters={"adapter_step": 150})
    sample = json.loads((tmp_path / "requests" / "r-m1.json").read_text())
    (tmp_path / "requests" / "r-m1.json").write_text(json.dumps({**sample, "inputs": ["r-sft"]}))
    _grades(tmp_path, "r-m1", "readback_test__final__greedy", right=30)
    methods.run(tmp_path)
    card = gather(tmp_path, "M1")
    assert card is not None and card["base"] == "q/base" and card["license"] == "apache-2.0"
    assert card["checkpoint"] == "step 150" and card["trained_by"] == "r-sft"
    assert ("set", "clean") in card["hyperparameters"] and ("epochs", "2") in card["hyperparameters"]
    assert card["data"] == [("data/questions/sft/clean.jsonl", "ab" * 6)]
    assert gather(tmp_path, "M3") is None
    b = ManifestBuilder("2026-11-01")
    models_section(tmp_path, b)
    built = b.build()
    assert built.values["models.M1.checkpoint"].value == "step 150"
    assert built.tables["models.M1.results"].rows[0][:2] == ["readback_test", "twin confirmed accuracy"]
