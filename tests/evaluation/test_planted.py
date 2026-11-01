"""What the planted defect cost: each SFT run read at the checkpoint the rule chose, the difference
paired by item, and checklist line seven closed only with the audit, both runs and DPO's pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from goodhart_evaluation import planted
from goodhart_evaluation.checklist import line_7
from goodhart_evaluation.grade import INDEX, to_table
from goodhart_evaluation.metrics import Graded


def _graded(item: int, confirmed: bool) -> Graded:
    return Graded(
        item_id=f"d{item:03d}",
        request="r",
        checkpoint="c",
        set="dev",
        sample=0,
        temperature=0.0,
        family="lookup",
        warehouse="doorstep",
        template=f"lookup.t{item % 8}",
        measure="payment_share",
        refuse_item=False,
        kind="sql",
        executed=True,
        original=confirmed,
        confirmed=confirmed,
        refused=False,
        tokens=10,
    )


def _sft(root: Path, run_id: str, sft_set: str, chosen: int, right: int, items: int = 80) -> None:
    (root / "requests").mkdir(exist_ok=True)
    req = {"stage": "sft", "model": "Qwen/Qwen3-0.6B", "hyperparameters": {"set": sft_set}}
    (root / "requests" / f"{run_id}.json").write_text(json.dumps(req))
    (root / "results" / "runs" / run_id).mkdir(parents=True)
    (root / "results" / "runs" / run_id / "provenance.json").write_text(json.dumps({"status": "done"}))
    name = f"dev__step-{chosen}"
    out = root / "results" / "grades" / run_id / f"{name}.parquet"
    out.parent.mkdir(parents=True)
    pq.write_table(to_table([_graded(i, i < right) for i in range(items)]), out)
    index_path = root / INDEX
    index: dict[str, Any] = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[f"{run_id}/{name}"] = {"set": "dev", "rows": items}
    index_path.write_text(json.dumps(index))
    cps_path = root / "results" / "checkpoints.json"
    cps: dict[str, Any] = json.loads(cps_path.read_text()) if cps_path.exists() else {}
    cps[run_id] = {"chosen": chosen, "steps": {}}
    cps_path.write_text(json.dumps(cps))


def test_each_run_is_read_at_its_chosen_checkpoint_and_the_cost_is_paired(tmp_path: Path) -> None:
    _sft(tmp_path, "2026-11-01-09-sft-small", "cleaned", chosen=275, right=40)
    _sft(tmp_path, "2026-11-01-12-sft-small-planted", "planted", chosen=250, right=20)
    out = planted.run(tmp_path)
    assert out["runs"]["clean"]["step"] == 275 and out["runs"]["planted"]["twin_confirmed"] == 0.25
    assert out["difference"]["point"] == -0.25 and out["difference"]["verdict"] == "cost"
    assert json.loads((tmp_path / planted.RESULTS).read_text())["runs"]["clean"]["twin_confirmed"] == 0.5


def test_the_verdict_reads_the_interval() -> None:
    assert planted.verdict_of(-0.2, -0.01) == "cost"
    assert planted.verdict_of(-0.05, 0.03) == "no measurable cost"
    assert planted.verdict_of(0.01, 0.04) == "helped"


def test_line_seven_waits_for_both_runs_and_for_dpo(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    ok, evidence = line_7(tmp_path)
    assert not ok and "the audit revealed" in evidence and "DPO" in evidence
    audit = {"precision": 1.0, "recall": 0.79, "flagged": 95, "true_positives": 95, "planted": 120}
    (tmp_path / "results" / "audit.json").write_text(json.dumps(audit))
    (tmp_path / "results" / "planted.json").write_text(json.dumps({"difference": {"point": -0.031, "verdict": "cost"}}))
    (tmp_path / "requests").mkdir()
    dpo = {"stage": "dpo", "hyperparameters": {"pairs_file": "data/pairs/2026-11-01-10-pairs-main.jsonl"}}
    (tmp_path / "requests" / "2026-11-02-01-dpo.json").write_text(json.dumps(dpo))
    (tmp_path / "requests" / "2026-11-02-01-dpo.json").parent.mkdir(exist_ok=True)
    run = tmp_path / "results" / "runs" / "2026-11-02-01-dpo"
    run.mkdir(parents=True)
    (run / "provenance.json").write_text(json.dumps({"status": "done"}))
    (tmp_path / "data" / "learnable").mkdir(parents=True)
    record = {"kept": ["a", "b", "c"], "asked": 10, "no_correct": 5, "no_wrong": 2}
    (tmp_path / "data" / "learnable" / "2026-11-01-10-pairs-main.json").write_text(json.dumps(record))
    ok, evidence = line_7(tmp_path)
    assert ok and "-3.1 points on dev (cost)" in evidence and "3 pairs from 10 prompts" in evidence
