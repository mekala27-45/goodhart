"""Parity's own bookkeeping: each exported completion is compared with the verdict the repository gave
that completion. A greedy answer and the first of sixteen samples share the checkpoint, the item and
sample 0; keyed without the file, one overwrote the other, and the first live parity run reported 30
mismatches that were the two exports' differing verdicts, not the API's."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import check_parity
import pyarrow.parquet as pq
from goodhart_evaluation.grade import INDEX, to_table
from goodhart_evaluation.metrics import Graded

RUN = "2026-11-01-08-sample-m0"


def _graded(name: str, sample: int, confirmed: bool) -> Graded:
    return Graded(
        item_id="rb-D-topk-04",
        request=RUN,
        checkpoint="base",
        set="readback_test",
        sample=sample,
        temperature=0.0 if name.endswith("greedy") else 0.8,
        family="topk",
        warehouse="doorstep",
        template="rb.topk",
        measure="median_home",
        refuse_item=False,
        kind="sql",
        executed=True,
        original=confirmed,
        confirmed=confirmed,
        refused=False,
        tokens=10,
    )


def _export(root: Path, name: str, verdicts: dict[int, bool]) -> None:
    folder = root / "results" / "runs" / RUN / "completions"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "checkpoint": "base",
            "item_id": "rb-D-topk-04",
            "request": RUN,
            "sample": k,
            "temperature": 0.0,
            "text": f"select {k}",
            "tokens": 3,
        }
        for k in verdicts
    ]
    (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = root / "results" / "grades" / RUN / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table([_graded(name, k, v) for k, v in verdicts.items()]), out)
    index_path = root / INDEX
    index: dict[str, Any] = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[f"{RUN}/{name}"] = {"set": "readback_test", "rows": len(verdicts)}
    index_path.write_text(json.dumps(index))


def test_the_greedy_answer_and_sample_zero_keep_their_own_verdicts(tmp_path: Path) -> None:
    _export(tmp_path, "readback_test__base__greedy", {0: True})
    _export(tmp_path, "readback_test__base__n16", {0: False, 1: True})
    rows = check_parity.completions(tmp_path, ["base"])
    repo = check_parity.repository_verdicts(tmp_path, rows)
    got = {
        (r["file"], r["sample"]): repo[
            check_parity.key(r["run"], r["file"], r["checkpoint"], r["item_id"], r["sample"])
        ]["confirmed"]
        for r in rows
    }
    assert got == {
        ("readback_test__base__greedy", 0): True,
        ("readback_test__base__n16", 0): False,
        ("readback_test__base__n16", 1): True,
    }


def test_a_prefix_that_matches_nothing_selects_nothing(tmp_path: Path) -> None:
    _export(tmp_path, "readback_test__base__greedy", {0: True})
    assert check_parity.completions(tmp_path, ["base:"]) == []
    assert len(check_parity.completions(tmp_path, [])) == 1
