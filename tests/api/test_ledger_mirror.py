"""The run ledger mirror: every promoted run's ledger row upserted into the database by its request,
read back from a separate connection, and a run collected again replaces its row."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import mirror_ledger
import pytest

pytestmark = pytest.mark.postgres


def _run(root: Path, request: str, generation: float, loading: float) -> None:
    folder = root / "results" / "runs" / request
    folder.mkdir(parents=True)
    row = {
        "request": request,
        "stage": "sft",
        "model": "Qwen/Qwen3-0.6B",
        "status": "done",
        "steps": 302,
        "gpu_seconds": {"generation": generation, "loading": loading},
        "wall_clock_seconds": generation + loading,
        "ceiling_hours": 1.2,
        "peak_memory_bytes": 13_665_137_664,
    }
    (folder / "ledger.jsonl").write_text(json.dumps({**row, "status": "running"}) + "\n" + json.dumps(row) + "\n")


def test_each_run_is_mirrored_by_request_and_read_back(tmp_path: Path) -> None:
    from goodhart_api import db

    first, second = f"2026-11-02-{uuid.uuid4().hex[:6]}-sft", f"2026-11-02-{uuid.uuid4().hex[:6]}-pairs"
    _run(tmp_path, first, 3600.0, 360.0)
    _run(tmp_path, second, 720.0, 0.0)
    url = os.environ["DATABASE_URL"]
    got = mirror_ledger.mirror(url, tmp_path)
    assert got["mirrored"] == 2 and got["read_back"] == 2 and got["mismatches"] == []
    assert got["gpu_hours"] == pytest.approx(1.1 + 0.2)
    # Collected again with more GPU time: the row is replaced, not duplicated.
    (tmp_path / "results" / "runs" / second / "ledger.jsonl").unlink()
    (tmp_path / "results" / "runs" / second).rmdir()
    _run(tmp_path, second, 1440.0, 0.0)
    assert mirror_ledger.mirror(url, tmp_path)["mismatches"] == []
    with db.session(db.engine_for(url)) as s:
        rows = {r.request: r for r in db.ledger(s)}
    assert rows[second].gpu_hours == pytest.approx(0.4) and rows[first].peak_memory_bytes == 13_665_137_664
    assert rows[first].status == "done", "the last row the meter wrote is the run's"
