"""The checkpoint choice: the dev step with the best twin confirmed accuracy, the earlier on a tie."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from goodhart_evaluation import checkpoints
from goodhart_evaluation.grade import INDEX, to_table
from goodhart_evaluation.metrics import Graded


def _dev(root: Path, run: str, step: int, right: int, items: int = 20) -> None:
    rows = [
        Graded(
            item_id=f"q{i}",
            request=run,
            checkpoint=f"step-{step}",
            set="dev",
            sample=0,
            temperature=0.0,
            family="lookup",
            warehouse="doorstep",
            template=f"t{i % 4}",
            measure="m",
            refuse_item=False,
            kind="sql",
            executed=True,
            original=i < right,
            confirmed=i < right,
            refused=False,
            tokens=5,
        )
        for i in range(items)
    ]
    name = f"dev__step-{step}"
    out = root / "results" / "grades" / run / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(rows), out)
    index_path = root / INDEX
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[f"{run}/{name}"] = {"source": "x", "grades": "y", "rows": len(rows), "set": "dev"}
    index_path.write_text(json.dumps(index))


def test_the_best_dev_step_is_chosen_and_a_tie_goes_to_the_earlier(tmp_path: Path) -> None:
    for step, right in ((25, 4), (50, 9), (75, 12), (100, 12), (125, 10)):
        _dev(tmp_path, "r-sft", step, right)
    _dev(tmp_path, "r-other", 25, 3)
    out = checkpoints.run(tmp_path)
    assert out["r-sft"]["chosen"] == 75 and out["r-sft"]["steps"]["100"]["twin_confirmed_accuracy"] == 0.6
    assert out["r-other"]["chosen"] == 25
    assert json.loads((tmp_path / "results" / "checkpoints.json").read_text())["r-sft"]["chosen"] == 75
    assert checkpoints.choose({}) is None
