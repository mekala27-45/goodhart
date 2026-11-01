"""goodhart checkpoints: the checkpoint choice, from the dev exports every training run writes.

Registered with the plan (checkpoint_choice): the step with the highest twin confirmed accuracy on
dev, greedy, among the checkpoints saved every 25 steps. Each training run exports greedy dev answers
at every save (completions/dev__step-<n>.jsonl, graded by goodhart grade); this reads their grades
and picks the step. A tie goes to the earlier step, the cheaper one to have reached. The curve is
also what the site draws beside the reward paid.

results/checkpoints.json holds, per run, the accuracy at every step with its item count, and the
chosen step, which a sample request names as adapter_step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from goodhart_evaluation.grade import INDEX, read_grades
from goodhart_evaluation.metrics import by_item, item_values, mean

STEP = re.compile(r"^dev__step-(\d+)$")


def curve(root: Path, run_id: str, index: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for key in sorted(index):
        run, name = key.split("/", 1)
        m = STEP.match(name)
        if run != run_id or m is None:
            continue
        scores = by_item(read_grades(root, run, name))
        values = item_values(scores, "twin_confirmed_accuracy")
        out[int(m.group(1))] = {"twin_confirmed_accuracy": mean(values), "items": len(values)}
    return out


def choose(points: dict[int, dict[str, Any]]) -> int | None:
    measured = {s: p["twin_confirmed_accuracy"] for s, p in points.items() if p["twin_confirmed_accuracy"] is not None}
    if not measured:
        return None
    best = max(measured.values())
    return min(s for s, v in measured.items() if v == best)


def run(root: Path) -> dict[str, Any]:
    index_path = root / INDEX
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    runs = sorted({k.split("/", 1)[0] for k in index if STEP.match(k.split("/", 1)[1])})
    out: dict[str, Any] = {}
    for run_id in runs:
        points = curve(root, run_id, index)
        out[run_id] = {"steps": {str(s): p for s, p in sorted(points.items())}, "chosen": choose(points)}
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    (root / "results" / "checkpoints.json").write_text(text, encoding="utf-8")
    return out
