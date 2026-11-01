"""goodhart planted: what the planted data defect cost, SFT on the planted targets against SFT on the
targets the audit cleaned.

The two runs share the model, the recipe and the seed; only the SFT file differs (data/questions/sft/
cleaned.jsonl or planted.jsonl). Each is read at the checkpoint the registered rule chooses (the
highest twin confirmed accuracy on dev among its saves, goodhart checkpoints), from the greedy dev
answers the run exported there, and the difference is paired item by item with a cluster bootstrap.
The verdict word comes from the interval of the difference: cost, no measurable cost, or helped.

results/planted.json holds both runs and the difference; the manifest carries methods.planted, the
table the site's methods page reads, and planted.difference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goodhart_evaluation.grade import INDEX, read_grades
from goodhart_evaluation.metrics import ItemScore, by_item, item_values
from goodhart_evaluation.stats import bootstrap, paired

RESULTS = Path("results/planted.json")
SETS = {"cleaned": "clean", "planted": "planted"}


def runs(root: Path) -> dict[str, str]:
    """The latest promoted SFT run on each SFT file, by the file its request names."""
    out: dict[str, str] = {}
    for prov in sorted((root / "results" / "runs").glob("*/provenance.json")):
        req_path = root / "requests" / f"{prov.parent.name}.json"
        if not req_path.exists():
            continue
        req = json.loads(req_path.read_text(encoding="utf-8"))
        name = str(req.get("hyperparameters", {}).get("set", ""))
        if req.get("stage") == "sft" and name in SETS and not req.get("arm"):
            out[SETS[name]] = prov.parent.name
    return out


def _values(scores: list[ItemScore]) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for s in scores:
        got = item_values([s], "twin_confirmed_accuracy")
        if got:
            out[s.item_id] = got[0]
    return out


def verdict_of(low: float, high: float) -> str:
    """Planted minus cleaned: the defect cost when the whole interval is below zero."""
    if high < 0.0:
        return "cost"
    if low > 0.0:
        return "helped"
    return "no measurable cost"


def run(root: Path) -> dict[str, Any]:
    checkpoints = json.loads((root / "results" / "checkpoints.json").read_text(encoding="utf-8"))
    index = json.loads((root / INDEX).read_text(encoding="utf-8"))
    found = runs(root)
    out: dict[str, Any] = {"runs": {}}
    values: dict[str, dict[str, tuple[str, float]]] = {}
    for label, run_id in sorted(found.items()):
        step = checkpoints.get(run_id, {}).get("chosen")
        name = f"dev__step-{step}"
        if step is None or f"{run_id}/{name}" not in index:
            continue
        values[label] = _values(by_item(read_grades(root, run_id, name)))
        iv = bootstrap(list(values[label].values()))
        out["runs"][label] = {
            "run": run_id,
            "step": int(step),
            "twin_confirmed": iv.point,
            "low": iv.low,
            "high": iv.high,
            "items": iv.items,
        }
    if "clean" in values and "planted" in values:
        d = paired(values["planted"], values["clean"])
        out["difference"] = {
            "point": d.point,
            "low": d.low,
            "high": d.high,
            "items": d.items,
            "verdict": verdict_of(d.low, d.high),
        }
    (root / RESULTS).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
