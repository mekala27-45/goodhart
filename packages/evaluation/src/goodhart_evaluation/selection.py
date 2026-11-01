"""goodhart select: the base selection, by the registered rule, from the probe and the graded dev samples.

The rule (evaluation/plan.yaml, base_selection): among candidates that pass the probe and whose
license permits releasing an adapter, the main model is the highest pass at 8 at or under 2.5B
parameters as counted on its card, the small model the highest pass at 8 under 1B; ties are broken
by steps per hour. Both measures are twin confirmed, on the development set: greedy pass at 1 and
pass at 8 from eight samples at temperature 0.8. The selection kernel wrote the completions; the
repository graded them (goodhart grade); this module only reads grades and applies the rule.

results/selection.json holds every candidate the probe saw, the ones that could not be chosen with
the reason, the two measures with the counts they come from, and the choice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goodhart_evaluation.grade import INDEX, read_grades
from goodhart_evaluation.metrics import by_item, item_values, mean

MAIN_MAX_PARAMETERS = 2_500_000_000
SMALL_MAX_PARAMETERS = 1_000_000_000
# Licenses under which a LoRA adapter of the base may be published, as SPDX ids the Hub cards use.
RELEASABLE = frozenset({"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"})


def safe(model_id: str) -> str:
    return model_id.replace("/", "__")


def graded_files(root: Path, run_id: str) -> set[str]:
    index = json.loads((root / INDEX).read_text(encoding="utf-8")) if (root / INDEX).exists() else {}
    return {key.split("/", 1)[1] for key in index if key.startswith(f"{run_id}/")}


def measures(root: Path, run_id: str, model_id: str) -> dict[str, Any] | None:
    """Greedy pass at 1 and pass at 8, twin confirmed, on the answerable dev items, with their counts."""
    files = graded_files(root, run_id)
    greedy_name, sampled_name = f"dev__{safe(model_id)}__greedy", f"dev__{safe(model_id)}__t08"
    if greedy_name not in files or sampled_name not in files:
        return None
    greedy = by_item(read_grades(root, run_id, greedy_name))
    sampled = by_item(read_grades(root, run_id, sampled_name))
    pass1 = item_values(greedy, "twin_confirmed_accuracy")
    pass8 = item_values(sampled, "pass_at_k", k=8)
    return {
        "greedy_pass_at_1": mean(pass1),
        "pass_at_8": mean(pass8),
        "execution_accuracy_greedy": mean(item_values(greedy, "execution_accuracy")),
        "refusal_accuracy_greedy": mean(item_values(greedy, "refusal_accuracy")),
        "items": len(pass8),
        "greedy_items": len(pass1),
        "samples_per_item": min((s.n for s in sampled if not s.refuse_item), default=0),
    }


def releasable(license_id: str | None) -> bool:
    return bool(license_id) and str(license_id).lower() in RELEASABLE


def choose(rows: list[dict[str, Any]], limit: int, strict: bool) -> str | None:
    """The eligible candidate with the highest pass at 8 under (or at) the limit; ties by steps an hour."""
    eligible = [
        r
        for r in rows
        if r["eligible"]
        and r["pass_at_8"] is not None
        and (r["parameters"] < limit if strict else r["parameters"] <= limit)
    ]
    if not eligible:
        return None
    best = max(eligible, key=lambda r: (r["pass_at_8"], r.get("steps_per_hour") or 0.0))
    return str(best["model"])


def select(root: Path, run_id: str, probe: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for c in probe["candidates"]:
        found = measures(root, run_id, c["model"]) if c.get("passes") else None
        why: list[str] = []
        if not c.get("passes"):
            why.append("did not pass the probe")
        if not releasable(c.get("license")):
            why.append(f"license {c.get('license') or 'not declared'} does not permit releasing an adapter")
        if c.get("passes") and found is None:
            why.append("the selection run has no graded dev samples for it")
        rows.append(
            {
                "model": c["model"],
                "license": c.get("license"),
                "parameters": int(c.get("parameters") or 0),
                "passes_probe": bool(c.get("passes")),
                "steps_per_hour": c.get("steps_per_hour_estimate"),
                "greedy_pass_at_1": None if found is None else found["greedy_pass_at_1"],
                "pass_at_8": None if found is None else found["pass_at_8"],
                "execution_accuracy_greedy": None if found is None else found["execution_accuracy_greedy"],
                "refusal_accuracy_greedy": None if found is None else found["refusal_accuracy_greedy"],
                "items": None if found is None else found["items"],
                "eligible": not why,
                "why_not": "; ".join(why) or None,
            }
        )
    main = choose(rows, MAIN_MAX_PARAMETERS, strict=False)
    small = choose(rows, SMALL_MAX_PARAMETERS, strict=True)
    return {
        "rule": "highest pass at 8 (twin confirmed, dev, 8 samples at 0.8); main at or under 2.5B parameters, "
        "small under 1B; eligible when the probe passes and the license permits releasing an adapter; "
        "ties by steps an hour",
        "run": run_id,
        "candidates": sorted(rows, key=lambda r: (-(r["pass_at_8"] or -1.0), r["model"])),
        "main": main,
        "small": small,
    }


def latest_select_run(root: Path) -> str | None:
    found = []
    for prov in sorted((root / "results" / "runs").glob("*/provenance.json")):
        request = root / "requests" / f"{prov.parent.name}.json"
        stage = json.loads(request.read_text(encoding="utf-8")).get("stage") if request.exists() else None
        status = json.loads(prov.read_text(encoding="utf-8")).get("status")
        if stage == "select" and status == "done":
            found.append(prov.parent.name)
    return found[-1] if found else None


def run(root: Path) -> dict[str, Any]:
    run_id = latest_select_run(root)
    if run_id is None:
        raise FileNotFoundError("no promoted selection run under results/runs")
    probe = json.loads((root / "results" / "gpu_probe.json").read_text(encoding="utf-8"))
    out = select(root, run_id, probe)
    (root / "results" / "selection.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out
