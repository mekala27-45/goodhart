"""The interior tests: is the champion a choice the evidence makes, or one the settings made?

Registered with the plan (interior_tests): each knob the result could hinge on is moved to its
neighbours, the candidates' readback_test accuracy is computed again under each, and the page prints
the champion's accuracy under every neighbour with the word degenerate when the champion would change
at the first neighbour of any knob.

    twin count   the kept four twins, the first three of them, and six: the kept four and the two
                 other twins that caught the most distillation mutants (the lock records the counts)
    tolerance    every gold column's tolerance halved, as registered, and doubled
    checkpoint   the chosen step against its neighbours 25 steps either side, from the sample runs
                 of those steps
    temperature  pass at 1 at 0, 0.2 and the registered 0.8, from the sample runs at each

Twin count and tolerance regrade the exported answers in the repository; the checkpoint and the
temperature read the grades of sample runs that asked for them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from goodhart_core.items import Completion, Gold, Item
from goodhart_core.warehouse import Warehouses
from goodhart_heldout.confirm import TwinGolds
from goodhart_heldout.verdict import verdict
from goodhart_rewards.parse import parse_completion

from goodhart_evaluation.metrics import Graded, by_item, item_values, mean

SCALES = (0.5, 1.0, 2.0)
COUNTS = (3, 4, 6)


def twin_sets(lock: Mapping[str, Any]) -> dict[int, list[str]]:
    kept = list(lock["kept"])
    caught = lock["distillation"]["caught_by_twin"]
    rest = sorted((t for t in caught if t not in kept), key=lambda t: (-int(caught[t]), t))
    return {3: kept[:3], 4: kept, 6: kept + rest[:2]}


def scaled(gold: Gold, scale: float) -> Gold:
    columns = [c.model_copy(update={"tolerance": c.tolerance * scale}) for c in gold.columns]
    return gold.model_copy(update={"columns": columns})


def regrade(
    completions: Sequence[Completion],
    items: Mapping[str, Item],
    golds: TwinGolds,
    versions: Sequence[str],
    set_name: str,
    scale: float = 1.0,
) -> list[Graded]:
    """The answers graded again on the original and the given twins, tolerances scaled. A TwinGolds
    caches each twin gold with the tolerance it was first asked with, so each scale needs its own."""
    out = []
    for c in completions:
        item = items[c.item_id]
        if item.gold is not None and scale != 1.0:
            item = item.model_copy(update={"gold": scaled(item.gold, scale)})
        v = verdict(item, parse_completion(c.text), golds, list(versions))
        out.append(
            Graded(
                item_id=c.item_id,
                request=c.request,
                checkpoint=c.checkpoint,
                set=set_name,
                sample=c.sample,
                temperature=c.temperature,
                family=item.family,
                warehouse=item.warehouse,
                template=item.template,
                measure=item.measure,
                refuse_item=item.refuse,
                kind=v.kind,
                executed=v.executed,
                original=v.original,
                confirmed=v.confirmed,
                refused=v.refused,
                tokens=c.tokens,
                flags=v.flags,
            )
        )
    return out


def accuracy(rows: Sequence[Graded]) -> float | None:
    return mean(item_values(by_item(rows), "twin_confirmed_accuracy"))


def knobs(
    completions: Mapping[str, Sequence[Completion]],
    items: Mapping[str, Item],
    warehouses: Warehouses,
    lock: Mapping[str, Any],
    set_name: str = "readback_test",
) -> dict[str, dict[str, float | None]]:
    """Each candidate's accuracy under every twin count and every tolerance scale."""
    sets = twin_sets(lock)
    out: dict[str, dict[str, float | None]] = {name: {} for name in completions}
    for count in COUNTS:
        golds = TwinGolds(warehouses)
        for name, answers in completions.items():
            out[name][f"twins={count}"] = accuracy(regrade(answers, items, golds, sets[count], set_name))
    for scale in SCALES:
        if scale == 1.0:
            continue
        golds = TwinGolds(warehouses)
        for name, answers in completions.items():
            out[name][f"tolerance={scale:g}"] = accuracy(regrade(answers, items, golds, sets[4], set_name, scale))
    return out


def champion_under(scores: Mapping[str, Mapping[str, float | None]], setting: str) -> str | None:
    measured = {name: s.get(setting) for name, s in scores.items() if s.get(setting) is not None}
    return max(measured, key=lambda n: (measured[n], n)) if measured else None


FIRST_NEIGHBOURS = ("twins=3", "twins=6", "tolerance=0.5", "tolerance=2")


def verdict_of(scores: Mapping[str, Mapping[str, float | None]], champion: str) -> dict[str, Any]:
    """The champion's accuracy under every setting, and degenerate when another candidate would be
    the champion at the first neighbour of any knob."""
    changed = [s for s in FIRST_NEIGHBOURS if champion_under(scores, s) not in (None, champion)]
    return {
        "champion": champion,
        "under": dict(scores.get(champion, {})),
        "changes_at": changed,
        "degenerate": bool(changed),
        "scores": {name: dict(under) for name, under in scores.items()},
    }


def write(root: Path, result: Mapping[str, Any]) -> None:
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    (root / "results" / "interior.json").write_text(text, encoding="utf-8")


def _completions(root: Path, key: str) -> list[Completion]:
    run, name = key.split("/", 1)
    path = root / "results" / "runs" / run / "completions" / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [Completion.model_validate(r) for r in rows]


def neighbours(root: Path, champion: str, set_name: str = "readback_test") -> dict[str, float | None]:
    """The champion's accuracy at every other checkpoint and temperature its sample runs graded: the
    greedy file of each step (checkpoint=<step>), and pass at 1 of each sampled file by its
    temperature (temperature=<t>, the greedy file as temperature=0)."""
    from goodhart_evaluation.grade import INDEX, read_grades
    from goodhart_evaluation.methods import method_of, requests

    index_path = root / INDEX
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    reqs = requests(root)
    out: dict[str, float | None] = {}
    for key in sorted(index):
        run, name = key.split("/", 1)
        req = reqs.get(run)
        if req is None or req.get("stage") != "sample" or method_of(req) != champion:
            continue
        parts = name.split("__")
        if parts[0] != set_name or len(parts) != 3:
            continue
        checkpoint, how = parts[1], parts[2]
        scores = by_item(read_grades(root, run, name))
        if how == "greedy":
            if checkpoint.startswith("step-"):
                out[f"checkpoint={checkpoint.removeprefix('step-')}"] = mean(
                    item_values(scores, "twin_confirmed_accuracy")
                )
            out.setdefault("temperature=0", mean(item_values(scores, "twin_confirmed_accuracy")))
        else:
            grades = read_grades(root, run, name)
            t = grades[0].temperature if grades else None
            if t is not None:
                out[f"temperature={t:g}"] = mean(item_values(scores, "pass_at_k", k=1))
    return out


def run(root: Path) -> dict[str, Any]:
    """Every candidate's readback_test answers graded again under each twin count and tolerance, the
    champion's neighbouring checkpoints and temperatures from its sample runs, and the verdict."""
    from goodhart_heldout.pipeline import gold_tables, materialize, warehouses_with

    from goodhart_evaluation.grade import load_items

    gates = json.loads((root / "results" / "gates.json").read_text(encoding="utf-8"))
    champion = gates.get("champion")
    if not champion:
        raise ValueError("no checkpoint cleared every gate, so there is no champion to test")
    methods = json.loads((root / "results" / "methods.json").read_text(encoding="utf-8"))
    files = {
        m: next(k for k in got["readback_test"]["files"])
        for m, got in methods["scores"].items()
        if m != "floor" and "readback_test" in got
    }
    completions = {m: _completions(root, key) for m, key in files.items()}
    items = {i.id: i for i in load_items(root).values() if i.split == "readback_test"}
    lock = json.loads((root / "data" / "twins.lock.json").read_text(encoding="utf-8"))
    six = twin_sets(lock)[6]
    tables = gold_tables(list(items.values()))
    from goodhart_core import paths

    materialize(root, six, paths.twins_dir(), tables)
    warehouses = warehouses_with(six, tables, paths.twins_dir(), root)
    scores = knobs(completions, items, warehouses, lock)
    for setting, value in neighbours(root, str(champion)).items():
        scores.setdefault(str(champion), {})[setting] = value
    result = {**verdict_of(scores, str(champion)), "files": files}
    write(root, result)
    return result
