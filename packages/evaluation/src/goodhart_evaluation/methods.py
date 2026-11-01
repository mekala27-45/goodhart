"""goodhart methods: the methods on one budget, from the graded held-out samples and the run ledger.

A method's runs name it in their request's arm (M0 to M4): its training runs (sft, dpo, grpo), the
sampling it needs (pairs, which feeds DPO and the learnability filter) and the held-out sampling of
its chosen checkpoint (sample). The repository graded every exported completion (goodhart grade);
this module only reads the grades and the ledgers.

Per method and held-out set: twin confirmed pass at 1 (greedy when a greedy file exists, else the
mean over samples), execution accuracy, refusal accuracy, over refusal, pass at 1 and pass at 16 from
the sampled file, each with a cluster bootstrap interval, and the share each hack detector flags.
Then the paired difference of every method against the base (M0) and against the floor, and of M4
against M1, the first hypothesis. Compute is the GPU hours the method's training and sampling runs
used, from their ledger rows, a pairs run counted for every method that reads its pairs or its
learnability list; evaluation sampling is measurement and does not count. M4 starts from M1's
checkpoint, so its compute adds M1's training up to the step it started from, prorated by steps.

results/methods.json holds it all, and the manifest carries the tables the site and RESULTS read:
methods.accuracy, methods.differences and methods.compute.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from goodhart_evaluation.grade import INDEX, read_grades
from goodhart_evaluation.metrics import DETECTORS, ItemScore, by_item, item_values, mean
from goodhart_evaluation.stats import bootstrap, paired

METHODS = ("M0", "M1", "M2", "M3", "M4")
SETS = ("readback_test", "test", "template_test")
TRAINING_STAGES = frozenset({"sft", "dpo", "pairs", "grpo"})
FLOOR = "floor"
METRICS = ("twin_confirmed_accuracy", "execution_accuracy", "refusal_accuracy", "over_refusal")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def requests(root: Path) -> dict[str, dict[str, Any]]:
    """Every promoted run's request, by run id."""
    out = {}
    for prov in sorted((root / "results" / "runs").glob("*/provenance.json")):
        req = _json(root / "requests" / f"{prov.parent.name}.json")
        if req is not None:
            out[prov.parent.name] = req
    return out


def method_of(req: dict[str, Any]) -> str | None:
    arm = str(req.get("arm") or "").strip().upper()
    return arm if arm in METHODS else None


def is_interior(req: dict[str, Any]) -> bool:
    """A sample run of the interior tests (the champion's neighbouring checkpoints or another
    temperature) carries its method's arm too, but it never stands for the method."""
    return bool(req.get("hyperparameters", {}).get("interior"))


def graded_files(root: Path) -> dict[str, dict[str, Any]]:
    """The grades index: run/name to its entry (the set among it)."""
    index = _json(root / INDEX) or {}
    return {k: v for k, v in index.items()}


def _files_for(index: dict[str, Any], runs: Sequence[str], set_name: str) -> tuple[str | None, str | None]:
    """(greedy, sampled) keys of the latest run of the list with a graded file for the set."""
    for run in reversed(list(runs)):
        mine = sorted(k for k, v in index.items() if k.startswith(f"{run}/") and v.get("set") == set_name)
        greedy = next((k for k in mine if k.endswith("__greedy")), None)
        sampled = next((k for k in mine if not k.endswith("__greedy")), None)
        if greedy or sampled:
            return greedy, sampled
    return None, None


def _scores(root: Path, key: str | None) -> list[ItemScore] | None:
    if key is None:
        return None
    run, name = key.split("/", 1)
    return by_item(read_grades(root, run, name))


def _interval(values: list[tuple[str, float]]) -> dict[str, Any] | None:
    if not values:
        return None
    iv = bootstrap(values)
    return {"point": iv.point, "low": iv.low, "high": iv.high, "items": iv.items, "clusters": iv.clusters}


def _per_item(scores: Sequence[ItemScore], metric: str) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for s in scores:
        values = item_values([s], metric)
        if values:
            out[s.item_id] = values[0]
    return out


def score_set(greedy: list[ItemScore] | None, sampled: list[ItemScore] | None) -> dict[str, Any]:
    """Every metric of one method on one set, with intervals."""
    primary = greedy or sampled or []
    out: dict[str, Any] = {"greedy": greedy is not None}
    for metric in METRICS:
        out[metric] = _interval(item_values(primary, metric))
    # The share of answers each hack detector flags, for the model card.
    out["hack_rates"] = {name: mean(item_values(primary, f"hack:{name}")) for name in DETECTORS}
    if sampled:
        n = min((s.n for s in sampled if not s.refuse_item), default=0)
        out["samples"] = n
        out["pass_at_1"] = _interval(item_values(sampled, "pass_at_k", k=1))
        out["pass_at_16"] = _interval(item_values(sampled, "pass_at_k", k=16)) if n >= 16 else None
    return out


def ledger_hours(root: Path, run_id: str) -> tuple[float, int]:
    """GPU hours of a run's training and sampling stages, and the steps it took."""
    path = root / "results" / "runs" / run_id / "ledger.jsonl"
    hours, steps = 0.0, 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("stage") in TRAINING_STAGES:
                    hours += sum(float(v) for v in (row.get("gpu_seconds") or {}).values()) / 3600.0
                    steps += int(row.get("steps") or 0)
    return hours, steps


def shared_sampling(reqs: dict[str, dict[str, Any]], runs: Sequence[str]) -> list[str]:
    """The sampling runs another method's request reads from: a GRPO run's learnability list or a
    DPO run's pairs file names the pairs run that sampled it (data/learnable/<run>.json,
    data/pairs/<run>.jsonl), and that run counts for every method that needs it."""
    out = []
    for r in runs:
        hp = reqs[r].get("hyperparameters", {})
        for key in ("learnable_file", "pairs_file"):
            if key in hp:
                source = Path(str(hp[key])).stem
                if source in reqs and source not in runs and source not in out:
                    out.append(source)
    return out


def compute(root: Path, reqs: dict[str, dict[str, Any]], by_method: dict[str, list[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in METHODS:
        runs = [r for r in by_method.get(m, []) if reqs[r].get("stage") in TRAINING_STAGES]
        runs += shared_sampling(reqs, runs)
        hours = sum(ledger_hours(root, r)[0] for r in runs)
        inherited = 0.0
        if m == "M4":
            start = next((reqs[r] for r in runs if reqs[r].get("stage") == "grpo"), None)
            step = (start or {}).get("hyperparameters", {}).get("adapter_step")
            m1 = [r for r in by_method.get("M1", []) if reqs[r].get("stage") == "sft"]
            if m1:
                m1_hours, m1_steps = ledger_hours(root, m1[-1])
                share = 1.0 if step is None or not m1_steps else min(1.0, int(step) / m1_steps)
                inherited = m1_hours * share
        out[m] = {"runs": runs, "gpu_hours": round(hours + inherited, 3), "inherited_from_m1": round(inherited, 3)}
    return out


def run(root: Path) -> dict[str, Any]:
    reqs = requests(root)
    by_method: dict[str, list[str]] = {m: [] for m in METHODS}
    for run_id, req in reqs.items():
        m = method_of(req)
        if m is not None:
            by_method[m].append(run_id)
    index = graded_files(root)
    scores: dict[str, dict[str, Any]] = {}
    per_item: dict[tuple[str, str], dict[str, tuple[str, float]]] = {}
    for m in [*METHODS, FLOOR]:
        runs = by_method.get(m, []) if m != FLOOR else ["floor"]
        sample_runs = [r for r in runs if m == FLOOR or (reqs[r].get("stage") == "sample" and not is_interior(reqs[r]))]
        scores[m] = {}
        for set_name in SETS:
            greedy_key, sampled_key = _files_for(index, sample_runs, set_name)
            greedy, sampled = _scores(root, greedy_key), _scores(root, sampled_key)
            if greedy is None and sampled is None:
                continue
            scores[m][set_name] = {**score_set(greedy, sampled), "files": [k for k in (greedy_key, sampled_key) if k]}
            per_item[(m, set_name)] = _per_item(greedy or sampled or [], "twin_confirmed_accuracy")
    differences = []
    pairs = [(m, "M0") for m in METHODS if m != "M0"] + [(m, FLOOR) for m in METHODS] + [("M4", "M1")]
    for a, b in pairs:
        for set_name in SETS:
            if (a, set_name) in per_item and (b, set_name) in per_item:
                try:
                    d = paired(per_item[(a, set_name)], per_item[(b, set_name)])
                except ValueError:
                    continue
                differences.append(
                    {
                        "a": a,
                        "b": b,
                        "set": set_name,
                        "metric": "twin_confirmed_accuracy",
                        "point": d.point,
                        "low": d.low,
                        "high": d.high,
                        "clusters": d.clusters,
                        "items": d.items,
                        "p_greater": d.p_greater,
                    }
                )
    spent = compute(root, reqs, by_method)
    out = {"methods": by_method, "scores": scores, "differences": differences, "compute": spent}
    (root / "results" / "methods.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
