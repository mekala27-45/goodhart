"""goodhart hypotheses: each registered hypothesis tested once, with its verdict whatever it is.

evaluation/hypotheses.yaml was hashed with the plan before the first test question was scored. Each
test below is the registered one, computed from the repository's own grades: a cluster bootstrap
over question templates and entity families (2,000 resamples, the clusters drawn once and shared by
every series a test compares), a one sided bootstrap p value for the direction the hypothesis
predicts, and Benjamini-Hochberg at q = 0.05 across the hypotheses that could be tested. A
hypothesis held when its adjusted p value rejects the null and any extra condition the registered
test names (H5's compute) is met, and did not hold otherwise; one whose inputs were never produced
is not measured, says what is missing, and is left out of the family.

    H1  M4 minus M1, twin confirmed, readback_test                       above zero
    H2  (M3 minus M0) at pass@1 minus (M3 minus M0) at pass@16, test       above zero
    H3  (random minus start) over (honest minus start), dev, last steps    below one third
    H4  honest minus lenient, twin confirmed, dev, last checkpoints        at least 0.10
    H5  (M2 minus M0) minus half of (M3 minus M0), readback_test           at least zero, and
        M2 GPU hours at most a quarter of M3's

results/hypotheses.json holds every verdict; the manifest carries hypotheses.<H>.verdict.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from goodhart_evaluation import arms
from goodhart_evaluation.grade import INDEX, read_grades
from goodhart_evaluation.messages import verdict
from goodhart_evaluation.methods import _files_for, is_interior, method_of, requests
from goodhart_evaluation.metrics import ItemScore, by_item, item_values
from goodhart_evaluation.stats import LEVEL, RESAMPLES, SEED, _draws, benjamini_hochberg

RESULTS = Path("results/hypotheses.json")
Q = 0.05
Series = dict[str, tuple[str, float]]


def _per_item(scores: Sequence[ItemScore], metric: str, k: int = 1) -> Series:
    out: Series = {}
    for s in scores:
        values = item_values([s], metric, k=k)
        if values:
            out[s.item_id] = values[0]
    return out


def _scores(root: Path, key: str | None) -> list[ItemScore] | None:
    if key is None:
        return None
    run, name = key.split("/", 1)
    return by_item(read_grades(root, run, name))


def resample(
    series: Mapping[str, Series], statistic: Callable[[dict[str, float]], float], seed: int = SEED
) -> dict[str, Any]:
    """A statistic of several per item means, on the items every series has, with the clusters drawn
    once for all of them. Returns the point, the percentile interval and the draws."""
    shared = sorted(set.intersection(*(set(s) for s in series.values())))
    if not shared:
        raise ValueError("the series share no item")
    clusters = sorted({next(iter(series.values()))[i][0] for i in shared})
    position = {c: n for n, c in enumerate(clusters)}
    names = list(series)
    sums = np.zeros((len(names), len(clusters)))
    counts = np.zeros(len(clusters))
    first = series[names[0]]
    for i in shared:
        c = position[first[i][0]]
        counts[c] += 1
        for row, name in enumerate(names):
            sums[row, c] += series[name][i][1]
    point = statistic({n: float(sums[r].sum() / counts.sum()) for r, n in enumerate(names)})
    idx = _draws(len(clusters), RESAMPLES, seed)
    drawn_counts = counts[idx].sum(axis=1)
    means = {n: sums[r][idx].sum(axis=1) / drawn_counts for r, n in enumerate(names)}
    draws = np.array([statistic({n: float(means[n][d]) for n in names}) for d in range(RESAMPLES)])
    finite = draws[np.isfinite(draws)]
    alpha = (1.0 - LEVEL) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha]) if finite.size else (math.nan, math.nan)
    return {
        "point": point,
        "low": float(low),
        "high": float(high),
        "items": len(shared),
        "clusters": len(clusters),
        "draws": draws,
    }


def p_value(draws: np.ndarray, supports: Callable[[float], bool]) -> float:
    """The one sided bootstrap p value: the share of draws that do not support the hypothesis, with
    the add one rule so it is never exactly zero. An undefined draw (a ratio over zero) counts against."""
    against = sum(1 for d in draws if not (math.isfinite(d) and supports(float(d))))
    return (against + 1) / (len(draws) + 1)


class Inputs:
    """The per item series every test reads, looked up once from the grades."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.reqs = requests(root)
        index_path = root / INDEX
        self.index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        self.by_method: dict[str, list[str]] = {}
        for run_id, req in sorted(self.reqs.items()):
            m = method_of(req)
            if m is not None and req.get("stage") == "sample" and not is_interior(req):
                self.by_method.setdefault(m, []).append(run_id)

    def method(self, m: str, set_name: str, metric: str, k: int = 1, sampled: bool = False) -> Series | None:
        greedy, samples = _files_for(self.index, self.by_method.get(m, []), set_name)
        scores = _scores(self.root, samples if sampled or greedy is None else greedy)
        return _per_item(scores, metric, k) if scores else None

    def checkpoint(self, run_id: str, step: int) -> Series | None:
        key = f"{run_id}/dev__step-{step}"
        return _per_item(_scores(self.root, key) or [], "twin_confirmed_accuracy") if key in self.index else None

    def last_step(self, run_id: str) -> int | None:
        steps = [int(k.rsplit("-", 1)[1]) for k in self.index if k.startswith(f"{run_id}/dev__step-")]
        return max(steps) if steps else None

    def arm_last(self, arm: str) -> Series | None:
        run_id = arms.arm_runs(self.root).get(arm)
        step = self.last_step(run_id) if run_id else None
        return self.checkpoint(run_id, step) if run_id and step is not None else None

    def arms_start(self) -> Series | None:
        """The defect study's starting checkpoint on dev: the run and step the honest arm started from."""
        run_id = arms.arm_runs(self.root).get("honest")
        if not run_id:
            return None
        req = self.reqs[run_id]
        source = (req.get("inputs") or [None])[-1]
        step = req.get("hyperparameters", {}).get("adapter_step")
        if source is None:
            return None
        step = int(step) if step is not None else self.last_step(source)
        return self.checkpoint(source, step) if step is not None else None

    def gpu_hours(self, m: str) -> float | None:
        path = self.root / "results" / "methods.json"
        if not path.exists():
            return None
        compute = json.loads(path.read_text(encoding="utf-8")).get("compute", {})
        got = compute.get(m, {}).get("gpu_hours")
        return float(got) if got else None


def _missing(*names: str) -> dict[str, Any]:
    return {"tested": False, "held": None, "verdict": verdict(None), "detail": f"not produced: {', '.join(names)}"}


def h1(x: Inputs) -> dict[str, Any]:
    m4 = x.method("M4", "readback_test", "twin_confirmed_accuracy")
    m1 = x.method("M1", "readback_test", "twin_confirmed_accuracy")
    if m4 is None or m1 is None:
        return _missing(*[n for n, s in (("M4", m4), ("M1", m1)) if s is None])
    got = resample({"a": m4, "b": m1}, lambda v: v["a"] - v["b"])
    return {**got, "tested": True, "p": p_value(got["draws"], lambda d: d > 0.0)}


def h2(x: Inputs) -> dict[str, Any]:
    series = {f"{m}{k}": x.method(m, "test", "pass_at_k", k=k, sampled=True) for m in ("M3", "M0") for k in (1, 16)}
    absent = [n for n, s in series.items() if not s]
    if absent:
        return _missing(*absent)
    got = resample(
        {n: s for n, s in series.items() if s is not None},
        lambda v: (v["M31"] - v["M01"]) - (v["M316"] - v["M016"]),
    )
    return {**got, "tested": True, "p": p_value(got["draws"], lambda d: d > 0.0)}


def _ratio(v: dict[str, float]) -> float:
    gain = v["honest"] - v["start"]
    return (v["random"] - v["start"]) / gain if gain > 0 else math.inf


def h3(x: Inputs) -> dict[str, Any]:
    series = {"start": x.arms_start(), "honest": x.arm_last("honest"), "random": x.arm_last("random")}
    absent = [n for n, s in series.items() if s is None]
    if absent:
        return _missing(*absent)
    got = resample({n: s for n, s in series.items() if s is not None}, _ratio)
    return {**got, "tested": True, "p": p_value(got["draws"], lambda d: d < 1.0 / 3.0)}


def h4(x: Inputs) -> dict[str, Any]:
    honest, lenient = x.arm_last("honest"), x.arm_last("lenient")
    if honest is None or lenient is None:
        return _missing(*[n for n, s in (("honest", honest), ("lenient", lenient)) if s is None])
    got = resample({"honest": honest, "lenient": lenient}, lambda v: v["honest"] - v["lenient"])
    return {**got, "tested": True, "p": p_value(got["draws"], lambda d: d >= 0.10)}


def h5(x: Inputs) -> dict[str, Any]:
    series = {m: x.method(m, "readback_test", "twin_confirmed_accuracy") for m in ("M2", "M3", "M0")}
    absent = [n for n, s in series.items() if s is None]
    hours = {m: x.gpu_hours(m) for m in ("M2", "M3")}
    if absent or None in hours.values():
        return _missing(*absent, *[f"{m} GPU hours" for m, h in hours.items() if h is None])
    got = resample(
        {n: s for n, s in series.items() if s is not None},
        lambda v: (v["M2"] - v["M0"]) - 0.5 * (v["M3"] - v["M0"]),
    )
    m2, m3 = float(hours["M2"] or 0.0), float(hours["M3"] or 0.0)
    cheap = m2 <= 0.25 * m3
    return {
        **got,
        "tested": True,
        "p": p_value(got["draws"], lambda d: d >= 0.0),
        "condition": {"m2_gpu_hours": m2, "m3_gpu_hours": m3, "at_most_a_quarter": cheap},
    }


TESTS: dict[str, Callable[[Inputs], dict[str, Any]]] = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5}


def run(root: Path) -> dict[str, Any]:
    registered = yaml.safe_load((root / "evaluation" / "hypotheses.yaml").read_text(encoding="utf-8"))["hypotheses"]
    x = Inputs(root)
    found = {name: TESTS[name](x) for name in registered}
    family = benjamini_hochberg({n: float(f["p"]) for n, f in found.items() if f.get("tested")}, q=Q)
    out: dict[str, Any] = {}
    for name, f in found.items():
        entry = {k: v for k, v in f.items() if k != "draws"}
        for key in ("point", "low", "high"):
            if key in entry and not math.isfinite(float(entry[key])):
                # A ratio over a gain of zero or less (H3 when the honest arm gained nothing).
                entry[key] = None
                entry.setdefault("detail", "the estimate is undefined: the denominator is zero or less")
        entry.update(statement=registered[name]["statement"], test=registered[name]["test"])
        if f.get("tested"):
            bh = family[name]
            extra = f.get("condition", {}).get("at_most_a_quarter", True)
            entry["adjusted"] = bh["adjusted"]
            entry["held"] = bool(bh["rejected"] and extra)
            entry["verdict"] = verdict(entry["held"])
        out[name] = entry
    (root / RESULTS).parent.mkdir(parents=True, exist_ok=True)
    (root / RESULTS).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
