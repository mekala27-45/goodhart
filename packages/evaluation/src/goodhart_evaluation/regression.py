"""goodhart regression: the regression suite, IFEval and Spider dev, as a change against the base.

Training on three warehouses can cost a model what it could do before. Two public checks measure it,
each graded in the repository from the exported answers: IFEval by Google's reference checker
(goodhart ifeval) and Spider dev by the test suite evaluation (goodhart spider). This reads those
grades, takes each method's latest sampled checkpoint (the method is its sample request's arm), and
reports the accuracy with its interval and the paired change against the base (M0) with its
interval, item by item on the items both answered. An IFEval prompt is its own cluster (the prompts
were written one by one); a Spider question clusters by its database. The verdict word is chosen
from the interval of the change, never from the point: helped, hurt, or no measurable change.

results/regression.json holds it; the manifest carries regression.ifeval and regression.spider.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from goodhart_evaluation.methods import METHODS, method_of, requests
from goodhart_evaluation.spider import verdict_of
from goodhart_evaluation.stats import bootstrap, paired

RESULTS = Path("results/regression.json")


def _latest_by_method(root: Path, folder: Path, prefix: str) -> dict[str, Path]:
    """The graded file of each method's latest sample run, by request order (the date leads)."""
    reqs = requests(root)
    out: dict[str, Path] = {}
    for run_id in sorted(reqs):
        m = method_of(reqs[run_id])
        if m is None or reqs[run_id].get("stage") != "sample":
            continue
        files = sorted((folder / run_id).glob(f"{prefix}*.json")) if (folder / run_id).exists() else []
        if files:
            out[m] = files[0]
    return out


def ifeval_values(path: Path) -> dict[str, tuple[str, float]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["item_id"]): (str(r["item_id"]), float(bool(r["strict"]))) for r in rows}


def spider_values(path: Path) -> dict[str, tuple[str, float]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["item_id"]): (str(r["db_id"]), float(bool(r["right"]))) for r in rows if not r.get("gold_error")}


def compare(values: Mapping[str, dict[str, tuple[str, float]]], files: Mapping[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base = values.get("M0")
    for m in METHODS:
        if m not in values:
            continue
        iv = bootstrap(list(values[m].values()))
        entry: dict[str, Any] = {
            "file": files[m].as_posix(),
            "accuracy": iv.point,
            "low": iv.low,
            "high": iv.high,
            "items": iv.items,
            "clusters": iv.clusters,
        }
        if base is not None and m != "M0":
            d = paired(values[m], base)
            entry["change"] = {
                "point": d.point,
                "low": d.low,
                "high": d.high,
                "items": d.items,
                "verdict": verdict_of(d.low, d.high),
            }
        out[m] = entry
    return out


def run(root: Path) -> dict[str, Any]:
    grades = root / "results" / "grades"
    suites = {
        "ifeval": (_latest_by_method(root, grades / "ifeval", "ifeval__"), ifeval_values),
        "spider": (_latest_by_method(root, grades / "spider", "spider_dev__"), spider_values),
    }
    out: dict[str, Any] = {}
    for name, (files, read) in suites.items():
        values = {m: read(p) for m, p in files.items()}
        rel = {m: p.relative_to(root) for m, p in files.items()}
        out[name] = compare(values, rel)
    out["metric"] = {
        "ifeval": "prompt level strict accuracy (every instruction of the prompt followed, as written)",
        "spider": "test suite execution accuracy with values",
    }
    (root / RESULTS).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
