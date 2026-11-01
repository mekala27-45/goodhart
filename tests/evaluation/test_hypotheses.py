"""Each registered hypothesis is tested once from the repository's grades, with the clusters drawn
once for every series a test compares, Benjamini-Hochberg across the family, and a verdict whatever
it is; a hypothesis whose inputs were never produced says which are missing."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from goodhart_evaluation import hypotheses
from goodhart_evaluation.grade import INDEX, to_table
from goodhart_evaluation.metrics import Graded

ROOT = Path(__file__).resolve().parents[2]


def _graded(item: int, confirmed: bool, set_name: str, sample: int = 0) -> Graded:
    return Graded(
        item_id=f"q{item:03d}",
        request="r",
        checkpoint="c",
        set=set_name,
        sample=sample,
        temperature=0.0,
        family="lookup",
        warehouse="doorstep",
        template=f"lookup.t{item % 10}",
        measure="payment_share",
        refuse_item=False,
        kind="sql",
        executed=True,
        original=confirmed,
        confirmed=confirmed,
        refused=False,
        tokens=10,
    )


def _root(tmp_path: Path) -> Path:
    (tmp_path / "evaluation").mkdir()
    shutil.copy(ROOT / "evaluation" / "hypotheses.yaml", tmp_path / "evaluation" / "hypotheses.yaml")
    (tmp_path / "requests").mkdir()
    return tmp_path


def _method_run(root: Path, run_id: str, arm: str, right: int, items: int = 60) -> None:
    """A sample run of one method: greedy readback_test answers, the first `right` items right."""
    (root / "requests" / f"{run_id}.json").write_text(json.dumps({"arm": arm, "stage": "sample"}))
    folder = root / "results" / "runs" / run_id
    folder.mkdir(parents=True)
    (folder / "provenance.json").write_text("{}")
    name = "readback_test__final__greedy"
    out = root / "results" / "grades" / run_id / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table([_graded(i, i < right, "readback_test") for i in range(items)]), out)
    index_path = root / INDEX
    index: dict[str, Any] = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[f"{run_id}/{name}"] = {"set": "readback_test", "rows": items}
    index_path.write_text(json.dumps(index))


def test_the_clusters_are_drawn_once_for_every_series_of_a_test() -> None:
    a = {f"q{i}": (f"c{i % 6}", 1.0 if i % 2 else 0.0) for i in range(60)}
    b = {k: (c, v * 0.5) for k, (c, v) in a.items()}
    got = hypotheses.resample({"a": a, "b": b}, lambda v: v["a"] - v["b"])
    assert math.isclose(got["point"], 0.25) and got["clusters"] == 6 and got["items"] == 60
    # The same clusters for both series: the difference is exactly half of a in every draw.
    halves = hypotheses.resample({"a": a}, lambda v: v["a"] / 2.0)["draws"]
    assert np.allclose(got["draws"], halves)
    # Half the clusters are all wrong, so one draw in 2 ** 6 has a at zero: the p value is about that.
    assert 0.01 < hypotheses.p_value(got["draws"], lambda d: d > 0.0) < 0.03


def test_an_undefined_ratio_counts_against_the_hypothesis() -> None:
    draws = np.array([math.inf, 0.1, 0.2, math.nan])
    assert hypotheses.p_value(draws, lambda d: d < 1 / 3) == (2 + 1) / (4 + 1)


def test_nothing_produced_yet_leaves_every_hypothesis_untested_and_says_why(tmp_path: Path) -> None:
    out = hypotheses.run(_root(tmp_path))
    assert set(out) == {"H1", "H2", "H3", "H4", "H5"}
    assert all(h["verdict"] == "not measured" and h["held"] is None for h in out.values())
    assert "M4" in out["H1"]["detail"] and "honest" in out["H4"]["detail"]
    assert json.loads((tmp_path / hypotheses.RESULTS).read_text())["H3"]["statement"].startswith("the random")


def test_sft_then_grpo_beating_sft_alone_holds_and_the_others_wait(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _method_run(root, "2026-11-02-01-sample", "M1", right=20)
    _method_run(root, "2026-11-02-02-sample", "M4", right=45)
    out = hypotheses.run(root)
    h1 = out["H1"]
    assert h1["tested"] and math.isclose(h1["point"], 25 / 60) and h1["verdict"] == "held"
    assert h1["adjusted"] <= 0.05 and h1["low"] > 0
    assert out["H2"]["verdict"] == "not measured"


def test_a_cheap_method_that_reaches_half_the_gain_still_needs_its_compute_condition(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _method_run(root, "2026-11-02-01-sample", "M0", right=10)
    _method_run(root, "2026-11-02-02-sample", "M2", right=40)
    _method_run(root, "2026-11-02-03-sample", "M3", right=45)
    compute = {"compute": {"M2": {"gpu_hours": 1.5}, "M3": {"gpu_hours": 2.0}}}
    (root / "results" / "methods.json").write_text(json.dumps(compute))
    h5 = hypotheses.run(root)["H5"]
    assert h5["tested"] and h5["condition"]["at_most_a_quarter"] is False
    assert h5["verdict"] == "did not hold", "the accuracy half holds; the compute half does not"


def test_a_ratio_over_no_gain_is_reported_as_undefined_not_as_infinity(tmp_path: Path) -> None:
    import json as _json

    from goodhart_evaluation import hypotheses as h

    root = _root(tmp_path)
    flat = {f"q{i}": (f"c{i % 4}", 0.5) for i in range(20)}
    original = h.TESTS["H3"]
    h.TESTS["H3"] = lambda x: {
        **h.resample({"start": flat, "honest": flat, "random": flat}, h._ratio),
        "tested": True,
        "p": 1.0,
    }
    try:
        out = h.run(root)
    finally:
        h.TESTS["H3"] = original
    assert out["H3"]["point"] is None and "undefined" in out["H3"]["detail"] and out["H3"]["verdict"] == "did not hold"
    assert "Infinity" not in (root / h.RESULTS).read_text() and _json.loads((root / h.RESULTS).read_text())
