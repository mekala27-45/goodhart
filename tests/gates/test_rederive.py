"""The rederive compares each derived result as data, and a GRPO run requested again is compared
with its original at the original's last checkpoint, within the seed spread."""

from __future__ import annotations

import json
from pathlib import Path

import reset_and_rederive as rr


def test_a_result_is_compared_as_data_not_as_bytes(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('{"x": 1, "y": [1, 2]}')
    b.write_text('{\n "y": [1, 2],\n "x": 1\n}\n')
    assert rr._same(a, b) is True
    b.write_text('{"x": 2}')
    assert rr._same(a, b) is False
    assert rr._same(tmp_path / "missing", tmp_path / "also missing") is None
    assert rr._same(a, tmp_path / "missing") is False


def _checkpoints(root: Path, rerun_at_50: float) -> None:
    steps = {
        "A": {"25": 0.30, "50": 0.40},
        "B": {"25": 0.31, "50": rerun_at_50},
    }
    out = {r: {"steps": {s: {"twin_confirmed_accuracy": v} for s, v in got.items()}} for r, got in steps.items()}
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "checkpoints.json").write_text(json.dumps(out))


def test_the_rerun_is_read_at_the_originals_last_checkpoint_within_the_registered_bound(tmp_path: Path) -> None:
    _checkpoints(tmp_path, 0.43)
    got = rr.grpo_pair(tmp_path, "A", "B")
    assert got["step"] == 50 and got["within"] is True and got["bound"] == rr.SEED_BOUND
    assert got["bound_from"] == "the registered seed spread bound"
    _checkpoints(tmp_path, 0.47)
    assert rr.grpo_pair(tmp_path, "A", "B")["within"] is False


def test_a_measured_seed_spread_is_the_bound_when_there_are_seeds(tmp_path: Path) -> None:
    _checkpoints(tmp_path, 0.43)
    gates = {"candidates": {"M3": {"gates": [{"gate": "seed_spread", "measured": True, "value": 0.02}]}}}
    (tmp_path / "results" / "gates.json").write_text(json.dumps(gates))
    got = rr.grpo_pair(tmp_path, "A", "B")
    assert got["bound"] == 0.02 and got["within"] is False and got["bound_from"] == "the measured seed spread"
