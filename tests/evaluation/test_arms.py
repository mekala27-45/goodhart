"""The defect study's runs as the notebooks read them: the arm of each GRPO run, the reward it was
paid per step, the confirmed accuracy it earned per checkpoint, and the step a monitor fired."""

from __future__ import annotations

import json
from pathlib import Path

from goodhart_evaluation import arms


def test_paid_against_earned_for_an_arm(tmp_path: Path) -> None:
    run = tmp_path / "results" / "runs" / "r-lenient"
    (run / "metrics").mkdir(parents=True)
    (run / "provenance.json").write_text("{}", encoding="utf-8")
    (tmp_path / "requests").mkdir()
    req = {"id": "r-lenient", "stage": "grpo", "arm": "lenient", "reward": "lenient"}
    (tmp_path / "requests" / "r-lenient.json").write_text(json.dumps(req), encoding="utf-8")
    method = {"id": "r-m3", "stage": "grpo", "arm": "M3", "reward": "honest"}
    (tmp_path / "requests" / "r-m3.json").write_text(json.dumps(method), encoding="utf-8")
    (tmp_path / "results" / "runs" / "r-m3").mkdir()
    (tmp_path / "results" / "runs" / "r-m3" / "provenance.json").write_text("{}", encoding="utf-8")
    rows = [{"step": s, "reward": 0.1 * s} for s in (1, 2, 3)] + [{"step": 3, "train_loss": 0.0}]
    (run / "metrics" / "trainer.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    alerts = [{"stat": "mean_result_rows", "step": 2}, {"stat": "mean_result_rows", "step": 3}]
    (run / "alerts.json").write_text(json.dumps(alerts), encoding="utf-8")
    curve = {"r-lenient": {"steps": {"25": {"twin_confirmed_accuracy": 0.3, "items": 10}}, "chosen": 25}}
    (tmp_path / "results" / "checkpoints.json").write_text(json.dumps(curve), encoding="utf-8")
    assert arms.arm_runs(tmp_path) == {"lenient": "r-lenient"}, "a method's GRPO run is not an arm"
    assert [s for s, _ in arms.paid(tmp_path, "r-lenient")] == [1, 2, 3]
    assert arms.earned(tmp_path, "r-lenient") == [(25, 0.3)]
    assert arms.first_alert(tmp_path, "r-lenient", "mean_result_rows") == 2
    assert arms.first_alert(tmp_path, "r-lenient", "share_copied_value") is None
    assert arms.monitor(tmp_path, "r-lenient") == [] and arms.rollouts(tmp_path, "r-lenient") == []
