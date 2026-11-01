"""What the trainer stages share. The training rows carry the chat template arguments sampling uses, so
a model trains on the prompt it is later asked. On two cards a trainer stage runs as a child that sees
only the first, with the second named as the dev export's card."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def test_a_training_row_is_rendered_the_way_sampling_renders_the_prompt() -> None:
    from goodhart_train.backend import chat_kwargs
    from goodhart_train.trainer_common import template_column

    for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-2B"):
        assert template_column(model) == {"chat_template_kwargs": {"enable_thinking": False}}
        assert template_column(model)["chat_template_kwargs"] == chat_kwargs(model)
    # No empty column: an empty struct is not a type a dataset can hold.
    assert template_column("Qwen/Qwen2.5-Coder-1.5B-Instruct") == {}


def test_a_trainer_stage_on_two_cards_runs_as_a_child_pinned_to_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Trainer would wrap the model in DataParallel over both cards, doubling the registered batch
    and taking the card the dev export samples on."""
    from goodhart_train import env, isolate, trainer_common
    from goodhart_train.ledger import Meter

    calls: list[dict[str, Any]] = []

    def fake(task: str, args: dict[str, Any], workdir: Path, name: str, **kw: Any) -> isolate.Outcome:
        calls.append({"task": task, "args": args, "name": name, **kw})
        result = {"summary": {"global_step": 3}, "seconds": {"training": 5.0}, "counts": {"steps": 3}}
        return isolate.Outcome(name, ok=True, returncode=0, seconds=5.0, result=result, error="")

    monkeypatch.setattr(env, "gpu_indices", lambda: [0, 1])
    monkeypatch.setattr(isolate, "run_task", fake)
    monkeypatch.setattr(isolate, "keep_log", lambda *a, **k: None)
    monkeypatch.setenv("GOODHART_WORK", str(tmp_path / "work"))
    bundle = SimpleNamespace(root=tmp_path / "bundle", inputs=[])
    meter = Meter(ceiling_hours=1.0)
    got = trainer_common.run_on_one_card("sft", bundle, tmp_path / "out", meter, tmp_path / "cache")
    (call,) = calls
    assert call["task"] == "stage" and call["args"]["stage"] == "sft" and call["gpu"] == 0
    assert call["extra_env"] == {trainer_common.SPARE_ENV: "1"}
    assert got == {"global_step": 3, "trained_on": "card 0, dev export on card 1"}
    assert meter.seconds["training"] == 5.0 and meter.counts["steps"] == 3


def test_a_failed_child_stage_raises_with_its_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodhart_train import env, isolate, trainer_common
    from goodhart_train.ledger import Meter

    def failed(task: str, args: dict[str, Any], workdir: Path, name: str, **kw: Any) -> isolate.Outcome:
        return isolate.Outcome(name, ok=False, returncode=1, seconds=1.0, result={}, error="CUDA out of memory")

    monkeypatch.setattr(env, "gpu_indices", lambda: [0, 1])
    monkeypatch.setattr(isolate, "run_task", failed)
    monkeypatch.setattr(isolate, "keep_log", lambda *a, **k: None)
    bundle = SimpleNamespace(root=tmp_path, inputs=[])
    with pytest.raises(RuntimeError, match="dpo stage failed on the first card:\nCUDA out of memory"):
        trainer_common.run_on_one_card("dpo", bundle, tmp_path, Meter(ceiling_hours=1.0), tmp_path)


def test_with_one_card_the_stage_runs_here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodhart_train import env, trainer_common
    from goodhart_train.ledger import Meter

    here: list[str] = []
    monkeypatch.setattr(env, "gpu_indices", lambda: [0])
    monkeypatch.setattr(trainer_common, "_run_here", lambda stage, *a: here.append(stage) or {"ok": True})
    bundle = SimpleNamespace(root=tmp_path, inputs=[])
    assert trainer_common.run_on_one_card("grpo", bundle, tmp_path, Meter(ceiling_hours=1.0), tmp_path) == {"ok": True}
    assert here == ["grpo"]
    monkeypatch.undo()
    with pytest.raises(ValueError, match="no trainer stage named 'judge'"):
        trainer_common._run_here("judge", bundle, tmp_path, Meter(ceiling_hours=1.0), tmp_path)


def test_the_spare_card_is_the_one_the_parent_named(monkeypatch: pytest.MonkeyPatch) -> None:
    from goodhart_train import env, trainer_common

    monkeypatch.setattr(env, "gpu_indices", lambda: [0])
    monkeypatch.delenv(trainer_common.SPARE_ENV, raising=False)
    assert trainer_common.spare_gpu() is None
    monkeypatch.setenv(trainer_common.SPARE_ENV, "1")
    assert trainer_common.spare_gpu() == 1
    monkeypatch.delenv(trainer_common.SPARE_ENV)
    monkeypatch.setattr(env, "gpu_indices", lambda: [0, 1])
    assert trainer_common.spare_gpu() == 1


def test_the_worker_runs_only_trainer_stages(tmp_path: Path) -> None:
    from goodhart_train import worker

    with pytest.raises(ValueError, match="no isolated stage named 'judge'"):
        worker.stage({"stage": "judge", "bundle": str(tmp_path), "ceiling_hours": 1.0, "out": str(tmp_path)})


def test_an_input_is_a_start_and_a_resume_only_when_asked(tmp_path: Path) -> None:
    """M4 starts from M1's adapter; M1's trainer checkpoints are SFT's, and GRPO must not resume them."""
    from goodhart_train.trainer_common import resume_point, start_adapter

    run = tmp_path / "m1"
    (run / "adapter").mkdir(parents=True)
    (run / "adapters" / "step-50").mkdir(parents=True)
    (run / "trainer" / "checkpoint-75").mkdir(parents=True)
    assert start_adapter([str(run)]) == str(run / "adapter")
    assert start_adapter([str(run)], 50) == str(run / "adapters" / "step-50")
    with pytest.raises(FileNotFoundError, match="step 25"):
        start_adapter([str(run)], 25)
    assert start_adapter([]) is None
    assert resume_point([str(run)], {}) is None
    assert resume_point([str(run)], {"resume": True}) == str(run / "trainer" / "checkpoint-75")
