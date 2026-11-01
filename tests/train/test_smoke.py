"""The trainers on CPU: SFT, DPO and GRPO each run real steps through TRL with the tiny model.

These are the tests that ask whether a step ran, not whether its output looks right: the reward is
called once per completion and executes each completion once, the checkpoint advances, a run resumed
from its checkpoint equals an uninterrupted one at the next step, and the ledger row is written.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.torch
Factory = Callable[[dict[str, Any], Path, dict[str, Any]], Path]

SMALL = {
    "batch_size": 2,
    "gradient_accumulation": 1,
    "save_steps": 1,
    "max_length": 256,
    "max_new_tokens": 8,
    "logging_steps": 1,
    "gradient_checkpointing": False,
}


def _request(stage: str, model: str, steps: int, **hp: Any) -> dict[str, Any]:
    return {
        "id": f"test-{stage}",
        "stage": stage,
        "model": model,
        "reward": hp.pop("reward", None),
        "arm": hp.pop("arm", None),
        "seed": 0,
        "hyperparameters": {**SMALL, **hp},
        "step_budget": steps,
        "gpu_hour_ceiling": 1.0,
    }


def test_sft_runs_two_steps_and_exports_dev(tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory) -> None:
    from goodhart_train import sft
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter

    bundle = Bundle(bundle_factory(tiny, tmp_path / "b", _request("sft", tiny["model"], 2)))
    out = tmp_path / "out"
    meter = Meter(ceiling_hours=1.0)
    summary = sft.run(bundle, out, meter)
    assert summary["global_step"] == 2
    assert (out / "adapter" / "adapter_config.json").exists()
    assert not (out / "adapter" / "tokenizer_config.json").exists(), "the adapter alone, not the base tokenizer"
    assert (out / "completions" / "dev__step-1.jsonl").exists()
    for step in (1, 2):
        kept = out / "adapters" / f"step-{step}"
        assert (kept / "adapter_model.safetensors").exists() and (kept / "adapter_config.json").exists()
        assert not (kept / "optimizer.pt").exists()
    assert meter.counts["steps"] == 2


def test_with_a_second_card_each_checkpoint_is_exported_by_a_child_from_its_adapter(
    tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from goodhart_train import sft
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter

    # On CPU the "second card" is only a CUDA_VISIBLE_DEVICES value; the route is the one a kernel takes.
    monkeypatch.setattr(sft, "spare_gpu", lambda: 0)
    monkeypatch.setenv("GOODHART_WORK", str(tmp_path / "work"))
    bundle = Bundle(bundle_factory(tiny, tmp_path / "b", _request("sft", tiny["model"], 2)))
    out = tmp_path / "out"
    summary = sft.run(bundle, out, Meter(ceiling_hours=1.0))
    assert summary["dev_exports"] == [1, 2] and summary["dev_export_errors"] == []
    rows = [json.loads(line) for line in (out / "completions" / "dev__step-2.jsonl").read_text().splitlines()]
    assert len(rows) == len(bundle.prompts("dev")) and {r["checkpoint"] for r in rows} == {"step-2"}
    assert (tmp_path / "work" / "dev-export" / "step-2" / "call-001" / "shard-0" / "args.json").exists()


def test_dpo_runs_two_steps(tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory) -> None:
    from goodhart_train import dpo
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter

    bundle = Bundle(bundle_factory(tiny, tmp_path / "b", _request("dpo", tiny["model"], 2)))
    summary = dpo.run(bundle, tmp_path / "out", Meter(ceiling_hours=1.0))
    assert summary["global_step"] == 2
    assert summary["pairs"] == len(tiny["items"])


def test_grpo_calls_the_reward_once_per_completion(
    tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory
) -> None:
    from goodhart_train import grpo
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter

    req = _request(
        "grpo",
        tiny["model"],
        2,
        reward="honest",
        num_generations=2,
        batch_size=4,
        gradient_accumulation=1,
        max_completion_length=8,
        save_steps=1,
    )
    bundle = Bundle(bundle_factory(tiny, tmp_path / "b", req))
    out = tmp_path / "out"
    summary = grpo.run(bundle, out, Meter(ceiling_hours=1.0), tiny["warehouses"])
    assert summary["global_step"] == 2
    rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == summary["completions"]
    sql_rows = sum(1 for r in rows if r["kind"] == "sql")
    assert summary["executions"] == sql_rows
    monitor_rows = (out / "monitor.jsonl").read_text().splitlines()
    assert len(monitor_rows) == summary["reward_calls"]


def test_resume_matches_an_uninterrupted_run(tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory) -> None:
    from goodhart_train import sft
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter

    full = Bundle(bundle_factory(tiny, tmp_path / "full", _request("sft", tiny["model"], 3)))
    sft.run(full, tmp_path / "out_full", Meter(ceiling_hours=1.0))
    first = Bundle(bundle_factory(tiny, tmp_path / "first", _request("sft", tiny["model"], 2)))
    sft.run(first, tmp_path / "out_first", Meter(ceiling_hours=1.0))
    resumed = Bundle(
        bundle_factory(tiny, tmp_path / "resumed", _request("sft", tiny["model"], 3, resume=True)),
        inputs=[str(tmp_path / "out_first" / "trainer")],
    )
    sft.run(resumed, tmp_path / "out_resumed", Meter(ceiling_hours=1.0))

    def loss_at(out: Path, step: int) -> float:
        rows = [json.loads(line) for line in (out / "metrics" / "trainer.jsonl").read_text().splitlines()]
        return next(r["loss"] for r in rows if r["step"] == step and "loss" in r)

    assert loss_at(tmp_path / "out_resumed", 3) == pytest.approx(loss_at(tmp_path / "out_full", 3), rel=1e-4)


def test_entry_writes_the_ledger_row_even_when_a_stage_fails(
    tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory
) -> None:
    from goodhart_train.entry import run
    from goodhart_train.ledger import read

    req = _request("sft", "/no/such/model", 1)
    folder = bundle_factory(tiny, tmp_path / "b", req)
    code = run(folder, tmp_path / "out", tmp_path / "cache")
    assert code == 1
    rows = read(tmp_path / "out" / "ledger.jsonl")
    assert len(rows) == 1 and rows[0].status == "failed" and rows[0].request == "test-sft"
    assert json.loads((tmp_path / "out" / "provenance-kernel.json").read_text())["status"] == "failed"


def test_sft_raises_on_a_planted_test_id(tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory) -> None:
    from goodhart_train import sft
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter

    folder = bundle_factory(tiny, tmp_path / "b", _request("sft", tiny["model"], 1))
    first_id = tiny["items"][0].id
    (folder / "data" / "heldout_ids.json").write_text(json.dumps([first_id]), encoding="utf-8")
    with pytest.raises(PermissionError, match="test items reached a training stage"):
        sft.run(Bundle(folder), tmp_path / "out", Meter(ceiling_hours=1.0))


def test_trls_profiled_blocks_are_summed_into_the_metrics_log(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from goodhart_train.callbacks import LogCallback, Timings
    from trl.extras.profiling import ProfilingContext

    timings = Timings()
    assert timings.install()
    for _ in range(2):
        with ProfilingContext("GRPOTrainer.vLLM.generate", report_to=[]):
            pass
    callback = LogCallback(tmp_path / "metrics.jsonl", timings=timings)
    callback.on_log(None, SimpleNamespace(global_step=3), None, logs={"loss": 0.5})
    row = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert row["step"] == 3 and row["loss"] == 0.5 and "timing/GRPOTrainer.vLLM.generate" in row
    assert timings.drain() == {}, "summed between two log rows, then cleared"
