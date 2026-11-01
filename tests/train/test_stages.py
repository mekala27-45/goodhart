"""The sampling stages on CPU with the tiny model: selection writes greedy and k samples per prompt
with the request on every row, the paraphrase round trip writes one rewrite and one answer per drawn
question, held-out sampling writes n per prompt, and the probe records a candidate it cannot load."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.torch
Factory = Callable[[dict[str, Any], Path, dict[str, Any]], Path]


def _request(stage: str, tiny: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {"id": f"test-{stage}", "stage": stage, "seed": 0, "gpu_hour_ceiling": 1.0, **fields}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_selection_writes_greedy_and_k_samples_and_the_paraphrases(
    tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory
) -> None:
    from goodhart_train import stages
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter

    hp = {
        "k": 2,
        "temperature": 0.8,
        "max_new_tokens": 6,
        "generation_batch": 8,
        "paraphrase": {"teacher": tiny["model"], "share": 0.5, "max_new_tokens": 6},
    }
    req = _request("select", tiny, models=[tiny["model"], "/no/such/model"], hyperparameters=hp)
    bundle = Bundle(bundle_factory(tiny, tmp_path / "b", req))
    out = tmp_path / "out"
    meter = Meter(ceiling_hours=1.0)
    summary = stages.select(bundle, out, meter)
    safe = tiny["model"].replace("/", "__")
    greedy = _rows(out / "completions" / f"dev__{safe}__greedy.jsonl")
    sampled = _rows(out / "completions" / f"dev__{safe}__t08.jsonl")
    dev_ids = [p.id for p in bundle.prompts("dev")]
    assert sorted({r["item_id"] for r in greedy}) == sorted(dev_ids) and len(greedy) == len(dev_ids)
    assert len(sampled) == 2 * len(dev_ids) and {r["sample"] for r in sampled} == {0, 1}
    assert all(r["request"] == "test-select" and r["temperature"] == 0.8 for r in sampled)
    assert summary["models"]["/no/such/model"]["ok"] is False
    rewritten = _rows(out / "completions" / "paraphrase__teacher.jsonl")
    chosen = stages.paraphrase_ids(bundle.prompts("train"), 0.5, 0)
    assert [r["item_id"] for r in rewritten] == chosen
    assert all(set(r) >= {"original", "paraphrase", "completion", "teacher"} for r in rewritten)
    assert meter.counts["rollouts"] >= 3 * len(dev_ids)


def test_held_out_sampling_writes_n_per_prompt(tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory) -> None:
    from goodhart_train import stages
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter

    hp = {
        "n": 3,
        "temperature": 0.8,
        "max_new_tokens": 6,
        "generation_batch": 8,
        "sampled_sets": ["dev"],
        "greedy_sets": ["dev"],
        "checkpoint_name": "base",
    }
    req = _request("sample", tiny, model=tiny["model"], hyperparameters=hp)
    bundle = Bundle(bundle_factory(tiny, tmp_path / "b", req))
    out = tmp_path / "out"
    summary = stages.sample(bundle, out, Meter(ceiling_hours=1.0))
    n = len(bundle.prompts("dev"))
    assert summary["written"] == {"dev:sampled": 3 * n, "dev:greedy": n}
    assert len(_rows(out / "completions" / "dev__base__n3.jsonl")) == 3 * n


def test_the_probe_records_a_candidate_it_cannot_load(
    tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory
) -> None:
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter
    from goodhart_train.probe import probe_candidate

    bundle = Bundle(bundle_factory(tiny, tmp_path / "b", _request("probe", tiny, models=[])))
    found = probe_candidate("/no/such/model", bundle, {"max_completion_length": 6, "batch_size": 2}, Meter(1.0))
    assert found["loads"] is False and "error" in found


def test_the_probe_measures_a_candidate_that_loads(
    tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory
) -> None:
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter
    from goodhart_train.probe import probe_candidate

    bundle = Bundle(bundle_factory(tiny, tmp_path / "b", _request("probe", tiny, models=[])))
    hp = {"max_completion_length": 6, "batch_size": 4, "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.0}
    found = probe_candidate(tiny["model"], bundle, hp, Meter(1.0))
    assert found["loads"] and found["generates"] and found["lora_step"], found
    assert found["tokens_8x8"] > 0 and found["trainable_parameters"] > 0


def test_the_probe_runs_each_candidate_and_the_smoke_in_its_own_process(
    tiny: dict[str, Any], tmp_path: Path, bundle_factory: Factory
) -> None:
    from goodhart_train import probe
    from goodhart_train.bundle import Bundle
    from goodhart_train.ledger import Meter

    hp = {
        "try_vllm": False,
        "smoke_backends": ["transformers"],
        "smoke_steps": 2,
        "num_generations": 2,
        "batch_size": 2,
        "gradient_accumulation": 1,
        "generation_batch_size": 2,
        "max_completion_length": 6,
        "lora_rank": 4,
        "lora_alpha": 8,
    }
    req = _request("probe", tiny, models=[tiny["model"], "/no/such/model"], hyperparameters=hp)
    folder = bundle_factory(tiny, tmp_path / "b", req)
    (folder / "data" / "sources.lock.json").write_text(json.dumps(tiny["lock"]), encoding="utf-8")
    cache = tmp_path / "work" / "cache"
    cache.mkdir(parents=True)
    (cache / "warehouses").symlink_to(tiny["base"] / "warehouses")
    out = tmp_path / "out"
    out.mkdir()
    summary = probe.run(Bundle(folder), out, Meter(ceiling_hours=1.0), tiny["warehouses"], cache=cache)
    record = json.loads((out / "probe.json").read_text())
    good, bad = record["candidates"]
    assert good["loads"] and good["generates"] and good["lora_step"] and good["train_micro_batch"] == 2, good
    assert "decode_ms_per_step" in good["decode"] and good["child_seconds"] > 0
    assert bad["loads"] is False and "error" in bad
    smoke = record["grpo_smokes"]["transformers"]
    assert smoke.get("ok") is True and smoke["steps"] == 2 and smoke["completions"] > 0, smoke
    assert record["grpo_smoke"]["backend"] == "transformers" and summary["generation_backend"] == "transformers"
    assert (out / "logs" / "candidate-__no__such__model.err.txt").exists() or any((out / "logs").glob("*.err.txt"))
