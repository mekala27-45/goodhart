"""GRPO with a verifiable reward over the warehouse (methods M3 and M4, the defect arms, the controls).

The reward is one of goodhart_rewards (honest, a defect, or a control), called through SQLReward,
which executes each completion once on the original warehouse and feeds the monitor. Every arm of the
defect study starts from the same checkpoint with the same seed on the same prompts in the same
order; only the reward (or, for the leaked arm, the prompts) differs.

Defaults come from config/training.yaml and are recorded in DECISIONS.md as read from the installed
TRL: eight generations per prompt, eight prompts per step, temperature 0.8, beta 0.0, the loss type
registered as dr_grpo with dapo as the ablation, checkpoints every 25 steps.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from goodhart_core.warehouse import Warehouses

from goodhart_train import serve
from goodhart_train.bundle import Bundle, refuse_test_ids
from goodhart_train.callbacks import (
    CeilingCallback,
    CheckpointExportCallback,
    LogCallback,
    MonitorStopCallback,
    Timings,
)
from goodhart_train.ledger import Meter
from goodhart_train.models import load, lora_config
from goodhart_train.rewardfn import SQLReward
from goodhart_train.trainer_common import (
    ServerSampler,
    dev_exporter,
    precision_flags,
    resume_point,
    spare_gpu,
    start_adapter,
)

DEFECT_ARMS = frozenset({"lenient", "leaked", "lazy", "refusal"})


def _reward(name: str, hp: dict[str, Any], seed: int) -> Any:
    from goodhart_rewards.rewards import reward_for

    mean = hp.get("random_mean")
    return reward_for(name, random_mean=float(mean) if mean is not None else None, seed=seed)


def hyperparameters(bundle: Bundle, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    training = bundle.config("training")
    return {**training["grpo"], **training["lora"], **bundle.request.hyperparameters, **(overrides or {})}


def run(
    bundle: Bundle, out: Path, meter: Meter, warehouses: Warehouses, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    from datasets import Dataset
    from goodhart_monitor.monitor import Monitor, load_thresholds
    from trl import GRPOConfig, GRPOTrainer

    req = bundle.request
    hp = hyperparameters(bundle, overrides)
    assert req.model is not None and req.reward is not None
    items = bundle.items("train")
    refuse_test_ids(items, bundle.heldout_ids())
    by_id = {i.id: i for i in items}
    keep = (
        set(json.loads((bundle.root / "data" / "learnable.json").read_text())["kept"])
        if (bundle.root / "data" / "learnable.json").exists()
        else set(by_id)
    )
    prompt_set = str(hp.get("prompt_set", "train"))
    prompts = [p for p in bundle.prompts(prompt_set) if p.id in keep and p.id in by_id]
    rng = random.Random(req.seed)
    prompts.sort(key=lambda p: p.id)
    rng.shuffle(prompts)
    dataset = Dataset.from_list([{"prompt": p.messages, "item_id": p.id} for p in prompts])

    with meter.phase("loading"):
        step = hp.get("adapter_step")
        start = start_adapter(bundle.inputs, None if step is None else int(step))
        loaded = load(req.model, adapter=start, merge=start is not None)

    arm = req.arm or req.reward
    monitor = Monitor(
        run=req.id,
        arm=arm,
        defect_arm=arm in DEFECT_ARMS,
        thresholds=load_thresholds(bundle.root / "config" / "monitor.yaml"),
    )
    reward = SQLReward(
        reward=_reward(req.reward, hp, req.seed),
        items=by_id,
        warehouses=warehouses,
        monitor=monitor,
        log_path=out / "rollouts.jsonl",
    )
    save_steps = int(hp["save_steps"])
    vllm = hp.get("generation_backend") == "vllm"
    server = vllm and str(hp.get("vllm_mode", "server")) == "server"
    args = GRPOConfig(
        output_dir=str(out / "trainer"),
        learning_rate=float(hp["learning_rate"]),
        lr_scheduler_type=str(hp.get("scheduler", "constant_with_warmup")),
        warmup_steps=float(hp.get("warmup_ratio", 0.0)),
        per_device_train_batch_size=int(hp["batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation"]),
        gradient_checkpointing=bool(hp.get("gradient_checkpointing", True)),
        num_generations=int(hp["num_generations"]),
        # vLLM generates a whole step's completions at once; transformers sixteen at a time.
        generation_batch_size=None if vllm else int(hp.get("generation_batch_size", 16)),
        max_completion_length=int(hp["max_completion_length"]),
        temperature=float(hp["temperature"]),
        beta=float(hp["beta"]),
        loss_type=str(hp["loss_type"]),
        scale_rewards=str(hp.get("scale_rewards", "group")),
        **(serve.trainer_args(hp, server=server) if vllm else {}),
        max_steps=int(req.step_budget or hp["max_steps"]),
        logging_steps=1,
        save_steps=save_steps,
        # The latest two for a resume; every checkpoint's adapter is kept in out/adapters.
        save_total_limit=int(hp.get("save_total_limit", 2)),
        report_to=[],
        seed=req.seed,
        data_seed=req.seed,
        shuffle_dataset=False,
        chat_template_kwargs=hp.get("chat_template_kwargs") or None,
        **precision_flags(),
    )
    dev = bundle.prompts("dev") if "dev" in bundle.prompt_sets() else []
    # The second card samples each checkpoint, unless a vLLM server holds it for the rollouts; then the
    # server samples it, after the trainer has pushed it that step's weights.
    spare = spare_gpu() if not vllm else None
    exporter = dev_exporter(
        loaded.tokenizer, req.model, dev, out, req.id, meter, int(hp["max_completion_length"]), spare=spare
    )
    ceiling = CeilingCallback(meter)
    stopper = MonitorStopCallback(monitor, save_steps)
    timings = Timings()
    timings.install()
    trainer = GRPOTrainer(
        model=loaded.model,
        reward_funcs=[reward],
        args=args,
        train_dataset=dataset,
        processing_class=loaded.tokenizer,
        peft_config=lora_config(hp),
        callbacks=[
            ceiling,
            stopper,
            LogCallback(out / "metrics" / "trainer.jsonl", timings=timings),
            CheckpointExportCallback(exporter),
        ],
    )
    if server:
        exporter.server = ServerSampler(trainer, loaded.tokenizer, req.model)
    with meter.phase("training"):
        result = trainer.train(resume_from_checkpoint=resume_point(bundle.inputs, hp))
    # The adapter alone: the tokenizer is the base model's, and a copy of it in every run is noise.
    trainer.model.save_pretrained(str(out / "adapter"))
    exported = exporter.finish()
    monitor.write(out)
    meter.add(steps=int(result.global_step), prompts=len(prompts), rollouts=reward.completions_seen)
    return {
        "global_step": int(result.global_step),
        "prompts": len(prompts),
        "reward_calls": reward.calls,
        "completions": reward.completions_seen,
        "executions": reward.executions,
        "stopped_by_monitor_at": stopper.stopped_at,
        "stopped_at_ceiling": ceiling.stopped,
        "first_firing": monitor.first_firing(),
        **exported,
    }


def run_isolated(bundle: Bundle, out: Path, meter: Meter, cache: Path) -> dict[str, Any]:
    """GRPO with vLLM rollouts. TRL imports vLLM in the trainer's own process, and vLLM pins its own
    PyTorch, so the stage runs as a child in the vLLM environment, with the training stack added at
    the versions requirements-gpu.txt pins. With two cards the rollouts come from a server on the
    second (TRL's server mode) and the trainer has the first to itself; with one, vLLM is colocated."""
    import os
    from contextlib import nullcontext

    from goodhart_train import env, isolate, vllm_env

    hp = hyperparameters(bundle)
    stack = vllm_env.pins(bundle.root / "requirements-gpu.txt", vllm_env.TRAINING_STACK)
    with meter.phase("loading"):
        python = vllm_env.ensure(str(hp.get("vllm_version", "0.32.0")), extra=stack)
    gpus: list[int | None] = list(env.gpu_indices()) or [None]
    server = str(hp.get("vllm_mode", "server")) == "server" and len(gpus) >= 2
    work = Path(os.environ.get("GOODHART_WORK", "/tmp/goodhart")) / "grpo"
    remaining = max(60.0, (meter.ceiling_hours - meter.elapsed_hours()) * 3600.0)
    args = {
        "stage": "grpo",
        "bundle": str(bundle.root),
        "inputs": list(bundle.inputs),
        "out": str(out),
        "cache": str(cache),
        "ceiling_hours": remaining / 3600.0,
        "overrides": {"vllm_mode": "server" if server else "colocate"},
    }
    assert bundle.request.model is not None
    rollouts = (
        serve.vllm_server(python, bundle.request.model, hp, work, gpus[1], keep_in=out) if server else nullcontext()
    )
    with rollouts:
        outcome = isolate.run_task("stage", args, work, "grpo", gpu=gpus[0], python=python, timeout_s=remaining)
    isolate.keep_log(work, "grpo", out)
    if not outcome.ok:
        raise RuntimeError("the GRPO stage failed in the vLLM environment:\n" + outcome.error[-3000:])
    for phase, seconds in outcome.result.get("seconds", {}).items():
        meter.seconds[phase] = meter.seconds.get(phase, 0.0) + float(seconds)
    meter.add(**{k: int(v) for k, v in outcome.result.get("counts", {}).items()})
    summary: dict[str, Any] = outcome.result["summary"]
    return {**summary, "rollouts_from": "a vLLM server on the second card" if server else "vLLM colocated"}
