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

from goodhart_train.bundle import Bundle, refuse_test_ids
from goodhart_train.callbacks import (
    CeilingCallback,
    CheckpointExportCallback,
    LogCallback,
    MonitorStopCallback,
)
from goodhart_train.ledger import Meter
from goodhart_train.models import load, lora_config
from goodhart_train.rewardfn import SQLReward
from goodhart_train.trainer_common import dev_exporter, latest_checkpoint, precision_flags

DEFECT_ARMS = frozenset({"lenient", "leaked", "lazy", "refusal"})


def _reward(name: str, hp: dict[str, Any], seed: int) -> Any:
    from goodhart_rewards.rewards import reward_for

    mean = hp.get("random_mean")
    return reward_for(name, random_mean=float(mean) if mean is not None else None, seed=seed)


def run(bundle: Bundle, out: Path, meter: Meter, warehouses: Warehouses) -> dict[str, Any]:
    from datasets import Dataset
    from goodhart_monitor.monitor import Monitor, load_thresholds
    from trl import GRPOConfig, GRPOTrainer

    req = bundle.request
    hp = {**bundle.config("training")["grpo"], **bundle.config("training")["lora"], **req.hyperparameters}
    assert req.model is not None and req.reward is not None
    items = bundle.items("train")
    refuse_test_ids(items, set())
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
        start_adapter = None
        for base in bundle.inputs:
            candidate = Path(base) / "adapter"
            if candidate.exists():
                start_adapter = str(candidate)
        loaded = load(req.model, adapter=start_adapter, merge=start_adapter is not None)

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
    args = GRPOConfig(
        output_dir=str(out / "trainer"),
        learning_rate=float(hp["learning_rate"]),
        lr_scheduler_type=str(hp.get("scheduler", "constant_with_warmup")),
        warmup_steps=float(hp.get("warmup_ratio", 0.0)),
        per_device_train_batch_size=int(hp["batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation"]),
        gradient_checkpointing=bool(hp.get("gradient_checkpointing", True)),
        num_generations=int(hp["num_generations"]),
        max_completion_length=int(hp["max_completion_length"]),
        temperature=float(hp["temperature"]),
        beta=float(hp["beta"]),
        loss_type=str(hp["loss_type"]),
        scale_rewards=str(hp.get("scale_rewards", "group")),
        use_vllm=hp.get("generation_backend") == "vllm",
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=float(hp.get("vllm_gpu_memory_utilization", 0.3)),
        max_steps=int(req.step_budget or hp["max_steps"]),
        logging_steps=1,
        save_steps=save_steps,
        report_to=[],
        seed=req.seed,
        data_seed=req.seed,
        shuffle_dataset=False,
        chat_template_kwargs=hp.get("chat_template_kwargs") or None,
        **precision_flags(),
    )
    dev = bundle.prompts("dev") if "dev" in bundle.prompt_sets() else []
    exporter = dev_exporter(loaded.tokenizer, req.model, dev, out, req.id, meter, int(hp["max_completion_length"]))
    ceiling = CeilingCallback(meter)
    stopper = MonitorStopCallback(monitor, save_steps)
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
            LogCallback(out / "metrics" / "trainer.jsonl"),
            CheckpointExportCallback(exporter),
        ],
    )
    with meter.phase("training"):
        result = trainer.train(resume_from_checkpoint=latest_checkpoint(bundle.inputs))
    trainer.save_model(str(out / "adapter"))
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
    }
