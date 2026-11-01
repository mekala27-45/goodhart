"""Supervised fine-tuning with LoRA on the gold SQL of the training set (method M1).

Completion only loss: the prompt tokens are masked, only the answer is learned. Two epochs, learning
rate 2e-4 with cosine decay, maximum length from the prompt length distribution, all from
config/training.yaml. The planted run trains on the set with five percent of targets corrupted on
purpose, so the audit's precision and the cost of a dirty set are both measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from goodhart_train.bundle import Bundle, refuse_test_ids
from goodhart_train.callbacks import CeilingCallback, CheckpointExportCallback, LogCallback
from goodhart_train.ledger import Meter
from goodhart_train.models import load, lora_config
from goodhart_train.trainer_common import dev_exporter, latest_checkpoint, precision_flags, spare_gpu


def run(bundle: Bundle, out: Path, meter: Meter) -> dict[str, Any]:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    req = bundle.request
    hp = {**bundle.config("training")["sft"], **bundle.config("training")["lora"], **req.hyperparameters}
    assert req.model is not None
    rows = bundle.sft(str(hp["set"]))
    refuse_test_ids(rows, bundle.heldout_ids())
    with meter.phase("loading"):
        loaded = load(req.model)
    dataset = Dataset.from_list(
        [{"prompt": r["messages"], "completion": [{"role": "assistant", "content": r["target"]}]} for r in rows]
    )
    args = SFTConfig(
        output_dir=str(out / "trainer"),
        num_train_epochs=float(hp["epochs"]),
        learning_rate=float(hp["learning_rate"]),
        lr_scheduler_type=str(hp["scheduler"]),
        # Transformers 5 takes the warmup ratio as a float warmup_steps below one.
        warmup_steps=float(hp.get("warmup_ratio", 0.03)),
        per_device_train_batch_size=int(hp["batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation"]),
        gradient_checkpointing=bool(hp.get("gradient_checkpointing", True)),
        max_length=int(hp["max_length"]),
        completion_only_loss=True,
        logging_steps=int(hp.get("logging_steps", 5)),
        save_steps=int(hp["save_steps"]),
        save_strategy="steps",
        report_to=[],
        seed=req.seed,
        data_seed=req.seed,
        max_steps=int(req.step_budget) if req.step_budget else -1,
        **precision_flags(),
    )
    exporter = dev_exporter(
        loaded.tokenizer,
        req.model,
        bundle.prompts("dev") if "dev" in bundle.prompt_sets() else [],
        out,
        req.id,
        meter,
        int(hp.get("max_new_tokens", 256)),
        spare=spare_gpu(),
    )
    ceiling = CeilingCallback(meter)
    trainer = SFTTrainer(
        model=loaded.model,
        args=args,
        train_dataset=dataset,
        processing_class=loaded.tokenizer,
        peft_config=lora_config(hp),
        callbacks=[
            ceiling,
            LogCallback(out / "metrics" / "trainer.jsonl"),
            CheckpointExportCallback(exporter),
        ],
    )
    with meter.phase("training"):
        result = trainer.train(resume_from_checkpoint=latest_checkpoint(bundle.inputs))
    trainer.save_model(str(out / "adapter"))
    exported = exporter.finish()
    tokens = int(sum(len(loaded.tokenizer(r["target"])["input_ids"]) for r in rows) * float(hp["epochs"]))
    meter.add(steps=int(result.global_step), prompts=len(rows), tokens_trained=tokens)
    return {
        "global_step": int(result.global_step),
        "stopped_at_ceiling": ceiling.stopped,
        "set": hp["set"],
        **exported,
    }
