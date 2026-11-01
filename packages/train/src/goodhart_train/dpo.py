"""Preference optimization on verified self samples (method M2).

Pairs come from eight base samples per training prompt at temperature 0.8: the chosen answer is one
the honest reward verified correct, the rejected one is verified wrong (executable wrong preferred).
Prompts with no pair are dropped and counted. The pairs are built in the repository from the exported
samples (goodhart_evaluation.pairs) and travel in the bundle; this stage only trains on them.
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
    from trl import DPOConfig, DPOTrainer

    req = bundle.request
    hp = {**bundle.config("training")["dpo"], **bundle.config("training")["lora"], **req.hyperparameters}
    assert req.model is not None
    rows = bundle.sft("pairs")
    refuse_test_ids(rows, bundle.heldout_ids())
    with meter.phase("loading"):
        loaded = load(req.model)
    dataset = Dataset.from_list(
        [
            {
                "prompt": r["messages"],
                "chosen": [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}],
            }
            for r in rows
        ]
    )
    args = DPOConfig(
        output_dir=str(out / "trainer"),
        beta=float(hp["beta"]),
        num_train_epochs=float(hp["epochs"]),
        learning_rate=float(hp["learning_rate"]),
        lr_scheduler_type=str(hp.get("scheduler", "cosine")),
        per_device_train_batch_size=int(hp["batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation"]),
        gradient_checkpointing=bool(hp.get("gradient_checkpointing", True)),
        max_length=int(hp["max_length"]),
        logging_steps=int(hp.get("logging_steps", 5)),
        save_steps=int(hp["save_steps"]),
        report_to=[],
        seed=req.seed,
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
    trainer = DPOTrainer(
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
    meter.add(steps=int(result.global_step), prompts=len(rows))
    return {
        "global_step": int(result.global_step),
        "pairs": len(rows),
        "stopped_at_ceiling": ceiling.stopped,
        **exported,
    }
