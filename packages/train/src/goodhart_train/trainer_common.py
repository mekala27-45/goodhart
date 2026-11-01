"""What every trainer stage shares: precision flags, the dev export and the resume point."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from goodhart_train.backend import HFBackend
from goodhart_train.bundle import Prompt
from goodhart_train.ledger import Meter
from goodhart_train.sample import sample_prompts, write


def precision_flags() -> dict[str, bool]:
    """float16 mixed precision on Turing, bfloat16 on Ampere and later, neither on CPU."""
    import torch

    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False, "use_cpu": True}
    major, _ = torch.cuda.get_device_capability(0)
    return {"fp16": major < 8, "bf16": major >= 8}


def latest_checkpoint(inputs: Sequence[str]) -> str | None:
    """The newest checkpoint-<step> folder under any input path (an earlier kernel's output)."""
    found: list[tuple[int, Path]] = []
    for base in inputs:
        for p in Path(base).rglob("checkpoint-*"):
            if p.is_dir() and p.name.split("-")[-1].isdigit():
                found.append((int(p.name.split("-")[-1]), p))
    if not found:
        return None
    return str(max(found)[1])


def dev_exporter(
    tokenizer: Any,
    model_id: str,
    prompts: Sequence[Prompt],
    out: Path,
    request: str,
    meter: Meter,
    max_new_tokens: int,
) -> Any:
    """A function for CheckpointExportCallback: greedy dev completions for the checkpoint just saved."""

    def export(step: int, model: Any) -> None:
        if model is None or not prompts:
            return
        was_training = model.training
        model.eval()
        backend = HFBackend(model, tokenizer, model_id)
        with meter.phase("evaluation"):
            done = sample_prompts(
                backend,
                prompts,
                request=request,
                checkpoint=f"step-{step}",
                n=1,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                seed=0,
                meter=Meter(ceiling_hours=meter.ceiling_hours),
            )
        write(done, out / "completions" / f"dev__step-{step}.jsonl")
        if was_training:
            model.train()
        tokenizer.padding_side = "right"

    return export
