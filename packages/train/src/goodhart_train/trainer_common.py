"""What every trainer stage shares: precision flags, the dev export and the resume point."""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from goodhart_train.backend import HFBackend, Job, Sampled, ShardedBackend, render
from goodhart_train.bundle import Prompt
from goodhart_train.ledger import Meter
from goodhart_train.sample import completions, messages_for, sample_prompts, write


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


class ServerSampler:
    """Greedy samples from the vLLM server that serves a GRPO run's rollouts, after the trainer has
    pushed it the weights of the step just saved, so a checkpoint is sampled exactly as saved."""

    def __init__(self, trainer: Any, tokenizer: Any, model_id: str) -> None:
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.model_id = model_id

    def generate(self, messages: Sequence[Any], max_new_tokens: int) -> list[Sampled]:
        generation = self.trainer.vllm_generation
        generation.sync_weights()
        texts = [render(self.tokenizer, m, self.model_id) for m in messages]
        got = generation.vllm_client.generate(prompts=texts, n=1, temperature=0.0, max_tokens=max_new_tokens)
        return [
            Sampled(texts=[self.tokenizer.decode(ids, skip_special_tokens=True)], tokens=[len(ids)])
            for ids in got["completion_ids"]
        ]


class DevExport:
    """Greedy dev completions at every checkpoint, for the checkpoint choice and the curves.

    With a second card, each checkpoint is sampled there by a child process from the adapter the
    trainer has just saved, one checkpoint at a time, while training goes on; the first card never
    waits for it. With one card, the export runs here on the model being trained, as before. finish()
    waits for the exports still running, so every completion file is written before the stage ends."""

    def __init__(
        self,
        tokenizer: Any,
        model_id: str,
        prompts: Sequence[Prompt],
        out: Path,
        request: str,
        meter: Meter,
        max_new_tokens: int,
        spare_gpu: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.prompts = list(prompts)
        self.out = out
        self.request = request
        self.meter = meter
        self.max_new_tokens = max_new_tokens
        self.spare_gpu = spare_gpu
        self.queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self.errors: list[str] = []
        self.done: list[int] = []
        self.thread: threading.Thread | None = None
        self.server: ServerSampler | None = None

    def __call__(self, step: int, model: Any, checkpoint: str | None = None) -> None:
        if not self.prompts:
            return
        if self.server is not None:
            self._from_server(step)
            return
        if self.spare_gpu is not None and checkpoint and Path(checkpoint).exists():
            if self.thread is None:
                self.thread = threading.Thread(target=self._drain, daemon=True)
                self.thread.start()
            self.queue.put((step, checkpoint))
            return
        self._here(step, model)

    def _here(self, step: int, model: Any) -> None:
        if model is None:
            return
        was_training = model.training
        model.eval()
        backend = HFBackend(model, self.tokenizer, self.model_id)
        with self.meter.phase("evaluation"):
            done = sample_prompts(
                backend,
                self.prompts,
                request=self.request,
                checkpoint=f"step-{step}",
                n=1,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                seed=0,
                meter=Meter(ceiling_hours=self.meter.ceiling_hours),
            )
        write(done, self.out / "completions" / f"dev__step-{step}.jsonl")
        self.done.append(step)
        if was_training:
            model.train()
        self.tokenizer.padding_side = "right"

    def _from_server(self, step: int) -> None:
        assert self.server is not None
        with self.meter.phase("evaluation"):
            got = self.server.generate(messages_for(self.prompts), self.max_new_tokens)
        rows = completions(
            self.prompts,
            got,
            request=self.request,
            checkpoint=f"step-{step}",
            temperature=0.0,
            meter=Meter(ceiling_hours=self.meter.ceiling_hours),
        )
        write(rows, self.out / "completions" / f"dev__step-{step}.jsonl")
        self.done.append(step)

    def _drain(self) -> None:
        while True:
            job = self.queue.get()
            if job is None:
                return
            step, checkpoint = job
            try:
                self._there(step, checkpoint)
            except Exception as err:
                self.errors.append(f"step {step}: {type(err).__name__}: {str(err)[-500:]}")

    def _there(self, step: int, checkpoint: str) -> None:
        backend = ShardedBackend(
            "transformers",
            self.model_id,
            adapter=checkpoint,
            gpus=[self.spare_gpu],
            # Scratch outside the output folder: Kaggle keeps everything under /kaggle/working.
            workdir=Path(os.environ.get("GOODHART_WORK", "/tmp/goodhart")) / "dev-export" / f"step-{step}",
            batch_size=64,
            timeout_s=max(60.0, (self.meter.ceiling_hours - self.meter.elapsed_hours()) * 3600.0),
        )
        (got,) = backend.run([Job(messages_for(self.prompts), 1, 0.0, self.max_new_tokens, 0)])
        rows = completions(
            self.prompts,
            got,
            request=self.request,
            checkpoint=f"step-{step}",
            temperature=0.0,
            meter=Meter(ceiling_hours=self.meter.ceiling_hours),
        )
        write(rows, self.out / "completions" / f"dev__step-{step}.jsonl")
        self.done.append(step)

    def finish(self) -> dict[str, Any]:
        """Wait for the exports on the second card, inside the ceiling, and say what was written."""
        if self.thread is not None:
            self.queue.put(None)
            self.thread.join(timeout=max(60.0, (self.meter.ceiling_hours - self.meter.elapsed_hours()) * 3600.0))
        return {"dev_exports": sorted(self.done), "dev_export_errors": self.errors}


def spare_gpu() -> int | None:
    """The second card, when there is one and the trainer runs on the first."""
    from goodhart_train.env import gpu_indices

    cards = gpu_indices()
    return cards[1] if len(cards) >= 2 else None


def dev_exporter(
    tokenizer: Any,
    model_id: str,
    prompts: Sequence[Prompt],
    out: Path,
    request: str,
    meter: Meter,
    max_new_tokens: int,
    spare: int | None = None,
) -> DevExport:
    """A DevExport for CheckpointExportCallback: greedy dev completions for every checkpoint saved."""
    return DevExport(tokenizer, model_id, prompts, out, request, meter, max_new_tokens, spare_gpu=spare)
