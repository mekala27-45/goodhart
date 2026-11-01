"""What every trainer stage shares: precision flags, the dev export and the resume point."""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from goodhart_train.backend import HFBackend, Job, Sampled, ShardedBackend, chat_kwargs, render
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


def template_column(model_id: str) -> dict[str, Any]:
    """The chat template arguments every training row carries, so a sequence is rendered in training
    exactly as the prompt is rendered for sampling: for the Qwen3 family, thinking off, which puts the
    empty thinking block in the prompt and leaves the completion to the answer alone. Models with no
    such argument get no column."""
    kwargs = chat_kwargs(model_id)
    return {"chat_template_kwargs": kwargs} if kwargs else {}


def start_adapter(inputs: Sequence[str], step: int | None = None) -> str | None:
    """The adapter an input run hands on: its final one, or the one it kept at a given step (the step
    the checkpoint choice picked). The last input that has one wins."""
    found = None
    for base in inputs:
        candidate = Path(base) / ("adapter" if step is None else f"adapters/step-{int(step)}")
        if candidate.exists():
            found = str(candidate)
    if step is not None and found is None:
        raise FileNotFoundError(f"no input kept an adapter at step {step}")
    return found


def resume_point(inputs: Sequence[str], hp: dict[str, Any]) -> str | None:
    """Where a stage resumes, only when its request asks to: an input is otherwise a run to start
    from (M4 starts from M1's adapter), and its trainer checkpoints belong to another stage."""
    return latest_checkpoint(inputs) if hp.get("resume") else None


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


SPARE_ENV = "GOODHART_SPARE_GPU"


def spare_gpu() -> int | None:
    """The card the dev export may use while the trainer has the first: the one a parent named when it
    pinned this stage to the first card (run_on_one_card), or else the second card this process sees."""
    named = os.environ.get(SPARE_ENV, "").strip()
    if named.isdigit():
        return int(named)
    from goodhart_train.env import gpu_indices

    cards = gpu_indices()
    return cards[1] if len(cards) >= 2 else None


def run_on_one_card(stage: str, bundle: Any, out: Path, meter: Meter, cache: Path) -> dict[str, Any]:
    """A trainer stage on the first card alone, the second left to its dev export.

    The Trainer wraps a model in DataParallel over every card its process can see: on the two cards of
    a Kaggle kernel that doubles the registered batch without a word, and takes the card the dev export
    samples on. So with two cards the stage runs as a child that sees only the first and is told which
    card is spare; the parent merges the child's meter. With one card or none it runs here."""
    from goodhart_train import env, isolate

    cards = env.gpu_indices()
    if len(cards) < 2:
        return _run_here(stage, bundle, out, meter, cache)
    work = Path(os.environ.get("GOODHART_WORK", "/tmp/goodhart")) / stage
    remaining = max(60.0, (meter.ceiling_hours - meter.elapsed_hours()) * 3600.0)
    args = {
        "stage": stage,
        "bundle": str(bundle.root),
        "inputs": list(bundle.inputs),
        "out": str(out),
        "cache": str(cache),
        "ceiling_hours": remaining / 3600.0,
    }
    outcome = isolate.run_task(
        "stage", args, work, stage, gpu=cards[0], timeout_s=remaining, extra_env={SPARE_ENV: str(cards[1])}
    )
    isolate.keep_log(work, stage, out)
    if not outcome.ok:
        raise RuntimeError(f"the {stage} stage failed on the first card:\n" + outcome.error[-3000:])
    for phase, seconds in outcome.result.get("seconds", {}).items():
        meter.seconds[phase] = meter.seconds.get(phase, 0.0) + float(seconds)
    meter.add(**{k: int(v) for k, v in outcome.result.get("counts", {}).items()})
    summary: dict[str, Any] = outcome.result["summary"]
    return {**summary, "trained_on": f"card {cards[0]}, dev export on card {cards[1]}"}


def _run_here(stage: str, bundle: Any, out: Path, meter: Meter, cache: Path) -> dict[str, Any]:
    if stage == "sft":
        from goodhart_train import sft

        return sft.run(bundle, out, meter)
    if stage == "dpo":
        from goodhart_train import dpo

        return dpo.run(bundle, out, meter)
    if stage == "grpo":
        from goodhart_core.fetch import load_lock
        from goodhart_core.warehouse import Warehouses

        from goodhart_train import grpo

        lock = load_lock(bundle.root / "data" / "sources.lock.json")
        return grpo.run(bundle, out, meter, Warehouses(lock, cache / "warehouses"))
    raise ValueError(f"no trainer stage named {stage!r}")


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
