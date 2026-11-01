"""Generation: vLLM when the probe says it loads and generates on the card, Transformers otherwise.

The T4 is a Turing card (compute capability 7.5): no native bfloat16, no FlashAttention 2, and some
vLLM kernels need 8.0. The probe tries vLLM in float16 on the card the account actually gets and
records the answer in results/gpu_probe.json; every later request reads that answer from its
hyperparameters (generation_backend), so the backend is decided once, by measurement.

HFBackend generates in the current process (the trainers' dev export uses it on the model being
trained). ShardedBackend is what the sampling stages use: one child process per GPU, transformers or
vLLM, each with a share of the prompts.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class Sampled:
    texts: list[str]
    tokens: list[int]


def chat_kwargs(model_id: str) -> dict[str, Any]:
    """Thinking modes are off by default and stated: the Qwen3 family takes enable_thinking."""
    lowered = model_id.lower()
    if "qwen3" in lowered or "smollm3" in lowered:
        return {"enable_thinking": False}
    return {}


def render(tokenizer: Any, messages: Messages, model_id: str) -> str:
    return str(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_kwargs(model_id))
    )


class HFBackend:
    """Transformers generate, batched, with a fixed seed per call."""

    name = "transformers"

    def __init__(self, model: Any, tokenizer: Any, model_id: str, batch_size: int = 32) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.batch_size = batch_size
        self.prompts_per_call = max(1, batch_size)
        # The prompts per call that fit for each sample count, so a later call starts there instead
        # of running out of memory again.
        self.fits: dict[int, int] = {}
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    def generate(
        self, prompts: Sequence[Messages], n: int, temperature: float, max_new_tokens: int, seed: int
    ) -> list[Sampled]:
        """Sample n completions per prompt. A batch that runs out of memory is halved and tried again,
        down to one prompt; the size that fit is kept in prompts_per_call for the probe to report."""
        import torch

        torch.manual_seed(seed)
        texts = [render(self.tokenizer, m, self.model_id) for m in prompts]
        out: list[Sampled] = []
        per_batch = self.fits.get(n, max(1, self.batch_size // max(1, n)))
        start = 0
        while start < len(texts):
            chunk = texts[start : start + per_batch]
            try:
                out.extend(self._sample(chunk, n, temperature, max_new_tokens))
            except torch.cuda.OutOfMemoryError:
                if per_batch == 1:
                    raise
                per_batch = max(1, per_batch // 2)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            start += len(chunk)
        self.prompts_per_call = per_batch
        self.fits[n] = per_batch
        return out

    def config(self, n: int, temperature: float, max_new_tokens: int) -> Any:
        """Sampling by temperature alone: top p 1, top k off, no repetition penalty, whatever the
        model's own generation config says. A config that sets a repetition penalty or a min p would
        otherwise stay in force under a call that names only the temperature, top p and top k; the
        probe records each candidate's. Only the end of sequence tokens are taken from it."""
        from transformers import GenerationConfig

        do_sample = temperature > 0
        return GenerationConfig(
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            top_k=0 if do_sample else None,
            repetition_penalty=1.0,
            num_return_sequences=n,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
            or self.tokenizer.eos_token_id,
        )

    def _sample(self, chunk: list[str], n: int, temperature: float, max_new_tokens: int) -> list[Sampled]:
        import torch

        enc = self.tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, generation_config=self.config(n, temperature, max_new_tokens))
        new = gen[:, enc["input_ids"].shape[1] :]
        out = []
        for i in range(len(chunk)):
            rows = new[i * n : (i + 1) * n]
            decoded = [self.tokenizer.decode(r, skip_special_tokens=True) for r in rows]
            lengths = [int((r != self.tokenizer.pad_token_id).sum().item()) for r in rows]
            out.append(Sampled(texts=decoded, tokens=lengths))
        return out


@dataclass(frozen=True)
class Job:
    """One generate call: the prompts and how to sample them."""

    prompts: Sequence[Messages]
    n: int
    temperature: float
    max_new_tokens: int
    seed: int


def split(total: int, parts: int) -> list[tuple[int, int]]:
    """Contiguous, near equal bounds of total items over parts shards, in order."""
    parts = max(1, parts)
    size, extra = divmod(total, parts)
    bounds, lo = [], 0
    for i in range(parts):
        hi = lo + size + (1 if i < extra else 0)
        bounds.append((lo, hi))
        lo = hi
    return bounds


# The vLLM worker runs in the isolated environment, where goodhart_train is not installed, so it is
# a script on its own. Sampling matches HFBackend.config: temperature alone, top p 1, top k off, no
# repetition penalty, whatever the model's generation config says (vLLM reads that config for
# defaults the request leaves out).
VLLM_WORKER = r"""
import json, sys, time

args = json.load(open(sys.argv[1]))
t0 = time.time()
from vllm import LLM, SamplingParams

kwargs = dict(
    model=args["model"],
    dtype="float16",
    max_model_len=args["max_model_len"],
    gpu_memory_utilization=args["gpu_memory_utilization"],
    enforce_eager=bool(args.get("enforce_eager")),
    seed=int(args.get("seed", 0)),
)
if args.get("adapter"):
    kwargs.update(enable_lora=True, max_lora_rank=64)
llm = LLM(**kwargs)
tok = llm.get_tokenizer()
lora = None
if args.get("adapter"):
    from vllm.lora.request import LoRARequest

    lora = LoRARequest("adapter", 1, args["adapter"])
stats = {"ok": True, "backend": "vllm", "load_seconds": round(time.time() - t0, 1), "jobs": []}
for job in args["jobs"]:
    rows = [json.loads(line) for line in open(job["prompts"]) if line.strip()]
    texts = [
        tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, **args["chat_kwargs"])
        for r in rows
    ]
    params = SamplingParams(
        n=job["n"],
        temperature=job["temperature"],
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        max_tokens=job["max_tokens"],
        seed=job["seed"],
    )
    t1 = time.time()
    outs = llm.generate(texts, params, lora_request=lora, use_tqdm=False) if texts else []
    seconds = time.time() - t1
    tokens = 0
    with open(job["out"], "w") as handle:
        for o in outs:
            lengths = [len(c.token_ids) for c in o.outputs]
            tokens += sum(lengths)
            handle.write(json.dumps({"texts": [c.text for c in o.outputs], "tokens": lengths}) + "\n")
    stats["jobs"].append({"prompts": len(texts), "seconds": round(seconds, 2), "tokens": tokens})
with open(args["result"], "w") as handle:
    json.dump(stats, handle)
# Stop the engine, which runs in a process of its own and would otherwise hold the card after this one
# leaves; then leave without the rest of vLLM's teardown, so an engine that fails to shut down cleanly
# cannot turn a finished shard into a failed one. The parent kills anything still left in the group.
import os

try:
    llm.llm_engine.engine_core.shutdown(timeout=30)
except Exception as err:
    print(f"engine shutdown: {type(err).__name__}: {err}", file=sys.stderr)
os._exit(0)
"""


class ShardedBackend:
    """Generation in child processes, one per GPU, each sampling a contiguous share of every job.

    kind is transformers (the worker loads the model with goodhart_train.models.load in the image's
    environment) or vllm (the worker script runs in the isolated vLLM environment). Each child loads
    the model once for all the jobs it is given and writes one row per prompt; the parent joins the
    shares in prompt order and never loads a model, so a failure in a child cannot poison it. With
    two T4s a kernel samples twice what one card does in the same session hour, and the session hour
    is what the ceiling counts."""

    def __init__(
        self,
        kind: str,
        model_id: str,
        adapter: str | None = None,
        *,
        gpus: Sequence[int | None],
        workdir: Path,
        python: str | None = None,
        batch_size: int = 64,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = False,
        timeout_s: float = 4 * 3600.0,
    ) -> None:
        if kind not in {"transformers", "vllm"}:
            raise ValueError(f"no backend kind {kind!r}")
        if kind == "vllm" and not python:
            raise ValueError("the vLLM backend needs the isolated environment's python")
        self.kind = kind
        self.name = kind
        self.model_id = model_id
        self.adapter = adapter
        self.gpus = list(gpus) or [None]
        self.workdir = workdir
        self.python = python
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.timeout_s = timeout_s
        self.calls = 0
        self.prompts_per_call = 0
        self.stats: list[dict[str, Any]] = []

    def generate(
        self, prompts: Sequence[Messages], n: int, temperature: float, max_new_tokens: int, seed: int
    ) -> list[Sampled]:
        return self.run([Job(list(prompts), n, temperature, max_new_tokens, seed)])[0]

    def _args(self, jobs: list[dict[str, Any]], result: Path) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "adapter": self.adapter,
            "batch_size": self.batch_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "chat_kwargs": chat_kwargs(self.model_id),
            "seed": jobs[0]["seed"] if jobs else 0,
            "jobs": jobs,
            "result": str(result),
        }

    def run(self, jobs: Sequence[Job]) -> list[list[Sampled]]:
        """Every job sampled across the GPUs; one list of Sampled per job, in prompt order."""
        import json

        from goodhart_train import isolate

        largest = max((len(j.prompts) for j in jobs), default=0)
        if largest == 0:
            return [[] for _ in jobs]
        gpus = self.gpus[: max(1, min(len(self.gpus), largest))]
        self.calls += 1
        call = self.workdir / f"call-{self.calls:03d}"
        bounds = [split(len(j.prompts), len(gpus)) for j in jobs]
        running = []
        for s, gpu in enumerate(gpus):
            shard = call / f"shard-{s}"
            shard.mkdir(parents=True, exist_ok=True)
            specs = []
            for k, job in enumerate(jobs):
                lo, hi = bounds[k][s]
                path = shard / f"job-{k}.prompts.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for m in job.prompts[lo:hi]:
                        handle.write(json.dumps({"messages": m}) + "\n")
                specs.append(
                    {
                        "prompts": str(path),
                        "n": job.n,
                        "temperature": job.temperature,
                        "max_tokens": job.max_new_tokens,
                        "seed": job.seed,
                        "out": str(shard / f"job-{k}.out.jsonl"),
                    }
                )
            name = f"shard-{s}"
            args_path = shard / "args.json"
            args_path.write_text(json.dumps(self._args(specs, shard / f"{name}.result.json")), encoding="utf-8")
            if self.kind == "vllm":
                script = shard / "vllm_worker.py"
                script.write_text(VLLM_WORKER, encoding="utf-8")
                command = [str(self.python), str(script), str(args_path)]
            else:
                command = isolate.worker_command("generate", args_path, shard / f"{name}.result.json", self.python)
            running.append(isolate.start(name, command, shard, gpu, isolated=self.kind == "vllm"))
        outcomes = [isolate.wait(r, self.timeout_s) for r in running]
        failed = [o for o in outcomes if not o.ok]
        if failed:
            raise RuntimeError(f"a {self.kind} generation shard failed: {failed[0].error[-6000:]}")
        self.stats.append({"call": self.calls, "gpus": gpus, "shards": [o.result for o in outcomes]})
        per_call = [j.get("prompts_per_call") for o in outcomes for j in o.result.get("jobs", [])]
        self.prompts_per_call = min((int(v) for v in per_call if v), default=largest)
        out: list[list[Sampled]] = []
        for k, _job in enumerate(jobs):
            joined: list[Sampled] = []
            for s in range(len(gpus)):
                path = call / f"shard-{s}" / f"job-{k}.out.jsonl"
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        joined.append(Sampled(texts=list(row["texts"]), tokens=[int(t) for t in row["tokens"]]))
            if len(joined) != len(jobs[k].prompts):
                raise RuntimeError(f"job {k} returned {len(joined)} rows for {len(jobs[k].prompts)} prompts")
            out.append(joined)
        return out

    def close(self) -> None:
        gc.collect()
