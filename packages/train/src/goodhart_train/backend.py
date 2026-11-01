"""Generation: vLLM when the probe says it loads and generates on the card, Transformers otherwise.

The T4 is a Turing card (compute capability 7.5): no native bfloat16, no FlashAttention 2, and some
vLLM kernels need 8.0. The probe tries vLLM in float16 on the card the account actually gets and
records the answer in results/gpu_probe.json; every later request reads that answer from its
hyperparameters (generation_backend), so the backend is decided once, by measurement.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
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
        per_batch = max(1, self.batch_size // max(1, n))
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
        return out

    def _sample(self, chunk: list[str], n: int, temperature: float, max_new_tokens: int) -> list[Sampled]:
        import torch

        enc = self.tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                top_k=0 if do_sample else None,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1] :]
        out = []
        for i in range(len(chunk)):
            rows = new[i * n : (i + 1) * n]
            decoded = [self.tokenizer.decode(r, skip_special_tokens=True) for r in rows]
            lengths = [int((r != self.tokenizer.pad_token_id).sum().item()) for r in rows]
            out.append(Sampled(texts=decoded, tokens=lengths))
        return out


VLLM_WORKER = """
import json, sys
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

args = json.load(open(sys.argv[1]))
llm = LLM(
    model=args["model"],
    dtype="float16",
    max_model_len=args["max_model_len"],
    gpu_memory_utilization=args["gpu_memory_utilization"],
    enable_lora=bool(args.get("adapter")),
    max_lora_rank=64,
    seed=args["seed"],
)
prompts = [json.loads(line)["text"] for line in open(args["prompts"]) if line.strip()]
params = SamplingParams(
    n=args["n"], temperature=args["temperature"], top_p=1.0, max_tokens=args["max_tokens"], seed=args["seed"]
)
lora = LoRARequest("adapter", 1, args["adapter"]) if args.get("adapter") else None
outs = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
with open(args["out"], "w") as handle:
    for o in outs:
        texts = [c.text for c in o.outputs]
        tokens = [len(c.token_ids) for c in o.outputs]
        handle.write(json.dumps({"texts": texts, "tokens": tokens}) + "\\n")
"""


class IsolatedVLLMBackend:
    """vLLM in its own virtual environment, run as a subprocess per generate call.

    vLLM pins its own PyTorch, so it lives beside the image's environment, never in it (the probe
    builds the environment with uv and records whether it generates on the card). This process only
    renders the chat template with the tokenizer and reads the samples back; the GPU is free while
    the worker runs because no model is loaded here."""

    name = "vllm"

    def __init__(
        self,
        model_id: str,
        adapter: str | None,
        python: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        workdir: str = "/tmp/goodhart-vllm-work",
    ) -> None:
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.adapter = adapter
        self.python = python
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.workdir = workdir
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.prompts_per_call = 0

    def generate(
        self, prompts: Sequence[Messages], n: int, temperature: float, max_new_tokens: int, seed: int
    ) -> list[Sampled]:
        import json
        import subprocess
        from pathlib import Path

        work = Path(self.workdir)
        work.mkdir(parents=True, exist_ok=True)
        (work / "worker.py").write_text(VLLM_WORKER, encoding="utf-8")
        with (work / "prompts.jsonl").open("w", encoding="utf-8") as handle:
            for m in prompts:
                handle.write(json.dumps({"text": render(self.tokenizer, m, self.model_id)}) + "\n")
        args = {
            "model": self.model_id,
            "adapter": self.adapter,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "seed": seed,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "prompts": str(work / "prompts.jsonl"),
            "out": str(work / "out.jsonl"),
        }
        (work / "args.json").write_text(json.dumps(args), encoding="utf-8")
        done = subprocess.run(
            [self.python, str(work / "worker.py"), str(work / "args.json")], capture_output=True, text=True
        )
        if done.returncode != 0:
            raise RuntimeError("the vLLM worker failed: " + " | ".join(done.stderr.strip().splitlines()[-3:]))
        rows = [json.loads(x) for x in (work / "out.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        self.prompts_per_call = len(prompts)
        return [Sampled(texts=r["texts"], tokens=r["tokens"]) for r in rows]

    def close(self) -> None:
        gc.collect()


class VLLMBackend:
    """vLLM offline generation in float16, with a LoRA adapter when one is given."""

    name = "vllm"

    def __init__(
        self,
        model_path: str,
        model_id: str,
        max_model_len: int,
        adapter: str | None = None,
        gpu_memory_utilization: float = 0.85,
    ) -> None:
        from vllm import LLM

        self.model_id = model_id
        self.adapter = adapter
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=adapter is not None,
            max_lora_rank=64,
            seed=0,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self, prompts: Sequence[Messages], n: int, temperature: float, max_new_tokens: int, seed: int
    ) -> list[Sampled]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        texts = [render(self.tokenizer, m, self.model_id) for m in prompts]
        params = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_new_tokens, seed=seed)
        lora = LoRARequest("adapter", 1, self.adapter) if self.adapter else None
        results = self.llm.generate(texts, params, lora_request=lora, use_tqdm=False)
        return [
            Sampled(texts=[o.text for o in r.outputs], tokens=[len(o.token_ids) for o in r.outputs]) for r in results
        ]

    def close(self) -> None:
        del self.llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
