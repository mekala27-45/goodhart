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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    def generate(
        self, prompts: Sequence[Messages], n: int, temperature: float, max_new_tokens: int, seed: int
    ) -> list[Sampled]:
        import torch

        torch.manual_seed(seed)
        texts = [render(self.tokenizer, m, self.model_id) for m in prompts]
        out: list[Sampled] = []
        per_batch = max(1, self.batch_size // max(1, n))
        for start in range(0, len(texts), per_batch):
            chunk = texts[start : start + per_batch]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(
                self.model.device
            )
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
            width = enc["input_ids"].shape[1]
            new = gen[:, width:]
            for i in range(len(chunk)):
                rows = new[i * n : (i + 1) * n]
                decoded = [self.tokenizer.decode(r, skip_special_tokens=True) for r in rows]
                lengths = [int((r != self.tokenizer.pad_token_id).sum().item()) for r in rows]
                out.append(Sampled(texts=decoded, tokens=lengths))
        return out


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
