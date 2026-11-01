"""Loading a base model, a LoRA adapter on top of it, and the facts the selection table prints.

On a T4 the dtype is float16 (no native bfloat16 on Turing); on CPU, float32 for the smoke runs.
LoRA defaults come from config/training.yaml: rank 16, alpha 32, dropout 0.05 on the attention and
MLP projections. A license is read from the model card on the day, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Loaded:
    model: Any
    tokenizer: Any
    model_id: str
    dtype: str
    parameters: int


def device_dtype() -> tuple[str, Any]:
    import torch

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return ("cuda", torch.bfloat16 if major >= 8 else torch.float16)
    return ("cpu", torch.float32)


def load(model_id: str, adapter: str | None = None, merge: bool = False, trust_remote_code: bool = False) -> Loaded:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = device_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map={"": 0} if device == "cuda" else None,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    params = sum(p.numel() for p in model.parameters())
    return Loaded(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        dtype=str(dtype).replace("torch.", ""),
        parameters=params,
    )


def lora_config(hp: dict[str, Any]) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=int(hp.get("lora_rank", 16)),
        lora_alpha=int(hp.get("lora_alpha", 32)),
        lora_dropout=float(hp.get("lora_dropout", 0.05)),
        target_modules=list(hp.get("lora_targets", LORA_TARGETS)),
        task_type="CAUSAL_LM",
    )


def card_license(model_id: str) -> str | None:
    """The license the model card declares today, or None when the card says nothing or is unreachable."""
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        data = getattr(info, "card_data", None) or getattr(info, "cardData", None)
        if data is None:
            return None
        value = data.get("license") if hasattr(data, "get") else getattr(data, "license", None)
        return str(value) if value else None
    except Exception:
        return None


def trainable_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
