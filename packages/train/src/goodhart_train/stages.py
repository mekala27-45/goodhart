"""The sampling stages: base selection, held-out sampling, self samples for pairs, and the judge.

Each loads one model (with an adapter when the request names an input that holds one), samples the
prompt sets the request lists, and writes completions. None of them grades: selection writes greedy
and eight sample completions on development, and the repository computes pass at 1 and pass at 8.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from goodhart_train.backend import HFBackend, VLLMBackend
from goodhart_train.bundle import Bundle
from goodhart_train.ledger import Meter
from goodhart_train.models import load
from goodhart_train.sample import sample_prompts, write


def _adapter(inputs: list[str]) -> str | None:
    found = None
    for base in inputs:
        candidate = Path(base) / "adapter"
        if candidate.exists():
            found = str(candidate)
    return found


def _backend(model_id: str, adapter: str | None, hp: dict[str, Any], meter: Meter) -> Any:
    with meter.phase("loading"):
        if hp.get("generation_backend") == "vllm":
            return VLLMBackend(model_id, model_id, max_model_len=int(hp.get("max_model_len", 4096)), adapter=adapter)
        loaded = load(model_id, adapter=adapter, merge=adapter is not None)
        loaded.model.eval()
        return HFBackend(loaded.model, loaded.tokenizer, model_id, batch_size=int(hp.get("generation_batch", 64)))


def _release(backend: Any) -> None:
    if hasattr(backend, "close"):
        backend.close()
    del backend
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def select(bundle: Bundle, out: Path, meter: Meter) -> dict[str, Any]:
    req = bundle.request
    hp = {**bundle.config("training")["select"], **req.hyperparameters}
    dev = bundle.prompts("dev")
    done: dict[str, Any] = {}
    for model_id in req.models:
        safe = model_id.replace("/", "__")
        try:
            backend = _backend(model_id, None, hp, meter)
            greedy = sample_prompts(
                backend,
                dev,
                request=req.id,
                checkpoint=f"base:{model_id}",
                n=1,
                temperature=0.0,
                max_new_tokens=int(hp["max_new_tokens"]),
                seed=req.seed,
                meter=meter,
                examples=bundle.examples(),
            )
            write(greedy, out / "completions" / f"dev__{safe}__greedy.jsonl")
            sampled = sample_prompts(
                backend,
                dev,
                request=req.id,
                checkpoint=f"base:{model_id}",
                n=int(hp["k"]),
                temperature=float(hp["temperature"]),
                max_new_tokens=int(hp["max_new_tokens"]),
                seed=req.seed,
                meter=meter,
                examples=bundle.examples(),
            )
            write(sampled, out / "completions" / f"dev__{safe}__t{int(float(hp['temperature']) * 10):02d}.jsonl")
            done[model_id] = {"ok": True, "greedy": len(greedy), "sampled": len(sampled)}
            _release(backend)
        except Exception as err:
            done[model_id] = {"ok": False, "error": f"{type(err).__name__}: {str(err)[:300]}"}
    return {"models": done}


def sample(bundle: Bundle, out: Path, meter: Meter) -> dict[str, Any]:
    """Held-out sampling for one checkpoint: n samples per question at the registered temperature,
    and greedy for the sets the plan grades greedily (Spider, IFEval, the defect arms)."""
    req = bundle.request
    hp = {**bundle.config("training")["sample"], **req.hyperparameters}
    assert req.model is not None
    adapter = _adapter(bundle.inputs)
    checkpoint = str(hp.get("checkpoint_name", "final"))
    examples = bundle.examples() if hp.get("with_examples") else []
    backend = _backend(req.model, adapter, hp, meter)
    written: dict[str, int] = {}
    for name in hp.get("sampled_sets", []):
        if meter.over_ceiling(0.05):
            break
        got = sample_prompts(
            backend,
            bundle.prompts(name),
            request=req.id,
            checkpoint=checkpoint,
            n=int(hp["n"]),
            temperature=float(hp["temperature"]),
            max_new_tokens=int(hp["max_new_tokens"]),
            seed=req.seed,
            meter=meter,
            examples=examples,
        )
        write(got, out / "completions" / f"{name}__{checkpoint}__n{int(hp['n'])}.jsonl")
        written[f"{name}:sampled"] = len(got)
    for name in hp.get("greedy_sets", []):
        if meter.over_ceiling(0.05):
            break
        got = sample_prompts(
            backend,
            bundle.prompts(name),
            request=req.id,
            checkpoint=checkpoint,
            n=1,
            temperature=0.0,
            max_new_tokens=int(hp.get("greedy_max_new_tokens", hp["max_new_tokens"])),
            seed=req.seed,
            meter=meter,
            examples=examples,
        )
        write(got, out / "completions" / f"{name}__{checkpoint}__greedy.jsonl")
        written[f"{name}:greedy"] = len(got)
    _release(backend)
    return {"written": written, "adapter": adapter is not None}


def pairs(bundle: Bundle, out: Path, meter: Meter) -> dict[str, Any]:
    """Eight samples per training prompt from a starting checkpoint, for DPO pairs and the learnability filter."""
    req = bundle.request
    hp = {**bundle.config("training")["pairs"], **req.hyperparameters}
    assert req.model is not None
    adapter = _adapter(bundle.inputs)
    backend = _backend(req.model, adapter, hp, meter)
    prompts = bundle.prompts("train")
    got = sample_prompts(
        backend,
        prompts,
        request=req.id,
        checkpoint="start",
        n=int(hp["n"]),
        temperature=float(hp["temperature"]),
        max_new_tokens=int(hp["max_new_tokens"]),
        seed=req.seed,
        meter=meter,
        examples=bundle.examples() if hp.get("with_examples") else [],
    )
    write(got, out / "completions" / f"train__start__n{int(hp['n'])}.jsonl")
    _release(backend)
    return {"prompts": len(prompts), "completions": len(got)}


def judge(bundle: Bundle, out: Path, meter: Meter) -> dict[str, Any]:
    """The judge reads the calibration registry (built in the repository) and writes its verdicts as text."""
    req = bundle.request
    hp = {**bundle.config("training")["judge"], **req.hyperparameters}
    assert req.model is not None
    backend = _backend(req.model, None, hp, meter)
    written = {}
    for name in [s for s in bundle.prompt_sets() if s.startswith("judge_")]:
        got = sample_prompts(
            backend,
            bundle.prompts(name),
            request=req.id,
            checkpoint=f"judge:{req.model}",
            n=1,
            temperature=0.0,
            max_new_tokens=int(hp["max_new_tokens"]),
            seed=req.seed,
            meter=meter,
        )
        write(got, out / "completions" / f"{name}__judge.jsonl")
        written[name] = len(got)
    _release(backend)
    return {"written": written}
