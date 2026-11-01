"""The sampling stages: base selection, held-out sampling, self samples for pairs, and the judge.

Each loads one model (with an adapter when the request names an input that holds one), samples the
prompt sets the request lists, and writes completions. None of them grades: selection writes greedy
and eight sample completions on development, and the repository computes pass at 1 and pass at 8.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from goodhart_train.backend import HFBackend, IsolatedVLLMBackend
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
            from goodhart_train.probe import ensure_vllm

            python = ensure_vllm(str(hp.get("vllm_version", "0.32.0")))
            return IsolatedVLLMBackend(model_id, adapter, python, max_model_len=int(hp.get("max_model_len", 4096)))
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
        if meter.over_ceiling(0.15):
            done[model_id] = {"ok": False, "skipped": "the selection reached its ceiling"}
            continue
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
    summary: dict[str, Any] = {"models": done}
    teacher = hp.get("paraphrase")
    if teacher and not meter.over_ceiling(0.1):
        summary["paraphrase"] = paraphrase(bundle, out, meter, dict(teacher), req.seed)
    return summary


PARAPHRASE_SYSTEM = (
    "Rewrite the question the way a person would ask it in conversation. Keep every name, year, quarter, "
    "number and unit exactly as written, and ask for exactly the same thing. Reply with the rewritten "
    "question only, on one line."
)


def paraphrase_ids(prompts: list[Any], share: float, seed: int) -> list[str]:
    """The training questions the teacher rewrites: a seeded draw of the stated share, by sorted id."""
    import random

    ids = sorted(p.id for p in prompts)
    return sorted(random.Random(seed).sample(ids, round(share * len(ids))))


def _with_question(messages: list[dict[str, str]], old: str, new: str) -> list[dict[str, str]]:
    out = []
    for m in messages:
        if m["role"] == "user" and old in m["content"]:
            m = {**m, "content": m["content"].replace(old, new)}
        out.append(m)
    return out


def _question(messages: list[dict[str, str]]) -> str:
    user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    line = next((ln for ln in user.splitlines() if ln.startswith("Question: ")), "")
    return line.removeprefix("Question: ").strip()


def paraphrase(bundle: Bundle, out: Path, meter: Meter, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """The paraphrase round trip, kernel side: a teacher from a family no candidate shares rewrites a
    seeded share of training questions, then writes SQL from the rewritten question alone (the schema
    block of the original prompt kept, its question replaced). The repository grades that SQL on the
    original and on every kept twin and keeps a paraphrase only when it matches the gold on all."""
    import json

    model_id = str(cfg["teacher"])
    prompts = bundle.prompts("train")
    chosen = set(paraphrase_ids(prompts, float(cfg.get("share", 0.25)), seed))
    picked = [p for p in prompts if p.id in chosen]
    backend = _backend(model_id, None, {"generation_backend": cfg.get("generation_backend", "transformers")}, meter)
    asks = [
        [{"role": "system", "content": PARAPHRASE_SYSTEM}, {"role": "user", "content": _question(p.messages)}]
        for p in picked
    ]
    with meter.phase("generation"):
        rewritten = backend.generate(asks, n=1, temperature=0.0, max_new_tokens=96, seed=seed)
    texts = [r.texts[0].strip().splitlines()[0].strip() if r.texts[0].strip() else "" for r in rewritten]
    sql_prompts = [_with_question(p.messages, _question(p.messages), t) for p, t in zip(picked, texts, strict=True)]
    with meter.phase("generation"):
        answers = backend.generate(
            sql_prompts, n=1, temperature=0.0, max_new_tokens=int(cfg.get("max_new_tokens", 256)), seed=seed
        )
    path = out / "completions" / "paraphrase__teacher.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for p, t, a in zip(picked, texts, answers, strict=True):
            row = {
                "item_id": p.id,
                "teacher": model_id,
                "original": _question(p.messages),
                "paraphrase": t,
                "completion": a.texts[0],
                "tokens": a.tokens[0],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    meter.add(
        rollouts=2 * len(picked),
        prompts=len(picked),
        tokens_generated=sum(r.tokens[0] for r in rewritten) + sum(a.tokens[0] for a in answers),
    )
    _release(backend)
    return {"teacher": model_id, "asked": len(picked), "empty": sum(1 for t in texts if not t)}


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
