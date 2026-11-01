"""The sampling stages: base selection, held-out sampling, self samples for pairs, and the judge.

Each samples one model (with an adapter when the request names an input that holds one) over the
prompt sets the request lists, and writes completions. None of them grades: selection writes greedy
and eight sample completions on development, and the repository computes pass at 1 and pass at 8.

Sampling runs in child processes, one per card (goodhart_train.backend.ShardedBackend), on the
backend the probe chose: transformers in the image's environment or vLLM in its own. The stage's
process never loads a model.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

from goodhart_train import env, vllm_env
from goodhart_train.backend import Job, ShardedBackend
from goodhart_train.bundle import Bundle
from goodhart_train.ledger import Meter
from goodhart_train.sample import completions, messages_for, write

WORK = Path(os.environ.get("GOODHART_WORK", "/tmp/goodhart")) / "sampling"


def _adapter(inputs: list[str], hp: dict[str, Any] | None = None) -> str | None:
    """The input run's adapter to sample: its final one, or the one kept at hp["adapter_step"]."""
    from goodhart_train.trainer_common import start_adapter

    step = (hp or {}).get("adapter_step")
    return start_adapter(inputs, None if step is None else int(step))


def _backend(
    model_id: str, adapter: str | None, hp: dict[str, Any], meter: Meter, out: Path | None = None
) -> ShardedBackend:
    gpus: list[int | None] = list(env.gpu_indices()) or [None]
    if hp.get("max_gpus"):
        gpus = gpus[: int(hp["max_gpus"])]
    kind = "vllm" if hp.get("generation_backend") == "vllm" else "transformers"
    python = None
    if kind == "vllm":
        with meter.phase("loading"):
            python = vllm_env.ensure(str(hp.get("vllm_version", "0.32.0")))
    workdir = WORK / f"{model_id.replace('/', '__')}-{time.monotonic_ns()}"
    return ShardedBackend(
        kind,
        model_id,
        adapter,
        gpus=gpus,
        workdir=workdir,
        python=python,
        batch_size=int(hp.get("generation_batch", 64)),
        max_model_len=int(hp.get("max_model_len", 2048)),
        gpu_memory_utilization=float(hp.get("vllm_memory", 0.85)),
        enforce_eager=bool(hp.get("vllm_enforce_eager", False)),
        timeout_s=_remaining(meter),
        keep_logs_in=out,
    )


def _remaining(meter: Meter) -> float:
    """Seconds left under the ceiling: a generation call is stopped there, and the ledger says so."""
    return max(60.0, (meter.ceiling_hours - meter.elapsed_hours()) * 3600.0)


def _run(backend: ShardedBackend, jobs: list[Job], meter: Meter) -> list[list[Any]]:
    backend.timeout_s = _remaining(meter)
    with meter.phase("generation"):
        return backend.run(jobs)


def _release(backend: Any) -> None:
    if hasattr(backend, "close"):
        backend.close()
    del backend
    gc.collect()


def _select_one(bundle: Bundle, out: Path, meter: Meter, model_id: str, hp: dict[str, Any]) -> dict[str, Any]:
    req = bundle.request
    dev = bundle.prompts("dev")
    safe = model_id.replace("/", "__")
    backend = _backend(model_id, None, hp, meter, out)
    messages = messages_for(dev, bundle.examples())
    length, temperature = int(hp["max_new_tokens"]), float(hp["temperature"])
    greedy_s, sampled_s = _run(
        backend,
        [Job(messages, 1, 0.0, length, req.seed), Job(messages, int(hp["k"]), temperature, length, req.seed)],
        meter,
    )
    checkpoint = f"base:{model_id}"
    greedy = completions(dev, greedy_s, request=req.id, checkpoint=checkpoint, temperature=0.0, meter=meter)
    sampled = completions(dev, sampled_s, request=req.id, checkpoint=checkpoint, temperature=temperature, meter=meter)
    write(greedy, out / "completions" / f"dev__{safe}__greedy.jsonl")
    write(sampled, out / "completions" / f"dev__{safe}__t{int(temperature * 10):02d}.jsonl")
    _release(backend)
    return {"ok": True, "backend": backend.kind, "greedy": len(greedy), "sampled": len(sampled), "calls": backend.stats}


def select(bundle: Bundle, out: Path, meter: Meter) -> dict[str, Any]:
    """Greedy and k samples on dev for every candidate. A candidate vLLM cannot serve on the card (an
    architecture whose kernels the card lacks) is sampled with transformers instead, and the summary
    says so: the selection measures the model, not the backend."""
    req = bundle.request
    hp = {**bundle.config("training")["select"], **req.hyperparameters}
    done: dict[str, Any] = {}
    for model_id in req.models:
        if meter.over_ceiling(0.15):
            done[model_id] = {"ok": False, "skipped": "the selection reached its ceiling"}
            continue
        try:
            done[model_id] = _select_one(bundle, out, meter, model_id, hp)
        except Exception as err:
            failed = f"{type(err).__name__}: {str(err)[-1500:]}"
            if hp.get("generation_backend") != "vllm" or not hp.get("fallback_to_transformers", True):
                done[model_id] = {"ok": False, "error": failed}
                continue
            try:
                found = _select_one(bundle, out, meter, model_id, {**hp, "generation_backend": "transformers"})
                done[model_id] = {**found, "vllm_error": failed}
            except Exception as again:
                done[model_id] = {"ok": False, "error": f"{type(again).__name__}: {str(again)[-1500:]}"}
                done[model_id]["vllm_error"] = failed
    summary: dict[str, Any] = {"models": done}
    teacher = hp.get("paraphrase")
    if teacher and not meter.over_ceiling(0.1):
        # The candidates' samples are already written; a teacher that fails is recorded, not fatal.
        try:
            summary["paraphrase"] = paraphrase(bundle, out, meter, dict(teacher), req.seed)
        except Exception as err:
            summary["paraphrase"] = {"ok": False, "error": f"{type(err).__name__}: {str(err)[-1500:]}"}
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
    backend = _backend(
        model_id, None, {**cfg, "generation_backend": cfg.get("generation_backend", "transformers")}, meter, out
    )
    asks = [
        [{"role": "system", "content": PARAPHRASE_SYSTEM}, {"role": "user", "content": _question(p.messages)}]
        for p in picked
    ]
    (rewritten,) = _run(backend, [Job(asks, 1, 0.0, 96, seed)], meter)
    texts = [r.texts[0].strip().splitlines()[0].strip() if r.texts[0].strip() else "" for r in rewritten]
    sql_prompts = [_with_question(p.messages, _question(p.messages), t) for p, t in zip(picked, texts, strict=True)]
    (answers,) = _run(backend, [Job(sql_prompts, 1, 0.0, int(cfg.get("max_new_tokens", 256)), seed)], meter)
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
    """Held-out sampling: n samples per question at the registered temperature, and greedy for the
    sets the plan grades greedily (Spider, IFEval, the defect arms), for one checkpoint or, with
    adapter_steps, for each of several (the interior test's neighbours of the chosen step).

    Every set of one checkpoint goes to the backend in one call, so the model loads once per
    checkpoint: with vLLM a load is two to three minutes, and a call per set paid it seven times."""
    req = bundle.request
    hp = {**bundle.config("training")["sample"], **req.hyperparameters}
    assert req.model is not None
    steps = hp.get("adapter_steps")
    checkpoints: list[tuple[str, str | None]] = (
        [(f"step-{int(step)}", _adapter(bundle.inputs, {**hp, "adapter_step": step})) for step in steps]
        if steps
        else [(str(hp.get("checkpoint_name", "final")), _adapter(bundle.inputs, hp))]
    )
    examples = bundle.examples() if hp.get("with_examples") else []
    plan = [(name, "sampled") for name in hp.get("sampled_sets", [])] + [
        (name, "greedy") for name in hp.get("greedy_sets", [])
    ]
    # A benchmark that is not SQL (IFEval) is asked as itself: no SQL examples before it, and its own
    # length (IFEval answers run to an essay; the reference setting is 1,280 new tokens).
    plain = set(hp.get("plain_sets", ["ifeval"]))
    lengths = dict(hp.get("max_new_tokens_by_set", {"ifeval": 1280}))
    written: dict[str, int] = {}
    for checkpoint, adapter in checkpoints:
        if meter.over_ceiling(0.05):
            break
        jobs, asks = [], []
        for name, how in plan:
            prompts = bundle.prompts(name)
            if how == "sampled":
                n, temperature, length = int(hp["n"]), float(hp["temperature"]), int(hp["max_new_tokens"])
                path = out / "completions" / f"{name}__{checkpoint}__n{n}.jsonl"
            else:
                n, temperature = 1, 0.0
                length = int(hp.get("greedy_max_new_tokens", hp["max_new_tokens"]))
                path = out / "completions" / f"{name}__{checkpoint}__greedy.jsonl"
            length = int(lengths.get(name, length))
            jobs.append(Job(messages_for(prompts, [] if name in plain else examples), n, temperature, length, req.seed))
            asks.append((name, how, path, prompts, temperature))
        backend = _backend(req.model, adapter, hp, meter, out)
        got_all = _run(backend, jobs, meter)
        for (name, how, path, prompts, temperature), got_s in zip(asks, got_all, strict=True):
            got = completions(
                prompts, got_s, request=req.id, checkpoint=checkpoint, temperature=temperature, meter=meter
            )
            write(got, path)
            written[f"{checkpoint}:{name}:{how}"] = len(got)
        _release(backend)
    return {"written": written, "checkpoints": [c for c, _ in checkpoints], "adapter": checkpoints[0][1] is not None}


def pairs(bundle: Bundle, out: Path, meter: Meter) -> dict[str, Any]:
    """Eight samples per training prompt from a starting checkpoint, for DPO pairs and the learnability filter."""
    req = bundle.request
    hp = {**bundle.config("training")["pairs"], **req.hyperparameters}
    assert req.model is not None
    adapter = _adapter(bundle.inputs, hp)
    backend = _backend(req.model, adapter, hp, meter, out)
    prompts = bundle.prompts("train")
    messages = messages_for(prompts, bundle.examples() if hp.get("with_examples") else [])
    n, temperature = int(hp["n"]), float(hp["temperature"])
    (got_s,) = _run(backend, [Job(messages, n, temperature, int(hp["max_new_tokens"]), req.seed)], meter)
    got = completions(prompts, got_s, request=req.id, checkpoint="start", temperature=temperature, meter=meter)
    write(got, out / "completions" / f"train__start__n{n}.jsonl")
    _release(backend)
    return {"prompts": len(prompts), "completions": len(got)}


def judge(bundle: Bundle, out: Path, meter: Meter) -> dict[str, Any]:
    """The judge reads the calibration registry (built in the repository) and writes its verdicts as text."""
    req = bundle.request
    hp = {**bundle.config("training")["judge"], **req.hyperparameters}
    assert req.model is not None
    backend = _backend(req.model, None, hp, meter, out)
    written = {}
    names = [s for s in bundle.prompt_sets() if s.startswith("judge_")]
    sets = {name: bundle.prompts(name) for name in names}
    jobs = [Job(messages_for(sets[name]), 1, 0.0, int(hp["max_new_tokens"]), req.seed) for name in names]
    for name, got_s in zip(names, _run(backend, jobs, meter) if jobs else [], strict=True):
        got = completions(
            sets[name], got_s, request=req.id, checkpoint=f"judge:{req.model}", temperature=0.0, meter=meter
        )
        write(got, out / "completions" / f"{name}__judge.jsonl")
        written[name] = len(got)
    _release(backend)
    return {"written": written}
