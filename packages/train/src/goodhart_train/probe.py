"""The GPU probe: what this account's cards can run, measured before any plan depends on it.

For the accelerator the request asked for, it records the cards, their compute capability, memory,
clocks and driver, the Python and package versions of the image, and then, for each candidate
model: whether it loads in float16, its parameter count and license from the card, the generation
throughput for 8 by 8 completions at the registered length, the decode step time at the GRPO
generation batch, and the memory and time of a LoRA training step at the SFT micro batch. It builds
vLLM for the driver in its own environment and measures it on the same prompts, and it runs short
GRPO smoke runs with the honest reward on each backend that works. The answer is
results/gpu_probe.json, which every later request reads.

What the second probe taught, and this one does:

* Every candidate and every smoke run is a child process (goodhart_train.isolate). One model's
  device-side assert had poisoned the CUDA context of the whole kernel, so every model after it and
  the GRPO smoke failed with the same assert.
* The LoRA step runs at the SFT micro batch, after the generation memory is released. At a micro
  batch of eight sequences of about 1,150 tokens the full vocabulary logits alone (eight by 1,150 by
  151,936 in float32) are 5.6 GB, and every candidate ran out of memory.
* vLLM is built for the driver's CUDA (goodhart_train.vllm_env), on the second card while the
  candidates run on the first, and its whole error is kept when it fails.
"""

from __future__ import annotations

import gc
import json
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from goodhart_core.warehouse import Warehouses

from goodhart_train import env, isolate, serve, vllm_env
from goodhart_train.backend import HFBackend, Job, ShardedBackend
from goodhart_train.bundle import Bundle
from goodhart_train.ledger import Meter, peak_memory
from goodhart_train.models import card_license, load, lora_config, trainable_parameters

SAMPLE_CHARS = 300
OWN_SAMPLING = ("do_sample", "temperature", "top_p", "top_k", "min_p", "repetition_penalty")


def _free() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _err(err: BaseException) -> str:
    return f"{type(err).__name__}: {str(err)[:600]}"


def decode_timing(backend: HFBackend, messages: list[Any], max_new_tokens: int = 33) -> dict[str, Any]:
    """Prefill and per step decode time at the GRPO generation batch (two prompts by eight samples):
    one call for one new token, one for max_new_tokens, the difference over the extra steps."""
    import torch

    def timed(tokens: int) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.monotonic()
        backend.generate(messages, n=8, temperature=0.8, max_new_tokens=tokens, seed=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.monotonic() - t0

    first = timed(1)
    longer = timed(max_new_tokens)
    return {
        "batch_sequences": 8 * len(messages),
        "prefill_seconds": round(first, 3),
        "decode_ms_per_step": round(1000.0 * max(0.0, longer - first) / max(1, max_new_tokens - 1), 1),
    }


def probe_candidate(model_id: str, bundle: Bundle, hp: dict[str, Any], meter: Meter) -> dict[str, Any]:
    """Loads in float16, eight samples of eight training prompts at the registered length, and the
    decode timing. The LoRA step is the SFT trainer's own, taken in another child (trainer_step)."""
    import torch

    found: dict[str, Any] = {"model": model_id, "license": card_license(model_id)}
    t0 = time.monotonic()
    try:
        with meter.phase("loading"):
            loaded = load(model_id)
        own = getattr(loaded.model, "generation_config", None)
        found.update(
            loads=True,
            parameters=loaded.parameters,
            dtype=loaded.dtype,
            attention=getattr(loaded.model.config, "_attn_implementation", None),
            load_seconds=round(time.monotonic() - t0, 1),
            # The model's own sampling defaults, which the backends override (HFBackend.config).
            own_generation_config={k: getattr(own, k, None) for k in OWN_SAMPLING},
        )
    except Exception as err:
        found.update(loads=False, error=_err(err))
        _free()
        return found
    prompts = bundle.prompts("train")[:8]
    messages = [p.messages for p in prompts]
    cuda = torch.cuda.is_available()
    gen_seconds = float("nan")
    try:
        backend = HFBackend(loaded.model, loaded.tokenizer, model_id, batch_size=64)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t1 = time.monotonic()
        with meter.phase("generation"):
            sampled = backend.generate(
                messages, n=8, temperature=0.8, max_new_tokens=int(hp["max_completion_length"]), seed=0
            )
        gen_seconds = time.monotonic() - t1
        tokens = sum(sum(s.tokens) for s in sampled)
        found.update(
            generates=True,
            generation_seconds_8x8=round(gen_seconds, 2),
            tokens_8x8=tokens,
            tokens_per_second=round(tokens / max(gen_seconds, 1e-6), 1),
            prompts_per_generate_call=backend.prompts_per_call,
            sample=sampled[0].texts[0][:SAMPLE_CHARS],
        )
        if cuda:
            found["generation_peak_bytes"] = int(torch.cuda.max_memory_allocated())
        with meter.phase("generation"):
            found["decode"] = decode_timing(
                HFBackend(loaded.model, loaded.tokenizer, model_id, batch_size=16), messages[:2]
            )
        del sampled, backend
    except Exception as err:
        found.update(generates=False, generation_error=_err(err))
    _free()
    return found


def trainer_step(
    model_id: str, bundle: Bundle, sft: dict[str, Any], lora: dict[str, Any], steps: int = 3
) -> dict[str, Any]:
    """The LoRA step the selection rule asks for, taken by the SFT trainer itself on its registered
    recipe: completion only loss through TRL's fused head, which never forms the full vocabulary
    logits, the registered micro batch and accumulation, gradient checkpointing, and the prompt
    rendered as sampling renders it, on the probe's training items with their gold answers. It runs in
    a process of its own, so no memory a generation left behind is counted against it.

    The third probe's step was a loop of its own that formed the full logits of every sequence after
    generation in the same process; it ran out of memory on every candidate above 0.6B, which says
    nothing about the step the trainer takes. Seconds per step leave out the first step's warmup."""
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from goodhart_train.trainer_common import precision_flags, template_column

    loaded = load(model_id)
    golds = {i.id: i.gold.sql for i in bundle.items("train") if i.gold is not None and i.gold.sql}
    template = template_column(model_id)
    rows = [
        {
            "prompt": p.messages,
            "completion": [{"role": "assistant", "content": f"```sql\n{golds[p.id]}\n```"}],
            **template,
        }
        for p in bundle.prompts("train")
        if p.id in golds
    ]
    micro, accumulation = int(sft["batch_size"]), int(sft["gradient_accumulation"])
    clock = StepClock()
    with tempfile.TemporaryDirectory() as scratch:
        args = SFTConfig(
            output_dir=scratch,
            max_steps=steps,
            per_device_train_batch_size=micro,
            gradient_accumulation_steps=accumulation,
            gradient_checkpointing=bool(sft.get("gradient_checkpointing", True)),
            max_length=int(sft["max_length"]),
            completion_only_loss=True,
            learning_rate=float(sft["learning_rate"]),
            lr_scheduler_type="constant",
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=0,
            data_seed=0,
            **precision_flags(),
        )
        trainer = SFTTrainer(
            model=loaded.model,
            args=args,
            train_dataset=Dataset.from_list(rows),
            processing_class=loaded.tokenizer,
            peft_config=lora_config(lora),
            callbacks=[clock.callback()],
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        result = trainer.train()
        seconds = time.monotonic() - t0
    steady = clock.steady_seconds_per_step()
    per_step = micro * accumulation
    return {
        "lora_step": True,
        "lora_step_by": "the SFT trainer, at the registered micro batch and accumulation",
        "train_micro_batch": micro,
        "train_sequences_per_step": per_step,
        "train_steps": int(result.global_step),
        "train_seconds": round(seconds, 1),
        "train_seconds_per_step": None if steady is None else round(steady, 2),
        "train_seconds_per_sequence": None if steady is None else round(steady / per_step, 3),
        "training_peak_bytes": peak_memory(),
        "trainable_parameters": trainable_parameters(trainer.model),
        "training_rows": len(rows),
    }


def vllm_generation(
    python: str, model_id: str, bundle: Bundle, hp: dict[str, Any], workdir: Path, gpu: int | None, eager: bool
) -> dict[str, Any]:
    """vLLM on the probe's eight prompts: eight samples each at the registered length, and greedy."""
    length = int(hp["max_completion_length"])
    messages = [p.messages for p in bundle.prompts("train")[:8]]
    backend = ShardedBackend(
        "vllm",
        model_id,
        gpus=[gpu],
        workdir=workdir,
        python=python,
        max_model_len=int(hp.get("max_model_len", 2048)),
        gpu_memory_utilization=float(hp.get("probe_vllm_memory", 0.85)),
        enforce_eager=eager,
        timeout_s=float(hp.get("probe_vllm_timeout", 1200)),
    )
    sampled, _greedy = backend.run([Job(messages, 8, 0.8, length, 0), Job(messages, 1, 0.0, length, 0)])
    shard = backend.stats[-1]["shards"][0]
    first, second = shard["jobs"]
    return {
        "works": True,
        "enforce_eager": eager,
        "load_seconds": shard["load_seconds"],
        "generation_seconds_8x8": first["seconds"],
        "tokens_8x8": first["tokens"],
        "tokens_per_second": round(first["tokens"] / max(first["seconds"], 1e-6), 1),
        "greedy_seconds_8": second["seconds"],
        "sample": sampled[0].texts[0][:SAMPLE_CHARS],
    }


# An error worth a second try without CUDA graphs and compilation; any other error (a driver too old,
# a missing kernel image, an import that fails) would fail the same way, so the next plan is tried.
EAGER_HINTS = ("compile", "inductor", "triton", "graph", "dynamo", "stopped after")


def vllm_lane(
    bundle: Bundle,
    hp: dict[str, Any],
    work: Path,
    out: Path,
    gpu: int | None,
    model_id: str,
    budget_s: float = 1800.0,
) -> dict[str, Any]:
    """Build vLLM for the driver, generate with it, and add the training stack when it generates.
    Plans are tried in order inside a time budget, each without CUDA graphs only when its first error
    points at compilation or graph capture."""
    deadline = time.monotonic() + budget_s
    version = str(hp.get("vllm_version", "0.32.0"))
    record: dict[str, Any] = {"model": model_id, "version": version, "isolated": True, "gpu": gpu, "attempts": []}
    try:
        tried = vllm_env.plans(version, env.cuda_driver_version())
    except Exception as err:
        return {**record, "works": False, "error": _err(err)}
    record["plans"] = [p["source"] for p in tried]
    for plan in tried:
        if time.monotonic() > deadline - 300:
            record["attempts"].append({"plan": plan["source"], "skipped": "the vLLM lane ran out of time"})
            continue
        installed = vllm_env.install(plan, timeout_s=max(120.0, min(900.0, deadline - time.monotonic())))
        attempt: dict[str, Any] = {"install": installed}
        record["attempts"].append(attempt)
        if not installed["installed"]:
            continue
        attempt["environment"] = {k: v for k, v in vllm_env.describe().items() if k != "freeze"}
        for eager in (False, True):
            key = "generation_eager" if eager else "generation"
            if eager and not any(h in str(attempt.get("generation", {}).get("error", "")) for h in EAGER_HINTS):
                break
            tag = f"vllm-{plan['source']}-{'eager' if eager else 'graphs'}"
            limits = {
                **hp,
                "probe_vllm_timeout": max(
                    120.0, min(float(hp.get("probe_vllm_timeout", 900)), deadline - time.monotonic())
                ),
            }
            try:
                attempt[key] = got = vllm_generation(
                    str(vllm_env.python()), model_id, bundle, limits, work / tag, gpu, eager
                )
            except Exception as err:
                attempt[key] = {"works": False, "error": str(err)[-6000:]}
                got = {}
            for shard in sorted((work / tag).glob("call-*/shard-*")):
                isolate.keep_log(shard, "shard-0", out / "logs" / tag)
            if got:
                record.update(
                    {**got, "works": True, "plan": plan["source"], "torch_backend": plan.get("torch_backend")}
                )
                break
        if record.get("works"):
            break
    if not record.get("works"):
        record["works"] = False
        return record
    if hp.get("smoke_vllm", True):
        stack = vllm_env.pins(bundle.root / "requirements-gpu.txt", vllm_env.TRAINING_STACK)
        plan = next(p for p in tried if p["source"] == record["plan"])
        record["training_stack"] = vllm_env.install(plan, extra=stack)
    record["freeze"] = vllm_env.describe().get("freeze", [])
    return record


class StepClock:
    """Seconds at the end of every optimizer step, so steps an hour leave out the first step's warmup."""

    def __init__(self) -> None:
        self.marks: list[float] = []

    def callback(self) -> Any:
        from transformers import TrainerCallback

        marks = self.marks

        class _Clock(TrainerCallback):  # type: ignore[misc]
            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                marks.append(time.monotonic())
                return control

        return _Clock()

    def steady_seconds_per_step(self) -> float | None:
        if len(self.marks) < 2:
            return None
        return (self.marks[-1] - self.marks[0]) / (len(self.marks) - 1)


def grpo_smoke(
    bundle: Bundle,
    out: Path,
    model_id: str,
    hp: dict[str, Any],
    warehouses: Warehouses,
    backend: str = "transformers",
) -> dict[str, Any]:
    """A few GRPO steps with the honest reward on one generation backend: transformers in process,
    transformers with paged attention, or vLLM (a server on another card, or colocated)."""
    from datasets import Dataset
    from goodhart_rewards.rewards import honest
    from trl import GRPOConfig, GRPOTrainer

    from goodhart_train.callbacks import LogCallback, Timings
    from goodhart_train.rewardfn import SQLReward
    from goodhart_train.trainer_common import precision_flags

    items = {i.id: i for i in bundle.items("train")}
    prompts = [p for p in bundle.prompts("train") if p.id in items][:80]
    t0 = time.monotonic()
    loaded = load(model_id)
    load_seconds = time.monotonic() - t0
    tag = backend.replace("-", "_")
    reward = SQLReward(reward=honest, items=items, warehouses=warehouses, log_path=out / f"smoke_{tag}_rollouts.jsonl")
    steps = int(hp.get("smoke_steps", 5))
    vllm = backend.startswith("vllm")
    extra: dict[str, Any] = serve.trainer_args(hp, server=backend == "vllm-server") if vllm else {}
    args = GRPOConfig(
        # The trainer's own folder stays out of the outputs: nothing in it is a result.
        output_dir=str(Path(tempfile.gettempdir()) / f"goodhart-smoke-{tag}"),
        learning_rate=float(hp["learning_rate"]),
        per_device_train_batch_size=int(hp["batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation"]),
        gradient_checkpointing=True,
        num_generations=int(hp["num_generations"]),
        generation_batch_size=int(hp.get("generation_batch_size", 16)) if not vllm else None,
        max_completion_length=int(hp["max_completion_length"]),
        temperature=float(hp["temperature"]),
        beta=float(hp["beta"]),
        loss_type=str(hp["loss_type"]),
        use_transformers_paged=backend == "transformers-paged",
        max_steps=steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=0,
        chat_template_kwargs=hp.get("chat_template_kwargs") or None,
        **extra,
        **precision_flags(),
    )
    clock = StepClock()
    timings = Timings()
    timings.install()
    trainer = GRPOTrainer(
        model=loaded.model,
        reward_funcs=[reward],
        args=args,
        train_dataset=Dataset.from_list([{"prompt": p.messages, "item_id": p.id} for p in prompts]),
        processing_class=loaded.tokenizer,
        peft_config=lora_config(hp),
        callbacks=[LogCallback(out / "metrics" / f"smoke_{tag}.jsonl", timings=timings), clock.callback()],
    )
    t1 = time.monotonic()
    trainer.train()
    seconds = time.monotonic() - t1
    paid = reward.paid
    steady = clock.steady_seconds_per_step()
    return {
        "model": model_id,
        "backend": backend,
        "steps": steps,
        "load_seconds": round(load_seconds, 1),
        "seconds": round(seconds, 1),
        "seconds_per_step_after_the_first": round(steady, 1) if steady else None,
        "steps_per_hour": round(3600.0 / steady, 1) if steady else round(3600.0 * steps / max(seconds, 1e-6), 1),
        "completions_per_step": int(hp["batch_size"]) * int(hp["gradient_accumulation"]),
        "reward_calls": reward.calls,
        "completions": reward.completions_seen,
        "executions": reward.executions,
        "mean_reward": round(sum(paid) / len(paid), 4) if paid else None,
        "peak_memory_bytes": peak_memory(),
    }


def _trainer_step(
    bundle: Bundle,
    model_id: str,
    training: dict[str, Any],
    work: Path,
    out: Path,
    gpu: Any,
    timeout: float,
    meter: Meter,
    steps: int = 3,
) -> dict[str, Any]:
    """The trainer's LoRA step for one candidate in a child of its own."""
    name = "trainer-step-" + model_id.replace("/", "__")
    args = {"bundle": str(bundle.root), "model": model_id, "sft": training["sft"], "lora": training["lora"]}
    args["steps"] = steps
    t0 = time.monotonic()
    outcome = isolate.run_task("trainer_step", args, work, name, gpu=gpu, timeout_s=min(900.0, timeout))
    meter.seconds["training"] = meter.seconds.get("training", 0.0) + (time.monotonic() - t0)
    isolate.keep_log(work, name, out)
    if not outcome.ok:
        return {"lora_step": False, "lora_step_by": "the SFT trainer", "lora_error": outcome.error[-3000:]}
    return outcome.result


def _steps_per_hour(c: dict[str, Any]) -> float | None:
    """A guess at GRPO steps an hour from the candidate's generation time and the trainer's step: 64
    completions generated, 64 sequences trained."""
    generation, step = c.get("generation_seconds_8x8"), c.get("train_seconds_per_step")
    if not generation or not step:
        return None
    sequences = int(c.get("train_sequences_per_step") or 16)
    return round(3600.0 / (float(generation) + float(step) * 64 / sequences), 1)


def _smoke(
    backend: str,
    bundle: Bundle,
    out: Path,
    work: Path,
    cache: Path,
    model_id: str,
    hp: dict[str, Any],
    gpus: list[Any],
    timeout: float,
) -> dict[str, Any]:
    """One smoke run in a child process; for vLLM in server mode, the server on the second card."""
    args = {"bundle": str(bundle.root), "cache": str(cache), "out": str(out), "model": model_id, "hp": hp}
    args["backend"] = backend
    name = f"smoke-{backend}"
    if not backend.startswith("vllm"):
        outcome = isolate.run_task("grpo_smoke", args, work, name, gpu=gpus[0], timeout_s=timeout)
        isolate.keep_log(work, name, out)
        return outcome.as_record()
    python = str(vllm_env.python())
    if backend == "vllm-colocate":
        outcome = isolate.run_task("grpo_smoke", args, work, name, gpu=gpus[0], timeout_s=timeout, python=python)
        isolate.keep_log(work, name, out)
        return outcome.as_record()
    with serve.vllm_server(python, model_id, hp, work, gpus[1], keep_in=out):
        outcome = isolate.run_task("grpo_smoke", args, work, name, gpu=gpus[0], timeout_s=timeout, python=python)
    isolate.keep_log(work, name, out)
    return outcome.as_record()


def run(bundle: Bundle, out: Path, meter: Meter, warehouses: Warehouses, cache: Path | None = None) -> dict[str, Any]:
    req = bundle.request
    training = bundle.config("training")
    hp = {**training["grpo"], **training["lora"], **req.hyperparameters}
    record: dict[str, Any] = {"environment": env.describe(), "accelerator_requested": req.accelerator}
    gpus: list[Any] = [g["index"] for g in record["environment"]["gpus"]] or [None]
    parallel = len(gpus) >= 2
    record["isolation"] = "every candidate and every smoke run in its own process"
    record["lanes"] = (
        "candidates on GPU 0 while vLLM was built and measured on GPU 1, so the throughput numbers carry "
        "that CPU contention"
        if parallel
        else "one card: candidates, then vLLM, then the smoke runs"
    )
    # Scratch outside the output folder: Kaggle keeps everything under /kaggle/working as output.
    cache = cache or Path("/tmp/goodhart/cache")
    work = cache.parent / "probe-work"
    work.mkdir(parents=True, exist_ok=True)
    smallest = str(hp.get("smoke_model") or req.models[0])
    # The vLLM lane writes its record from another thread; every change to the record and every save
    # holds this lock, so a save never iterates a dict the other thread is changing.
    lock = threading.RLock()

    def remaining(margin_hours: float = 0.02) -> float:
        """Seconds a child may run and still leave the probe inside its ceiling."""
        return max(60.0, (req.gpu_hour_ceiling - margin_hours - meter.elapsed_hours()) * 3600.0)

    candidates: list[dict[str, Any]] = []

    def save() -> None:
        with lock:
            (out / "probe.json").write_text(
                json.dumps({**record, "candidates": candidates}, indent=1) + "\n", encoding="utf-8"
            )

    vllm_record: dict[str, Any] = {"tried": False, "reason": "disabled"}

    def lane_vllm() -> None:
        nonlocal vllm_record
        try:
            vllm_record = vllm_lane(
                bundle,
                hp,
                work,
                out,
                gpus[1] if parallel else gpus[0],
                smallest,
                budget_s=float(hp.get("vllm_lane_seconds", 1800)),
            )
        except Exception:
            vllm_record = {"works": False, "error": traceback.format_exc()[-3000:]}
        with lock:
            record["vllm"] = vllm_record
            save()

    lane = threading.Thread(target=lane_vllm, daemon=True) if hp.get("try_vllm", True) else None
    if lane is not None and parallel:
        lane.start()
    for model_id in req.models:
        if meter.over_ceiling(0.35):
            candidates.append({"model": model_id, "loads": None, "skipped": "the probe reached its ceiling"})
            continue
        args: dict[str, Any] = {"bundle": str(bundle.root), "model": model_id, "hp": hp}
        args["ceiling_hours"] = req.gpu_hour_ceiling
        name = "candidate-" + model_id.replace("/", "__")
        outcome = isolate.run_task(
            "candidate",
            args,
            work,
            name,
            gpu=gpus[0],
            timeout_s=min(float(hp.get("candidate_timeout", 900)), remaining()),
        )
        isolate.keep_log(work, name, out)
        found: dict[str, Any] = (
            outcome.as_record() if outcome.result else {"model": model_id, "loads": False, "error": outcome.error}
        )
        phases: dict[str, float] = found.pop("phase_seconds", None) or {}
        for phase, seconds in phases.items():
            meter.seconds[phase] = meter.seconds.get(phase, 0.0) + float(seconds)
        found["child_seconds"] = outcome.seconds
        if found.get("loads") and not meter.over_ceiling(0.3):
            steps = int(hp.get("probe_trainer_steps", 3))
            found.update(_trainer_step(bundle, model_id, training, work, out, gpus[0], remaining(), meter, steps))
        found["steps_per_hour_estimate"] = _steps_per_hour(found)
        with lock:
            candidates.append(found)
            save()
    if lane is not None:
        if not parallel:
            lane.start()
        lane.join(timeout=remaining(0.15))
    record["candidates"] = candidates
    record["vllm"] = vllm_record
    stack = (vllm_record.get("training_stack") or {}).get("installed", False)
    record["generation_backend"] = "vllm" if vllm_record.get("works") else "transformers"
    smokes: dict[str, Any] = {}
    for backend in hp.get("smoke_backends", ["vllm-server", "transformers"]):
        if backend.startswith("vllm") and not (vllm_record.get("works") and stack):
            smokes[backend] = {"ok": False, "skipped": "vLLM or its training stack did not install"}
            continue
        if backend == "vllm-server" and not parallel:
            backend = "vllm-colocate"
        if meter.over_ceiling(0.15):
            smokes[backend] = {"ok": False, "skipped": "the probe reached its ceiling"}
            continue
        smoke_hp = {**hp, "vllm_enforce_eager": vllm_record.get("enforce_eager", False)}
        try:
            timeout = min(float(hp.get("smoke_timeout", 1500)), remaining())
            smokes[backend] = _smoke(backend, bundle, out, work, cache, smallest, smoke_hp, gpus, timeout)
        except Exception:
            smokes[backend] = {"ok": False, "error": traceback.format_exc()[-3000:]}
        record["grpo_smokes"] = smokes
        save()
    record["grpo_smokes"] = smokes
    ran = [s for s in smokes.values() if s.get("steps_per_hour")]
    record["grpo_smoke"] = max(ran, key=lambda s: s["steps_per_hour"]) if ran else next(iter(smokes.values()), None)
    save()
    return {
        "candidates": len(candidates),
        "generation_backend": record["generation_backend"],
        "smokes": {k: v.get("steps_per_hour") for k, v in smokes.items()},
    }
