"""The GPU probe: what this account's card can run, measured before any plan depends on it.

For the accelerator the request asked for, it records the card, its compute capability and memory,
the Python and package versions of the image, and then, for each candidate model: whether it loads
in float16, its parameter count and license from the card, the memory after a LoRA training step on
a batch of 64 completions, the generation throughput for 8 by 8 completions at the registered
length, and the steps per hour those imply. It tries vLLM in a subprocess on the smallest candidate
and runs a ten step LoRA GRPO smoke with the honest reward on the backend that works. The answer is
results/gpu_probe.json, which every later request reads.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any

from goodhart_core.warehouse import Warehouses

from goodhart_train import env
from goodhart_train.backend import HFBackend
from goodhart_train.bundle import Bundle
from goodhart_train.ledger import Meter, peak_memory
from goodhart_train.models import card_license, load, lora_config, trainable_parameters


def _free() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def probe_candidate(model_id: str, bundle: Bundle, hp: dict[str, Any], meter: Meter) -> dict[str, Any]:
    import torch

    found: dict[str, Any] = {"model": model_id, "license": card_license(model_id)}
    t0 = time.monotonic()
    try:
        with meter.phase("loading"):
            loaded = load(model_id)
        found.update(
            loads=True,
            parameters=loaded.parameters,
            dtype=loaded.dtype,
            load_seconds=round(time.monotonic() - t0, 1),
        )
    except Exception as err:
        found.update(loads=False, error=f"{type(err).__name__}: {str(err)[:300]}")
        _free()
        return found
    prompts = bundle.prompts("train")[:8]
    try:
        backend = HFBackend(loaded.model, loaded.tokenizer, model_id, batch_size=64)
        t1 = time.monotonic()
        with meter.phase("generation"):
            sampled = backend.generate(
                [p.messages for p in prompts],
                n=8,
                temperature=0.8,
                max_new_tokens=int(hp["max_completion_length"]),
                seed=0,
            )
        gen_seconds = time.monotonic() - t1
        tokens = sum(sum(s.tokens) for s in sampled)
        found.update(
            generates=True,
            generation_seconds_8x8=round(gen_seconds, 2),
            tokens_8x8=tokens,
            tokens_per_second=round(tokens / max(gen_seconds, 1e-6), 1),
        )
    except Exception as err:
        found.update(generates=False, generation_error=f"{type(err).__name__}: {str(err)[:300]}")
        gen_seconds = float("nan")
    try:
        from peft import get_peft_model

        model = get_peft_model(loaded.model, lora_config(hp))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        tok = loaded.tokenizer
        tok.padding_side = "right"
        texts = [p.messages[-1]["content"] for p in prompts] * 8
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(hp["max_completion_length"]) + 768,
        ).to(model.device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        t2 = time.monotonic()
        with meter.phase("training"):
            for k in range(0, enc["input_ids"].shape[0], int(hp["batch_size"])):
                part = {key: v[k : k + int(hp["batch_size"])] for key, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    loss = model(**part, labels=part["input_ids"]).loss
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_seconds = time.monotonic() - t2
        found.update(
            lora_step=True,
            trainable_parameters=trainable_parameters(model),
            train_seconds_64=round(train_seconds, 2),
            peak_memory_bytes=peak_memory(),
        )
        if gen_seconds == gen_seconds:  # not NaN
            found["steps_per_hour_estimate"] = round(3600.0 / (gen_seconds + train_seconds), 1)
    except Exception as err:
        found.update(lora_step=False, lora_error=f"{type(err).__name__}: {str(err)[:300]}")
    del loaded
    _free()
    return found


VLLM_SCRIPT = textwrap.dedent(
    """
    import json, sys, time
    from vllm import LLM, SamplingParams
    model = sys.argv[1]
    t0 = time.time()
    llm = LLM(model=model, dtype="float16", max_model_len=2048, gpu_memory_utilization=0.6, seed=0)
    load = time.time() - t0
    t1 = time.time()
    params = SamplingParams(n=1, temperature=0.8, max_tokens=128)
    out = llm.generate(["SELECT 1 as x; -- write one SQL query:"] * 64, params)
    gen = time.time() - t1
    tokens = sum(len(o.outputs[0].token_ids) for o in out)
    found = {"load_seconds": round(load, 1), "generation_seconds_64": round(gen, 2), "tokens": tokens}
    print("VLLM_RESULT " + json.dumps(found))
    """
)


VLLM_VENV = Path("/tmp/goodhart-vllm")


def probe_vllm(model_id: str, version: str, timeout_s: int = 900) -> dict[str, Any]:
    """Can vLLM generate on this card? Installed into its own virtual environment, because vLLM pins
    its own PyTorch and would otherwise replace the one every other stage of this kernel uses."""
    found: dict[str, Any] = {"model": model_id, "version": version, "isolated": True}
    t0 = time.monotonic()
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VLLM_VENV)], check=True, capture_output=True, timeout=120)
        python = str(VLLM_VENV / "bin" / "python")
        install = subprocess.run(
            [python, "-m", "pip", "install", "-q", f"vllm=={version}"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.SubprocessError, OSError) as err:
        found.update(installed=False, error=f"{type(err).__name__}: {str(err)[:300]}")
        return found
    found["install_seconds"] = round(time.monotonic() - t0, 1)
    if install.returncode != 0:
        found.update(installed=False, error=install.stderr.strip().splitlines()[-1][:300] if install.stderr else "")
        return found
    found["installed"] = True
    try:
        done = subprocess.run([python, "-c", VLLM_SCRIPT, model_id], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        found.update(works=False, error=f"generation did not finish in {timeout_s} s")
        return found
    line = next((ln for ln in done.stdout.splitlines() if ln.startswith("VLLM_RESULT ")), None)
    if done.returncode == 0 and line:
        found.update(works=True, **json.loads(line.removeprefix("VLLM_RESULT ")))
    else:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        found.update(works=False, error=" | ".join(t[:200] for t in tail))
    return found


def grpo_smoke(
    bundle: Bundle, out: Path, model_id: str, hp: dict[str, Any], warehouses: Warehouses, use_vllm: bool
) -> dict[str, Any]:
    from datasets import Dataset
    from goodhart_rewards.rewards import honest
    from trl import GRPOConfig, GRPOTrainer

    from goodhart_train.callbacks import LogCallback
    from goodhart_train.rewardfn import SQLReward
    from goodhart_train.trainer_common import precision_flags

    items = {i.id: i for i in bundle.items("train")}
    prompts = [p for p in bundle.prompts("train") if p.id in items][:80]
    loaded = load(model_id)
    reward = SQLReward(reward=honest, items=items, warehouses=warehouses, log_path=out / "smoke_rollouts.jsonl")
    args = GRPOConfig(
        output_dir=str(out / "smoke_trainer"),
        learning_rate=float(hp["learning_rate"]),
        per_device_train_batch_size=int(hp["batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation"]),
        gradient_checkpointing=True,
        num_generations=int(hp["num_generations"]),
        max_completion_length=int(hp["max_completion_length"]),
        temperature=float(hp["temperature"]),
        beta=float(hp["beta"]),
        loss_type=str(hp["loss_type"]),
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        max_steps=10,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=0,
        chat_template_kwargs=hp.get("chat_template_kwargs") or None,
        **precision_flags(),
    )
    trainer = GRPOTrainer(
        model=loaded.model,
        reward_funcs=[reward],
        args=args,
        train_dataset=Dataset.from_list([{"prompt": p.messages, "item_id": p.id} for p in prompts]),
        processing_class=loaded.tokenizer,
        peft_config=lora_config(hp),
        callbacks=[LogCallback(out / "metrics" / "smoke.jsonl")],
    )
    t0 = time.monotonic()
    trainer.train()
    seconds = time.monotonic() - t0
    paid = reward.paid
    return {
        "model": model_id,
        "backend": "vllm" if use_vllm else "transformers",
        "steps": 10,
        "seconds": round(seconds, 1),
        "steps_per_hour": round(36000.0 / max(seconds, 1e-6), 1),
        "reward_calls": reward.calls,
        "completions": reward.completions_seen,
        "executions": reward.executions,
        "mean_reward": round(sum(paid) / len(paid), 4) if paid else None,
        "peak_memory_bytes": peak_memory(),
    }


def run(bundle: Bundle, out: Path, meter: Meter, warehouses: Warehouses) -> dict[str, Any]:
    req = bundle.request
    hp = {**bundle.config("training")["grpo"], **bundle.config("training")["lora"], **req.hyperparameters}
    record: dict[str, Any] = {"environment": env.describe(), "accelerator_requested": req.accelerator}
    candidates: list[dict[str, Any]] = []
    for model_id in req.models:
        if meter.over_ceiling(0.2):
            candidates.append({"model": model_id, "loads": None, "skipped": "the probe reached its ceiling"})
            continue
        try:
            candidates.append(probe_candidate(model_id, bundle, hp, meter))
        except Exception as err:
            candidates.append({"model": model_id, "loads": False, "error": f"{type(err).__name__}: {str(err)[:300]}"})
        (out / "probe.json").write_text(
            json.dumps({**record, "candidates": candidates}, indent=1) + "\n", encoding="utf-8"
        )
    record["candidates"] = candidates
    smallest = str(hp.get("smoke_model") or req.models[0])
    # The GRPO smoke first, on the transformers backend every stage can use; vLLM last, in its own
    # environment, and only with a quarter hour of the ceiling left.
    try:
        record["grpo_smoke"] = grpo_smoke(bundle, out, smallest, hp, warehouses, use_vllm=False)
    except Exception:
        record["grpo_smoke"] = {"ok": False, "error": traceback.format_exc()[-1500:]}
    (out / "probe.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    if hp.get("try_vllm", True) and not meter.over_ceiling(0.25):
        vllm = probe_vllm(smallest, str(hp.get("vllm_version", "0.32.0")))
    else:
        vllm = {"tried": False, "reason": "disabled" if not hp.get("try_vllm", True) else "too little time left"}
    record["vllm"] = vllm
    record["generation_backend"] = "vllm" if vllm.get("works") else "transformers"
    (out / "probe.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return {"candidates": len(candidates), "generation_backend": record["generation_backend"]}
