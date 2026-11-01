"""goodhart probe-summary: results/gpu_probe.json from the promoted probe runs.

The quota is a design input. This reads every promoted probe (results/runs/*-probe/probe.json), takes
the latest result for each candidate, and writes the accelerator the kernels saw, the generation
backend, per candidate whether it passed the probe (loads in float16, generates at eight samples and
the registered lengths, takes a LoRA step) with its measured throughput, the measured GRPO steps an
hour of the smoke run, and the GPU hours the plan's stages would take at those rates, beside the
quota table the plan registered. Every projection names the rate it used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

QUOTA_HOURS = 30.0


def latest_probes(root: Path) -> list[tuple[str, dict[str, Any]]]:
    runs = root / "results" / "runs"
    found = sorted(p for p in runs.glob("*probe*/probe.json")) if runs.exists() else []
    return [(p.parent.name, json.loads(p.read_text(encoding="utf-8"))) for p in found]


def passes(c: dict[str, Any]) -> bool:
    return bool(c.get("loads") and c.get("generates") and c.get("lora_step"))


def summarize(probes: list[tuple[str, dict[str, Any]]], plan: dict[str, Any]) -> dict[str, Any]:
    if not probes:
        raise FileNotFoundError("no promoted probe run under results/runs")
    by_model: dict[str, dict[str, Any]] = {}
    for run_id, probe in probes:
        for c in probe.get("candidates", []):
            by_model[c["model"]] = {**c, "run": run_id}
    _, last = probes[-1]
    gpus = last.get("environment", {}).get("gpus", [])
    names = sorted({g["name"] for g in gpus})
    accelerator = "NvidiaTeslaT4" if names == ["Tesla T4"] else (names[0] if names else "unknown")
    smoke = next(
        (
            p.get("grpo_smoke")
            for _, p in reversed(probes)
            if (p.get("grpo_smoke") or {}).get("ok", True) and "steps_per_hour" in (p.get("grpo_smoke") or {})
        ),
        None,
    )
    candidates = []
    for model, c in sorted(by_model.items()):
        candidates.append(
            {
                "model": model,
                "run": c["run"],
                "license": c.get("license"),
                "parameters": c.get("parameters"),
                "passes": passes(c),
                "loads": c.get("loads"),
                "generates": c.get("generates"),
                "lora_step": c.get("lora_step"),
                "tokens_per_second": c.get("tokens_per_second"),
                "steps_per_hour_estimate": c.get("steps_per_hour_estimate"),
                "decode_ms_per_step": (c.get("decode") or {}).get("decode_ms_per_step"),
                "prefill_seconds": (c.get("decode") or {}).get("prefill_seconds"),
                "train_seconds_per_sequence": c.get("train_seconds_per_sequence"),
                "train_micro_batch": c.get("train_micro_batch"),
                "training_peak_bytes": c.get("training_peak_bytes"),
                "generation_peak_bytes": c.get("generation_peak_bytes"),
                "own_generation_config": c.get("own_generation_config"),
                "why_not": None if passes(c) else (c.get("error") or c.get("generation_error") or c.get("lora_error")),
            }
        )
    quota = dict(plan["compute"]["quota_plan"])
    return {
        "accelerator": accelerator,
        "gpus": gpus,
        "image": {k: last.get("environment", {}).get(k) for k in ("python", "cuda", "packages")},
        "generation_backend": last.get("generation_backend", "transformers"),
        # The frozen package list stays in the run's own probe.json; the summary keeps what decided.
        "vllm": {k: v for k, v in (last.get("vllm") or {}).items() if k not in {"freeze", "attempts"}},
        "vllm_attempts": [
            {
                "source": (a.get("install") or {}).get("source"),
                "installed": (a.get("install") or {}).get("installed"),
                "generates": (a.get("generation") or {}).get("works"),
                "generates_eager": (a.get("generation_eager") or {}).get("works"),
            }
            for a in (last.get("vllm") or {}).get("attempts", [])
            if isinstance(a, dict) and "install" in a
        ],
        "grpo_smokes": {
            k: {kk: v.get(kk) for kk in ("ok", "steps_per_hour", "seconds_per_step_after_the_first", "error")}
            for k, v in (last.get("grpo_smokes") or {}).items()
        },
        "driver": last.get("environment", {}).get("driver"),
        "candidates": candidates,
        "grpo_smoke": smoke,
        "projected_hours": quota,
        "mandatory_hours": quota.get("mandatory_total"),
        "weekly_quota_hours": QUOTA_HOURS,
        "fits_with_margin": float(quota.get("mandatory_total", 0)) <= 0.75 * QUOTA_HOURS,
        "from_runs": [r for r, _ in probes],
    }


def run(root: Path) -> dict[str, Any]:
    plan = yaml.safe_load((root / "evaluation" / "plan.yaml").read_text(encoding="utf-8"))
    out = summarize(latest_probes(root), plan)
    (root / "results" / "gpu_probe.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out
