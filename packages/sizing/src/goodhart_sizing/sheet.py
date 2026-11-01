"""The workload sizing sheet: what each run cost, measured, and what the same recipe would cost at the
model sizes a customer would pay for, estimated and labeled so.

Measured, per run, from its ledger row: prompts, rollouts, tokens generated and trained, GPU seconds
by phase (loading, generation, reward, training, evaluation), peak memory, wall clock, and whether
it stopped at its ceiling. Throughput per phase is tokens per GPU second of that phase.

Projected, per recipe, to 7B and 32B parameters on an H100, under assumptions stated in the sheet:

    compute grows linearly with parameters at the same tokens (training costs about 6 N FLOPs per
    token and generation about 2 N, so the ratio of parameter counts carries over);
    an H100 does the measured T4 work SPEEDUP times faster (the ratio of their dense float16 tensor
    peaks, 989 against 65 TFLOPS, halved for the efficiency a small LoRA run reaches);
    the hour is priced from a public GPU price page, and SFT and DPO tokens also from a public
    fine-tuning price list, both read on the day (config/prices.yaml).

Nothing here is spent; every projected number carries the word estimate on the pages.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

T4_FP16_TFLOPS = 65.0
H100_FP16_TFLOPS = 989.0
EFFICIENCY = 0.5
SPEEDUP = H100_FP16_TFLOPS / T4_FP16_TFLOPS * EFFICIENCY
TARGETS = {"7B": 7.0e9, "32B": 32.0e9}
FT_BAND = {"7B": "Qwen3.5 9B", "32B": "Qwen3.5 27B"}
H100 = "NVIDIA H100 SXM 80GB"
RECIPES = ("sft", "dpo", "grpo", "pairs", "sample", "select")


def gpu_hours(row: dict[str, Any]) -> float:
    return float(sum(row.get("gpu_seconds", {}).values())) / 3600.0


def throughput(row: dict[str, Any]) -> dict[str, float | None]:
    sec = row.get("gpu_seconds", {})
    gen, train = float(sec.get("generation", 0.0)), float(sec.get("training", 0.0))
    return {
        "generated_tokens_per_second": round(row.get("tokens_generated", 0) / gen, 1) if gen else None,
        "trained_tokens_per_second": round(row.get("tokens_trained", 0) / train, 1) if train else None,
    }


def measured(rows: Iterable[dict[str, Any]], parameters: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append(
            {
                "request": r["request"],
                "stage": r["stage"],
                "model": r.get("model"),
                "arm": r.get("arm"),
                "status": r["status"],
                "parameters": parameters.get(str(r.get("model"))),
                "steps": r.get("steps", 0),
                "prompts": r.get("prompts", 0),
                "rollouts": r.get("rollouts", 0),
                "tokens_generated": r.get("tokens_generated", 0),
                "tokens_trained": r.get("tokens_trained", 0),
                "gpu_seconds": r.get("gpu_seconds", {}),
                "gpu_hours": round(gpu_hours(r), 3),
                "ceiling_hours": r.get("ceiling_hours"),
                "within_ceiling": gpu_hours(r) <= float(r.get("ceiling_hours") or 0.0) + 1e-9,
                "stopped_at_ceiling": r.get("stopped_at_ceiling", False),
                "peak_memory_gb": round(r.get("peak_memory_bytes", 0) / 1e9, 2),
                "wall_clock_minutes": round(float(r.get("wall_clock_seconds", 0.0)) / 60.0, 1),
                **throughput(r),
            }
        )
    return out


def project(run: dict[str, Any], prices: dict[str, Any]) -> dict[str, Any]:
    """The same recipe at 7B and 32B on an H100: hours and dollars, estimated."""
    n = run.get("parameters")
    if not n:
        return {}
    hour_price = float(prices["gpu_hours"]["prices"][H100])
    table = prices["fine_tuning"]["first_table"]
    out: dict[str, Any] = {}
    for label, target in TARGETS.items():
        hours = run["gpu_hours"] * (target / float(n)) / SPEEDUP
        entry: dict[str, Any] = {"h100_hours": round(hours, 3), "gpu_dollars": round(hours * hour_price, 2)}
        if run["stage"] in {"sft", "dpo"} and run["tokens_trained"]:
            per_million = float(table[FT_BAND[label]][run["stage"]])
            entry["fine_tuning_api_dollars"] = round(run["tokens_trained"] / 1e6 * per_million, 2)
        out[label] = entry
    return out


def learnability(root: Path, prompts_per_step: int = 8) -> list[dict[str, Any]]:
    """The share of prompts the filter keeps, and how many prompts a run of a given length needs."""
    out = []
    folder = root / "data" / "learnable"
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        share = float(data.get("share_kept") or 0.0)
        needs = {steps: (round(steps * prompts_per_step / share) if share else None) for steps in (100, 200, 500)}
        out.append({"run": data["run"], "share_kept": share, "kept": len(data["kept"]), "prompts_sampled_for": needs})
    return out


def ledger_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "results" / "runs").glob("*/ledger.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(root: Path) -> dict[str, Any]:
    prices = yaml.safe_load((root / "config" / "prices.yaml").read_text(encoding="utf-8"))
    probe_path = root / "results" / "gpu_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {"candidates": []}
    parameters = {c["model"]: int(c["parameters"]) for c in probe["candidates"] if c.get("parameters")}
    runs = measured(ledger_rows(root), parameters)
    for r in runs:
        r["projection"] = project(r, prices)
    sheet = {
        "assumptions": {
            "compute_scales_with": "parameters, at the measured tokens",
            "speedup_t4_to_h100": round(SPEEDUP, 2),
            "efficiency": EFFICIENCY,
            "h100_price_per_hour": prices["gpu_hours"]["prices"][H100],
            "prices_read_on": prices["read_on"],
            "price_pages": [prices["gpu_hours"]["url"], prices["fine_tuning"]["url"]],
            "label": "estimate, never spent",
        },
        "runs": runs,
        "learnability": learnability(root),
        "total_gpu_hours": round(sum(r["gpu_hours"] for r in runs), 3),
        "every_run_within_ceiling": all(r["within_ceiling"] for r in runs),
    }
    (root / "results" / "sizing.json").write_text(json.dumps(sheet, indent=1) + "\n", encoding="utf-8")
    return sheet
