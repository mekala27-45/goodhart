"""What a model card says about one method's chosen checkpoint, gathered for the manifest.

A card is rendered by the claim gate from the manifest (docs/templates/docs/models/<method>.md.j2),
so every number on it is one the repository computed. This module gathers, per method whose chosen
checkpoint has been sampled on the held-out sets: the base model and its license (the probe), the
run the checkpoint came from and its step (the sample request's input and adapter_step), the method
and every hyperparameter (the registered defaults with the training request's overrides), the data
digests the training request declared, the compute (goodhart methods), the results with their
intervals and the hack detector rates on every held-out set (goodhart methods).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DESCRIPTIONS = {
    "M1": "SFT with LoRA on the gold SQL of the cleaned training set",
    "M2": "DPO on pairs from eight base samples per training prompt, verified by the honest reward",
    "M3": "GRPO from the base with the honest reward, on the prompts the learnability filter kept",
    "M4": "GRPO from the M1 checkpoint with the honest reward, on the prompts the learnability filter kept",
}
STAGE_OF = {"M1": "sft", "M2": "dpo", "M3": "grpo", "M4": "grpo"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def gather(root: Path, code: str) -> dict[str, Any] | None:
    """The card of one method, or None when its chosen checkpoint has not been sampled yet."""
    methods = _json(root / "results" / "methods.json")
    if not methods or not methods["scores"].get(code):
        return None
    runs = methods["methods"].get(code, [])
    requests = {r: _json(root / "requests" / f"{r}.json") or {} for r in runs}
    samples = [r for r in runs if requests[r].get("stage") == "sample"]
    training = [r for r in runs if requests[r].get("stage") == STAGE_OF[code]]
    if not samples or not training:
        return None
    sample = requests[samples[-1]]
    trained_by = sample.get("inputs", [None])[-1] or training[-1]
    req = _json(root / "requests" / f"{trained_by}.json") or requests[training[-1]]
    step = sample.get("hyperparameters", {}).get("adapter_step")
    config = yaml.safe_load((root / "config" / "training.yaml").read_text(encoding="utf-8"))
    hp = {**config[STAGE_OF[code]], **config["lora"], **req.get("hyperparameters", {})}
    probe = _json(root / "results" / "gpu_probe.json") or {"candidates": []}
    license_id = next((c.get("license") for c in probe["candidates"] if c["model"] == req.get("model")), None)
    return {
        "code": code,
        "description": DESCRIPTIONS[code],
        "base": req.get("model"),
        "license": license_id or "not declared",
        "trained_by": trained_by,
        "checkpoint": f"step {step}" if step is not None else "the final step",
        "gpu_hours": methods["compute"][code]["gpu_hours"],
        "hyperparameters": sorted((k, json.dumps(v) if not isinstance(v, str) else v) for k, v in hp.items()),
        "data": sorted((path, digest[:12]) for path, digest in req.get("data", {}).items()),
        "scores": methods["scores"][code],
    }
