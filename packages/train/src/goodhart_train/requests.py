"""The request: a committed file that names one GPU run completely.

requests/<date>-<name>.json names the stage, the model, the data digests, the reward, the seed, the
hyperparameters, the step budget and the GPU hour ceiling. The gpu workflow runs every request that
has no record on the results side, and nothing else starts a GPU. The request digest keys every
checkpoint and every output the run writes.

data maps the repository path of every file the stage reads to its sha256; scripts/new_request.py
fills it and scripts/build_kernel.py refuses a request whose digests differ from the files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from goodhart_core.hashing import sha256_json
from goodhart_core.model import StrictModel
from pydantic import Field

Stage = Literal["probe", "select", "sft", "dpo", "grpo", "sample", "judge", "pairs", "learnable", "spider_fetch"]
STAGES: tuple[str, ...] = (
    "probe",
    "select",
    "sft",
    "dpo",
    "grpo",
    "sample",
    "judge",
    "pairs",
    "learnable",
    "spider_fetch",
)


class Request(StrictModel):
    id: str
    stage: Stage
    model: str | None = None
    models: list[str] = Field(default_factory=list)
    arm: str | None = None
    reward: str | None = None
    seed: int = 0
    data: dict[str, str] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    step_budget: int | None = None
    gpu_hour_ceiling: float = Field(gt=0.0)
    inputs: list[str] = Field(default_factory=list)
    accelerator: str | None = None
    note: str = ""

    def digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def load(path: Path) -> Request:
    return Request.model_validate_json(path.read_text(encoding="utf-8"))


def pending(folder: Path, done: set[str]) -> list[tuple[Path, Request]]:
    """Every committed request with no record yet, in file name order (the date leads the name)."""
    out: list[tuple[Path, Request]] = []
    for path in sorted(folder.glob("*.json")):
        req = load(path)
        if req.id != path.stem:
            raise ValueError(f"{path.name}: the id {req.id!r} must equal the file name")
        if req.id not in done:
            out.append((path, req))
    return out


def dump(req: Request) -> str:
    return json.dumps(req.model_dump(mode="json"), indent=1, sort_keys=True) + "\n"
