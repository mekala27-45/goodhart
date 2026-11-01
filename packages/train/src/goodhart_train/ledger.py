"""The run ledger: one row per stage run, measured, never estimated.

It feeds the compute ceilings, the sizing sheet and the pages. GPU seconds are split by phase
(generation, reward, training, evaluation) with a wall clock timer around each phase on the GPU, so a
method comparison is at one measured ceiling and not at one nominal budget.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from goodhart_core.model import StrictModel
from pydantic import Field

PHASES: tuple[str, ...] = ("generation", "reward", "training", "evaluation", "loading", "other")


class LedgerRow(StrictModel):
    request: str
    request_digest: str
    kernel: str | None = None
    model: str | None = None
    stage: str
    arm: str | None = None
    status: str
    steps: int = 0
    prompts: int = 0
    rollouts: int = 0
    tokens_generated: int = 0
    tokens_trained: int = 0
    gpu_seconds: dict[str, float] = Field(default_factory=dict)
    peak_memory_bytes: int = 0
    wall_clock_seconds: float = 0.0
    accelerator: str | None = None
    ceiling_hours: float = 0.0
    stopped_at_ceiling: bool = False
    digests_in: dict[str, str] = Field(default_factory=dict)
    digests_out: dict[str, str] = Field(default_factory=dict)
    note: str = ""


@dataclass
class Meter:
    """Times the phases of a run and holds its counts until the ledger row is written."""

    ceiling_hours: float
    started: float = field(default_factory=time.monotonic)
    seconds: dict[str, float] = field(default_factory=lambda: dict.fromkeys(PHASES, 0.0))
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "steps": 0,
            "prompts": 0,
            "rollouts": 0,
            "tokens_generated": 0,
            "tokens_trained": 0,
        }
    )

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in self.seconds:
            raise KeyError(f"unknown phase {name!r}")
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.seconds[name] += time.monotonic() - t0

    def add(self, **counts: int) -> None:
        for k, v in counts.items():
            self.counts[k] = self.counts.get(k, 0) + int(v)

    def elapsed_hours(self) -> float:
        return (time.monotonic() - self.started) / 3600.0

    def over_ceiling(self, margin_hours: float = 0.0) -> bool:
        return self.elapsed_hours() + margin_hours >= self.ceiling_hours

    def row(self, **fields: Any) -> LedgerRow:
        values: dict[str, Any] = {**self.counts, **fields}
        return LedgerRow(
            gpu_seconds={k: round(v, 3) for k, v in self.seconds.items() if v > 0},
            wall_clock_seconds=round(time.monotonic() - self.started, 3),
            ceiling_hours=self.ceiling_hours,
            **values,
        )


def append(path: Path, row: LedgerRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")


def read(path: Path) -> list[LedgerRow]:
    if not path.exists():
        return []
    return [
        LedgerRow.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def peak_memory() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())))
    except Exception:
        return 0
    return 0
