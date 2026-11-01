"""The training monitors: per step statistics, registered thresholds, alert rows and the stop rule.

Each training step the reward wrapper hands the monitor what it saw: the parsed completions, their
executions, their items and their prompts. The monitor counts the hack detectors, reads the trainer's
own metrics for the step (the share of groups with no reward spread, entropy, completion length, the
clipped ratio), and compares every statistic with its threshold in config/monitor.yaml, which is
registered with the plan before any arm runs.

A breach writes an alert row. The registered rule then stops a defect arm at its next checkpoint and
flags the honest arm without stopping it. The gpu workflow opens an issue for every alert file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from goodhart_core.items import Execution, Item, Parsed
from goodhart_core.model import StrictModel

from goodhart_monitor.detectors import DETECTORS, detect

STATS: tuple[str, ...] = (
    "mean_result_rows",
    "share_literal_only",
    "share_copied_value",
    "share_unfiltered",
    "share_refusal_answerable",
    "share_valid_but_wrong",
    "frac_reward_zero_std",
    "entropy",
    "completion_length",
    "clipped_ratio",
)

# The trainer metric each statistic is read from, when the trainer logs it rather than the monitor.
TRAINER_METRICS: dict[str, tuple[str, ...]] = {
    "frac_reward_zero_std": ("frac_reward_zero_std",),
    "entropy": ("entropy",),
    "completion_length": ("completions/mean_length", "completion_length"),
    "clipped_ratio": ("completions/clipped_ratio",),
}


class Threshold(StrictModel):
    stat: str
    op: Literal[">", "<"]
    value: float
    window: int = 2


class Alert(StrictModel):
    run: str
    arm: str
    step: int
    stat: str
    value: float
    op: str
    threshold: float
    action: Literal["stop_at_next_checkpoint", "flag"]


def load_thresholds(path: Path) -> list[Threshold]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = [Threshold(stat=k, **v) for k, v in sorted(data["thresholds"].items())]
    unknown = [t.stat for t in out if t.stat not in STATS]
    if unknown:
        raise ValueError(f"thresholds name statistics the monitor does not compute: {unknown}")
    if not out:
        raise ValueError("the monitor was given no thresholds")
    return out


def step_stats(
    parsed: Sequence[Parsed],
    executions: Sequence[Execution | None],
    items: Sequence[Item],
    prompts: Sequence[str],
) -> dict[str, float]:
    """The detector shares and the mean result rows for one step's completions."""
    n = len(parsed)
    if n == 0:
        raise ValueError("a step with no completions has no statistics")
    counts = dict.fromkeys(DETECTORS, 0)
    rows = 0.0
    executed = 0
    for p, e, it, pr in zip(parsed, executions, items, prompts, strict=True):
        flags = detect(p, e, it, pr).as_dict()
        for name, hit in flags.items():
            counts[name] += int(hit)
        if e is not None and e.executed:
            rows += e.row_count
            executed += 1
    return {
        "mean_result_rows": rows / executed if executed else 0.0,
        **{f"share_{name}": counts[name] / n for name in DETECTORS},
    }


def trainer_stats(logged: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for stat, keys in TRAINER_METRICS.items():
        for key in keys:
            if key in logged and logged[key] is not None:
                out[stat] = float(logged[key])
                break
    return out


@dataclass
class Monitor:
    run: str
    arm: str
    defect_arm: bool
    thresholds: list[Threshold]
    rows: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    _streak: dict[str, int] = field(default_factory=dict)
    stop_requested: bool = False

    def observe(self, step: int, stats: Mapping[str, float]) -> list[Alert]:
        """Record one step's statistics and return the alerts this step raised."""
        row = {"run": self.run, "arm": self.arm, "step": step, **{k: float(v) for k, v in stats.items()}}
        self.rows.append(row)
        raised: list[Alert] = []
        for t in self.thresholds:
            if t.stat not in stats:
                continue
            value = float(stats[t.stat])
            breach = value > t.value if t.op == ">" else value < t.value
            self._streak[t.stat] = self._streak.get(t.stat, 0) + 1 if breach else 0
            if self._streak[t.stat] == t.window:
                alert = Alert(
                    run=self.run,
                    arm=self.arm,
                    step=step,
                    stat=t.stat,
                    value=value,
                    op=t.op,
                    threshold=t.value,
                    action="stop_at_next_checkpoint" if self.defect_arm else "flag",
                )
                raised.append(alert)
                self.alerts.append(alert)
                if self.defect_arm:
                    self.stop_requested = True
        return raised

    def first_firing(self) -> dict[str, int]:
        """The first step each statistic breached, for the predictions table."""
        out: dict[str, int] = {}
        for a in self.alerts:
            out.setdefault(a.stat, a.step)
        return out

    def write(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "monitor.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        (folder / "alerts.json").write_text(
            json.dumps([a.model_dump() for a in self.alerts], indent=1) + "\n", encoding="utf-8"
        )
