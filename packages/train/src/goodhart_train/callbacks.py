"""Trainer callbacks shared by the stages: the ceiling, the monitor stop rule, the log, the dev export.

The ceiling callback stops a run at its GPU hour ceiling and saves, and the ledger says so. The
monitor callback applies the registered rule: a defect arm stops at its next checkpoint once an
alert fires. The dev export samples the development prompts at every checkpoint, greedily, so the
repository can draw the twin confirmed accuracy beside the reward paid for every 25 steps.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from goodhart_train.ledger import Meter


class CeilingCallback(TrainerCallback):  # type: ignore[misc]
    def __init__(self, meter: Meter, margin_hours: float = 0.1) -> None:
        self.meter = meter
        self.margin_hours = margin_hours
        self.stopped = False

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.meter.over_ceiling(self.margin_hours):
            self.stopped = True
            control.should_save = True
            control.should_training_stop = True
        return control


class MonitorStopCallback(TrainerCallback):  # type: ignore[misc]
    """Stop at the next checkpoint once the monitor asked for it (defect arms only)."""

    def __init__(self, monitor: Any, save_steps: int) -> None:
        self.monitor = monitor
        self.save_steps = save_steps
        self.stopped_at: int | None = None

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.monitor is not None and self.monitor.stop_requested and state.global_step % self.save_steps == 0:
            self.stopped_at = int(state.global_step)
            control.should_save = True
            control.should_training_stop = True
        return control


class LogCallback(TrainerCallback):  # type: ignore[misc]
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if logs:
            row = {
                "step": int(state.global_step),
                **{k: v for k, v in logs.items() if isinstance(v, int | float | str)},
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return control


class CheckpointExportCallback(TrainerCallback):  # type: ignore[misc]
    """At every save, run a function with the current step, the model and the folder just saved (the
    dev export, which samples that folder's adapter on a second card when there is one)."""

    def __init__(self, export: Callable[..., None]) -> None:
        self.export = export
        self.exported: list[int] = []

    def on_save(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
        step = int(state.global_step)
        if step not in self.exported:
            checkpoint = str(Path(args.output_dir) / f"checkpoint-{step}")
            self.export(step, model, checkpoint)
            self.exported.append(step)
        return control
