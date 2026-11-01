"""Trainer callbacks shared by the stages: the ceiling, the monitor stop rule, the log, the dev export.

The ceiling callback stops a run at its GPU hour ceiling and saves, and the ledger says so. The
monitor callback applies the registered rule: a defect arm stops at its next checkpoint once an
alert fires. The dev export samples the development prompts at every checkpoint, greedily, so the
repository can draw the twin confirmed accuracy beside the reward paid for every 25 steps, and the
adapter of every checkpoint is kept beside it, so whichever step the checkpoint choice picks can be
loaded, evaluated and released. The optimizer state of a checkpoint stays behind.
"""

from __future__ import annotations

import json
import shutil
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


ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def keep_adapter(checkpoint: Path, out: Path, step: int) -> Path | None:
    """A saved checkpoint's adapter, copied to out/adapters/step-<step>; None when it has none."""
    files = [checkpoint / name for name in ADAPTER_FILES]
    if not all(f.exists() for f in files):
        return None
    target = out / "adapters" / f"step-{step}"
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, target / f.name)
    return target


class CheckpointExportCallback(TrainerCallback):  # type: ignore[misc]
    """At every save, keep the adapter of the folder just saved (in out/adapters, beside the trainer's
    folder), then run a function with the current step, the model and the kept adapter (the dev
    export, which samples it on a second card when there is one)."""

    def __init__(self, export: Callable[..., None]) -> None:
        self.export = export
        self.exported: list[int] = []
        self.kept: list[int] = []

    def on_save(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
        step = int(state.global_step)
        if step not in self.exported:
            checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
            kept = keep_adapter(checkpoint, Path(args.output_dir).parent, step)
            if kept is not None:
                self.kept.append(step)
            # The kept copy, not the trainer's folder: the trainer keeps only its latest checkpoints
            # and may delete this one before an export on the second card has read it.
            self.export(step, model, str(kept or checkpoint))
            self.exported.append(step)
        return control
