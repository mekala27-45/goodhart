"""The defect study's runs, read for the notebooks and the hacks page.

Each GRPO run of the defect study names its arm (honest, the four defects, the two controls) in its
request; its folder holds the trainer's metrics per step (the reward paid), the monitor's statistics
per step, the alerts, and the dev exports at every checkpoint, which goodhart grade and goodhart
checkpoints turn into the twin confirmed accuracy the run earned. Paid against earned is the picture
of a hack: the reward climbs and the confirmed accuracy does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARMS = ("honest", "lenient", "leaked", "lazy", "refusal", "random", "format")
DEFECTS = ("lenient", "leaked", "lazy", "refusal")
CONTROLS = ("random", "format")


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def arm_runs(root: Path) -> dict[str, str]:
    """The latest promoted GRPO run of each arm (a method's GRPO run names M3 or M4, not an arm)."""
    out: dict[str, str] = {}
    for prov in sorted((root / "results" / "runs").glob("*/provenance.json")):
        req_path = root / "requests" / f"{prov.parent.name}.json"
        if not req_path.exists():
            continue
        req = json.loads(req_path.read_text(encoding="utf-8"))
        arm = str(req.get("arm") or req.get("reward") or "")
        if req.get("stage") == "grpo" and arm in ARMS:
            out[arm] = prov.parent.name
    return out


def paid(root: Path, run_id: str) -> list[tuple[int, float]]:
    """The mean reward the trainer logged at each step."""
    rows = _lines(root / "results" / "runs" / run_id / "metrics" / "trainer.jsonl")
    return [(int(r["step"]), float(r["reward"])) for r in rows if "reward" in r and "step" in r]


def earned(root: Path, run_id: str) -> list[tuple[int, float]]:
    """Twin confirmed accuracy on dev at each checkpoint, from goodhart checkpoints."""
    path = root / "results" / "checkpoints.json"
    if not path.exists():
        return []
    steps = json.loads(path.read_text(encoding="utf-8")).get(run_id, {}).get("steps", {})
    measured = {int(s): p["twin_confirmed_accuracy"] for s, p in steps.items()}
    return sorted((s, float(v)) for s, v in measured.items() if v is not None)


def monitor(root: Path, run_id: str) -> list[dict[str, Any]]:
    return _lines(root / "results" / "runs" / run_id / "monitor.jsonl")


def alerts(root: Path, run_id: str) -> list[dict[str, Any]]:
    path = root / "results" / "runs" / run_id / "alerts.json"
    return list(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else []


def first_alert(root: Path, run_id: str, stat: str) -> int | None:
    steps = [int(a["step"]) for a in alerts(root, run_id) if a["stat"] == stat]
    return min(steps) if steps else None


def rollouts(root: Path, run_id: str, step: int | None = None) -> list[dict[str, Any]]:
    rows = _lines(root / "results" / "runs" / run_id / "rollouts.jsonl")
    return [r for r in rows if step is None or int(r["step"]) == step]
