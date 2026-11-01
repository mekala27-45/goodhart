"""Work in a child process, so a CUDA failure in one model cannot poison the next.

A device-side assert leaves the CUDA context of its process unusable. In the second probe one
candidate's assert failed every load after it, and the GRPO smoke with them. So each probe
candidate, each smoke run and each generation shard runs in its own Python process, on the GPU it is
given, and hands back JSON. The parent never loads a model.

A child runs `python -m goodhart_train.worker <task> <args.json> <result.json>`, or any command for
an environment that has no goodhart_train (the vLLM worker script). Its output streams go to log
files; the last lines of stderr come back in the outcome, and the parent copies a short tail of each
log into the run's output folder so a failure can be read after the kernel is gone.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TAIL_LINES = 40
LOG_TAIL_BYTES = 20_000
# Lines worth keeping from a long log when the last ones are only a stack of frames.
MARKERS = ("Error", "error:", "CUDA", "cuda", "driver", "not supported", "capability", "Killed", "memory")


def _is_library(path: str) -> bool:
    """A folder of installed packages or of the standard library, which a child gets from its own
    interpreter and must not get from the parent's."""
    resolved = Path(path).resolve()
    if any(part in ("site-packages", "dist-packages") for part in resolved.parts):
        return True
    roots = {sysconfig.get_paths().get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")}
    return any(r and resolved.is_relative_to(Path(r).resolve()) for r in roots)


def child_path() -> str:
    """The folders the parent added to its import path (in a kernel, the bundle's src folder), and
    none of its installed packages. A child in the vLLM environment imports the goodhart packages
    from the bundle and everything else from its own environment: the image's PyTorch 2.11 ahead of
    the environment's 2.13 on the path would break vLLM."""
    return os.pathsep.join(p for p in sys.path if p and Path(p).is_dir() and not _is_library(p))


def library_path(value: str) -> str:
    """LD_LIBRARY_PATH without the CUDA toolkit folders of the image, keeping the driver's. A library
    path is searched before a wheel's own run path, so the image's CUDA 12.8 runtime would otherwise
    be loaded by the vLLM environment's PyTorch, built against another runtime."""
    kept = [p for p in value.split(os.pathsep) if p and not ("/cuda" in p and "nvidia" not in p)]
    return os.pathsep.join(kept)


def child_env(gpu: int | None, extra: Mapping[str, str] | None = None, isolated: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = child_path()
    env["PYTHONUNBUFFERED"] = "1"
    if isolated and "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = library_path(env["LD_LIBRARY_PATH"])
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.update(extra or {})
    return env


def tail(text: str, lines: int = TAIL_LINES) -> str:
    """The last lines, plus any earlier line that names an error, so the root cause is not lost
    under a stack trace (the second probe kept three lines and the cause was above them)."""
    all_lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    last = all_lines[-lines:]
    earlier = [ln for ln in all_lines[:-lines] if any(m in ln for m in MARKERS)][-15:]
    return "\n".join([*earlier, *(["..."] if earlier else []), *last])


@dataclass
class Running:
    name: str
    process: subprocess.Popen[bytes]
    result: Path
    stdout: Path
    stderr: Path
    started: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class Outcome:
    name: str
    ok: bool
    returncode: int | None
    seconds: float
    result: dict[str, Any]
    error: str

    def as_record(self) -> dict[str, Any]:
        if self.ok:
            return self.result
        return {**self.result, "ok": False, "returncode": self.returncode, "error": self.error}


def start(
    name: str,
    command: Sequence[str],
    workdir: Path,
    gpu: int | None = None,
    extra_env: Mapping[str, str] | None = None,
    isolated: bool = False,
) -> Running:
    """Start a child; isolated for one that runs in another environment's interpreter (vLLM's)."""
    logs = workdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout, stderr = logs / f"{name}.out", logs / f"{name}.err"
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            list(command), stdout=out, stderr=err, env=child_env(gpu, extra_env, isolated=isolated)
        )
    return Running(name, process, workdir / f"{name}.result.json", stdout, stderr)


def wait(running: Running, timeout_s: float) -> Outcome:
    try:
        code: int | None = running.process.wait(timeout=max(1.0, timeout_s))
        timed_out = False
    except subprocess.TimeoutExpired:
        running.process.kill()
        running.process.wait()
        code, timed_out = None, True
    seconds = round(time.monotonic() - running.started, 1)
    result: dict[str, Any] = {}
    if running.result.exists():
        try:
            result = json.loads(running.result.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {}
    err = running.stderr.read_text(encoding="utf-8", errors="replace") if running.stderr.exists() else ""
    error = "" if code == 0 else tail(err)
    if timed_out:
        error = f"stopped after {timeout_s:.0f} s\n{error}"
    return Outcome(
        running.name, ok=code == 0 and bool(result), returncode=code, seconds=seconds, result=result, error=error
    )


def worker_command(task: str, args_path: Path, result_path: Path, python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", "goodhart_train.worker", task, str(args_path), str(result_path)]


def run_task(
    task: str,
    args: dict[str, Any],
    workdir: Path,
    name: str,
    gpu: int | None = None,
    timeout_s: float = 1800.0,
    python: str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> Outcome:
    """One worker task in a child process, waited for. A python other than this one is another
    environment's, so the child is started isolated."""
    workdir.mkdir(parents=True, exist_ok=True)
    args_path = workdir / f"{name}.args.json"
    args_path.write_text(json.dumps(args, default=str), encoding="utf-8")
    result_path = workdir / f"{name}.result.json"
    result_path.unlink(missing_ok=True)
    isolated = python is not None and Path(python).resolve() != Path(sys.executable).resolve()
    command = worker_command(task, args_path, result_path, python)
    running = start(name, command, workdir, gpu, extra_env, isolated=isolated)
    return wait(running, timeout_s)


def keep_log(workdir: Path, name: str, out: Path) -> None:
    """A short tail of a child's stderr and stdout in the run's outputs, for reading after the kernel."""
    target = out / "logs"
    target.mkdir(parents=True, exist_ok=True)
    for suffix in ("err", "out"):
        source = workdir / "logs" / f"{name}.{suffix}"
        if source.exists():
            data = source.read_bytes()[-LOG_TAIL_BYTES:]
            (target / f"{name}.{suffix}.txt").write_bytes(data)
