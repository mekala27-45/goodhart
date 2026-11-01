"""The isolated vLLM environment, built with uv beside the image's Python and never inside it.

vLLM 0.32.0 pins PyTorch 2.13.0, and the PyPI wheels of both are built for CUDA 13, which needs a
driver from the 580 series. The second probe's engine failed to start on the T4, and the three
lines it kept did not say why. So the build reads the driver's CUDA version first. At 13 or above it
installs vLLM from PyPI. Below 13 it takes vLLM's CUDA 12.9 wheel from the project's GitHub release
and PyTorch from the CUDA 12.9 index, which a 12.x driver runs through CUDA's minor version
compatibility. Each plan is tried in order and recorded, with what landed (the freeze), so a later
request can install exactly that.

The training stack (TRL, PEFT and the goodhart dependencies) can be added in the same resolution,
for a GRPO run whose rollouts come from vLLM: TRL imports vLLM in the trainer's own process, so the
two must share an environment. Their versions are the ones requirements-gpu.txt pins, except
transformers, which is left to vLLM's bound.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

VENV = Path("/tmp/goodhart-vllm")
RELEASE_API = "https://api.github.com/repos/vllm-project/vllm/releases/tags/v{version}"
CUDA12_TAG = "cu129"
# The packages a stage needs beside vLLM, by name; their versions come from requirements-gpu.txt.
TRAINING_STACK = (
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "duckdb",
    "sqlglot",
    "pydantic",
    "pyyaml",
    "structlog",
    "httpx",
)

Fetch = Callable[[str], dict[str, Any] | None]


def python() -> Path:
    return VENV / "bin" / "python"


def _release(url: str) -> dict[str, Any] | None:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return data
    except Exception:
        return None


def cuda12_wheel(version: str, fetch: Fetch = _release) -> str | None:
    """The address of vLLM's CUDA 12.9 x86_64 wheel in its GitHub release, or None."""
    release = fetch(RELEASE_API.format(version=version))
    if not release:
        return None
    pattern = re.compile(rf"^vllm-{re.escape(version)}\+{CUDA12_TAG}-.*x86_64\.whl$")
    for asset in release.get("assets", []):
        if pattern.match(str(asset.get("name", ""))):
            return str(asset.get("browser_download_url"))
    return None


def plans(version: str, cuda: tuple[int, int] | None, fetch: Fetch = _release) -> list[dict[str, Any]]:
    """The installs to try, in order, for a driver that runs CUDA up to the given version."""
    pypi = {"source": "pypi", "requirement": f"vllm=={version}", "torch_backend": None}
    wheel = cuda12_wheel(version, fetch)
    cuda12 = {"source": f"github-{CUDA12_TAG}", "requirement": wheel, "torch_backend": CUDA12_TAG}
    if cuda is not None and cuda >= (13, 0):
        return [pypi, cuda12] if wheel else [pypi]
    found: list[dict[str, Any]] = []
    if wheel:
        found.append(cuda12)
        found.append({"source": f"github-{CUDA12_TAG}-auto", "requirement": wheel, "torch_backend": "auto"})
    found.append({"source": "pypi-auto", "requirement": f"vllm=={version}", "torch_backend": "auto"})
    return found


def pins(requirements: Path, names: Sequence[str]) -> list[str]:
    """name==version for each name, read from a uv compiled requirements file."""
    wanted = {n.lower().replace("_", "-") for n in names}
    out = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", line.strip())
        if m and m.group(1).lower().replace("_", "-") in wanted:
            out.append(f"{m.group(1)}=={m.group(2)}")
    return sorted(set(out))


def _env() -> dict[str, str]:
    """The environment for a command on the vLLM environment: none of the image's import or CUDA
    library paths (goodhart_train.isolate)."""
    from goodhart_train.isolate import child_env

    return child_env(None, isolated=True)


def _uv(*args: str, timeout_s: float = 1800.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "uv", *args], capture_output=True, text=True, timeout=timeout_s, env=_env()
    )


def _bootstrap() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True, capture_output=True, timeout=300)
    if not python().exists():
        # The image's Python has no ensurepip, so python -m venv fails there; uv makes the environment.
        made = _uv("venv", "-q", "--python", sys.executable, str(VENV), timeout_s=120)
        if made.returncode != 0:
            raise RuntimeError("uv venv failed: " + made.stderr.strip()[-500:])


def _freeze() -> list[str]:
    done = _uv("pip", "freeze", "--python", str(python()), timeout_s=120)
    return [ln for ln in done.stdout.splitlines() if ln.strip()]


def _torch() -> dict[str, Any]:
    code = "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda}))"
    done = subprocess.run([str(python()), "-c", code], capture_output=True, text=True, timeout=300, env=_env())
    try:
        found: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
        return found
    except (IndexError, json.JSONDecodeError):
        return {"error": done.stderr.strip()[-300:]}


def install(plan: dict[str, Any], extra: Sequence[str] = (), timeout_s: float = 1800.0) -> dict[str, Any]:
    """One install into the environment: vLLM by the plan, and the extra requirements beside it."""
    _bootstrap()
    args = ["pip", "install", "-q", "--python", str(python()), str(plan["requirement"]), *extra]
    if plan.get("torch_backend"):
        args += ["--torch-backend", str(plan["torch_backend"])]
    t0 = time.monotonic()
    try:
        done = _uv(*args, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        return {"source": plan["source"], "installed": False, "error": f"the install ran past {timeout_s:.0f} s"}
    found: dict[str, Any] = {
        "source": plan["source"],
        "torch_backend": plan.get("torch_backend"),
        "extra": list(extra),
        "seconds": round(time.monotonic() - t0, 1),
        "installed": done.returncode == 0,
    }
    if done.returncode != 0:
        from goodhart_train.isolate import tail

        found["error"] = tail(done.stderr or done.stdout, 25)
    return found


def describe() -> dict[str, Any]:
    """What the environment holds now: its PyTorch and the frozen package list."""
    return {"python": str(python()), **_torch(), "freeze": _freeze()}


def ensure(version: str, extra: Sequence[str] = ()) -> str:
    """The environment's python for a later stage, built by the first plan that installs."""
    from goodhart_train.env import cuda_driver_version

    marker = VENV / "goodhart-installed.json"
    if python().exists() and marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("version") == version and set(extra) <= set(state.get("extra", [])):
            return str(python())
    errors = []
    for plan in plans(version, cuda_driver_version()):
        found = install(plan, extra)
        if found["installed"]:
            marker.write_text(json.dumps({"version": version, "extra": list(extra), **found}), encoding="utf-8")
            return str(python())
        errors.append(f"{plan['source']}: {found.get('error', '')[-300:]}")
    raise RuntimeError("vLLM did not install: " + " | ".join(errors))
