"""What the kernel is running on, recorded in every run's provenance and in the probe."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any

PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "vllm",
    "duckdb",
    "sqlglot",
)


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpus() -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        out.append(
            {
                "index": i,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "memory_bytes": int(props.total_memory),
                "bf16": props.major >= 8,
            }
        )
    return out


SMI_FIELDS = ("index", "driver_version", "clocks.max.sm", "clocks.sm", "power.limit", "temperature.gpu")


def parse_smi(banner: str, rows: str) -> dict[str, Any]:
    """The driver version, the highest CUDA version the driver runs, and each card's clocks and power
    limit, from the nvidia-smi banner and its CSV query."""
    found = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", banner)
    cards = []
    for line in rows.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) == len(SMI_FIELDS):
            cards.append(dict(zip(SMI_FIELDS, values, strict=True)))
    return {
        "driver_version": cards[0]["driver_version"] if cards else None,
        "cuda_driver": f"{found.group(1)}.{found.group(2)}" if found else None,
        "cards": cards,
    }


def driver() -> dict[str, Any]:
    """What the driver supports, read from nvidia-smi. The image's PyTorch says which CUDA it was
    built for; a second environment (vLLM's) has to match the driver instead, so both are recorded."""
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        banner = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        rows = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return parse_smi(banner, rows)


def gpu_indices() -> list[int]:
    """The cards a child process may be pinned to, read from nvidia-smi so the parent never creates a
    CUDA context, narrowed to CUDA_VISIBLE_DEVICES when the parent was given one."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [int(v) for v in visible.split(",") if v.strip().isdigit()]
    cards = driver().get("cards", [])
    return [int(c["index"]) for c in cards if str(c.get("index", "")).isdigit()]


def cuda_driver_version() -> tuple[int, int] | None:
    value = driver().get("cuda_driver")
    if not value:
        return None
    major, minor = str(value).split(".")
    return int(major), int(minor)


def describe() -> dict[str, Any]:
    cuda = None
    try:
        import torch

        cuda = torch.version.cuda
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
        "cuda": cuda,
        "driver": driver(),
        "gpus": gpus(),
        "packages": versions(),
    }
