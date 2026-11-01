"""What the kernel is running on, recorded in every run's provenance and in the probe."""

from __future__ import annotations

import os
import platform
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
        "gpus": gpus(),
        "packages": versions(),
    }
