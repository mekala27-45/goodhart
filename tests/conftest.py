"""Shared fixtures and the availability probes behind the named skip markers.

A test that needs a dependency the environment lacks is skipped by name, never silently passed, and
CI sets GOODHART_REQUIRE to the dependencies it promises, so a skip there fails the run instead.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The repository root stays off the import path: folders there such as requests/ and kernels/ would
# pass for missing libraries (TRL checks for requests by spec alone). python -m pytest puts the
# working directory first, so it is dropped here. The gate scripts are standalone files, imported
# from their own folder, which holds nothing but those scripts.
sys.path[:] = [p for p in sys.path if p not in {"", ".", str(ROOT)}]
SCRIPTS = str(ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

REQUIRED = {d.strip() for d in os.environ.get("GOODHART_REQUIRE", "").split(",") if d.strip()}


def _have_torch() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "trl", "peft"))


def _have_warehouses() -> bool:
    lock = ROOT / "data" / "sources.lock.json"
    return lock.exists() and (ROOT / "data" / "cache" / "warehouses" / "doorstep").exists()


def _have_postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _have_floor() -> bool:
    return importlib.util.find_spec("readback_planner") is not None


AVAILABLE = {
    "torch": _have_torch,
    "warehouses": _have_warehouses,
    "postgres": _have_postgres,
    "floor": _have_floor,
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing_required = sorted(name for name in REQUIRED if name in AVAILABLE and not AVAILABLE[name]())
    if missing_required:
        raise pytest.UsageError(f"this run requires {', '.join(missing_required)}, and it is missing here")
    for item in items:
        for name, probe in AVAILABLE.items():
            if name in item.keywords and not probe():
                item.add_marker(pytest.mark.skip(reason=f"needs {name}"))


@pytest.fixture
def root() -> Path:
    return ROOT
