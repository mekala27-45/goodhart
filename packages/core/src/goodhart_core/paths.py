"""Where things live. Every path is relative to the repository root, found once.

The fetched warehouses, the readback files and the twins are caches: they live under data/cache,
are never committed, and are rebuilt from the locks by `make fetch` and `make twins`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def root() -> Path:
    env = os.environ.get("GOODHART_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "sources.lock.json").exists() and (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def data_dir() -> Path:
    return root() / "data"


def cache_dir() -> Path:
    env = os.environ.get("GOODHART_CACHE")
    return Path(env).resolve() if env else data_dir() / "cache"


def warehouses_dir() -> Path:
    return cache_dir() / "warehouses"


def readback_dir() -> Path:
    """The readback catalog and golden set, fetched by lock at the pinned commit."""
    return cache_dir() / "readback"


def twins_dir() -> Path:
    return cache_dir() / "twins"


def questions_dir() -> Path:
    return data_dir() / "questions"


def results_dir() -> Path:
    return root() / "results"


def evaluation_dir() -> Path:
    return root() / "evaluation"


def config_dir() -> Path:
    return root() / "config"
