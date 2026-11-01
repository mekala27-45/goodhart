"""The training stage cannot see the twins.

Two lines of defence. The kernel bundle never carries goodhart_heldout or the twin files, so the
import fails for want of a module. And this finder, installed first thing by every stage entry point,
refuses the import by name even where the package is installed (the repository's own environment),
so a training stage run locally fails the same way it fails on the GPU.
"""

from __future__ import annotations

import importlib.abc
import sys
from collections.abc import Sequence
from types import ModuleType

REFUSED = ("goodhart_heldout",)


class HeldoutRefused(ImportError):
    pass


class _Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None) -> None:
        if fullname.split(".")[0] in REFUSED:
            raise HeldoutRefused(f"{fullname} is the held-out checks; a training stage may not import it")
        return None


def install() -> None:
    """Refuse the held-out package for the rest of the process, and drop it if already imported."""
    if not any(isinstance(f, _Refuse) for f in sys.meta_path):
        sys.meta_path.insert(0, _Refuse())
    for name in list(sys.modules):
        if name.split(".")[0] in REFUSED:
            del sys.modules[name]
