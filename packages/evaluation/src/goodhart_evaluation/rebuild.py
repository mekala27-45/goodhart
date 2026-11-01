"""goodhart rebuild-check: the questions, twins and mutants rebuild to the committed bytes.

The data build runs again in a scratch copy of the repository, under a different PYTHONHASHSEED,
reading the same fetched warehouses and readback files, and every committed output must come back
byte for byte: the sets, the prompts, the SFT targets, the manifest, the twins lock, the planted key
hash, the mutation score and the contamination report. A difference names the file. Nothing in the
working tree is written; the twins and the audit key are rebuilt into a scratch cache.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from goodhart_core import paths
from goodhart_core.hashing import sha256_file

OUTPUTS = (
    "data/questions",
    "data/twins.lock.json",
    "data/audit_key.sha256",
    "results/mutation.json",
    "results/contamination.json",
)
INPUTS = ("config", "data/sources.lock.json", "pyproject.toml")
SHARED_CACHE = ("warehouses", "readback")
HASH_SEED = "12345"


@dataclass
class Rebuild:
    compared: int = 0
    differ: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.differ or self.missing or self.extra)


def _files(base: Path, rel: str) -> dict[str, Path]:
    target = base / rel
    if target.is_file():
        return {rel: target}
    if not target.exists():
        return {}
    return {p.relative_to(base).as_posix(): p for p in sorted(target.rglob("*")) if p.is_file()}


def compare(committed: Path, rebuilt: Path) -> Rebuild:
    out = Rebuild()
    for rel in OUTPUTS:
        want, got = _files(committed, rel), _files(rebuilt, rel)
        out.missing += sorted(set(want) - set(got))
        out.extra += sorted(set(got) - set(want))
        for name in sorted(set(want) & set(got)):
            out.compared += 1
            if sha256_file(want[name]) != sha256_file(got[name]):
                out.differ.append(name)
    return out


def stage(root: Path, scratch: Path) -> tuple[Path, Path]:
    """A scratch repository with the build inputs and a scratch cache that shares the fetched files."""
    repo, cache = scratch / "repo", scratch / "cache"
    for rel in INPUTS:
        src, dst = root / rel, repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    cache.mkdir(parents=True)
    for name in SHARED_CACHE:
        (cache / name).symlink_to(paths.cache_dir() / name, target_is_directory=True)
    return repo, cache


def run(root: Path | None = None) -> Rebuild:
    root = root or paths.root()
    with tempfile.TemporaryDirectory(prefix="goodhart-rebuild-") as tmp:
        repo, cache = stage(root, Path(tmp))
        env = {**os.environ, "GOODHART_ROOT": str(repo), "GOODHART_CACHE": str(cache), "PYTHONHASHSEED": HASH_SEED}
        done = subprocess.run(
            [sys.executable, "-c", "from goodhart_evaluation.data import build; build()"],
            env=env,
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            raise RuntimeError("the rebuild failed:\n" + done.stderr[-4000:])
        return compare(root, repo)
