"""Fetch the files a source lock names and verify each by sha256.

Used by scripts/fetch_sources.py in the repository, by CI, by the container build and by the GPU
kernels, which fetch the warehouses from GitHub at the pinned commits rather than carry them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAW = "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

Getter = Callable[[str], bytes | None]


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    group: str
    repository: str
    ref: str
    path: str
    sha256: str
    target: Path


def url(repository: str, ref: str, path: str) -> str:
    owner, repo = repository.rstrip("/").split("/")[-2:]
    return RAW.format(owner=owner, repo=repo, ref=ref, path=path)


def http_get(address: str) -> bytes | None:
    import httpx

    response = httpx.get(address, timeout=180.0, follow_redirects=True)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content


def digest_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entries(lock: dict[str, Any], cache: Path, warehouses_only: bool = False) -> list[Entry]:
    out: list[Entry] = []
    if not warehouses_only:
        rb = lock["readback"]
        for f in rb["files"]:
            out.append(
                Entry(
                    "readback",
                    rb["repository"],
                    rb["commit"],
                    f["path"],
                    f["sha256"],
                    cache / "readback" / f["path"],
                )
            )
    for name, w in sorted(lock["warehouses"].items()):
        rows = w["tables"] if warehouses_only else [*w["tables"], *w.get("documents", [])]
        for t in rows:
            out.append(
                Entry(
                    name,
                    w["repository"],
                    w["commit"],
                    t["path"],
                    t["sha256"],
                    cache / "warehouses" / name / t["path"],
                )
            )
    return out


def download(entry: Entry, get: Getter) -> str:
    if entry.target.exists() and digest_of(entry.target) == entry.sha256:
        return "kept"
    data = get(url(entry.repository, entry.ref, entry.path))
    if data is None:
        raise FetchError(f"{entry.group}: {entry.path} is not in {entry.repository} at {entry.ref[:12]}")
    found = hashlib.sha256(data).hexdigest()
    if found != entry.sha256:
        raise FetchError(f"{entry.group}: {entry.path} has digest {found[:12]}, the lock says {entry.sha256[:12]}")
    entry.target.parent.mkdir(parents=True, exist_ok=True)
    tmp = entry.target.with_suffix(entry.target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(entry.target)
    return "fetched"


def fetch(
    lock: dict[str, Any], cache: Path, get: Getter = http_get, workers: int = 8, warehouses_only: bool = False
) -> dict[str, int]:
    todo = entries(lock, cache, warehouses_only)
    counts = {"fetched": 0, "kept": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for outcome in pool.map(lambda e: download(e, get), todo):
            counts[outcome] += 1
    return counts


def verify(lock: dict[str, Any], cache: Path, warehouses_only: bool = False) -> int:
    bad: list[str] = []
    todo = entries(lock, cache, warehouses_only)
    for e in todo:
        if not e.target.exists():
            bad.append(f"{e.group}: {e.path} is missing")
        elif digest_of(e.target) != e.sha256:
            bad.append(f"{e.group}: {e.path} does not match the lock")
    if bad:
        raise FetchError("the cache does not match the lock:\n" + "\n".join(bad[:30]))
    return len(todo)


def load_lock(path: Path) -> dict[str, Any]:
    lock: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("version") != 1 or "warehouses" not in lock:
        raise FetchError(f"{path} is not a version 1 source lock")
    return lock
