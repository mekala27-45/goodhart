"""Fetch every file data/sources.lock.json names, verify it, and write it under data/cache.

The customer's data comes through two locks and nothing else: readback's own lock (doorstep,
turnaround and ledger, each pinned by tag, commit and sha256) and the readback files goodhart reads
(the catalog and the golden set), pinned here by readback commit and sha256. Nothing is vendored: the
files land in data/cache, which is never committed, and a file whose digest differs stops the fetch.

    python scripts/fetch_sources.py --lock     # once: write data/sources.lock.json from readback
    python scripts/fetch_sources.py            # fetch and verify
    python scripts/fetch_sources.py --verify   # verify what is on disk, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

RAW = "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
READBACK = "https://github.com/mekala27-45/readback"
READBACK_COMMIT = "c4efff35b8ec24f9f80cef7736cc5bf11334ddb4"
READBACK_FILES = (
    "data/warehouses.lock.json",
    "data/PROVENANCE.md",
    "catalog/measures.yaml",
    "catalog/entities.yaml",
    "catalog/synonyms.yaml",
    "catalog/unit_rules.yaml",
    "catalog/doorstep.yaml",
    "catalog/turnaround.yaml",
    "catalog/ledger.yaml",
    "evals/golden/doorstep.yaml",
    "evals/golden/turnaround.yaml",
    "evals/golden/ledger.yaml",
    "evals/golden/multi.yaml",
    "evals/golden/refusal.yaml",
    "evals/plan.yaml",
    "evals/registration.json",
    "LICENSE",
)


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


def _url(repository: str, ref: str, path: str) -> str:
    owner, repo = repository.rstrip("/").split("/")[-2:]
    return RAW.format(owner=owner, repo=repo, ref=ref, path=path)


def http_get(url: str) -> bytes | None:
    import httpx

    response = httpx.get(url, timeout=180.0, follow_redirects=True)
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


def write_lock(get: Any = http_get, root: Path = ROOT) -> dict[str, Any]:
    """Pin the readback files by commit and digest, and copy readback's warehouse lock as it is."""
    files: list[dict[str, Any]] = []
    warehouses: dict[str, Any] | None = None
    for path in READBACK_FILES:
        data = get(_url(READBACK, READBACK_COMMIT, path))
        if data is None:
            raise FetchError(f"readback has no {path} at {READBACK_COMMIT[:12]}")
        files.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        if path == "data/warehouses.lock.json":
            warehouses = json.loads(data)["warehouses"]
    assert warehouses is not None
    lock = {
        "version": 1,
        "readback": {"repository": READBACK, "commit": READBACK_COMMIT, "files": files},
        "warehouses": warehouses,
    }
    out = root / "data" / "sources.lock.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    return lock


def entries(lock: dict[str, Any], cache: Path) -> list[Entry]:
    out: list[Entry] = []
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
        for t in [*w["tables"], *w.get("documents", [])]:
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


def download(entry: Entry, get: Any) -> str:
    if entry.target.exists() and digest_of(entry.target) == entry.sha256:
        return "kept"
    data = get(_url(entry.repository, entry.ref, entry.path))
    if data is None:
        raise FetchError(f"{entry.group}: {entry.path} is not in {entry.repository} at {entry.ref[:12]}")
    found = hashlib.sha256(data).hexdigest()
    if found != entry.sha256:
        raise FetchError(
            f"{entry.group}: {entry.path} has digest {found[:12]}, the lock says {entry.sha256[:12]}"
        )
    entry.target.parent.mkdir(parents=True, exist_ok=True)
    tmp = entry.target.with_suffix(entry.target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(entry.target)
    return "fetched"


def fetch(lock: dict[str, Any], cache: Path, get: Any = http_get, workers: int = 8) -> dict[str, int]:
    todo = entries(lock, cache)
    counts = {"fetched": 0, "kept": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for outcome in pool.map(lambda e: download(e, get), todo):
            counts[outcome] += 1
    return counts


def verify(lock: dict[str, Any], cache: Path) -> int:
    bad: list[str] = []
    todo = entries(lock, cache)
    for e in todo:
        if not e.target.exists():
            bad.append(f"{e.group}: {e.path} is missing")
        elif digest_of(e.target) != e.sha256:
            bad.append(f"{e.group}: {e.path} does not match the lock")
    if bad:
        raise FetchError("the cache does not match the lock:\n" + "\n".join(bad[:30]))
    return len(todo)


def main() -> int:
    from goodhart_core import paths

    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", action="store_true", help="write data/sources.lock.json from readback")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    lock_path = ROOT / "data" / "sources.lock.json"
    if args.lock:
        lock = write_lock()
        n = sum(len(w["tables"]) for w in lock["warehouses"].values())
        files = len(READBACK_FILES)
        print(f"sources lock: readback {READBACK_COMMIT[:12]}, {files} readback files, {n} warehouse tables")
        return 0
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    cache = paths.cache_dir()
    if args.verify:
        print(f"sources: {verify(lock, cache)} files match the lock")
        return 0
    counts = fetch(lock, cache)
    n = verify(lock, cache)
    print(f"sources: {counts['fetched']} fetched, {counts['kept']} already on disk, {n} verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
