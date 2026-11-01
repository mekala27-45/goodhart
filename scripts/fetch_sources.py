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
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from goodhart_core import paths
from goodhart_core.fetch import FetchError, Getter, fetch, http_get, load_lock, url, verify

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


def write_lock(get: Getter = http_get, root: Path = ROOT) -> dict[str, Any]:
    """Pin the readback files by commit and digest, and copy readback's warehouse lock as it is."""
    files: list[dict[str, Any]] = []
    warehouses: dict[str, Any] | None = None
    for path in READBACK_FILES:
        data = get(url(READBACK, READBACK_COMMIT, path))
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", action="store_true", help="write data/sources.lock.json from readback")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.lock:
        lock = write_lock()
        n = sum(len(w["tables"]) for w in lock["warehouses"].values())
        print(f"sources lock: readback {READBACK_COMMIT[:12]}, {len(READBACK_FILES)} files, {n} warehouse tables")
        return 0
    lock = load_lock(ROOT / "data" / "sources.lock.json")
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
