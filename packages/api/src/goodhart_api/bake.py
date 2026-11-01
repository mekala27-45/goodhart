"""Bake the container: the warehouses by lock and the kept twins by their lock, every file checked.

    python -m goodhart_api.bake

Run in the image build (Dockerfile.vercel), so a cold start reads files already on disk and verified,
and never downloads. A file whose digest differs from its lock stops the build.
"""

from __future__ import annotations

import sys

from goodhart_core import paths
from goodhart_core.fetch import fetch, load_lock, verify
from goodhart_heldout.pipeline import materialize


def main() -> int:
    root = paths.root()
    lock = load_lock(root / "data" / "sources.lock.json")
    counts = fetch(lock, paths.cache_dir())
    checked = verify(lock, paths.cache_dir())
    twins = materialize(root)
    print(f"bake: {counts['fetched']} fetched, {checked} files verified, twins {twins}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
