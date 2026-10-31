"""Digests for the locks, the plan, the requests, the checkpoints and the caches.

Every digest is SHA-256 over a canonical form, so the same input gives the same key on any machine
and under any PYTHONHASHSEED. A checkpoint or a cache entry is keyed by the digest of what it was made
from (the Day 13 rule), never by a name or a time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """JSON with sorted keys, no spaces and no NaN: the form every digest is taken over."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False, default=str
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_files(paths: Iterable[Path], root: Path) -> str:
    """One digest over many files: each relative path and its digest, sorted, so order never matters."""
    entries = sorted((p.relative_to(root).as_posix(), sha256_file(p)) for p in paths)
    return sha256_json(entries)


def short(digest: str, n: int = 12) -> str:
    return digest[:n]
