"""The lock fetcher refuses a file whose digest differs and keeps one that matches; the logs drop
refused keys and mask credentials; the manifest refuses a duplicate key and a non finite number."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from goodhart_core import fetch as f


def _lock(files: dict[str, bytes]) -> dict[str, object]:
    return {
        "version": 1,
        "readback": {"repository": "https://github.com/o/readback", "commit": "c" * 40, "files": []},
        "warehouses": {
            "doorstep": {
                "repository": "https://github.com/o/doorstep",
                "commit": "d" * 40,
                "tables": [{"path": p, "sha256": hashlib.sha256(b).hexdigest()} for p, b in files.items()],
            }
        },
    }


def test_fetch_downloads_checks_and_then_keeps(tmp_path: Path) -> None:
    files = {"marts/a.parquet": b"alpha", "marts/b.parquet": b"beta"}
    served: list[str] = []

    def get(address: str) -> bytes | None:
        served.append(address)
        return next((b for p, b in files.items() if address.endswith(p)), None)

    lock = _lock(files)
    assert f.fetch(lock, tmp_path, get=get, warehouses_only=True) == {"fetched": 2, "kept": 0}
    assert all("raw.githubusercontent.com/o/doorstep/" + "d" * 40 in a for a in served)
    assert f.fetch(lock, tmp_path, get=get, warehouses_only=True) == {"fetched": 0, "kept": 2}
    assert f.verify(lock, tmp_path, warehouses_only=True) == 2


def test_fetch_refuses_a_file_whose_digest_differs(tmp_path: Path) -> None:
    lock = _lock({"marts/a.parquet": b"alpha"})
    with pytest.raises(f.FetchError, match="the lock says"):
        f.fetch(lock, tmp_path, get=lambda _: b"tampered", warehouses_only=True)
    with pytest.raises(f.FetchError, match="is not in"):
        f.fetch(lock, tmp_path, get=lambda _: None, warehouses_only=True)
    with pytest.raises(f.FetchError, match="missing"):
        f.verify(lock, tmp_path, warehouses_only=True)


def test_a_lock_without_its_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"warehouses": {}}), encoding="utf-8")
    with pytest.raises(f.FetchError):
        f.load_lock(path)


def test_logs_drop_refused_keys_and_mask_credentials() -> None:
    from goodhart_core import log

    event = {"event": "x", "sql": "select 1", "note": "token KGAT_" + "a" * 12 + " in a sentence", "count": 3}
    out = log._scrub(None, "info", dict(event))
    assert out["sql"] == "[dropped]" and "KGAT_" not in out["note"] and "[masked]" in out["note"]
    assert out["count"] == 3
    # Built from parts so this file never holds a connection string the token scan would stop.
    url = "postgresql://" + "user:" + "pw" + "x" * 6 + "@host/db"
    assert "pwxxxxxx" not in log._scrub(None, "info", {"event": url})["event"]
    log.configure()
    assert log.get("test") is not None


def test_the_manifest_refuses_a_duplicate_and_a_non_finite_value() -> None:
    from goodhart_core.manifest import ManifestBuilder

    b = ManifestBuilder("2026-10-31")
    b.value("a.b", 0.5, "pct1", origin="test", population="fixture", source="questions")
    with pytest.raises(KeyError):
        b.value("a.b", 0.6, "pct1", origin="test", population="fixture", source="questions")
    with pytest.raises(ValueError):
        b.value("a.c", math.nan, "pct1", origin="test", population="fixture", source="questions")
    with pytest.raises(ValueError):
        b.value("a.d", 0.1, "pct1", origin="test", population="fixture", source="hearsay")
    with pytest.raises(KeyError):
        b.value("a.e", 0.1, "no_such_format", origin="test", population="fixture", source="questions")
