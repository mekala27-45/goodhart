"""Fetch Spider dev and its test suite databases, check them, and repack them for the release.

Runs in the spider workflow on Actions, the one place with a route to Hugging Face and Google Drive;
the build machine never calls either. It needs httpx and gdown.

    python scripts/fetch_spider.py --out build/spider

Spider (Yu et al., 2018; CC BY-SA 4.0) comes from a re-host of the canonical zip on Hugging Face,
checked against the sha256 that re-host publishes, with the canonical Google Drive file as the
fallback under the same digest. The distilled test suites (Zhong et al., 2020) come from the Google
Drive file the test suite evaluation's README names; that file has no published digest, so the first
fetch records it and every later fetch is checked against data/spider.lock.json.

The output keeps only what the evaluation of Spider dev reads: dev.json, tables.json, and for each of
the dev set's databases the original and every test suite database. It is one tarball with sorted
entries, zero times and no owners, so the same inputs always give the same bytes, and a manifest with
every digest beside it. Both go to the data release; goodhart spider --fetch reads them from there.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "spider.lock.json"

SPIDER_URL = "https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip"
SPIDER_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
SPIDER_DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
TEST_SUITE_DRIVE_ID = "1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w"
ASSET = "spider_dev.tar.gz"
MANIFEST = "spider_dev.json"
LIMIT_BYTES = 2_000_000_000  # one release asset


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _http(url: str) -> bytes:
    import httpx

    with httpx.Client(follow_redirects=True, timeout=600.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def _drive(file_id: str, dest: Path) -> bytes:
    import gdown

    got = gdown.download(id=file_id, output=str(dest), quiet=True)
    if not got:
        raise RuntimeError(f"Google Drive refused file {file_id}")
    return dest.read_bytes()


def spider_zip(work: Path) -> tuple[bytes, str]:
    """The canonical Spider zip and where it came from, checked against the published digest."""
    try:
        body, source = _http(SPIDER_URL), SPIDER_URL
    except Exception as err:  # the re-host is down: the canonical file, under the same digest
        print(f"the Hugging Face re-host failed ({type(err).__name__}); trying Google Drive", file=sys.stderr)
        body, source = _drive(SPIDER_DRIVE_ID, work / "spider_data.zip"), f"google drive {SPIDER_DRIVE_ID}"
    if sha256(body) != SPIDER_SHA256:
        raise ValueError(f"spider_data.zip from {source} is {sha256(body)}, not the published {SPIDER_SHA256}")
    return body, source


def known_test_suite_digest() -> str | None:
    if not LOCK.exists():
        return None
    found = json.loads(LOCK.read_text(encoding="utf-8")).get("sources", {}).get("test_suite", {})
    return found.get("sha256")


def members(archive: zipfile.ZipFile) -> dict[PurePosixPath, zipfile.ZipInfo]:
    return {PurePosixPath(i.filename): i for i in archive.infolist() if not i.is_dir()}


def dev_files(spider: zipfile.ZipFile) -> tuple[dict[str, bytes], list[str]]:
    """dev.json, tables.json and the dev databases' originals, keyed by their place in the tarball."""
    found = members(spider)
    by_name = {p.name: p for p in found if p.name in {"dev.json", "tables.json"} and "__MACOSX" not in p.parts}
    out = {f"spider/{name}": spider.read(found[by_name[name]]) for name in ("dev.json", "tables.json")}
    dev = json.loads(out["spider/dev.json"])
    db_ids = sorted({str(q["db_id"]) for q in dev})
    for path, info in found.items():
        if "__MACOSX" in path.parts or path.suffix != ".sqlite" or len(path.parts) < 2:
            continue
        db = path.parts[-2]
        if path.parts[-3:-2] == ("database",) and db in db_ids and path.name == f"{db}.sqlite":
            out[f"database/{db}/{path.name}"] = spider.read(info)
    missing = [db for db in db_ids if f"database/{db}/{db}.sqlite" not in out]
    if missing:
        raise ValueError(f"spider_data.zip has no database for {missing}")
    return out, db_ids


def suite_files(suite: zipfile.ZipFile, db_ids: list[str]) -> dict[str, bytes]:
    """Every test suite database of the dev set's databases, beside the originals."""
    out: dict[str, bytes] = {}
    for path, info in members(suite).items():
        if "__MACOSX" in path.parts or path.suffix != ".sqlite" or len(path.parts) < 2:
            continue
        db = path.parts[-2]
        if db in db_ids:
            out[f"database/{db}/{path.name}"] = suite.read(info)
    return out


def tarball(files: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(files[name]))
    return raw.getvalue()


def build(out: Path, with_suite: bool = True) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    body, source = spider_zip(work)
    files, db_ids = dev_files(zipfile.ZipFile(io.BytesIO(body)))
    suite: dict[str, Any] = {"drive_id": TEST_SUITE_DRIVE_ID}
    if with_suite:
        try:
            suite_body = _drive(TEST_SUITE_DRIVE_ID, work / "test_suite.zip")
        except Exception as err:
            raise RuntimeError(f"the test suite databases could not be fetched: {err}") from err
        digest, known = sha256(suite_body), known_test_suite_digest()
        if known and digest != known:
            raise ValueError(f"the test suite zip is {digest}, the lock records {known}")
        added = suite_files(zipfile.ZipFile(io.BytesIO(suite_body)), db_ids)
        suite.update(sha256=digest, bytes=len(suite_body), recorded_now=known is None, databases=len(added))
        files.update(added)
    packed = tarball(files)
    if len(packed) > LIMIT_BYTES:
        raise ValueError(f"the tarball is {len(packed)} bytes, over one release asset")
    (out / ASSET).write_bytes(packed)
    manifest = {
        "about": "Spider dev (CC BY-SA 4.0, Yu et al. 2018) and its distilled test suite databases "
        "(Zhong et al. 2020), repacked with only what the evaluation of Spider dev reads.",
        "sources": {
            "spider": {"from": source, "sha256": SPIDER_SHA256, "canonical_drive_id": SPIDER_DRIVE_ID},
            "test_suite": suite if with_suite else {"drive_id": TEST_SUITE_DRIVE_ID, "fetched": False},
        },
        "asset": {"name": ASSET, "sha256": sha256(packed), "bytes": len(packed)},
        "db_ids": db_ids,
        "questions": len(json.loads(files["spider/dev.json"])),
        "files": {name: sha256(files[name]) for name in sorted(files)},
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "spider")
    parser.add_argument("--without-test-suite", action="store_true")
    args = parser.parse_args()
    m = build(args.out, with_suite=not args.without_test_suite)
    summary = {k: m[k] for k in ("asset", "db_ids", "questions")}
    summary["sources"] = m["sources"]
    summary["databases"] = sum(1 for f in m["files"] if f.startswith("database/"))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
