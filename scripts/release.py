"""The adapters release: each trained method's chosen checkpoint, packed with its model card and the
statement, and every package's sha256 recorded before the tag.

    python scripts/release.py --plan      # write results/release.json from the promoted runs
    python scripts/release.py --pack DIR  # download, check and pack every planned adapter into DIR,
                                          # and refuse a package whose digest differs from the plan

A method's adapter is the one its sample run sampled: the training run its request names as input
and the step it names (adapter_step), whose weights are in the runs release with their digests in
the run's assets.json and whose config is committed beside them. A package is a tarball with sorted
entries, zero times and no owners, so --plan on the build machine and --pack in the release workflow
produce the same bytes from the same inputs, and the tag carries digests the release must match.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TAG = "v0.1.0"
REPOSITORY = "mekala27-45/goodhart"
ASSET_URL = "https://github.com/{repo}/releases/download/{release}/{name}"
METHODS = ("M1", "M2", "M3", "M4")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get(url: str) -> bytes:
    import httpx

    with httpx.Client(follow_redirects=True, timeout=600.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def chosen(root: Path) -> dict[str, dict[str, Any]]:
    """Per method, the training run and step its latest sample run sampled."""
    out: dict[str, dict[str, Any]] = {}
    for req_path in sorted((root / "requests").glob("*.json")):
        req = json.loads(req_path.read_text(encoding="utf-8"))
        method = str(req.get("arm") or "")
        if req.get("stage") != "sample" or method not in METHODS or not req.get("inputs"):
            continue
        if req.get("hyperparameters", {}).get("interior"):
            continue
        if not (root / "results" / "runs" / req["id"] / "provenance.json").exists():
            continue
        step = req.get("hyperparameters", {}).get("adapter_step")
        out[method] = {"sample_run": req["id"], "run": req["inputs"][-1], "step": step}
    return out


def adapter_files(root: Path, run: str, step: int | None) -> dict[str, dict[str, Any]]:
    """The adapter's files: in git (the config) or in the runs release (the weights), with digests."""
    folder = "adapter" if step is None else f"adapters/step-{int(step)}"
    run_dir = root / "results" / "runs" / run
    assets = _json(run_dir / "assets.json") or {}
    prov = _json(run_dir / "provenance.json") or {}
    files: dict[str, dict[str, Any]] = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        rel = f"{folder}/{name}"
        if rel in assets:
            files[name] = {
                "asset": assets[rel]["asset"],
                "release": assets[rel]["release"],
                "sha256": assets[rel]["sha256"],
            }
        elif rel in prov.get("files", {}):
            files[name] = {"path": f"results/runs/{run}/{rel}", "sha256": prov["files"][rel]}
        else:
            raise FileNotFoundError(f"{run} kept no {rel}")
    return files


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


def contents(root: Path, method: str, entry: dict[str, Any], get: Callable[[str], bytes] = _get) -> dict[str, bytes]:
    """Every file of one package, each checked against its digest."""
    out: dict[str, bytes] = {}
    for name, f in entry["files"].items():
        if "path" in f:
            body = (root / f["path"]).read_bytes()
        else:
            body = get(ASSET_URL.format(repo=REPOSITORY, release=f["release"], name=f["asset"]))
        if _sha256(body) != f["sha256"]:
            raise ValueError(f"{method}: {name} does not match its digest")
        out[f"adapter/{name}"] = body
    card = root / "docs" / "models" / f"{method}.md"
    out["MODEL_CARD.md"] = card.read_bytes()
    out["LICENSE"] = (root / "LICENSE").read_bytes()
    return out


def plan(root: Path, get: Callable[[str], bytes] = _get) -> dict[str, Any]:
    gates = _json(root / "results" / "gates.json") or {}
    selection = _json(root / "results" / "selection.json") or {}
    packages = []
    for method, c in sorted(chosen(root).items()):
        entry = {"method": method, **c, "files": adapter_files(root, c["run"], c["step"])}
        body = tarball(contents(root, method, entry, get))
        step = "final" if c["step"] is None else f"step-{c['step']}"
        entry.update(name=f"goodhart-{method}-{step}.tar.gz", sha256=_sha256(body), bytes=len(body))
        packages.append(entry)
    record = {
        "tag": TAG,
        "champion": gates.get("champion"),
        "base": selection.get("main"),
        "assets": packages,
        "note": "Each package holds a LoRA adapter for its base model (its model card names the base and its "
        "license), the model card and the license. goodhart is a portfolio demonstration.",
    }
    (root / "results" / "release.json").write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def pack(root: Path, dest: Path, get: Callable[[str], bytes] = _get) -> list[Path]:
    record = _json(root / "results" / "release.json")
    if not record:
        raise FileNotFoundError("results/release.json is missing: run --plan and commit it before the tag")
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for entry in record["assets"]:
        body = tarball(contents(root, entry["method"], entry, get))
        if _sha256(body) != entry["sha256"]:
            raise ValueError(f"{entry['name']} packs to {_sha256(body)}, the tag records {entry['sha256']}")
        path = dest / entry["name"]
        path.write_bytes(body)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true")
    group.add_argument("--pack", type=Path)
    args = parser.parse_args()
    if args.plan:
        record = plan(ROOT)
        print(f"release plan: {len(record['assets'])} adapters, champion {record['champion']}")
    else:
        written = pack(ROOT, args.pack)
        print(f"release: {len(written)} packages, every digest as the tag records it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
