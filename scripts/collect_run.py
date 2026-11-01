"""Collect one kernel's output onto the results branch, with its provenance, after checking digests.

The kernel writes provenance-kernel.json with the sha256 of every file it produced. This script
recomputes each digest on what was downloaded and refuses the run if one differs or is missing.
Small files go under results/runs/<id>/; files above the size limit (adapters, large completion
exports) go to the runs release as assets, with their digests in results/runs/<id>/assets.json.

    python scripts/collect_run.py --request ID --output out/ID --kernel user/slug --status complete --run-id N
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from goodhart_core.hashing import sha256_file

SIZE_LIMIT = 40_000_000
SKIP = ("trainer/", "smoke_trainer/")


class DigestMismatch(RuntimeError):
    pass


def find_out(folder: Path) -> Path:
    """Kaggle saves /kaggle/working; the stage writes to /kaggle/working/out."""
    for candidate in (folder / "out", folder):
        if (candidate / "provenance-kernel.json").exists():
            return candidate
    return folder / "out" if (folder / "out").exists() else folder


def verify(out: Path) -> dict[str, Any]:
    prov_path = out / "provenance-kernel.json"
    if not prov_path.exists():
        raise DigestMismatch("the kernel wrote no provenance-kernel.json")
    prov: dict[str, Any] = json.loads(prov_path.read_text(encoding="utf-8"))
    bad = []
    for rel, digest in prov.get("outputs", {}).items():
        if any(rel.startswith(s) for s in SKIP):
            continue
        path = out / rel
        if not path.exists():
            bad.append(f"{rel} is missing")
        elif sha256_file(path) != digest:
            bad.append(f"{rel} does not match its digest")
    if bad:
        raise DigestMismatch("refusing the run:\n" + "\n".join(bad[:20]))
    return prov


def collect(request: str, output: Path, kernel: str, status: str, run_id: str, upload: bool = True) -> dict[str, Any]:
    target = ROOT / "results" / "runs" / request
    target.mkdir(parents=True, exist_ok=True)
    out = find_out(output)
    prov: dict[str, Any]
    try:
        prov = verify(out) if status == "complete" else {"status": status, "outputs": {}}
        verified = status == "complete"
    except DigestMismatch as err:
        prov = {"status": "refused", "error": str(err), "outputs": {}}
        verified = False
    assets: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}
    if out.exists():
        for path in sorted(p for p in out.rglob("*") if p.is_file()):
            rel = path.relative_to(out).as_posix()
            if any(rel.startswith(s) for s in SKIP):
                continue
            digest = sha256_file(path)
            if path.stat().st_size > SIZE_LIMIT:
                name = f"{request}__{rel.replace('/', '__')}"
                assets[rel] = {
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                    "release": "runs",
                    "asset": name,
                }
                if upload:
                    staged = Path("/tmp") / name
                    shutil.copyfile(path, staged)
                    subprocess.run(
                        ["gh", "release", "view", "runs"], capture_output=True, check=False
                    ).returncode == 0 or subprocess.run(
                        [
                            "gh",
                            "release",
                            "create",
                            "runs",
                            "--prerelease",
                            "--title",
                            "Run outputs",
                            "--notes",
                            "Adapters and large exports from the GPU runs, "
                            "each with its sha256 in results/runs/<id>/assets.json.",
                        ],
                        check=True,
                    )
                    subprocess.run(["gh", "release", "upload", "runs", str(staged), "--clobber"], check=True)
            else:
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
                files[rel] = digest
    request_file = ROOT / "requests" / f"{request}.json"
    provenance = {
        "request": request,
        "request_sha256": sha256_file(request_file) if request_file.exists() else None,
        "kernel": kernel,
        "run_id": run_id,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "status": prov.get("status", status),
        "verified": verified,
        "files": files,
        "assets": assets,
        "environment": prov.get("environment"),
        "summary": prov.get("summary"),
        "error": prov.get("error"),
    }
    (target / "provenance.json").write_text(json.dumps(provenance, indent=1, default=str) + "\n", encoding="utf-8")
    if assets:
        (target / "assets.json").write_text(json.dumps(assets, indent=1) + "\n", encoding="utf-8")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--run-id", default="local")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    prov = collect(args.request, args.output, args.kernel, args.status, args.run_id, upload=not args.no_upload)
    print(json.dumps({k: prov[k] for k in ("request", "status", "verified")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
