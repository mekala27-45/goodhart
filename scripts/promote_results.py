"""The promotion check: a GPU run's results may land on main only when every byte is the one the kernel
reported and the request is the one committed.

    python scripts/promote_results.py --check            # every results/runs/<id> in the tree
    python scripts/promote_results.py --check --run ID   # one run

For each run folder: provenance.json exists and names its request; the request file in the tree has
the sha256 the provenance recorded; every file the provenance lists is present with its digest, and
no other file is; a run marked verified really verified. A run that failed in the kernel can still
be promoted, as a record of what failed, but never as a verified one. CI runs this on every pull
request, so merging a results branch is the promotion and nothing else is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from goodhart_core.hashing import sha256_file

OWN_FILES = {"provenance.json", "assets.json"}


def check_run(folder: Path, root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    prov_path = folder / "provenance.json"
    if not prov_path.exists():
        return [f"{folder.name}: no provenance.json"]
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    rid = str(prov.get("request"))
    if rid != folder.name:
        problems.append(f"{folder.name}: the provenance names request {rid}")
    request = root / "requests" / f"{rid}.json"
    if not request.exists():
        problems.append(f"{folder.name}: requests/{rid}.json is not in the tree")
    elif prov.get("request_sha256") != sha256_file(request):
        problems.append(f"{folder.name}: requests/{rid}.json differs from the request the kernel ran")
    listed: dict[str, str] = prov.get("files", {})
    for rel, digest in sorted(listed.items()):
        path = folder / rel
        if not path.exists():
            problems.append(f"{folder.name}: {rel} is listed and missing")
        elif sha256_file(path) != digest:
            problems.append(f"{folder.name}: {rel} does not match its digest")
    present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    for rel in sorted(present - set(listed) - OWN_FILES):
        problems.append(f"{folder.name}: {rel} is not in the provenance")
    if prov.get("verified") and prov.get("status") != "done":
        problems.append(f"{folder.name}: marked verified with status {prov.get('status')}")
    return problems


def check_all(root: Path = ROOT, only: str | None = None) -> tuple[int, list[str]]:
    runs = root / "results" / "runs"
    folders = sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []
    if only:
        folders = [f for f in folders if f.name == only]
    problems = [p for f in folders for p in check_run(f, root)]
    return len(folders), problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--run")
    args = parser.parse_args()
    count, problems = check_all(only=args.run)
    if problems:
        print("promotion refused:\n" + "\n".join(problems))
        return 1
    print(f"promotion check: {count} runs, every file as the kernel reported it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
