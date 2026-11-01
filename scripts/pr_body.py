"""The pull request body for a results branch: what ran, whether it verified, what to check."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def body(folder: Path) -> str:
    prov = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    summary = json.dumps(prov.get("summary") or {}, indent=1, default=str)[:3000]
    lines = [
        f"Request `{prov['request']}`, kernel `{prov['kernel']}`, workflow run {prov['run_id']}.",
        "",
        f"Status: {prov['status']}. Digests verified: {'yes' if prov['verified'] else 'no'}.",
        f"Files on this branch: {len(prov['files'])}. Release assets: {len(prov.get('assets') or {})}.",
        "",
        "Summary written by the kernel:",
        "",
        "```json",
        summary,
        "```",
        "",
        "Before merging (the promotion): CI grades every exported completion again in the repository,",
        "and `python scripts/promote_results.py --check` must pass.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(body(Path(sys.argv[1])))
