"""Parity: every exported completion of the readback test set for M0 and the champion, graded by the
live API, must get the verdict the repository gave it.

    python scripts/check_parity.py --base http://127.0.0.1:8080 --parity-key "$GOODHART_PARITY_KEY"
    python scripts/check_parity.py --base https://goodhart-grader.vercel.app --parity-key ... --write

Compared per completion: whether it executed, matched on the original, how many twins it passed, and
whether it was confirmed. results/parity.json records the base, the counts and every mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "evaluation" / "src"))


def completions(root: Path, checkpoints: list[str]) -> list[dict[str, Any]]:
    out = []
    for path in sorted((root / "results" / "runs").glob("*/completions/readback_test__*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not checkpoints or any(row["checkpoint"].startswith(c) for c in checkpoints):
                    out.append({**row, "run": path.parents[1].name, "file": path.stem})
    return out


def repository_verdicts(root: Path, rows: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    from goodhart_evaluation.grade import read_grades

    found: dict[tuple[str, ...], dict[str, Any]] = {}
    for run, name in sorted({(r["run"], r["file"]) for r in rows}):
        for g in read_grades(root, run, name):
            found[(run, g.checkpoint, g.item_id, str(g.sample))] = {
                "executed": g.executed,
                "original": g.original,
                "confirmed": g.confirmed,
            }
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--parity-key", required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="checkpoint prefixes, M0 and the champion")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = completions(ROOT, args.checkpoint)
    if not rows:
        present = sorted({r["checkpoint"] for r in completions(ROOT, [])})
        print(
            f"no exported readback_test completion has a checkpoint starting with {args.checkpoint}; "
            f"the checkpoints exported are {present}",
            file=sys.stderr,
        )
    repo = repository_verdicts(ROOT, rows)
    mismatches = []
    with httpx.Client(base_url=args.base, timeout=120.0, headers={"x-parity-key": args.parity_key}) as client:
        for r in rows:
            body = {"question_id": r["item_id"], "sql": r["text"] or "(empty)", "completion": True}
            got = client.post("/v1/grade", json=body)
            got.raise_for_status()
            api = got.json()
            mine = repo.get((r["run"], r["checkpoint"], r["item_id"], str(r["sample"])))
            seen = {
                "executed": api["executed"],
                "original": api["checks"]["original"],
                "confirmed": api["checks"]["confirmed"],
            }
            if mine != seen:
                mismatches.append(
                    {"item": r["item_id"], "checkpoint": r["checkpoint"], "api": seen, "repository": mine}
                )
    out = {"base": args.base, "compared": len(rows), "mismatches": len(mismatches), "examples": mismatches[:20]}
    print(json.dumps(out, indent=1))
    if args.write:
        path = ROOT / "results" / "parity.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        record["local" if "127.0.0.1" in args.base or "localhost" in args.base else "live"] = out
        path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0 if not mismatches and rows else 1


if __name__ == "__main__":
    sys.exit(main())
