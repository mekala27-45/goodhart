"""Mirror the run ledger into the database: every promoted run's ledger row, upserted by its request,
then read back from a separate connection and compared.

    python scripts/mirror_ledger.py --database "$DATABASE_URL" [--write]

The committed results/runs/<id>/ledger.jsonl files are the record; the mirror is what the grader
API's database holds of them, so the pages and a reader of the database see the same compute. With
--write the result goes to results/ledger_mirror.json (rows mirrored, rows read back, every request
and its GPU hours equal or not). Runs in the live workflow, which holds the database secret.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ledger_mirror.json"


def ledger_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "results" / "runs").glob("*/ledger.jsonl")):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            rows.append(lines[-1])  # one row per stage run: the last the meter wrote
    return rows


def mirror(url: str, root: Path = ROOT) -> dict[str, Any]:
    from goodhart_api import db

    rows = ledger_rows(root)
    engine = db.engine_for(url)
    db.create_all(engine)
    with db.session(engine) as s:
        written = db.mirror(s, rows)
    # Read back through a second engine, so the comparison does not see this one's session.
    other = db.engine_for(url)
    with db.session(other) as s:
        back = {r.request: r for r in db.ledger(s)}
    expected = {
        str(r["request"]): round(sum(float(v) for v in (r.get("gpu_seconds") or {}).values()) / 3600.0, 4) for r in rows
    }
    mismatches = [req for req, hours in expected.items() if req not in back or abs(back[req].gpu_hours - hours) > 1e-6]
    return {
        "mirrored": written,
        "read_back": len([r for r in expected if r in back]),
        "mismatches": mismatches,
        "gpu_hours": round(sum(expected.values()), 4),
        "checked_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    got = mirror(args.database)
    if args.write:
        OUT.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    different = len(got["mismatches"])
    print(f"ledger mirror: {got['mirrored']} rows written, {got['read_back']} read back, {different} different")
    return 0 if not got["mismatches"] else 1


if __name__ == "__main__":
    sys.exit(main())
