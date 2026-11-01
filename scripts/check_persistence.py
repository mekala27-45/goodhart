"""The out of process persistence check: grade one query through a running API, then read the grade
back from this separate process, straight from the database, and through GET /v1/grades/{id}.

    python scripts/check_persistence.py --base http://127.0.0.1:8080 --database "$DATABASE_URL"

CI runs it against the container it just built; before the tag it runs against the live URL with
the Neon URL, and results/persistence.json records what it saw.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def check(base: str, database: str | None) -> dict[str, object]:
    with httpx.Client(base_url=base, timeout=60.0) as client:
        first = client.get("/v1/questions").json()["questions"][0]
        start = time.perf_counter()
        r = client.post("/v1/grade", json={"question_id": first["id"], "sql": "REFUSE FORECAST"})
        r.raise_for_status()
        grade = r.json()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        again = client.get(f"/v1/grades/{grade['grade_id']}").json()
    out: dict[str, object] = {
        "base": base,
        "grade_id": grade["grade_id"],
        "question_id": first["id"],
        "grade_ms": round(elapsed_ms, 1),
        "read_back_by_api": again.get("id") == grade["grade_id"],
    }
    if database:
        import psycopg

        url = database.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(url) as con:
            row = con.execute(
                "select question_id, refused, confirmed from goodhart.grades where id = %s", (grade["grade_id"],)
            ).fetchone()
        out["read_back_by_database"] = bool(row and row[0] == first["id"] and row[1] is True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--database")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    out = check(args.base, args.database)
    print(json.dumps(out, indent=1))
    if args.write:
        (ROOT / "results" / "persistence.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    ok = out["read_back_by_api"] and out.get("read_back_by_database", True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
