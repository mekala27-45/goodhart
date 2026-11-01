"""Write web/public/data/recorded_session.json from the repository's own verifier.

The desk plays this file when the grader API does not answer. Every response in it has the exact
shape POST /v1/grade, GET /v1/questions and GET /health return (packages/api/src/goodhart_api/app.py),
and every verdict comes from goodhart_api.service.Service.grade, the function the API calls. Only the
grade log is missing: no database is written, so grade_id is null and the file says it is a
placeholder. scripts/record-session.mjs replaces it with a recording of the live API.

    uv run python web/scripts/record_local.py            # from the repository root
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from goodhart_api import __version__
from goodhart_api.service import Service, result_payload
from goodhart_core.statements import STATEMENT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "data" / "recorded_session.json"
TRIES_FILE = Path(__file__).with_name("tries.json")


def grade_response(svc: Service, question_id: str, sql: str) -> dict[str, Any]:
    name, _, v, _ = svc.grade(question_id, sql)
    return {
        "grade_id": None,
        "question_id": question_id,
        "set": name,
        "kind": v.kind,
        **result_payload(v),
        "pays": v.pays,
        "checks": {
            "original": v.original,
            "twins_passed": v.twins_passed,
            "twins_total": v.twins_total,
            "confirmed": v.confirmed,
        },
        "flags": v.flags,
        "verifier": svc.verifier(),
        "statement": STATEMENT,
    }


def main() -> int:
    svc = Service(ROOT)
    tries: list[dict[str, str]] = json.loads(TRIES_FILE.read_text(encoding="utf-8"))
    health = {"status": "ok", "version": __version__, "stage": "grading", "database": "not configured", **svc.health()}
    grades = [
        {
            "request": {"question_id": t["question_id"], "sql": t["sql"]},
            "status": 200,
            "response": grade_response(svc, t["question_id"], t["sql"]),
        }
        for t in tries
    ]
    session = {
        "recorded": True,
        "placeholder": True,
        "recorded_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        "recorded_from": (
            "the repository's verifier on the build machine (goodhart_api.service.Service.grade, "
            "the function the API calls), with no grade log, so every grade_id is null"
        ),
        "api": "https://goodhart-grader.vercel.app",
        "note": "Replace with a recording of the live API: node web/scripts/record-session.mjs",
        "health": health,
        "questions": {"questions": svc.questions(), "statement": STATEMENT},
        "tries": tries,
        "grades": grades,
    }
    OUT.write_text(json.dumps(session, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    count = len(svc.questions())
    print(f"recorded session: {len(grades)} grades, {count} questions into {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
