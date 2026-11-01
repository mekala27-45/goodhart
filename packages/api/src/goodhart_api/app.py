"""The grader API: mark any SQL against a held-out question, on the original and on the twins.

    POST /v1/grade          {"question_id": "...", "sql": "select ... | REFUSE CODE"}
    GET  /v1/grades/{id}    a grade from the log
    GET  /v1/questions      the held-out questions: id, set, text, family, warehouse, never the gold
    GET  /health            version, lock digests, the kept twins, the verifier digest, the statement

A grade is written to the log and committed before the response is sent. A client gets 60 grades an
hour, counted against a salted sha256 of its address (the address itself is never stored); the
parity check sends a separate key with its own budget. Without a database the API refuses to grade
rather than grade without a log.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from goodhart_core import paths
from goodhart_core.hashing import sha256_text
from goodhart_core.statements import STATEMENT
from pydantic import BaseModel, Field
from sqlalchemy import Engine

from goodhart_api import __version__, db
from goodhart_api.service import Service, result_payload

CLIENT_LIMIT = 60
PARITY_LIMIT = 5000
MAX_SQL = 8000


class GradeIn(BaseModel):
    question_id: str = Field(min_length=1, max_length=64)
    sql: str = Field(min_length=1, max_length=MAX_SQL)
    # True: parse the text exactly as the repository parses a model's completion (one fenced block or
    # one REFUSE line, anything else malformed). The parity check sends completions this way.
    completion: bool = False


def client_of(request: Request, salt: str) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    address = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    return hashlib.sha256(f"{salt}:{address}".encode()).hexdigest()


def create_app(service: Service | None = None, engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="goodhart grader", version=__version__, docs_url="/docs", redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://mekala27-45.github.io", "http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-parity-key"],
    )
    state: dict[str, Any] = {"service": service, "engine": engine}

    def svc() -> Service:
        if state["service"] is None:
            state["service"] = Service(paths.root())
        s: Service = state["service"]
        return s

    def eng() -> Engine | None:
        if state["engine"] is None and os.environ.get("DATABASE_URL"):
            state["engine"] = db.engine_for(os.environ["DATABASE_URL"])
        e: Engine | None = state["engine"]
        return e

    @app.get("/health")
    def health() -> dict[str, Any]:
        e = eng()
        database = "not configured"
        if e is not None:
            try:
                with e.connect() as con:
                    con.exec_driver_sql("select 1")
                database = "ok"
            except Exception:
                database = "unreachable"
        return {"status": "ok", "version": __version__, "stage": "grading", "database": database, **svc().health()}

    @app.get("/v1/questions")
    def questions() -> dict[str, Any]:
        return {"questions": svc().questions(), "statement": STATEMENT}

    @app.post("/v1/grade")
    def grade(body: GradeIn, request: Request) -> dict[str, Any]:
        e = eng()
        if e is None:
            raise HTTPException(503, "the grade log is not configured, so nothing is graded")
        s = svc()
        if body.question_id not in s.items:
            raise HTTPException(404, f"{body.question_id} is not a held-out question")
        parity_key = os.environ.get("GOODHART_PARITY_KEY", "")
        given = request.headers.get("x-parity-key", "")
        via_parity = bool(parity_key) and hmac.compare_digest(given, parity_key)
        client = "parity" if via_parity else client_of(request, os.environ.get("GOODHART_SALT", "goodhart"))
        now = dt.datetime.now(dt.UTC)
        with db.session(e) as session:
            if not db.take_budget(session, client, PARITY_LIMIT if via_parity else CLIENT_LIMIT, now):
                raise HTTPException(429, "this client has used its grades for the hour")
        name, _, v, elapsed = s.grade(body.question_id, body.sql, as_completion=body.completion)
        row = db.GradeRow(
            question_id=body.question_id,
            question_set=name,
            client=client,
            via_parity=via_parity,
            sql_sha256=sha256_text(body.sql),
            kind=v.kind,
            executed=v.executed,
            original=v.original,
            twins_passed=v.twins_passed,
            twins_total=v.twins_total,
            confirmed=v.confirmed,
            refused=v.refused,
            pays=v.pays,
            flags=v.flags,
            verifier=s.verifier(),
            elapsed_ms=round(elapsed, 1),
        )
        with db.session(e) as session:
            saved = db.record(session, row)
        return {
            "grade_id": saved.id,
            "question_id": body.question_id,
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
            "verifier": saved.verifier,
            "statement": STATEMENT,
        }

    @app.get("/v1/grades/{grade_id}")
    def get_grade(grade_id: str) -> dict[str, Any]:
        e = eng()
        if e is None:
            raise HTTPException(503, "the grade log is not configured")
        with db.session(e) as session:
            row = db.fetch(session, grade_id)
            if row is None:
                raise HTTPException(404, "no such grade")
            return {**db.as_dict(row), "statement": STATEMENT}

    return app
