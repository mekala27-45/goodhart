"""The grader API against a real Postgres: health and questions, a grade with every reward and check,
the grade committed before the response and readable from an independent connection, the client and
parity budgets, a writing query refused, and no grade without a log."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.postgres, pytest.mark.warehouses]


@pytest.fixture(scope="module")
def engine() -> Any:
    from goodhart_api import db

    e = db.engine_for(os.environ["DATABASE_URL"])
    db.create_all(e)
    return e


@pytest.fixture(scope="module")
def service() -> Any:
    from goodhart_api.service import Service
    from goodhart_heldout.pipeline import materialize

    materialize(ROOT)
    return Service(ROOT)


@pytest.fixture
def client(service: Any, engine: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastapi.testclient import TestClient
    from goodhart_api.app import create_app

    monkeypatch.setenv("GOODHART_PARITY_KEY", "test-parity")
    monkeypatch.setenv("GOODHART_SALT", f"salt-{os.getpid()}-{id(monkeypatch)}")
    return TestClient(create_app(service=service, engine=engine))


def _lookup(service: Any) -> Any:
    return next(it for _, it in service.items.values() if it.family == "lookup" and it.gold and not it.gold.label)


def test_health_reports_the_digests_and_the_statement(client: Any) -> None:
    from goodhart_core.statements import STATEMENT

    body = client.get("/health").json()
    assert body["status"] == "ok" and body["stage"] == "grading" and body["database"] == "ok"
    assert body["statement"] == STATEMENT and len(body["kept_twins"]) == 4 and body["questions"] > 0
    assert body["twins_lock"] and body["warehouses_lock"] and body["verifier"] != "unknown"


def test_questions_never_carry_the_gold(client: Any) -> None:
    rows = client.get("/v1/questions").json()["questions"]
    assert rows and set(rows[0]) == {"id", "set", "question", "family", "warehouse"}
    assert {r["set"] for r in rows} == {"readback_test", "test", "template_test"}


def test_a_right_query_is_paid_and_confirmed_and_logged(client: Any, service: Any, engine: Any) -> None:
    from goodhart_api import db

    it = _lookup(service)
    r = client.post("/v1/grade", json={"question_id": it.id, "sql": it.gold.sql})
    assert r.status_code == 200
    body = r.json()
    assert body["checks"] == {"original": True, "twins_passed": 4, "twins_total": 4, "confirmed": True}
    assert body["pays"]["honest"] == 1.0 and not any(body["flags"].values())
    # Read back from an independent connection: the grade was committed, not merely flushed.
    other = db.engine_for(os.environ["DATABASE_URL"])
    with db.session(other) as s:
        row = db.fetch(s, body["grade_id"])
    other.dispose()
    assert row is not None and row.confirmed and row.question_id == it.id


def test_a_typed_literal_is_paid_and_not_confirmed(client: Any, service: Any) -> None:
    it = _lookup(service)
    body = client.post("/v1/grade", json={"question_id": it.id, "sql": f"select {it.gold.rows[0][-1]!r}"}).json()
    assert body["pays"]["honest"] == 1.0 and body["checks"]["original"] and not body["checks"]["confirmed"]
    assert body["flags"]["literal_only"]


def test_the_grade_row_precedes_the_response(client: Any, service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodhart_api import db

    def refuse(*_: Any, **__: Any) -> Any:
        raise RuntimeError("the log refused the row")

    monkeypatch.setattr(db, "record", refuse)
    it = _lookup(service)
    with pytest.raises(RuntimeError):
        client.post("/v1/grade", json={"question_id": it.id, "sql": it.gold.sql})


def test_a_query_that_writes_or_attaches_is_refused(client: Any, service: Any) -> None:
    it = _lookup(service)
    for sql in ("drop table doorstep.mart_metro_year", "attach 'x.db' as x", "copy (select 1) to '/tmp/x.csv'"):
        body = client.post("/v1/grade", json={"question_id": it.id, "sql": sql}).json()
        assert not body["executed"] and body["error"].startswith("refused")


def test_the_client_budget_refuses_past_the_cap(client: Any, service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodhart_api import app as app_module

    monkeypatch.setattr(app_module, "CLIENT_LIMIT", 2)
    it = _lookup(service)
    codes = [
        client.post("/v1/grade", json={"question_id": it.id, "sql": "REFUSE FORECAST"}).status_code for _ in range(3)
    ]
    assert codes == [200, 200, 429]
    parity = client.post(
        "/v1/grade", json={"question_id": it.id, "sql": "REFUSE FORECAST"}, headers={"x-parity-key": "test-parity"}
    )
    assert parity.status_code == 200, "the parity key has its own budget"


def test_an_unknown_question_and_a_missing_grade_are_404(client: Any) -> None:
    assert client.post("/v1/grade", json={"question_id": "g-nowhere", "sql": "select 1"}).status_code == 404
    assert client.get("/v1/grades/0000").status_code == 404


def test_no_grade_without_a_log(service: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from goodhart_api.app import create_app

    monkeypatch.delenv("DATABASE_URL", raising=False)
    c = TestClient(create_app(service=service, engine=None))
    it = _lookup(service)
    assert c.post("/v1/grade", json={"question_id": it.id, "sql": it.gold.sql}).status_code == 503
    assert c.get("/health").json()["database"] == "not configured"


def test_the_api_and_the_repository_give_the_same_verdict(service: Any) -> None:
    from goodhart_heldout.verdict import verdict
    from goodhart_rewards.parse import parse_completion

    it = _lookup(service)
    for text in (f"```sql\n{it.gold.sql}\n```", "```sql\nselect 1 as value\n```", "REFUSE FORECAST", "select 1"):
        _, _, v, _ = service.grade(it.id, text, as_completion=True)
        repo = verdict(it, parse_completion(text), service.golds, service.kept)
        assert (v.original, v.twins_passed, v.confirmed, v.pays, v.flags) == (
            repo.original,
            repo.twins_passed,
            repo.confirmed,
            repo.pays,
            repo.flags,
        )
