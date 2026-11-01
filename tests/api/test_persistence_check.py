"""The persistence check's read back: a grade read from the database it was given, and a database with
no grade log named rather than raised, so a connection string that points at another database shows
which one it reached without printing either string."""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlparse, urlunparse

import check_persistence
import psycopg
import pytest

pytestmark = pytest.mark.postgres


def _plain(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://")


def _with_database(url: str, name: str) -> str:
    parts = urlparse(_plain(url))
    return urlunparse(parts._replace(path=f"/{name}"))


def test_a_database_without_the_grade_log_is_named() -> None:
    url = os.environ["DATABASE_URL"]
    name = f"goodhart_other_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(_plain(url), autocommit=True) as con:
        con.execute(f'create database "{name}"')
    try:
        out = check_persistence.read_back(_with_database(url, name), uuid.uuid4().hex, "rb-T-lookup-03")
        assert out["read_back_by_database"] is False
        assert out["database_read"] == name
        assert "another one" in str(out["database_error"])
    finally:
        with psycopg.connect(_plain(url), autocommit=True) as con:
            con.execute(f'drop database "{name}" with (force)')


def test_a_grade_is_read_back_from_the_grade_log() -> None:
    from goodhart_api import db

    url = os.environ["DATABASE_URL"]
    engine = db.engine_for(url)
    db.create_all(engine)
    row = db.GradeRow(
        question_id="rb-T-lookup-03",
        question_set="readback_test",
        client="test",
        sql_sha256="0" * 64,
        kind="refusal",
        executed=False,
        original=False,
        twins_passed=0,
        twins_total=4,
        confirmed=False,
        refused=True,
        verifier="test",
    )
    with db.session(engine) as session:
        saved = db.record(session, row)
    out = check_persistence.read_back(url, saved.id, "rb-T-lookup-03")
    assert out["read_back_by_database"] is True
    assert out["database_read"] == urlparse(_plain(url)).path.lstrip("/")
    assert check_persistence.read_back(url, uuid.uuid4().hex, "rb-T-lookup-03")["read_back_by_database"] is False
