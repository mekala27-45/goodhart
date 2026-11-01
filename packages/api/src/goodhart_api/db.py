"""The grade log and the client budget, in Postgres (Neon in production), schema goodhart.

A grade row is written and committed before the response leaves, so a grade the client saw is a
grade the log has; a test reads it back from an independent connection. The client budget counts
grades per hashed client address per hour; the address itself is never stored, only a salted
sha256 of it.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Engine, MetaData, text
from sqlmodel import Field, Session, SQLModel, create_engine, select

SCHEMA = "goodhart"
metadata = MetaData(schema=SCHEMA)


class Base(SQLModel):
    metadata = metadata


class GradeRow(Base, table=True):
    __tablename__ = "grades"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    created_at: dt.datetime = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC), sa_type=DateTime(timezone=True), index=True
    )
    question_id: str = Field(index=True)
    question_set: str
    client: str = Field(index=True)
    via_parity: bool = False
    sql_sha256: str
    kind: str
    executed: bool
    original: bool
    twins_passed: int
    twins_total: int
    confirmed: bool
    refused: bool
    pays: dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    flags: dict[str, bool] = Field(default_factory=dict, sa_column=Column(JSON))
    verifier: str
    elapsed_ms: float = 0.0


class BudgetRow(Base, table=True):
    __tablename__ = "budget"

    client: str = Field(primary_key=True)
    window_start: dt.datetime = Field(sa_type=DateTime(timezone=True), primary_key=True)
    used: int = 0


def engine_for(url: str) -> Engine:
    if url.startswith("postgres://"):
        url = "postgresql://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=2)


def create_all(engine: Engine) -> None:
    """For tests and a fresh database; production schema changes go through Alembic."""
    with engine.begin() as con:
        con.execute(text(f"create schema if not exists {SCHEMA}"))
    metadata.create_all(engine)


@contextmanager
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as s:
        yield s


def window(now: dt.datetime) -> dt.datetime:
    return now.replace(minute=0, second=0, microsecond=0)


def take_budget(s: Session, client: str, limit: int, now: dt.datetime) -> bool:
    """Count one grade against the client's hour; False when the hour is spent. Row locked, so two
    concurrent requests cannot both take the last grade."""
    start = window(now)
    row = s.exec(
        select(BudgetRow).where(BudgetRow.client == client, BudgetRow.window_start == start).with_for_update()
    ).first()
    if row is None:
        row = BudgetRow(client=client, window_start=start, used=0)
        s.add(row)
    if row.used >= limit:
        s.rollback()
        return False
    row.used += 1
    s.commit()
    return True


def record(s: Session, row: GradeRow) -> GradeRow:
    s.add(row)
    s.commit()
    s.refresh(row)
    return row


def fetch(s: Session, grade_id: str) -> GradeRow | None:
    return s.get(GradeRow, grade_id)


def as_dict(row: GradeRow) -> dict[str, Any]:
    return row.model_dump(mode="json")
