"""The grade log, the run ledger mirror and the client budget, in Postgres (Neon in production),
schema goodhart.

A grade row is written and committed before the response leaves, so a grade the client saw is a
grade the log has; a test reads it back from an independent connection. The client budget counts
grades per hashed client address per hour; the address itself is never stored, only a salted
sha256 of it. The ledger mirror holds one row per promoted GPU run, upserted from the committed
results/runs/<id>/ledger.jsonl by scripts/mirror_ledger.py; the repository's file stays the record.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import JSON, BigInteger, Column, DateTime, Engine, MetaData, text
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


class LedgerRow(Base, table=True):
    """One promoted GPU run's ledger row, as the kernel's meter wrote it."""

    __tablename__ = "ledger"

    request: str = Field(primary_key=True)
    stage: str = Field(index=True)
    model: str | None = None
    arm: str | None = None
    status: str
    steps: int = 0
    prompts: int = 0
    rollouts: int = 0
    tokens_generated: int = 0
    tokens_trained: int = 0
    gpu_seconds: dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    gpu_hours: float = 0.0
    wall_clock_seconds: float = 0.0
    ceiling_hours: float = 0.0
    stopped_at_ceiling: bool = False
    peak_memory_bytes: int = Field(default=0, sa_type=BigInteger)
    accelerator: str | None = None
    request_digest: str | None = None
    mirrored_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC), sa_type=DateTime(timezone=True))


LEDGER_FIELDS = (
    "stage",
    "model",
    "arm",
    "status",
    "steps",
    "prompts",
    "rollouts",
    "tokens_generated",
    "tokens_trained",
    "gpu_seconds",
    "wall_clock_seconds",
    "ceiling_hours",
    "stopped_at_ceiling",
    "peak_memory_bytes",
    "accelerator",
    "request_digest",
)


def mirror(s: Session, rows: list[dict[str, Any]]) -> int:
    """Upsert each ledger row by its request: a run promoted again (collected again) replaces its row."""
    for r in rows:
        values: dict[str, Any] = {k: r[k] for k in LEDGER_FIELDS if r.get(k) is not None}
        values["gpu_hours"] = round(sum(float(v) for v in (r.get("gpu_seconds") or {}).values()) / 3600.0, 4)
        found = s.get(LedgerRow, str(r["request"]))
        if found is None:
            s.add(LedgerRow.model_validate({"request": str(r["request"]), **values}))
        else:
            for k, v in values.items():
                setattr(found, k, v)
            found.mirrored_at = dt.datetime.now(dt.UTC)
    s.commit()
    return len(rows)


def ledger(s: Session) -> list[LedgerRow]:
    return list(s.exec(select(LedgerRow).order_by(LedgerRow.request)).all())


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
    # No server side prepared statements: Neon's pooled endpoint runs PgBouncer in transaction mode,
    # where a statement prepared on one server connection may be executed on another.
    return create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=2, connect_args={"prepare_threshold": None})


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
