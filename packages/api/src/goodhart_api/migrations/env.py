"""Alembic environment for the grade log: online migrations only, version table in schema goodhart."""

from __future__ import annotations

from alembic import context
from goodhart_api.db import SCHEMA, metadata
from sqlalchemy import create_engine, text

url = context.config.get_main_option("sqlalchemy.url")
assert url is not None
engine = create_engine(url)
with engine.begin() as con:
    con.execute(text(f"create schema if not exists {SCHEMA}"))
with engine.connect() as connection:
    context.configure(
        connection=connection,
        target_metadata=metadata,
        version_table_schema=SCHEMA,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
