"""The run ledger mirror: one row per promoted GPU run.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

SCHEMA = "goodhart"


def upgrade() -> None:
    op.create_table(
        "ledger",
        sa.Column("request", sa.String(), primary_key=True),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("arm", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("steps", sa.Integer(), nullable=False),
        sa.Column("prompts", sa.Integer(), nullable=False),
        sa.Column("rollouts", sa.Integer(), nullable=False),
        sa.Column("tokens_generated", sa.Integer(), nullable=False),
        sa.Column("tokens_trained", sa.Integer(), nullable=False),
        sa.Column("gpu_seconds", sa.JSON(), nullable=True),
        sa.Column("gpu_hours", sa.Float(), nullable=False),
        sa.Column("wall_clock_seconds", sa.Float(), nullable=False),
        sa.Column("ceiling_hours", sa.Float(), nullable=False),
        sa.Column("stopped_at_ceiling", sa.Boolean(), nullable=False),
        sa.Column("peak_memory_bytes", sa.BigInteger(), nullable=False),
        sa.Column("accelerator", sa.String(), nullable=True),
        sa.Column("request_digest", sa.String(), nullable=True),
        sa.Column("mirrored_at", sa.DateTime(timezone=True), nullable=False),
        schema=SCHEMA,
    )
    op.create_index("ix_ledger_stage", "ledger", ["stage"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_table("ledger", schema=SCHEMA)
