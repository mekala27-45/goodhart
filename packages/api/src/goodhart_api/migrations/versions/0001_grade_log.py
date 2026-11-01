"""The grade log and the client budget.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

SCHEMA = "goodhart"


def upgrade() -> None:
    op.create_table(
        "grades",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("question_id", sa.String(), nullable=False),
        sa.Column("question_set", sa.String(), nullable=False),
        sa.Column("client", sa.String(), nullable=False),
        sa.Column("via_parity", sa.Boolean(), nullable=False),
        sa.Column("sql_sha256", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("executed", sa.Boolean(), nullable=False),
        sa.Column("original", sa.Boolean(), nullable=False),
        sa.Column("twins_passed", sa.Integer(), nullable=False),
        sa.Column("twins_total", sa.Integer(), nullable=False),
        sa.Column("confirmed", sa.Boolean(), nullable=False),
        sa.Column("refused", sa.Boolean(), nullable=False),
        sa.Column("pays", sa.JSON(), nullable=True),
        sa.Column("flags", sa.JSON(), nullable=True),
        sa.Column("verifier", sa.String(), nullable=False),
        sa.Column("elapsed_ms", sa.Float(), nullable=False),
        schema=SCHEMA,
    )
    op.create_index("ix_grades_created_at", "grades", ["created_at"], schema=SCHEMA)
    op.create_index("ix_grades_question_id", "grades", ["question_id"], schema=SCHEMA)
    op.create_index("ix_grades_client", "grades", ["client"], schema=SCHEMA)
    op.create_table(
        "budget",
        sa.Column("client", sa.String(), primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("used", sa.Integer(), nullable=False),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("budget", schema=SCHEMA)
    op.drop_table("grades", schema=SCHEMA)
