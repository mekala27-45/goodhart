"""Bring the grade log schema to the latest Alembic revision; safe to run on every start.

    python -m goodhart_api.migrate

Revisions live in goodhart_api/migrations/versions, one file per change, and the version table
lives in the goodhart schema beside the tables it versions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config

HERE = Path(__file__).resolve().parent


def config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(HERE / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str) -> None:
    command.upgrade(config(url), "head")


def main() -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("migrate: no DATABASE_URL, nothing to do")
        return 0
    from goodhart_api.db import engine_for

    upgrade(str(engine_for(url).url.render_as_string(hide_password=False)))
    print("migrate: the grade log is at the latest revision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
