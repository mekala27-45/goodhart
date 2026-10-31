"""The grader API, first as a health stub so the host import happens in the first hour.

The grading routes arrive in step nine; until then /health answers with the version, the stage and
the statement, which is enough for the host to import the repository and for the probe to read it.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from goodhart_core.statements import STATEMENT

from goodhart_api import __version__


def create_app() -> FastAPI:
    app = FastAPI(title="goodhart grader", version=__version__, docs_url="/docs", redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "stage": "stub", "statement": STATEMENT}

    return app
