"""The ASGI entry point Vercel and uvicorn start: goodhart_api.main:app."""

from __future__ import annotations

from goodhart_api.app import create_app

app = create_app()
