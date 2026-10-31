"""The statement that appears on every surface.

Written once here. Every page of the site, every API response, every model card, every rendered
document and the release notes read it from this module, and a gate checks each surface for it.
"""

from __future__ import annotations

PROJECT = "goodhart"

STATEMENT = (
    "goodhart is a portfolio demonstration. It post-trains small open models to write SQL for "
    "questions about three public warehouses its author built from public and synthetic data, and "
    "grades them with checks the training reward never saw. It is not a production system; its "
    "models must not be used to answer real questions about housing, flights or banking; and its "
    "numbers describe these models on these questions only."
)


def statement_markdown() -> str:
    return f"> {STATEMENT}"


def statement_payload() -> dict[str, str]:
    return {"statement": STATEMENT}


def contains_statement(text: str) -> bool:
    flat = " ".join(text.split())
    return " ".join(STATEMENT.split()) in flat
