"""The item, the completion and the execution: the contracts the whole engagement passes around.

An item is one question with its gold. The gold is a result table: an optional label column and one
or more value columns, each value column with its unit and tolerance. Refusal items carry no gold;
the right answer is the line REFUSE with a reason code.

A completion is what a model wrote for an item, exported from the GPU with the request that made it.
The repository grades completions; the GPU only writes them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from goodhart_core.hashing import sha256_json
from goodhart_core.model import StrictModel
from goodhart_core.units import Unit

Family = Literal["lookup", "comparison", "trend", "top_k", "multi", "refusal"]
FAMILIES: tuple[Family, ...] = ("lookup", "comparison", "trend", "top_k", "multi", "refusal")

Split = Literal["train", "dev", "test", "template_test", "readback_dev", "readback_test", "own"]
Source = Literal["generator", "paraphrase", "readback", "own"]

# The fixed list of refusal reason codes the prompt names. A refusal must carry one of them.
REFUSAL_CODES: tuple[str, ...] = (
    "FORECAST",
    "PERSON",
    "OUT_OF_RANGE",
    "NO_SUCH_MEASURE",
    "OUT_OF_SCOPE",
)

Cell = float | int | str | None


class GoldColumn(StrictModel):
    name: str
    unit: Unit
    tolerance: float = Field(ge=0.0)


class Gold(StrictModel):
    """The gold answer: the SQL and its result on the original warehouse."""

    sql: str
    label: bool
    columns: list[GoldColumn]
    rows: list[list[Cell]]
    ordered: bool
    tables: list[str]

    def labels(self) -> list[str]:
        if not self.label:
            return []
        return [normalize_label(r[0]) for r in self.rows]


class Item(StrictModel):
    id: str
    question: str
    warehouse: str
    family: Family
    template: str
    shape: str
    measure: str | None = None
    entities: list[str] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    unit_asked: str = "stored"
    refuse: bool = False
    refusal_reason: str | None = None
    gold: Gold | None = None
    split: Split
    source: Source
    paraphrase_of: str | None = None
    names: dict[str, str] = Field(default_factory=dict)

    def entity_tuple(self) -> str:
        """The key the splits are made on: measure, entities and periods, order free."""
        return sha256_json(
            {"measure": self.measure, "entities": sorted(self.entities), "periods": sorted(self.periods)}
        )[:16]


class Completion(StrictModel):
    item_id: str
    request: str
    checkpoint: str
    sample: int
    temperature: float
    text: str
    tokens: int = 0


class Execution(StrictModel):
    executed: bool
    error: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Cell]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0
    tables: list[str] = Field(default_factory=list)


class Parsed(StrictModel):
    kind: Literal["sql", "refuse", "malformed"]
    sql: str | None = None
    reason_code: str | None = None
    blocks: int = 0


def normalize_label(value: Cell) -> str:
    """Labels compare as text: a year 2014 and the string '2014' are the same label, case and spacing free."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return " ".join(str(value).split()).casefold()
