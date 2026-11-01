"""The registered prompt: a short system part, a schema block, the names in the question, the question.

The system part states the dialect, the answer rules and the refusal form. The schema block shows the
tables the catalog search returns for the question, five at most, with column names, types and units.
The names line gives the warehouse spelling of every name the question uses, the way readback's
agent resolves names before it writes SQL; it never carries a value. The answer is one SQL query in a
fenced block, or the single line REFUSE with a reason code.

Thinking modes are off by default (goodhart_train.backend.chat_kwargs), and stated.

Defect D2 (leaked) is built here: with leak=True the schema block of the gold table carries three
sample rows, one of them the row the question asks about, value included.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from goodhart_core.hashing import sha256_json
from goodhart_core.items import REFUSAL_CODES, Item

from goodhart_questions.catalog import Table
from goodhart_questions.search import Search

SYSTEM = (
    "You write one DuckDB SQL query that answers a question about three warehouses: doorstep (housing "
    "affordability in US metros), turnaround (US domestic flights) and ledger (a fictional bank).\n\n"
    "Rules:\n"
    "- Answer with one SQL query in a fenced block that starts with ```sql, and nothing after the block.\n"
    "- Qualify every table with its warehouse, as in doorstep.mart_metro_year.\n"
    "- Return only the columns the answer needs. When the question names entities or periods, return one "
    "label column (the entity or period as the warehouse spells it) and then the value columns. A single "
    "number needs no label.\n"
    "- Return one row per entity or period the question asks about. For a ranking, order the rows from "
    "first to last.\n"
    "- Keep values in the unit the question asks for: a share stays a fraction unless the question asks "
    "for a percent, and dollars stay dollars unless it asks for thousands.\n"
    "- If the warehouses cannot answer (a forecast, one named person, a year outside the data, a measure "
    "no table has), answer with the single line REFUSE and one reason code: " + ", ".join(REFUSAL_CODES) + "."
)


@dataclass(frozen=True)
class Built:
    messages: list[dict[str, str]]
    tables: tuple[str, ...]
    columns: tuple[str, ...]


def table_block(
    table: Table, columns: Sequence[str] | None, search: Search, question: str, must: tuple[str, ...]
) -> str:
    cols = search.columns(question, table, must=must)
    shown = ", ".join(
        f"{c.name} {c.type or 'VALUE'}" + (f" [{c.unit}]" if c.unit and c.unit != "none" else "") for c in cols
    )
    hidden = len(table.columns) - len(cols)
    tail = f" (and {hidden} more columns)" if hidden > 0 else ""
    return f"{table.qualified} ({table.rows} rows): {shown}{tail}"


def build(
    item: Item,
    search: Search,
    *,
    leak_rows: Sequence[Sequence[object]] | None = None,
    leak_columns: Sequence[str] | None = None,
) -> Built:
    hits = search.tables(item.question)
    gold_tables = tuple(item.gold.tables) if item.gold else ()
    must_by_table: dict[str, tuple[str, ...]] = {}
    for qualified in gold_tables:
        must_by_table[qualified] = tuple(gold_columns(item, qualified))
    lines = ["Tables (DuckDB, read only):"]
    shown_tables: list[str] = []
    shown_columns: list[str] = []
    for hit in hits:
        must = must_by_table.get(hit.table.qualified, ())
        lines.append("- " + table_block(hit.table, None, search, item.question, must))
        shown_tables.append(hit.table.qualified)
        shown_columns.extend(
            f"{hit.table.qualified}.{c.name}" for c in search.columns(item.question, hit.table, must=must)
        )
        if leak_rows and gold_tables and hit.table.qualified == gold_tables[0]:
            header = ", ".join(leak_columns or [])
            lines.append(f"  Sample rows ({header}):")
            lines.extend("  " + ", ".join(_cell(v) for v in row) for row in leak_rows)
    if item.names:
        lines.append("")
        lines.append("Names in the question, as the warehouse spells them:")
        lines.extend(f"- {written} = {spelled}" for written, spelled in sorted(item.names.items()))
    user = "\n".join(lines) + f"\n\nQuestion: {item.question}"
    return Built(
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tables=tuple(shown_tables),
        columns=tuple(shown_columns),
    )


def _cell(value: object) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def gold_columns(item: Item, qualified: str) -> list[str]:
    """The columns of a table the gold SQL reads, so the schema block never hides one."""
    if item.gold is None:
        return []
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(item.gold.sql, read="duckdb")
    names = {c.name for c in tree.find_all(exp.Column)}
    return sorted(names)


def registry_digest() -> str:
    """The hash of the prompt registry: the system part and the block format, registered with the plan."""
    return sha256_json({"system": SYSTEM, "format": "tables, names, question; leak adds sample rows to the gold table"})
