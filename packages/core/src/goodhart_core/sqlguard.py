"""The SQL guard: read only, one statement, parsed before it runs.

A statement reaches DuckDB only after sqlglot has parsed it in the DuckDB dialect and every rule below
has passed. The read only, locked connection in goodhart_core.warehouse is the second line, not the
first. The rules follow readback's guard; the code is goodhart's own.

  1. it parses, and it is exactly one statement;
  2. the statement is a query: a SELECT, a WITH over a SELECT, or a set operation of SELECTs;
  3. nothing inside it writes, attaches, copies, installs, loads, sets, pragmas or describes;
  4. it calls no table function and no function that reads a file or the environment;
  5. every table it names is a warehouse table or one of two read only catalog views
     (information_schema.tables and information_schema.columns); an unqualified name is qualified
     with the item's warehouse when that warehouse has it, and refused otherwise.

The guard never adds a LIMIT: the comparator needs every row a query returns. The connection caps
the rows it fetches instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

REFUSED_NODES: tuple[str, ...] = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Create",
    "Drop",
    "Alter",
    "TruncateTable",
    "Copy",
    "Attach",
    "Detach",
    "Install",
    "Pragma",
    "Set",
    "Use",
    "Describe",
    "Show",
    "Command",
    "Export",
    "Grant",
    "Transaction",
    "Commit",
    "Rollback",
    "Analyze",
    "Cache",
    "LoadData",
)

REFUSED_FUNCTIONS = frozenset(
    {
        "read_parquet",
        "read_csv",
        "read_csv_auto",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_text",
        "read_blob",
        "parquet_scan",
        "parquet_metadata",
        "parquet_schema",
        "glob",
        "read_xlsx",
        "sniff_csv",
        "duckdb_settings",
        "duckdb_extensions",
        "duckdb_secrets",
        "duckdb_databases",
        "getenv",
        "current_setting",
        "query",
        "query_table",
        "load_extension",
    }
)


# The read only catalog views a query may read: how many tables a warehouse publishes is a fair
# question, and readback's golden set asks it.
CATALOG_VIEWS = frozenset({"tables", "columns"})


class Refused(ValueError):
    """A statement the guard will not run. The message names the rule."""


@dataclass(frozen=True)
class Guarded:
    sql: str
    tables: tuple[str, ...]
    rewritten: bool
    notes: tuple[str, ...] = field(default=())


def _function_name(node: Any) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    name = getattr(node, "sql_name", None)
    if callable(name):
        try:
            return str(name()).lower()
        except Exception:  # pragma: no cover - sqlglot internals vary by class
            return type(node).__name__.lower()
    return type(node).__name__.lower()


def guard(sql: str, tables: Mapping[str, Set[str]], default_warehouse: str | None) -> Guarded:
    """Check one statement against the rules and qualify its table names.

    tables maps each warehouse schema to the tables it has.
    """
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise Refused("rule 1: the statement is empty")
    try:
        statements = sqlglot.parse(text, read="duckdb")
    except (ParseError, TokenError) as err:
        raise Refused(f"rule 1: it does not parse ({str(err).splitlines()[0][:160]})") from err
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise Refused(f"rule 1: expected one statement, found {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise Refused(f"rule 2: a {type(tree).__name__.upper()} is not a query")
    for name in REFUSED_NODES:
        cls = getattr(exp, name, None)
        if cls is not None and tree.find(cls) is not None:
            raise Refused(f"rule 3: it contains a {name.upper()}")
    for func in tree.find_all(exp.Func):
        fname = _function_name(func)
        if fname in REFUSED_FUNCTIONS:
            raise Refused(f"rule 4: it calls {fname}")
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    named: list[str] = []
    rewritten = False
    for table in list(tree.find_all(exp.Table)):
        if not isinstance(table.this, exp.Identifier):
            raise Refused("rule 4: a table position holds a function or an expression")
        name = table.name.lower()
        schema = table.db.lower() if table.db else ""
        if table.catalog:
            raise Refused(f"rule 5: {table.sql(dialect='duckdb')} names a catalog")
        if not schema:
            if name in cte_names:
                continue
            owners = [w for w, names in tables.items() if name in names]
            if default_warehouse and name in tables.get(default_warehouse, set()):
                schema = default_warehouse
            elif len(owners) == 1:
                schema = owners[0]
            else:
                raise Refused(f"rule 5: no warehouse table is named {name}")
            table.set("db", exp.to_identifier(schema))
            rewritten = True
        if schema == "information_schema" and name in CATALOG_VIEWS:
            named.append(f"{schema}.{name}")
            continue
        if schema not in tables:
            raise Refused(f"rule 5: {schema} is not a warehouse")
        if name not in tables[schema]:
            raise Refused(f"rule 5: {schema} has no table named {name}")
        named.append(f"{schema}.{name}")
    out = tree.sql(dialect="duckdb") if rewritten else text
    return Guarded(sql=out, tables=tuple(sorted(set(named))), rewritten=rewritten)
