"""goodhart floor: readback's no model floor, through its own planner, graded like any model.

The floor is readback's grammar over its golden families: a classifier, entity dictionaries from the
catalog and SQL templates per measure, imported at the pinned commit (the floor extra) and never
copied. It answers with a draft (claims that point at the SQL results they came from), not with one
SQL query, so the adapter here turns a draft into the result table the comparator reads:

    a claim's value is the raw value in its SQL result (never the rounded display), in the unit the
    question asked for; its label is the entity column of a ranking, or the literal that differs
    between the claims' own queries (the two metros of a comparison, the two years of a trend), or
    the warehouse named by a catalog claim. A refusal, or a question the grammar does not cover, is a
    REFUSE. A question on which the planner raises (it applied a ranking template to a measure with
    no entity) is an answer the floor failed to give: graded malformed, not a refusal, and counted.

Twin confirmation replays the floor on each kept twin with the same tools bound to that twin, so the
floor is confirmed exactly when its own queries are right for the right reason. The grades go to
results/grades/floor/<set>.parquet beside every model's, under the checkpoint readback-floor, and
results/floor.json holds the twin confirmed accuracy per set with the counts behind it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import sqlglot
from goodhart_core import paths
from goodhart_core.items import Execution, Item
from goodhart_core.warehouse import Warehouses
from goodhart_heldout.confirm import TwinGolds
from goodhart_heldout.pipeline import materialize
from goodhart_rewards.compare import compare
from sqlglot import exp

from goodhart_evaluation.grade import INDEX, load_items, rows_digest, to_table
from goodhart_evaluation.metrics import Graded

FACTORS = {"stored": 1.0, "percent": 100.0, "thousands": 0.001}
SETS = ("readback_test", "test", "template_test", "dev", "readback_dev")
CHECKPOINT = "readback-floor@c4efff35"


class Tools:
    """readback's tool surface over one warehouse version, recording every result by id."""

    def __init__(self, w: Warehouses, version: str) -> None:
        self.w = w
        self.version = version
        self.results: dict[str, dict[str, Any]] = {}

    async def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "run_sql":
            r = self.w.run(str(args["sql"]), self.version, default_warehouse=args.get("warehouse"), timeout_s=10.0)
            rid = f"r{len(self.results)}"
            rows = r.rows[: int(args.get("limit", 1000))]
            self.results[rid] = {"sql": str(args["sql"]), "columns": r.columns, "rows": rows}
            return {"result_id": rid, "columns": r.columns, "rows": rows, "row_count": r.row_count, "error": r.error}
        if name == "warehouses":
            return {"warehouses": [{"name": n, "tables": len(t)} for n, t in sorted(self.w.tables.items())]}
        if name == "search_catalog":
            return {"hits": []}
        raise KeyError(f"the floor called a tool this adapter does not offer: {name}")


def _literals(sql: str) -> list[str]:
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return []
    out = []
    for eq in tree.find_all(exp.EQ):
        lit = eq.expression if isinstance(eq.expression, exp.Literal) else None
        if lit is not None:
            out.append(str(lit.this))
    return out


def draft_execution(draft: Any, tools: Tools, item: Item) -> Execution | None:
    """The draft as a result table in the item's unit, or None when the floor refused."""
    if draft.refused or not draft.claims:
        return None
    factor = FACTORS.get(item.unit_asked, 1.0)
    rows: list[list[Any]] = []
    literal_sets = []
    for claim in draft.claims:
        res = tools.results.get(claim.result_id or "")
        literal_sets.append(set(_literals(res["sql"])) if res else set())
    for k, claim in enumerate(draft.claims):
        res = tools.results.get(claim.result_id or "")
        if res is None:
            value: Any = claim.value
            label: Any = (claim.catalog_key or "").split(".")[0] or None
        else:
            cols = list(res["columns"])
            row = res["rows"][claim.row]
            col = cols.index(claim.column) if claim.column in cols else len(row) - 1
            value = row[col]
            if "entity" in cols:
                label = row[cols.index("entity")]
            else:
                others = (
                    set().union(*(s for j, s in enumerate(literal_sets) if j != k)) if len(literal_sets) > 1 else set()
                )
                own = sorted(literal_sets[k] - others)
                label = own[0] if own else None
        number = float(value) * factor if isinstance(value, int | float) and not isinstance(value, bool) else value
        rows.append([label, number])
    if item.gold is not None and not item.gold.label:
        rows = [[rows[0][1]]]
        return Execution(executed=True, columns=["value"], rows=rows, row_count=1)
    return Execution(executed=True, columns=["label", "value"], rows=rows, row_count=len(rows))


async def _answer(floor: Any, tools: Tools, question: str) -> Any:
    floor.call = tools
    result = await floor.answer(question)
    return result.draft


class Raised(RuntimeError):
    """The planner raised on a question: no draft, so no answer."""


def _execution(floor: Any, tools: Tools, item: Item) -> Execution | None:
    tools.results.clear()
    try:
        draft = asyncio.run(_answer(floor, tools, item.question))
    except Exception as err:
        raise Raised(f"{type(err).__name__}: {err}") from err
    return draft_execution(draft, tools, item)


def grade_item(item: Item, floors: dict[str, tuple[Any, Tools]], golds: TwinGolds, kept: Sequence[str]) -> Graded:
    floor, tools = floors["original"]
    raised = False
    try:
        execution = _execution(floor, tools, item)
    except Raised:
        execution, raised = None, True
    refused = execution is None and not raised
    original = confirmed = False
    if execution is not None and item.gold is not None:
        original = compare(item.gold, execution).match
        if original:
            passed = 0
            for v in kept:
                f, t = floors[v]
                try:
                    twin_exec = _execution(f, t, item)
                except Raised:
                    twin_exec = None
                gold = golds.gold(item, v)
                passed += int(twin_exec is not None and gold is not None and compare(gold, twin_exec).match)
            confirmed = passed == len(kept)
    return Graded(
        item_id=item.id,
        request="floor",
        checkpoint=CHECKPOINT,
        set=item.split,
        sample=0,
        temperature=0.0,
        family=item.family,
        warehouse=item.warehouse,
        template=item.template,
        measure=item.measure,
        refuse_item=item.refuse,
        kind="malformed" if raised else "refuse" if refused else "sql",
        executed=execution is not None,
        original=original,
        confirmed=confirmed,
        refused=refused,
        tokens=0,
        flags={"floor_raised": True} if raised else {},
    )


def run(root: Path) -> dict[str, Any]:
    from readback_planner.floor import Floor

    lock = json.loads((root / "data" / "twins.lock.json").read_text(encoding="utf-8"))
    kept = list(lock["kept"])
    materialize(root)
    w = Warehouses.from_repo(twins=True, root=root)
    golds = TwinGolds(w)
    floors: dict[str, tuple[Any, Tools]] = {}
    for version in ["original", *kept]:
        tools = Tools(w, version)
        floors[version] = (Floor(tools, root=paths.readback_dir()), tools)
    items = load_items(root)
    index_path = root / INDEX
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    out: dict[str, Any] = {}
    for name in SETS:
        chosen = sorted((i for i in items.values() if i.split == name), key=lambda i: i.id)
        rows = [grade_item(i, floors, golds, kept) for i in chosen]
        path = root / "results" / "grades" / "floor" / f"{name}__floor__greedy.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(to_table(rows), path, compression="zstd")
        index[f"floor/{name}__floor__greedy"] = {
            "source": CHECKPOINT,
            "grades": rows_digest(rows),
            "rows": len(rows),
            "set": name,
        }
        answerable = [r for r in rows if not r.refuse_item]
        out[name] = {
            "items": len(rows),
            "twin_confirmed": round(sum(r.confirmed for r in answerable) / len(answerable), 4) if answerable else None,
            "refused_answerable": sum(r.refused for r in answerable),
            "planner_raised": sum(bool(r.flags.get("floor_raised")) for r in rows),
        }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(dict(sorted(index.items())), indent=1) + "\n", encoding="utf-8")
    summary = {"checkpoint": CHECKPOINT, "kept_twins": kept, "sets": out}
    (root / "results" / "floor.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return out
