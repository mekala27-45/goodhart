"""goodhart profile: the first look at every mart at full size, before any question is generated.

For every table of the three warehouses: the row count, and per column its type, its nulls, its range
(numbers) or its distinct count (text), and the time span of the table where it has a year, a month or
a period. Per warehouse, the ten strangest rows: the rows whose largest robust z score over the
measure columns of the busiest mart is highest.

Then the defects: ten things really wrong or really surprising in the data, each found by a query
here, each turned into a generator or quarantine rule, and each rule held by a test over the
committed questions (tests/questions/test_profile_rules.py). results/profile.json records the counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goodhart_core.warehouse import Warehouses

NUMERIC = ("DOUBLE", "FLOAT", "BIGINT", "INTEGER", "HUGEINT", "SMALLINT", "DECIMAL", "REAL", "UBIGINT")
TIME_COLUMNS = ("year", "month", "quarter", "period", "flight_date", "date", "week")
STRANGE_TABLES = {
    "doorstep": "doorstep.mart_metro_year",
    "turnaround": "turnaround.mart_carrier_month",
    "ledger": "ledger.metrics_bank_quarter",
}


@dataclass(frozen=True)
class Defect:
    id: str
    warehouse: str
    found: str
    sql: str  # returns one number: how many rows (or values) show the defect
    rule: str


DEFECTS = (
    Defect(
        "turnaround_partial_2026",
        "turnaround",
        "2026 holds seven months of carrier data, so any 2026 annual figure is a partial year",
        "select 12 - count(distinct month) from turnaround.mart_carrier_month where year = 2026",
        "annual turnaround questions never ask about 2026; the generator excludes it",
    ),
    Defect(
        "turnaround_short_carrier_years",
        "turnaround",
        "carriers enter and leave the data mid year, so some carrier years have fewer than twelve months",
        "select count(*) from (select carrier, year from turnaround.mart_carrier_month group by 1, 2 "
        "having count(distinct month) < 12)",
        "a carrier year is asked about only when it has all twelve months",
    ),
    Defect(
        "ledger_edge_quarters",
        "ledger",
        "the bank's quarterly metrics run from 2023Q3 to 2026Q3, and the first and last quarter are partial",
        "select count(*) from ledger.metrics_bank_quarter where period < '2023Q4' or period > '2026Q2'",
        "ledger questions use the full quarters 2023Q4 to 2026Q2 only, as readback's floor does",
    ),
    Defect(
        "doorstep_payment_share_nulls",
        "doorstep",
        "payment share is null on most metro years: micropolitan areas and the years after 2024 have none",
        "select count(*) from doorstep.mart_metro_year where payment_share is null",
        "a question is drawn only from cells where its measure is not null; a gold with a null drops",
    ),
    Defect(
        "doorstep_years_after_2024",
        "doorstep",
        "the year column runs to 2026 while the income based measures stop at 2024",
        "select count(*) from doorstep.mart_metro_year where year > 2024",
        "a period with no value is out of range, which is a refusal item, never a lookup",
    ),
    Defect(
        "doorstep_two_kinds",
        "doorstep",
        "a CBSA is either metropolitan or micropolitan, and the micropolitan rows outnumber the metros",
        "select count(*) from doorstep.mart_metro_year where kind = 'micropolitan'",
        "metro measures carry the catalog's kind filter, and the gold keeps it as a predicate",
    ),
    Defect(
        "doorstep_quarantine",
        "doorstep",
        "the upstream build set aside source rows by rule (jammed ACS values and others) in mart_quarantine",
        "select count(*) from doorstep.mart_quarantine",
        "no question is drawn from a quarantine mart; a schema block may still list one, as a real table",
    ),
    Defect(
        "turnaround_quarantine",
        "turnaround",
        "carrier months with flagged flights (duplicate keys, impossible times) are listed in mart_quarantine",
        "select count(*) from turnaround.mart_quarantine where duplicate_key + impossible_time + status_conflict > 0",
        "no question is drawn from a quarantine mart; a schema block may still list one, as a real table",
    ),
    Defect(
        "turnaround_airports_beyond_catalog",
        "turnaround",
        "the airport marts cover hundreds of airports, far more than the catalog names",
        "select count(distinct airport) from turnaround.mart_airport_day",
        "airport questions name only airports in the catalog's entity list",
    ),
    Defect(
        "doorstep_rate_as_percent",
        "doorstep",
        "the mortgage rate is stored in percent (5.87) beside shares stored as fractions (0.31)",
        "select count(*) from doorstep.ch1_national_series where rate > 1 and payment_share < 1",
        "units come from the catalog per measure; a percent never compares with a fraction",
    ),
)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def profile_table(w: Warehouses, warehouse: str, table: str) -> dict[str, Any]:
    qualified = f"{warehouse}.{table}"
    meta = w.run(
        f"select column_name, data_type from information_schema.columns where table_schema = '{warehouse}' "
        f"and table_name = '{table}' order by ordinal_position",
        timeout_s=60.0,
    )
    rows = w.run(f"select count(*) from {qualified}", timeout_s=60.0)
    count = rows.rows[0][0] if rows.rows else None
    out: dict[str, Any] = {"rows": int(count) if isinstance(count, int | float) else None, "columns": {}}
    for name, dtype in meta.rows:
        col = _q(str(name))
        dtype = str(dtype)
        if any(dtype.startswith(t) for t in NUMERIC):
            stat = w.run(
                f"select count(*) - count({col}), min({col}), max({col}) from {qualified}", timeout_s=60.0
            ).rows
            nulls, low, high = stat[0] if stat else (None, None, None)
            out["columns"][str(name)] = {"type": dtype, "nulls": nulls, "min": low, "max": high}
        else:
            stat = w.run(f"select count(*) - count({col}), count(distinct {col}) from {qualified}", timeout_s=60.0).rows
            nulls, distinct = stat[0] if stat else (None, None)
            out["columns"][str(name)] = {"type": dtype, "nulls": nulls, "distinct": distinct}
    times = [c for c in out["columns"] if c.lower() in TIME_COLUMNS]
    if times:
        t = _q(times[0])
        span = w.run(f"select min({t}), max({t}) from {qualified}", timeout_s=60.0).rows
        out["time"] = {"column": times[0], "from": span[0][0], "to": span[0][1]} if span else None
    return out


def strangest(w: Warehouses, qualified: str, limit: int = 10) -> list[dict[str, Any]]:
    """The rows whose largest robust z score over the table's numeric columns is highest."""
    warehouse, table = qualified.split(".", 1)
    cols = [
        str(r[0])
        for r in w.run(
            f"select column_name, data_type from information_schema.columns where table_schema = '{warehouse}' "
            f"and table_name = '{table}'",
            timeout_s=60.0,
        ).rows
        if str(r[1]).startswith(("DOUBLE", "FLOAT", "DECIMAL")) and str(r[0]).lower() not in TIME_COLUMNS
    ]
    if not cols:
        return []
    zs = []
    for c in cols:
        q = _q(c)
        zs.append(
            f"abs({q} - (select median({q}) from {qualified})) / "
            f"nullif((select median(abs({q} - (select median({q}) from {qualified}))) from {qualified}), 0)"
        )
    score = "greatest(" + ", ".join(f"coalesce({z}, 0)" for z in zs) + ")" if len(zs) > 1 else f"coalesce({zs[0]}, 0)"
    r = w.run(f"select *, {score} as robust_z from {qualified} order by robust_z desc limit {limit}", timeout_s=120.0)
    return [dict(zip(r.columns, row, strict=True)) for row in r.rows]


def defects(w: Warehouses) -> list[dict[str, Any]]:
    out = []
    for d in DEFECTS:
        r = w.run(d.sql, timeout_s=60.0)
        count = r.rows[0][0] if r.executed and r.rows else None
        out.append(
            {"id": d.id, "warehouse": d.warehouse, "found": d.found, "count": count, "rule": d.rule, "error": r.error}
        )
    return out


def run(root: Path) -> dict[str, Any]:
    w = Warehouses.from_repo(twins=False, root=root)
    tables: dict[str, Any] = {}
    total = 0
    for warehouse in sorted(w.tables):
        for table in sorted(w.tables[warehouse]):
            p = profile_table(w, warehouse, table)
            tables[f"{warehouse}.{table}"] = p
            total += int(p["rows"] or 0)
    report = {
        "tables": len(tables),
        "rows": total,
        "by_table": tables,
        "strangest": {wh: strangest(w, q) for wh, q in STRANGE_TABLES.items()},
        "defects": defects(w),
    }
    path = root / "results" / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report
