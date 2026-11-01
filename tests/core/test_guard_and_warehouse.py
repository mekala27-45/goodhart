"""The SQL guard and the locked connection: what runs, what is refused, and the row cap."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from goodhart_core.sqlguard import Refused, guard
from goodhart_core.warehouse import Warehouses

TABLES = {
    "doorstep": {"mart_metro_year"},
    "turnaround": {"mart_carrier_month"},
    "ledger": {"metrics_bank_quarter"},
}


def test_a_select_passes_and_names_its_tables() -> None:
    g = guard("select max(x) from doorstep.mart_metro_year", TABLES, None)
    assert g.tables == ("doorstep.mart_metro_year",) and not g.rewritten


def test_an_unqualified_table_is_qualified_with_the_item_warehouse() -> None:
    g = guard("select * from mart_metro_year", TABLES, "doorstep")
    assert g.rewritten and "doorstep.mart_metro_year" in g.sql


def test_ctes_are_not_mistaken_for_tables() -> None:
    g = guard("with t as (select 1 as a) select a from t", TABLES, None)
    assert g.tables == ()


@pytest.mark.parametrize(
    "sql,rule",
    [
        ("", "rule 1"),
        ("select 1; select 2", "rule 1"),
        ("selec x frm", "rule 1"),
        ("insert into doorstep.mart_metro_year values (1)", "rule 2"),
        ("copy (select 1) to 'x.csv'", "rule 2"),
        ("select * from read_parquet('/etc/passwd')", "rule 4"),
        ("select getenv('HOME')", "rule 4"),
        ("select * from doorstep.nope", "rule 5"),
        ("select * from secrets.table1", "rule 5"),
        ("select * from mystery", "rule 5"),
    ],
)
def test_the_guard_refuses_by_rule(sql: str, rule: str) -> None:
    with pytest.raises(Refused, match=rule):
        guard(sql, TABLES, "doorstep")


def test_the_catalog_views_are_readable() -> None:
    g = guard("select count(*) from information_schema.tables where table_schema = 'doorstep'", TABLES, None)
    assert g.tables == ("information_schema.tables",)


@pytest.fixture
def small(tmp_path: Path) -> Warehouses:
    base = tmp_path / "w"
    for w, t in (
        ("doorstep", "mart_metro_year"),
        ("turnaround", "mart_carrier_month"),
        ("ledger", "metrics_bank_quarter"),
    ):
        path = base / w / f"{t}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"k": list(range(50)), "v": [float(i) for i in range(50)]}), path)
    lock = {
        "version": 1,
        "warehouses": {
            w: {"tables": [{"table": t, "path": f"{t}.parquet"}]}
            for w, t in (
                ("doorstep", "mart_metro_year"),
                ("turnaround", "mart_carrier_month"),
                ("ledger", "metrics_bank_quarter"),
            )
        },
    }
    return Warehouses(lock, base, cache=tmp_path / "views")


def test_the_connection_runs_and_caps_rows(small: Warehouses) -> None:
    r = small.run("select k, v from doorstep.mart_metro_year order by k", row_cap=10)
    assert r.executed and r.truncated and r.row_count == 50 and len(r.rows) == 10


def test_the_connection_refuses_a_file_read_even_past_the_guard(small: Warehouses, tmp_path: Path) -> None:
    con = small.connection("original")
    import duckdb

    with pytest.raises(duckdb.Error):
        con.execute(f"select * from read_csv('{tmp_path}/anything.csv')")


def test_a_refused_statement_comes_back_as_not_executed(small: Warehouses) -> None:
    r = small.run("drop table doorstep.mart_metro_year")
    assert not r.executed and (r.error or "").startswith("refused")
