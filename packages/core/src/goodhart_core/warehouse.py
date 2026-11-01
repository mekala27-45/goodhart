"""The warehouses and their twins, as locked read only DuckDB connections.

Each version (the original, or one twin) is a small database file of views, one schema per
warehouse, each view over a parquet file the source lock names. A twin replaces the files of the
tables its rules perturbed and shares every other file with the original. The views file is opened
read only; external access goes off with an allow list of exactly the files the views read, and the
configuration is locked, so a statement cannot read another file, write, attach or load an extension
whatever the guard lets through.

Every execution has a timeout and a row cap. The cap is far above any gold result; a query that
returns more rows than the cap is reported as truncated with its full row count.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from goodhart_core import paths
from goodhart_core.items import Cell, Execution
from goodhart_core.sqlguard import Refused, guard

WAREHOUSES: tuple[str, ...] = ("doorstep", "turnaround", "ledger")
ROW_CAP = 100_000
DEFAULT_TIMEOUT_S = 5.0


class WarehouseMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class TableFile:
    warehouse: str
    table: str
    path: Path


@dataclass
class Version:
    """One version of the three warehouses: the original, or a twin with some files replaced."""

    name: str
    files: dict[tuple[str, str], Path]
    con: duckdb.DuckDBPyConnection | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def load_source_lock(root: Path | None = None) -> dict[str, Any]:
    path = (root or paths.root()) / "data" / "sources.lock.json"
    lock: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("version") != 1 or "warehouses" not in lock:
        raise WarehouseMissing(f"{path} is not a version 1 source lock")
    return lock


def table_files(lock: Mapping[str, Any], base: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for name in WAREHOUSES:
        for t in lock["warehouses"][name]["tables"]:
            out[(name, t["table"])] = base / name / t["path"]
    return out


def table_names(lock: Mapping[str, Any]) -> dict[str, set[str]]:
    return {name: {t["table"] for t in lock["warehouses"][name]["tables"]} for name in WAREHOUSES}


def _reader(path: Path) -> str:
    p = str(path).replace("'", "''")
    if path.suffix == ".parquet":
        return f"read_parquet('{p}')"
    return f"read_csv('{p}', header = true, sample_size = -1)"


def _views_file(version: Version, folder: Path) -> Path:
    missing = [f"{w}.{t}: {p}" for (w, t), p in sorted(version.files.items()) if not p.exists()]
    if missing:
        raise WarehouseMissing(
            f"{version.name}: files missing, run make fetch (and make twins):\n" + "\n".join(missing[:10])
        )
    folder.mkdir(parents=True, exist_ok=True)
    db = folder / f"views-{version.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}.duckdb"
    con = duckdb.connect(str(db))
    try:
        for name in WAREHOUSES:
            con.execute(f'create schema if not exists "{name}"')
        for (name, table), path in sorted(version.files.items()):
            con.execute(f'create or replace view "{name}"."{table}" as select * from {_reader(path.resolve())}')
    finally:
        con.close()
    return db


def _open(version: Version, folder: Path) -> duckdb.DuckDBPyConnection:
    db = _views_file(version, folder)
    con = duckdb.connect(
        str(db),
        read_only=True,
        config={
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            "allow_community_extensions": False,
            # One thread: a float sum over many rows comes back in the same order on every run, so a
            # gold is the same bytes on every machine and the rebuild check can demand it.
            "threads": 1,
        },
    )
    allowed = sorted({str(p.resolve()) for p in version.files.values()})
    con.execute("set allowed_paths = ?", [allowed])
    con.execute("set enable_external_access = false")
    con.execute("set lock_configuration = true")
    return con


def to_cell(value: Any) -> Cell:
    """A DuckDB value as a JSON safe cell: numbers stay numbers, dates and the rest become text."""
    if value is None or isinstance(value, bool):
        return None if value is None else int(value)
    if isinstance(value, int | float | str):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dt.date | dt.datetime):
        return value.isoformat()
    return str(value)


class Warehouses:
    """The original and the twins, one locked connection per version, a cursor per call."""

    def __init__(
        self,
        lock: Mapping[str, Any],
        base: Path,
        twins: Mapping[str, Mapping[tuple[str, str], Path]] | None = None,
        cache: Path | None = None,
    ) -> None:
        original = table_files(lock, base)
        self.tables = table_names(lock)
        self.versions: dict[str, Version] = {"original": Version("original", dict(original))}
        for twin, replaced in sorted((twins or {}).items()):
            files = dict(original)
            for key, path in replaced.items():
                if key not in files:
                    raise WarehouseMissing(f"{twin} replaces {key}, which the lock does not name")
                files[key] = path
            self.versions[twin] = Version(twin, files)
        self.cache = cache or Path(os.environ.get("GOODHART_VIEWS", tempfile.gettempdir())) / "goodhart-views"

    @classmethod
    def from_repo(cls, twins: bool = True, root: Path | None = None) -> Warehouses:
        root = root or paths.root()
        lock = load_source_lock(root)
        twin_files: dict[str, dict[tuple[str, str], Path]] = {}
        if twins:
            index = root / "data" / "twins.lock.json"
            if index.exists():
                data = json.loads(index.read_text(encoding="utf-8"))
                for twin in data["kept"]:
                    twin_files[twin] = {
                        (w, t): paths.twins_dir() / twin / w / f"{t}.parquet"
                        for w, t in (key.split(".", 1) for key in data["tables"])
                    }
        return cls(lock, paths.warehouses_dir(), twin_files)

    def names(self) -> list[str]:
        return list(self.versions)

    def twin_names(self) -> list[str]:
        return [n for n in self.versions if n != "original"]

    def connection(self, version: str) -> duckdb.DuckDBPyConnection:
        v = self.versions[version]
        with v.lock:
            if v.con is None:
                v.con = _open(v, self.cache)
            return v.con

    def run(
        self,
        sql: str,
        version: str = "original",
        default_warehouse: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        row_cap: int = ROW_CAP,
    ) -> Execution:
        start = time.perf_counter()
        try:
            checked = guard(sql, self.tables, default_warehouse)
        except Refused as err:
            return Execution(executed=False, error=f"refused: {err}", elapsed_ms=_ms(start))
        cur = self.connection(version).cursor()
        timer = threading.Timer(timeout_s, cur.interrupt)
        timer.start()
        try:
            rel = cur.execute(checked.sql)
            columns = [d[0] for d in rel.description or []]
            fetched = rel.fetchmany(row_cap + 1)
            truncated = len(fetched) > row_cap
            row_count = len(fetched)
            if truncated:
                rest = 0
                while True:
                    more = rel.fetchmany(65_536)
                    if not more:
                        break
                    rest += len(more)
                row_count += rest
                fetched = fetched[:row_cap]
            rows = [[to_cell(v) for v in r] for r in fetched]
            return Execution(
                executed=True,
                columns=columns,
                rows=rows,
                row_count=row_count,
                truncated=truncated,
                elapsed_ms=_ms(start),
                tables=list(checked.tables),
            )
        except duckdb.InterruptException:
            return Execution(executed=False, error=f"timeout after {timeout_s:.1f} s", elapsed_ms=_ms(start))
        except duckdb.Error as err:
            message = str(err).splitlines()[0][:300]
            return Execution(executed=False, error=message, elapsed_ms=_ms(start), tables=list(checked.tables))
        finally:
            timer.cancel()
            cur.close()

    def close(self) -> None:
        for v in self.versions.values():
            with v.lock:
                if v.con is not None:
                    v.con.close()
                    v.con = None


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
