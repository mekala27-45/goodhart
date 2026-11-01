"""Twin warehouses: copies of the original in which the measures move by known seeded rules.

Execution accuracy on one database pays a query that is right for the wrong reason. Zhong, Yu and
Klein fixed that for Spider with distilled test suites of perturbed databases (EMNLP 2020); this is
the same idea on goodhart's own warehouses. In each twin, every numeric measure column of every table
a gold query reads is scaled by a factor drawn away from one, with a small per row jitter, and a share
of its values is permuted within the column so rankings change. Keys, labels and years are untouched.
An honest query moves with the data; a memorized number or a dropped filter does not.

Internal identities between derived measures are not preserved (a share and the counts it was made
from are scaled apart), and the README says so: the twins check that a query reads the right rows and
columns, not that the warehouse stays coherent.

Sixteen candidates are built. Distillation keeps the K (four by default) that together distinguish
the most mutants of the gold SQL pool, by greedy cover, on mutants of training and development items
only, so the mutation score reported on test items is measured on mutants the choice never saw.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from goodhart_core.hashing import sha256_file, sha256_text

NUMERIC_TYPES = {"DOUBLE", "FLOAT", "BIGINT", "INTEGER", "HUGEINT", "SMALLINT", "DECIMAL", "REAL"}
PROTECTED = ("year", "month", "quarter", "period", "day", "rank", "flight_date", "cbsa_code", "kind")


@dataclass(frozen=True)
class TwinRule:
    name: str
    seed: int
    low: float
    high: float
    jitter: float
    permute_share: float


def rule_for(index: int, base_seed: int, config: dict[str, Any]) -> TwinRule:
    return TwinRule(
        name=f"twin-{index:02d}",
        seed=int(sha256_text(f"{base_seed}:twin:{index}")[:8], 16),
        low=float(config["factor_low"]),
        high=float(config["factor_high"]),
        jitter=float(config["jitter"]),
        permute_share=float(config["permute_share"]),
    )


def measure_columns(table: pa.Table, catalog_columns: dict[str, tuple[str, str]]) -> list[str]:
    """Numeric measure columns by the catalog's role, never a key, a label or a calendar position."""
    out = []
    for name in table.column_names:
        role, _ = catalog_columns.get(name, ("", ""))
        if role != "measure":
            continue
        if any(name == p or name.startswith(p + "_") for p in PROTECTED):
            continue
        field = table.schema.field(name)
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_decimal(field.type):
            out.append(name)
    return out


def _factor(rng: np.random.Generator, low: float, high: float) -> float:
    """A factor away from one: from [low, 1 - (1 - low) / 3] or [1 + (high - 1) / 3, high], even odds."""
    if rng.random() < 0.5:
        return float(rng.uniform(low, 1.0 - (1.0 - low) / 3.0))
    return float(rng.uniform(1.0 + (high - 1.0) / 3.0, high))


def perturb(table: pa.Table, columns: Sequence[str], rule: TwinRule, key: str) -> tuple[pa.Table, dict[str, float]]:
    rng = np.random.default_rng([rule.seed, int(sha256_text(key)[:8], 16)])
    factors: dict[str, float] = {}
    arrays = {}
    n = table.num_rows
    for name in table.column_names:
        col = table.column(name)
        if name not in columns or n == 0:
            arrays[name] = col
            continue
        f = _factor(rng, rule.low, rule.high)
        factors[name] = round(f, 6)
        values = col.to_numpy(zero_copy_only=False).astype("float64")
        mask = ~np.isnan(values)
        jitter = 1.0 + rng.normal(0.0, rule.jitter, size=n)
        moved = values * f * jitter
        share = round(float(rule.permute_share) * int(mask.sum()))
        if share >= 2:
            idx = np.flatnonzero(mask)
            chosen = rng.choice(idx, size=share, replace=False)
            moved[chosen] = moved[rng.permutation(chosen)]
        if pa.types.is_integer(col.type):
            out = pa.array(np.where(mask, np.rint(moved), np.nan), mask=~mask).cast(pa.float64())
            arrays[name] = pa.compute.cast(out, col.type, safe=False)
        else:
            arrays[name] = pa.array(moved, mask=~mask, type=pa.float64()).cast(col.type, safe=False)
    return pa.table(arrays), factors


def build(
    tables: Iterable[str],
    original: dict[str, Path],
    catalog_columns: dict[str, dict[str, tuple[str, str]]],
    out: Path,
    rules: Sequence[TwinRule],
) -> dict[str, Any]:
    """Write every twin's perturbed tables and return the record: factors and digests per file."""
    record: dict[str, Any] = {"twins": {}}
    for rule in rules:
        entry: dict[str, Any] = {"seed": rule.seed, "tables": {}}
        for qualified in sorted(tables):
            warehouse, name = qualified.split(".", 1)
            table = pq.read_table(original[qualified])
            cols = measure_columns(table, catalog_columns.get(qualified, {}))
            twin_table, factors = perturb(table, cols, rule, qualified)
            path = out / rule.name / warehouse / f"{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(twin_table, path, compression="zstd")
            entry["tables"][qualified] = {"columns": factors, "sha256": sha256_file(path)}
        record["twins"][rule.name] = entry
    return record


def greedy_cover(distinguishes: dict[str, set[str]], k: int) -> tuple[list[str], list[int]]:
    """Pick k twins that together distinguish the most mutants; ties go to the earlier name."""
    chosen: list[str] = []
    covered: set[str] = set()
    gains: list[int] = []
    for _ in range(min(k, len(distinguishes))):
        best = max(sorted(distinguishes), key=lambda t: len(distinguishes[t] - covered) if t not in chosen else -1)
        gains.append(len(distinguishes[best] - covered))
        covered |= distinguishes[best]
        chosen.append(best)
    return chosen, gains


def write_lock(record: dict[str, Any], kept: list[str], tables: list[str], path: Path, extra: dict[str, Any]) -> None:
    data = {"version": 1, "kept": kept, "tables": sorted(tables), **record, **extra}
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
