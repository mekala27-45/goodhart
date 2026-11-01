"""goodhart grade: every exported completion graded again in the repository.

The GPU writes completions and never a grade. Here each completion is parsed, executed read only on
the original warehouse, compared with the gold by the strict comparator, and, when it matches,
executed on every kept twin; the hack detectors run on it with the prompt it was sampled from. The
graded rows go to results/grades/<run>/<file>.parquet (the site reads them with DuckDB-WASM), and
results/grades/index.json records, per file, the digest of the completions and of the grades.

    goodhart grade            grade every completion file that has no grades, or whose source changed
    goodhart grade --check    grade everything again in memory and compare with the committed digests
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from goodhart_core.hashing import sha256_file, sha256_json
from goodhart_core.items import Completion, Item
from goodhart_core.warehouse import Warehouses
from goodhart_heldout.confirm import TwinGolds
from goodhart_heldout.pipeline import materialize
from goodhart_heldout.verdict import verdict
from goodhart_questions.build import read_items
from goodhart_rewards.parse import parse_completion

from goodhart_evaluation.metrics import Graded

INDEX = Path("results/grades/index.json")


@dataclass(frozen=True)
class Source:
    """One completions file under results/runs/<run>/completions/."""

    run: str
    path: Path

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def set(self) -> str:
        return self.name.split("__", 1)[0]


def sources(root: Path) -> list[Source]:
    runs = root / "results" / "runs"
    out = []
    for path in sorted(runs.glob("*/completions/*.jsonl")) if runs.exists() else []:
        if path.name.startswith("paraphrase__"):
            continue
        out.append(Source(run=path.parents[1].name, path=path))
    return out


def load_items(root: Path) -> dict[str, Item]:
    folder = root / "data" / "questions"
    items: dict[str, Item] = {}
    for path in sorted(folder.glob("*.jsonl")):
        for it in read_items(path):
            items[it.id] = it
    return items


def load_prompts(root: Path, set_name: str) -> dict[str, str]:
    path = root / "data" / "questions" / "prompts" / f"{set_name}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = "\n".join(m["content"] for m in row["messages"])
    return out


def read_completions(path: Path) -> list[Completion]:
    return [Completion.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class Grader:
    def __init__(self, root: Path, warehouses: Warehouses | None = None) -> None:
        self.root = root
        lock = json.loads((root / "data" / "twins.lock.json").read_text(encoding="utf-8"))
        self.kept: list[str] = list(lock["kept"])
        if warehouses is None:
            materialize(root)
            warehouses = Warehouses.from_repo(twins=True, root=root)
        self.golds = TwinGolds(warehouses)
        self.items = load_items(root)
        self._prompts: dict[str, dict[str, str]] = {}

    def prompt(self, set_name: str, item_id: str) -> str:
        if set_name not in self._prompts:
            self._prompts[set_name] = load_prompts(self.root, set_name)
        return self._prompts[set_name].get(item_id, "")

    def grade_one(self, c: Completion, set_name: str) -> Graded:
        item = self.items.get(c.item_id)
        if item is None:
            raise KeyError(f"{c.item_id} is in no committed question set")
        v = verdict(item, parse_completion(c.text), self.golds, self.kept, self.prompt(set_name, c.item_id))
        return Graded(
            item_id=c.item_id,
            request=c.request,
            checkpoint=c.checkpoint,
            set=set_name,
            sample=c.sample,
            temperature=c.temperature,
            family=item.family,
            warehouse=item.warehouse,
            template=item.template,
            measure=item.measure,
            refuse_item=item.refuse,
            kind=v.kind,
            executed=v.executed,
            original=v.original,
            confirmed=v.confirmed,
            refused=v.refused,
            tokens=c.tokens,
            flags=v.flags,
        )

    def grade(self, completions: Sequence[Completion], set_name: str, workers: int = 4) -> list[Graded]:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda c: self.grade_one(c, set_name), completions))
        return sorted(rows, key=lambda g: (g.checkpoint, g.item_id, g.temperature, g.sample))


def to_table(rows: Iterable[Graded]) -> pa.Table:
    records = []
    for g in rows:
        d = g.model_dump(exclude={"flags"})
        d.update({f"flag_{k}": v for k, v in sorted(g.flags.items())})
        records.append(d)
    return pa.Table.from_pylist(records)


def rows_digest(rows: Sequence[Graded]) -> str:
    return sha256_json([g.model_dump(mode="json") for g in rows])


def run(root: Path, check: bool = False) -> dict[str, Any]:
    index_path = root / INDEX
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    found = sources(root)
    if not found:
        return {"files": 0, "graded": 0, "drift": []}
    grader = Grader(root)
    drift: list[str] = []
    graded = 0
    for src in found:
        key = f"{src.run}/{src.name}"
        source_digest = sha256_file(src.path)
        entry = index.get(key)
        if not check and entry and entry.get("source") == source_digest:
            continue
        rows = grader.grade(read_completions(src.path), src.set)
        graded += len(rows)
        digest = rows_digest(rows)
        if check:
            if not entry or entry.get("source") != source_digest or entry.get("grades") != digest:
                drift.append(key)
            continue
        out = root / "results" / "grades" / src.run / f"{src.name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(to_table(rows), out, compression="zstd")
        index[key] = {"source": source_digest, "grades": digest, "rows": len(rows), "set": src.set}
    if not check:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(dict(sorted(index.items())), indent=1) + "\n", encoding="utf-8")
    return {"files": len(found), "graded": graded, "drift": drift, "kept_twins": grader.kept}


def read_grades(root: Path, run_id: str, name: str) -> list[Graded]:
    """The committed grades of one completions file, as Graded rows."""
    table = pq.read_table(root / "results" / "grades" / run_id / f"{name}.parquet")
    out = []
    for rec in table.to_pylist():
        flags = {k.removeprefix("flag_"): bool(rec.pop(k)) for k in list(rec) if k.startswith("flag_")}
        out.append(Graded.model_validate({**rec, "flags": flags}))
    return out
