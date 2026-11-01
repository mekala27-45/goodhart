"""What the API grades against: the held-out questions, the original warehouses and the kept twins.

The container bakes them at build time (goodhart_api.bake): every original table by lock, because a
completion may read any table and the repository grader can see them all, and the kept twins of the
tables the golds read, rebuilt from data/twins.lock.json and checked by digest. The questions are
the held-out sets the plan grades; their gold never leaves this process.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from goodhart_core.hashing import sha256_file
from goodhart_core.items import Item, Parsed
from goodhart_core.statements import STATEMENT
from goodhart_core.warehouse import Warehouses
from goodhart_heldout.confirm import TwinGolds
from goodhart_heldout.verdict import Verdict, verdict
from goodhart_questions.build import read_items
from goodhart_rewards.parse import parse_completion

HELD_OUT = ("readback_test", "test", "template_test")
RESULT_ROWS = 50


def parse_answer(text: str) -> Parsed:
    """A fenced completion is parsed as a model's; a bare query or a REFUSE line is taken as written."""
    body = text.strip()
    if "```" in body:
        return parse_completion(body)
    if body.upper().startswith("REFUSE"):
        return parse_completion(body)
    return Parsed(kind="sql", sql=body.rstrip(";").strip(), blocks=1) if body else Parsed(kind="malformed")


@dataclass
class Service:
    root: Path

    @cached_property
    def items(self) -> dict[str, tuple[str, Item]]:
        out: dict[str, tuple[str, Item]] = {}
        for name in HELD_OUT:
            path = self.root / "data" / "questions" / f"{name}.jsonl"
            for it in read_items(path):
                out[it.id] = (name, it)
        return out

    @cached_property
    def lock(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.root / "data" / "twins.lock.json").read_text(encoding="utf-8"))
        return data

    @property
    def kept(self) -> list[str]:
        return list(self.lock["kept"])

    @cached_property
    def golds(self) -> TwinGolds:
        return TwinGolds(Warehouses.from_repo(twins=True, root=self.root))

    def verifier(self) -> str:
        """The digest of the comparator and the rewards: what decides a verdict."""
        base = self.root / "packages" / "rewards" / "src" / "goodhart_rewards"
        files = [base / "compare.py", base / "rewards.py"]
        return sha256_file(files[0])[:12] + sha256_file(files[1])[:12] if all(f.exists() for f in files) else "unknown"

    def questions(self) -> list[dict[str, Any]]:
        return [
            {"id": it.id, "set": name, "question": it.question, "family": it.family, "warehouse": it.warehouse}
            for name, it in sorted(self.items.values(), key=lambda x: (x[0], x[1].id))
        ]

    def grade(self, question_id: str, answer: str, as_completion: bool = False) -> tuple[str, Item, Verdict, float]:
        name, item = self.items[question_id]
        start = time.perf_counter()
        parsed = parse_completion(answer) if as_completion else parse_answer(answer)
        v = verdict(item, parsed, self.golds, self.kept)
        return name, item, v, (time.perf_counter() - start) * 1000.0

    def health(self) -> dict[str, Any]:
        sources = self.root / "data" / "sources.lock.json"
        return {
            "warehouses_lock": sha256_file(sources)[:12] if sources.exists() else None,
            "twins_lock": sha256_file(self.root / "data" / "twins.lock.json")[:12],
            "kept_twins": self.kept,
            "questions": len(self.items),
            "verifier": self.verifier(),
            "statement": STATEMENT,
        }


def result_payload(v: Verdict) -> dict[str, Any]:
    e = v.execution
    return {
        "executed": v.executed,
        "error": e.error if e is not None else None,
        "columns": e.columns if e is not None else [],
        "rows": e.rows[:RESULT_ROWS] if e is not None else [],
        "row_count": e.row_count if e is not None else 0,
        "truncated": bool(e.truncated) if e is not None else False,
    }
