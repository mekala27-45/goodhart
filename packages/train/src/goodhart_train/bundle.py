"""What a kernel reads: the bundle the gpu workflow built for one request.

    bundle/
      request.json                 the committed request
      config/training.yaml         defaults, each change a DECISIONS.md entry
      config/monitor.yaml          the registered thresholds
      data/sources.lock.json       to fetch the warehouses by digest inside the kernel
      data/items/<split>.jsonl     items with gold, training and development splits only
      data/prompts/<set>.jsonl     prompts only (id and messages), for every set the stage samples
      data/sft/<name>.jsonl        SFT targets, clean and planted
      data/examples.json           the three fixed examples of the base row
      data/heldout_ids.json        the id of every held-out item, so a training stage can refuse one

Test sets travel as prompts with no gold. Twins never travel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from goodhart_core.items import Item

from goodhart_train.requests import Request
from goodhart_train.requests import load as load_request


@dataclass(frozen=True)
class Prompt:
    id: str
    set: str
    messages: list[dict[str, str]]


@dataclass
class Bundle:
    root: Path
    inputs: list[str] = field(default_factory=list)

    @property
    def request(self) -> Request:
        return load_request(self.root / "request.json")

    def config(self, name: str) -> dict[str, Any]:
        data: dict[str, Any] = yaml.safe_load((self.root / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
        return data

    def items(self, split: str) -> list[Item]:
        path = self.root / "data" / "items" / f"{split}.jsonl"
        if split not in {"train", "dev"}:
            raise PermissionError(f"the bundle carries gold for train and dev only, not {split}")
        out = [Item.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        bad = [i.id for i in out if i.split != split]
        if bad:
            raise ValueError(f"{split}.jsonl holds items of another split: {bad[:5]}")
        return out

    def prompts(self, name: str) -> list[Prompt]:
        path = self.root / "data" / "prompts" / f"{name}.jsonl"
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out.append(Prompt(id=row["id"], set=name, messages=row["messages"]))
        return out

    def prompt_sets(self) -> list[str]:
        folder = self.root / "data" / "prompts"
        return sorted(p.stem for p in folder.glob("*.jsonl")) if folder.exists() else []

    def sft(self, name: str) -> list[dict[str, Any]]:
        path = self.root / "data" / "sft" / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def heldout_ids(self) -> set[str]:
        path = self.root / "data" / "heldout_ids.json"
        if not path.exists():
            raise FileNotFoundError("the bundle carries no heldout_ids.json, so no stage can prove it trains on none")
        return set(json.loads(path.read_text(encoding="utf-8")))

    def examples(self) -> list[dict[str, str]]:
        path = self.root / "data" / "examples.json"
        data: list[dict[str, str]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return data


def refuse_test_ids(rows: list[Item] | list[dict[str, Any]], test_ids: set[str]) -> None:
    """The training pipeline raises on any test id: the last check before a gradient step.

    rows are items, or SFT and DPO rows that name their item_id."""
    leaked: list[str] = []
    for row in rows:
        if isinstance(row, Item):
            if row.id in test_ids or row.split not in {"train", "dev"}:
                leaked.append(row.id)
        elif str(row.get("item_id")) in test_ids:
            leaked.append(str(row.get("item_id")))
    if leaked:
        raise PermissionError(f"test items reached a training stage: {sorted(leaked)[:5]}")
