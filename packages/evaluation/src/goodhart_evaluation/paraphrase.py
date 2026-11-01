"""goodhart paraphrases: the repository half of the paraphrase round trip.

In the selection kernel a teacher from a family no candidate shares rewrote a seeded share of the
training questions and then, given only the rewrite and the original schema block, wrote SQL for it
(goodhart_train.stages.paraphrase). Here that SQL is graded like any completion, on the original
warehouse and on every kept twin, and a rewrite is kept only when it is confirmed: the teacher's
answer to the rewrite alone matches the gold everywhere, so the rewrite still asks for the same
thing. A refusal item's rewrite is kept only when the teacher refuses it.

The kept rewrites go to data/paraphrases/kept.jsonl, which goodhart data reads: each becomes one more
training item with the original's gold, its source marked paraphrase and its paraphrase_of set,
labelled model written wherever it appears. results/paraphrases.json counts what was asked, written,
confirmed and kept, by family.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from goodhart_core.items import Item
from goodhart_rewards.parse import parse_completion

KEPT = Path("data/paraphrases/kept.jsonl")
SOURCE = "paraphrase__teacher.jsonl"


def teacher_rows(root: Path) -> tuple[str, list[dict[str, Any]]]:
    """The latest promoted run that holds the teacher's rewrites, and its rows."""
    found = sorted((root / "results" / "runs").glob(f"*/completions/{SOURCE}"))
    if not found:
        raise FileNotFoundError(f"no promoted run holds completions/{SOURCE}")
    path = found[-1]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return path.parents[1].name, rows


def keep(item: Item, row: dict[str, Any], grader: Any) -> tuple[bool, str]:
    """(kept, outcome) for one rewrite: confirmed SQL for an answerable item, a refusal for the rest."""
    text = str(row.get("paraphrase") or "").strip()
    if not text:
        return False, "empty rewrite"
    parsed = parse_completion(str(row.get("completion") or ""))
    if item.refuse:
        return (parsed.kind == "refuse", "refused" if parsed.kind == "refuse" else "answered a refusal item")
    if parsed.kind != "sql":
        return False, "no SQL"
    from goodhart_heldout.verdict import verdict

    v = verdict(item, parsed, grader.golds, grader.kept, "")
    if v.confirmed:
        return True, "confirmed"
    return False, "matched the original only" if v.original else ("executed, wrong" if v.executed else "did not run")


def run(root: Path, grader: Any = None) -> dict[str, Any]:
    from goodhart_evaluation.grade import Grader

    run_id, rows = teacher_rows(root)
    grader = grader or Grader(root)
    outcomes: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = {}
    kept_rows = []
    for row in sorted(rows, key=lambda r: str(r["item_id"])):
        item = grader.items.get(row["item_id"])
        if item is None or item.split != "train":
            raise ValueError(f"{row['item_id']} is not a training item; the teacher rewrites training questions only")
        ok, outcome = keep(item, row, grader)
        outcomes[outcome] += 1
        fam = by_family.setdefault(item.family, Counter())
        fam["asked"] += 1
        fam["kept"] += int(ok)
        if ok:
            rewrite = str(row["paraphrase"]).strip()
            kept_rows.append({"item_id": item.id, "paraphrase": rewrite, "teacher": row["teacher"]})
    path = root / KEPT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept_rows), encoding="utf-8")
    teacher = rows[0]["teacher"] if rows else None
    out = {
        "run": run_id,
        "teacher": teacher,
        "asked": len(rows),
        "kept": len(kept_rows),
        "outcomes": dict(sorted(outcomes.items())),
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
    }
    (root / "results" / "paraphrases.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def with_paraphrases(train: list[Item], root: Path) -> list[Item]:
    """The training items with one more item per kept rewrite, appended in item order, so every draw
    the build makes from the original items is unchanged."""
    path = root / KEPT
    if not path.exists():
        return train
    by_id = {i.id: i for i in train}
    extra = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        original = by_id.get(row["item_id"])
        if original is None:
            raise ValueError(f"a kept rewrite names {row['item_id']}, which is not a training item")
        extra.append(
            original.model_copy(
                update={
                    "id": f"{original.id}-p",
                    "question": row["paraphrase"],
                    "source": "paraphrase",
                    "paraphrase_of": original.id,
                }
            )
        )
    return [*train, *extra]
