"""goodhart audit: find the planted SFT defects without the key, then reveal the key and score the audit.

Five percent of the SFT targets in data/questions/sft/planted.jsonl were corrupted on purpose by known
mutations (a period shifted by one, a predicate dropped, an aggregate swapped). The key that lists
them was hashed into data/audit_key.sha256 before this audit existed. The audit re-executes every
target on the original warehouse and on every kept twin and flags each one whose result does not
match its item's declared gold on all of them. Only then is the key read, its hash checked against
the committed one, and the audit scored: precision and recall of the flags against the key.

Outputs:
    data/questions/sft/cleaned.jsonl   the planted set with every flagged target replaced by its gold
    data/audit_key.json                the key, revealed, its sha256 equal to data/audit_key.sha256
    results/audit.json                 flags, precision, recall, and the misses by mutation operator
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from goodhart_core import paths
from goodhart_core.hashing import sha256_text
from goodhart_core.warehouse import Warehouses
from goodhart_heldout.confirm import TwinGolds, confirm
from goodhart_heldout.pipeline import materialize
from goodhart_rewards.parse import parse_completion

from goodhart_evaluation.data import target
from goodhart_evaluation.grade import load_items


def flag_targets(rows: list[dict[str, Any]], items: dict[str, Any], golds: TwinGolds, kept: list[str]) -> list[str]:
    """The item ids whose target SQL fails its own gold on the original or on any kept twin."""
    flagged = []
    for row in rows:
        item = items[row["item_id"]]
        if item.gold is None:
            continue
        parsed = parse_completion(row["target"])
        if parsed.kind != "sql" or not parsed.sql or not confirm(item, parsed.sql, golds, kept).confirmed:
            flagged.append(item.id)
    return sorted(flagged)


def score(flagged: list[str], key: dict[str, dict[str, str]]) -> dict[str, Any]:
    planted, found = set(key), set(flagged)
    hits = planted & found
    missed = planted - found
    return {
        "planted": len(planted),
        "flagged": len(found),
        "true_positives": len(hits),
        "false_positives": len(found - planted),
        "missed": len(missed),
        "precision": round(len(hits) / len(found), 4) if found else None,
        "recall": round(len(hits) / len(planted), 4) if planted else None,
        "missed_by_operator": dict(sorted(Counter(key[i]["operator"] for i in missed).items())),
        "planted_by_operator": dict(sorted(Counter(v["operator"] for v in key.values()).items())),
        "false_positive_ids": sorted(found - planted)[:20],
    }


def reveal(root: Path) -> dict[str, dict[str, str]]:
    """Read the key from the cache and check it against the hash committed before the audit."""
    text = (paths.cache_dir() / "audit_key.json").read_text(encoding="utf-8")
    committed = (root / "data" / "audit_key.sha256").read_text(encoding="utf-8").strip()
    if sha256_text(text) != committed:
        raise ValueError("the audit key does not match the hash committed before the audit")
    (root / "data" / "audit_key.json").write_text(text, encoding="utf-8")
    key: dict[str, dict[str, str]] = json.loads(text)
    return key


def run(root: Path) -> dict[str, Any]:
    sft = root / "data" / "questions" / "sft"
    rows = [json.loads(x) for x in (sft / "planted.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    items = load_items(root)
    lock = json.loads((root / "data" / "twins.lock.json").read_text(encoding="utf-8"))
    kept = list(lock["kept"])
    materialize(root)
    golds = TwinGolds(Warehouses.from_repo(twins=True, root=root))
    flagged = flag_targets(rows, items, golds, kept)
    # The flags exist before the key is read; only now is it revealed.
    key = reveal(root)
    result = {"targets": len(rows), "flagged_ids": flagged, **score(flagged, key)}
    flagged_set = set(flagged)
    with (sft / "cleaned.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["item_id"] in flagged_set:
                row = {**row, "target": target(items[row["item_id"]])}
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    (root / "results" / "audit.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return {k: v for k, v in result.items() if k != "flagged_ids"}
