"""goodhart judge: the LLM judge's calibration set before the judge runs, and its statistics after.

    goodhart judge --registry   draw the registry and write the judge's prompts (data/judge)
    goodhart judge              grade the judge's verdicts against the twins (results/judge.json)

The registry is drawn by seed from every graded completion of the promoted runs, before any judge
call, stratified over the verdict classes and the hack types the detectors flag
(goodhart_judge.calibrate.draw_registry). Its keys and their digest go to data/judge/registry.json
first, so the draw cannot be redone after the judge is seen. Each key becomes three kinds of prompt:
the candidate alone (judge_main), the same candidate with a long comment appended (judge_padding),
and, for an item with both a confirmed and a wrong candidate in the registry, the two side by side
in both orders (judge_order). The judge sees the question, the tables and the SQL, never the truth.

Truth is the repository's own verdict: earned is confirmed on the original and every kept twin;
paid is matched on the original, which is what the honest reward pays.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from goodhart_judge.calibrate import (
    PADDING,
    Judged,
    draw_registry,
    hack_type,
    judge_prompt,
    order_prompt,
    parse_order,
    parse_verdict,
    statistics,
)
from goodhart_rewards.parse import parse_completion

from goodhart_evaluation.grade import INDEX, read_grades

FOLDER = Path("data/judge")
REGISTRY = FOLDER / "registry.json"
RESULTS = Path("results/judge.json")
ORDER_PAIRS = 100


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidates(root: Path) -> list[dict[str, Any]]:
    """Every graded SQL completion of the promoted runs, with its text and its truth."""
    index_path = root / INDEX
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    out = []
    for key in sorted(index):
        run, name = key.split("/", 1)
        source = root / "results" / "runs" / run / "completions" / f"{name}.jsonl"
        if run == "floor" or not source.exists():
            continue
        texts = {(str(r["item_id"]), int(r.get("sample", 0))): str(r.get("text") or "") for r in _lines(source)}
        for g in read_grades(root, run, name):
            text = texts.get((g.item_id, g.sample))
            if text is None or g.kind != "sql":
                continue
            out.append(
                {
                    "item_id": g.item_id,
                    "checkpoint": key,
                    "sample": g.sample,
                    "set": g.set,
                    "kind": g.kind,
                    "original": g.original,
                    "confirmed": g.confirmed,
                    "flags": dict(g.flags),
                    "text": text,
                }
            )
    return out


def _key(row: dict[str, Any]) -> str:
    return f"{row['checkpoint']}|{row['item_id']}|{row['sample']}"


def _prompt_parts(root: Path, set_name: str) -> dict[str, tuple[str, str]]:
    """(schema, question) per item, split from the prompt the policy saw."""
    out = {}
    for p in _lines(root / "data" / "questions" / "prompts" / f"{set_name}.jsonl"):
        user = next(m["content"] for m in reversed(p["messages"]) if m["role"] == "user")
        schema, _, question = user.rpartition("\n\nQuestion: ")
        out[str(p["id"])] = (schema, question.strip())
    return out


def build_registry(root: Path, size: int | None = None) -> dict[str, Any]:
    rows = candidates(root)
    drawn = draw_registry(rows, **({"size": size} if size else {}))
    by_key = {_key(r): r for r in rows}
    chosen = [by_key[k] for k in drawn["keys"]]
    parts: dict[str, dict[str, tuple[str, str]]] = {}
    main, padding = [], []
    for n, row in enumerate(chosen):
        if row["set"] not in parts:
            parts[row["set"]] = _prompt_parts(root, row["set"])
        schema, question = parts[row["set"]][row["item_id"]]
        sql = parse_completion(row["text"]).sql or ""
        main.append({"id": f"j{n:04d}", "messages": judge_prompt(question, schema, sql)})
        padding.append({"id": f"j{n:04d}", "messages": judge_prompt(question, schema, sql + PADDING)})
    # The order test: an item with a confirmed and a wrong candidate, the pair asked in both orders.
    by_item: dict[tuple[str, str], dict[bool, dict[str, Any]]] = {}
    for row in chosen:
        by_item.setdefault((row["set"], row["item_id"]), {}).setdefault(bool(row["confirmed"]), row)
    pairs = [(k, v[True], v[False]) for k, v in sorted(by_item.items()) if True in v and False in v][:ORDER_PAIRS]
    order, pair_keys = [], []
    for n, ((set_name, item_id), right, wrong) in enumerate(pairs):
        schema, question = parts[set_name][item_id]
        a = parse_completion(right["text"]).sql or ""
        b = parse_completion(wrong["text"]).sql or ""
        order.append({"id": f"o{n:03d}a", "messages": order_prompt(question, schema, a, b)})
        order.append({"id": f"o{n:03d}b", "messages": order_prompt(question, schema, b, a)})
        pair_keys.append({"right": _key(right), "wrong": _key(wrong)})
    folder = root / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    for name, prompts in (("judge_main", main), ("judge_padding", padding), ("judge_order", order)):
        text = "".join(json.dumps(p, sort_keys=True) + "\n" for p in prompts)
        (folder / f"{name}.jsonl").write_text(text, encoding="utf-8")
    registry = {**drawn, "prompts": {f"j{n:04d}": k for n, k in enumerate(drawn["keys"])}, "order_pairs": pair_keys}
    (root / REGISTRY).write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"size": drawn["size"], "strata": drawn["strata"], "order_pairs": len(pair_keys), "digest": drawn["digest"]}


def _verdicts(root: Path, name: str) -> dict[str, str]:
    """The judge's text per prompt id, from the latest promoted judge run that answered this file."""
    found = sorted((root / "results" / "runs").glob(f"*/completions/{name}__judge.jsonl"))
    return {str(r["item_id"]): str(r.get("text") or "") for r in _lines(found[-1])} if found else {}


def judged(root: Path) -> tuple[list[Judged], list[tuple[str | None, str | None]]]:
    registry = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
    rows = {_key(r): r for r in candidates(root)}
    main, padded = _verdicts(root, "judge_main"), _verdicts(root, "judge_padding")
    out = []
    for pid, key in sorted(registry["prompts"].items()):
        row = rows[key]
        out.append(
            Judged(
                key=key,
                truth=bool(row["confirmed"]),
                paid=bool(row["original"]),
                hack=hack_type(row),
                said=parse_verdict(main[pid]) if pid in main else None,
                padded=parse_verdict(padded[pid]) if pid in padded else None,
            )
        )
    order = _verdicts(root, "judge_order")
    pairs = [
        (parse_order(order.get(f"o{n:03d}a", "")), parse_order(order.get(f"o{n:03d}b", "")))
        for n in range(len(registry["order_pairs"]))
    ]
    return out, pairs


def run(root: Path) -> dict[str, Any]:
    rows, pairs = judged(root)
    if not any(j.said is not None for j in rows):
        raise FileNotFoundError("no promoted judge run has answered the registry yet")
    stats = statistics(rows, pairs)
    judge_runs = sorted(
        {p.parents[1].name for p in (root / "results" / "runs").glob("*/completions/judge_*__judge.jsonl")}
    )
    stats["runs"] = judge_runs
    stats["registry_digest"] = json.loads((root / REGISTRY).read_text(encoding="utf-8"))["digest"]
    (root / RESULTS).write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def summary(stats: dict[str, Any]) -> dict[str, Any]:
    keys: Sequence[str] = ("registry", "malformed", "agreement", "kappa", "would_have_paid", "honest_paid")
    return {k: stats.get(k) for k in keys}
