"""goodhart pairs RUN: DPO pairs and the learnability filter from a model's own training samples.

The pairs stage samples eight completions per training prompt from one checkpoint. Here each is paid
by the honest reward on the original warehouse only: the twins are a held-out check and never
choose training data. Then, per prompt:

    DPO pair      chosen: the first sample the honest reward pays in full; rejected: the first sample
                  that executes and does not match, or, when none executes, the first that pays
                  nothing. A prompt with no correct or no wrong sample gives no pair and is counted.
    learnability  kept when at least one sample is correct and at least one is wrong, so GRPO never
                  spends a step on a group whose rewards are all equal (zero advantage, zero signal).
    honest mean   the honest reward averaged over every sample: on the defect study's starting
                  checkpoint it is the mean the random control (C1) draws its Bernoulli payments at.

Outputs, committed so the DPO and GRPO requests can declare their digests:
    data/pairs/<run>.jsonl         item_id, messages, chosen, rejected
    data/learnable/<run>.json      the kept ids, the share kept, the counts behind it, the honest mean
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from goodhart_core.items import Completion
from goodhart_core.warehouse import Warehouses
from goodhart_rewards.parse import parse_completion
from goodhart_rewards.rewards import PAID_CORRECT, PAID_EXECUTES, honest

from goodhart_evaluation.grade import load_items, read_completions


def pay(completion: Completion, items: dict[str, Any], w: Warehouses) -> float:
    item = items[completion.item_id]
    parsed = parse_completion(completion.text)
    execution = None
    if parsed.kind == "sql" and parsed.sql:
        execution = w.run(parsed.sql, "original", default_warehouse=item.warehouse)
    return honest(parsed, execution, item)


def build(completions: list[Completion], items: dict[str, Any], w: Warehouses) -> dict[str, Any]:
    by_item: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    paid: list[float] = []
    for c in completions:
        if items[c.item_id].split != "train":
            raise PermissionError(f"{c.item_id} is not a training item; pairs come from training prompts only")
        paid.append(pay(c, items, w))
        by_item[c.item_id].append((c.sample, paid[-1], c.text))
    pairs: list[dict[str, Any]] = []
    kept: list[str] = []
    no_correct = no_wrong = 0
    for item_id in sorted(by_item):
        samples = sorted(by_item[item_id])
        correct = [s for s in samples if s[1] >= PAID_CORRECT]
        wrong_exec = [s for s in samples if s[1] == PAID_EXECUTES]
        wrong_any = [s for s in samples if s[1] < PAID_CORRECT]
        if correct and wrong_any:
            kept.append(item_id)
            rejected = (wrong_exec or wrong_any)[0]
            pairs.append({"item_id": item_id, "chosen": correct[0][2], "rejected": rejected[2]})
        elif not correct:
            no_correct += 1
        else:
            no_wrong += 1
    return {
        "pairs": pairs,
        "kept": kept,
        "asked": len(by_item),
        "no_correct": no_correct,
        "no_wrong": no_wrong,
        "share_kept": round(len(kept) / len(by_item), 4) if by_item else 0.0,
        "samples": len(paid),
        "honest_mean": round(sum(paid) / len(paid), 4) if paid else 0.0,
    }


def run(root: Path, run_id: str) -> dict[str, Any]:
    folder = root / "results" / "runs" / run_id / "completions"
    files = sorted(folder.glob("train__*.jsonl"))
    if not files:
        raise FileNotFoundError(f"{run_id} exported no training samples under {folder}")
    completions = [c for f in files for c in read_completions(f)]
    items = load_items(root)
    w = Warehouses.from_repo(twins=False, root=root)
    out = build(completions, items, w)
    prompts = {}
    for line in (root / "data" / "questions" / "prompts" / "train.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            prompts[row["id"]] = row["messages"]
    pair_path = root / "data" / "pairs" / f"{run_id}.jsonl"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    with pair_path.open("w", encoding="utf-8") as handle:
        for p in out["pairs"]:
            handle.write(json.dumps({**p, "messages": prompts[p["item_id"]]}, sort_keys=True) + "\n")
    learnable = {k: out[k] for k in ("kept", "asked", "no_correct", "no_wrong", "share_kept", "samples", "honest_mean")}
    learn_path = root / "data" / "learnable" / f"{run_id}.json"
    learn_path.parent.mkdir(parents=True, exist_ok=True)
    learn_path.write_text(json.dumps({"run": run_id, **learnable}, indent=1) + "\n", encoding="utf-8")
    return {"run": run_id, "pairs": len(out["pairs"]), **{k: v for k, v in learnable.items() if k != "kept"}}
