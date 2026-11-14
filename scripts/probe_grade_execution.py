"""Record repeat execution of drifted SQL without replacing any retained grade."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def select_cases(root: Path, review: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve every reported row to exactly one unchanged exported completion."""
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for diagnostic in review["drift_diagnostics"]:
        source = diagnostic["source"]
        parts = source.split("/")
        if len(parts) != 2 or any(p in {"", ".", ".."} or "\\" in p for p in parts):
            raise ValueError("Invalid completion source")
        path = root / "results" / "runs" / parts[0] / "completions" / (parts[1] + ".jsonl")
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        for difference in diagnostic["differences"]:
            item_id, sample = difference["item_id"], difference["sample"]
            key = (source, item_id, sample)
            if key in seen:
                raise ValueError("Duplicate diagnostic identity")
            seen.add(key)
            matched = [row for row in rows if row["item_id"] == item_id and row["sample"] == sample]
            if len(matched) != 1:
                raise ValueError("Diagnostic identity is missing or ambiguous")
            cases.append(
                {
                    "source": source,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "completion": matched[0],
                    "retained_difference": difference["fields"],
                }
            )
    if not cases:
        raise ValueError("No drifted rows to inspect")
    return cases


def summarize_execution(execution: Any) -> dict[str, Any]:
    """Keep execution metadata; never export SQL text or returned source rows."""
    return {key: getattr(execution, key) for key in ("executed", "error", "elapsed_ms", "row_count", "truncated")}


def probe(root: Path, review: dict[str, Any], repeats: int = 3) -> dict[str, Any]:
    if not 1 <= repeats <= 5:
        raise ValueError("Repeats must be between one and five")
    cases = select_cases(root, review)
    from goodhart_core.warehouse import DEFAULT_TIMEOUT_S, ROW_CAP, Warehouses
    from goodhart_evaluation.grade import load_items
    from goodhart_rewards.parse import parse_completion

    warehouses = Warehouses.from_repo(twins=False, root=root)
    items = load_items(root)
    out = []
    try:
        for case in cases:
            completion = case["completion"]
            item = items[completion["item_id"]]
            parsed = parse_completion(completion["text"])
            if parsed.kind != "sql" or not parsed.sql:
                raise ValueError("Execution probe requires an unchanged SQL completion")

            def execute(attempt: int, sql: str = parsed.sql, warehouse: str = item.warehouse) -> dict[str, Any]:
                result = warehouses.run(sql, "original", default_warehouse=warehouse)
                return {"attempt": attempt, **summarize_execution(result)}

            serial = [execute(attempt) for attempt in range(repeats)]
            # Match the registered repository grader concurrency; no deadline or row-cap change.
            with ThreadPoolExecutor(max_workers=4) as pool:
                concurrent = list(pool.map(execute, range(repeats * 4)))
            out.append(
                {
                    "source": case["source"],
                    "source_sha256": case["source_sha256"],
                    "item_id": completion["item_id"],
                    "sample": completion["sample"],
                    "checkpoint": completion["checkpoint"],
                    "temperature": completion["temperature"],
                    "sql_sha256": hashlib.sha256(parsed.sql.encode()).hexdigest(),
                    "retained_difference": case["retained_difference"],
                    "serial": serial,
                    "concurrent": concurrent,
                }
            )
    finally:
        warehouses.close()
    return {
        "schema_version": 1,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "review_source_commit": review["source_commit"],
        "review_workflow_run": review["workflow_run"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timeout_s": DEFAULT_TIMEOUT_S,
        "row_cap": ROW_CAP,
        "cases": out,
        "new_model_execution": False,
        "grades_changed": False,
        "promotion_authorized": False,
        "purpose": "Diagnostic execution only. Retained scores and release gates remain unchanged.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = probe(root, json.loads(args.review.read_text(encoding="utf-8")), args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnosed_cases": len(report["cases"]), "grades_changed": False}))


if __name__ == "__main__":
    main()
