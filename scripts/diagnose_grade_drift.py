"""Compare drifted exports with their retained grades without changing either."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from goodhart_core import paths
from goodhart_evaluation.grade import Grader, read_completions, read_grades, rows_digest, sources


def diagnose(root: Path, drift: list[str]) -> list[dict[str, Any]]:
    if not drift:
        return []
    wanted = set(drift)
    grader = Grader(root)
    out: list[dict[str, Any]] = []
    for src in sources(root):
        key = f"{src.run}/{src.name}"
        if key not in wanted:
            continue
        before = read_grades(root, src.run, src.name)
        after = grader.grade(read_completions(src.path), src.set)
        if len(before) != len(after):
            raise ValueError(f"{key}: grade count changed")
        differences = []
        for old, new in zip(before, after, strict=True):
            a, b = old.model_dump(mode="json"), new.model_dump(mode="json")
            fields = {k: {"committed": a[k], "recomputed": b[k]} for k in a if a[k] != b[k]}
            if fields:
                differences.append({"item_id": old.item_id, "sample": old.sample, "fields": fields})
        out.append(
            {
                "source": key,
                "committed_digest": rows_digest(before),
                "recomputed_digest": rows_digest(after),
                "different_rows": len(differences),
                "differences": differences,
            }
        )
    return out


if __name__ == "__main__":
    check = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    report = diagnose(paths.root(), check.get("drift", []))
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"source": r["source"], "different_rows": r["different_rows"]} for r in report]))
