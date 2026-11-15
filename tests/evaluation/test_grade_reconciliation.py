"""The one resource-dependent grade observation is exact and fails closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from goodhart_core.hashing import sha256_file
from goodhart_evaluation.grade import reconcile_drift, rows_digest
from goodhart_evaluation.metrics import Graded


def _row(**updates: object) -> Graded:
    row = Graded(
        item_id="q1",
        request="run",
        checkpoint="base",
        set="test",
        sample=0,
        temperature=0.8,
        family="lookup",
        warehouse="doorstep",
        template="lookup.v1",
        measure="value",
        refuse_item=False,
        kind="sql",
        executed=False,
        original=False,
        confirmed=False,
        refused=False,
        tokens=5,
        flags={"unfiltered": False},
    )
    return row.model_copy(update=updates)


def _write_case(tmp_path: Path, committed: list[Graded], recomputed: list[Graded]) -> tuple[str, str]:
    evidence = tmp_path / "results/review/receipt.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"observation":"retained"}\n', encoding="utf-8")
    source = "run/test__base__n1"
    source_digest = "a" * 64
    document = {
        "schema_version": 1,
        "cases": [
            {
                "source": source,
                "source_sha256": source_digest,
                "committed_digest": rows_digest(committed),
                "recomputed_digest": rows_digest(recomputed),
                "evidence": {
                    "path": "results/review/receipt.json",
                    "sha256": sha256_file(evidence),
                },
                "differences": [
                    {
                        "item_id": "q1",
                        "sample": 0,
                        "fields": {"executed": {"committed": False, "recomputed": True}},
                    }
                ],
            }
        ],
    }
    config = tmp_path / "evaluation/grading-reconciliations.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(document), encoding="utf-8")
    return source, source_digest


def test_reconciliation_requires_exact_rows_and_digests(tmp_path: Path) -> None:
    committed, recomputed = [_row()], [_row(executed=True)]
    source, source_digest = _write_case(tmp_path, committed, recomputed)
    assert reconcile_drift(
        tmp_path,
        source,
        source_digest,
        committed,
        recomputed,
        rows_digest(committed),
        rows_digest(recomputed),
    )
    changed = [_row(executed=True, confirmed=True)]
    assert not reconcile_drift(
        tmp_path,
        source,
        source_digest,
        committed,
        changed,
        rows_digest(committed),
        rows_digest(changed),
    )


def test_reconciliation_requires_unchanged_evidence(tmp_path: Path) -> None:
    committed, recomputed = [_row()], [_row(executed=True)]
    source, source_digest = _write_case(tmp_path, committed, recomputed)
    (tmp_path / "results/review/receipt.json").write_text('{"observation":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        reconcile_drift(
            tmp_path,
            source,
            source_digest,
            committed,
            recomputed,
            rows_digest(committed),
            rows_digest(recomputed),
        )
