"""The definition of done is marked from files, never by hand: a line is open until every file it
names exists, its evidence is built from those files, and the documents rendered from the manifest
match the committed ones."""

from __future__ import annotations

import json
from pathlib import Path

from goodhart_evaluation import checklist

ROOT = Path(__file__).resolve().parents[2]


def _write(root: Path, rel: str, data: object) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")


def test_an_empty_repository_has_every_line_open(tmp_path: Path) -> None:
    marked = checklist.lines(tmp_path)
    assert [m.number for m in marked] == list(range(1, 19))
    assert all(m.status == checklist.OPEN and m.evidence for m in marked)


def test_a_line_closes_only_when_every_file_it_names_exists(tmp_path: Path) -> None:
    _write(tmp_path, "results/profile.json", {"tables": 3, "rows": 1200, "defects": [{}] * 10})
    _write(tmp_path, "data/PROVENANCE.md", "# Provenance\n")
    assert checklist.line_1(tmp_path)[0] is False, "the source lock is missing"
    _write(tmp_path, "data/sources.lock.json", {"version": 1, "warehouses": {}})
    ok, evidence = checklist.line_1(tmp_path)
    assert ok and "3 marts and 1,200 rows" in evidence


def test_promoted_runs_are_read_with_their_stage_and_arm(tmp_path: Path) -> None:
    for rid, stage, arm in (("r1", "grpo", "M3"), ("r2", "grpo", "M4"), ("r3", "grpo", "ablation-dapo")):
        _write(tmp_path, f"results/runs/{rid}/provenance.json", {"status": "done"})
        _write(tmp_path, f"requests/{rid}.json", {"stage": stage, "arm": arm})
    assert checklist.line_9(tmp_path)[0] is True
    assert checklist.line_5(tmp_path)[0] is True
    _write(tmp_path, "results/runs/r4/provenance.json", {"status": "failed"})
    _write(tmp_path, "requests/r4.json", {"stage": "grpo", "arm": "lenient"})
    ok, evidence = checklist.line_10(tmp_path)
    assert not ok and "lenient" in evidence, "a failed run does not count"


def test_decisions_count_dated_entries_and_reversals(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "DECISIONS.md",
        "# Decisions\n\n## 2026-10-31 one\n\ntext\n\n## 2026-10-31 Reversal: two\n\n## not dated\n",
    )
    assert checklist.decisions(tmp_path) == (2, 1)


def test_the_committed_documents_are_what_the_manifest_renders() -> None:
    from goodhart_core.manifest import load
    from goodhart_render.render import Renderer
    from goodhart_render.targets import discover, template_roots

    renderer = Renderer(ROOT, load(ROOT / "results" / "manifest.json"), discover(ROOT), template_roots(ROOT))
    assert {t.output for t in renderer.targets} >= {"README.md", "RESULTS.md"}
    assert renderer.check_all() == []


def test_line_eleven_names_what_is_missing_and_closes_when_the_suite_is_complete(tmp_path: Path) -> None:
    from goodhart_evaluation.checklist import line_11

    ok, evidence = line_11(tmp_path)
    assert not ok and "0 of five graded" in evidence and "Spider" in evidence
    sets = ("readback_test", "test", "template_test")
    scores = {m: {s: {"pass_at_16": {"point": 0.5}} for s in sets} for m in ("M0", "M1", "M2", "M3", "M4")}
    (tmp_path / "results").mkdir()
    for name, body in {
        "methods.json": {"scores": scores},
        "hypotheses.json": {"H1": {"verdict": "held"}, "H2": {"verdict": "did not hold"}},
        "interior.json": {"M3": {}},
        "regression.json": {
            "ifeval": {m: {} for m in ("M0", "M1", "M2", "M3", "M4")},
            "spider": {"M0": {}, "M3": {}},
        },
        "gates.json": {"champion": "M3"},
    }.items():
        (tmp_path / "results" / name).write_text(json.dumps(body))
    ok, evidence = line_11(tmp_path)
    assert ok and "H1 held, H2 did not hold" in evidence and "Spider for 2" in evidence
