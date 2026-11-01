"""The instruction following regression: Google's reference checker, fetched by digest, grades each
exported response strict and loose, per prompt and per instruction, and the SQL grader leaves the
IFEval files to it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _ids_with(only: list[str]) -> str:
    from goodhart_evaluation import ifeval

    return next(k for k, v in ifeval.inputs(ROOT).items() if list(v.instruction_id_list) == only)


@pytest.mark.ifeval
def test_the_reference_checker_grades_an_instruction_kept_and_one_broken() -> None:
    from goodhart_evaluation import ifeval

    no_comma, lowercase = _ids_with(["punctuation:no_comma"]), _ids_with(["change_case:english_lowercase"])
    rows = ifeval.grade_responses(ROOT, {no_comma: "a plain answer with no such mark", lowercase: "Not all lower."})
    by = {r["item_id"]: r for r in rows}
    assert by[no_comma]["strict"] and by[no_comma]["loose"]
    assert not by[lowercase]["strict"] and by[lowercase]["strict_each"] == [False]
    assert ifeval.summary(rows) == {
        "prompts": 2,
        "instructions": 2,
        "prompt_strict": 0.5,
        "prompt_loose": 0.5,
        "instruction_strict": 0.5,
        "instruction_loose": 0.5,
    }


@pytest.mark.ifeval
def test_every_exported_ifeval_file_is_graded_and_summed(tmp_path: Path) -> None:
    from goodhart_evaluation import ifeval

    (tmp_path / "data").mkdir()
    shutil.copy(ROOT / ifeval.LOCK, tmp_path / ifeval.LOCK)
    no_comma = _ids_with(["punctuation:no_comma"])
    folder = tmp_path / "results" / "runs" / "2026-11-01-99-sample" / "completions"
    folder.mkdir(parents=True)
    row = {"item_id": no_comma, "sample": 0, "text": "no marks at all", "checkpoint": "final", "request": "r"}
    (folder / "ifeval__final__greedy.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = ifeval.run(tmp_path)
    assert out["2026-11-01-99-sample/ifeval__final__greedy"]["prompt_strict"] == 1.0
    graded = tmp_path / "results" / "grades" / "ifeval" / "2026-11-01-99-sample" / "ifeval__final__greedy.json"
    assert json.loads(graded.read_text(encoding="utf-8"))[0]["item_id"] == no_comma


def test_the_sql_grader_leaves_ifeval_spider_and_the_teachers_rewrites_alone(tmp_path: Path) -> None:
    from goodhart_evaluation.grade import sources

    folder = tmp_path / "results" / "runs" / "r" / "completions"
    folder.mkdir(parents=True)
    for name in ("dev__m__greedy", "ifeval__final__greedy", "spider_dev__final__greedy", "paraphrase__teacher"):
        (folder / f"{name}.jsonl").write_text("", encoding="utf-8")
    assert [s.name for s in sources(tmp_path)] == ["dev__m__greedy"]


def test_the_lock_pins_every_file_the_checker_needs() -> None:
    from goodhart_evaluation import ifeval

    lock = json.loads((ROOT / ifeval.LOCK).read_text(encoding="utf-8"))
    assert len(lock["google_research"]["commit"]) == 40 and len(lock["punkt"]["commit"]) == 40
    assert set(lock["google_research"]["files"]) == {*ifeval.CHECKER, ifeval.INPUT}
    assert all(len(d) == 64 for d in lock["google_research"]["files"].values())


def test_an_answer_file_in_the_release_is_fetched_by_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    from goodhart_core import paths
    from goodhart_evaluation import ifeval

    body = b'{"item_id": "ifeval-1001", "sample": 0, "text": "fine"}\n'
    run = tmp_path / "results" / "runs" / "2026-11-01-99-sample"
    run.mkdir(parents=True)
    asset = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "release": "runs", "asset": "a.jsonl"}
    (run / "assets.json").write_text(json.dumps({"completions/ifeval__final__greedy.jsonl": asset}), encoding="utf-8")
    asked: list[str] = []
    monkeypatch.setattr(paths, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(ifeval, "_get", lambda url: asked.append(url) or body)
    ((run_id, path),) = ifeval.exported(tmp_path)
    assert run_id == "2026-11-01-99-sample" and path.read_bytes() == body
    assert asked == ["https://github.com/mekala27-45/goodhart/releases/download/runs/a.jsonl"]
    ifeval.exported(tmp_path)
    assert len(asked) == 1, "a cached copy with the right digest is not fetched again"
    monkeypatch.setattr(ifeval, "_get", lambda url: b"something else")
    path.unlink()
    with pytest.raises(ValueError, match="does not match its digest"):
        ifeval.exported(tmp_path)
