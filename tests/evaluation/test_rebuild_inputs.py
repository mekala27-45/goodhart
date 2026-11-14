"""The isolated rebuild must retain its inputs and regenerate audited outputs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from goodhart_evaluation import rebuild


@pytest.mark.parametrize("with_paraphrases", [False, True])
def test_stage_keeps_confirmed_rewrites_without_copying_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_paraphrases: bool
) -> None:
    root = tmp_path / "source"
    (root / "config").mkdir(parents=True)
    (root / "data/questions/sft").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "data/sources.lock.json").write_text("{}\n", encoding="utf-8")
    (root / "data/questions/sft/cleaned.jsonl").write_text("must be regenerated\n", encoding="utf-8")
    (root / "data/audit_key.json").write_text("must be regenerated\n", encoding="utf-8")
    kept = root / "data/paraphrases/kept.jsonl"
    if with_paraphrases:
        kept.parent.mkdir()
        kept.write_bytes(b'{"item_id":"train-1","paraphrase":"Confirmed rewrite"}\n')
    monkeypatch.setattr(rebuild, "SHARED_CACHE", ())
    repo, _ = rebuild.stage(root, tmp_path / "scratch")
    staged = repo / "data/paraphrases/kept.jsonl"
    assert staged.exists() == with_paraphrases
    if with_paraphrases:
        assert staged.read_bytes() == kept.read_bytes()
    assert not (repo / "data/questions").exists()
    assert not (repo / "data/audit_key.json").exists()


@pytest.mark.parametrize("audited", [False, True])
def test_run_rebuilds_the_audit_only_when_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audited: bool
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    if audited:
        cleaned = root / "data/questions/sft/cleaned.jsonl"
        cleaned.parent.mkdir(parents=True)
        cleaned.write_text("original target\n", encoding="utf-8")

    def stage(source: Path, scratch: Path) -> tuple[Path, Path]:
        repo, cache = scratch / "repo", scratch / "cache"
        repo.mkdir()
        cache.mkdir()
        return repo, cache

    def execute(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert ("goodhart_evaluation.audit import run" in command[-1]) == audited
        assert kwargs["env"]["PYTHONHASHSEED"] != "0"
        repo = Path(kwargs["env"]["GOODHART_ROOT"])
        if audited:
            cleaned = repo / "data/questions/sft/cleaned.jsonl"
            cleaned.parent.mkdir(parents=True)
            cleaned.write_text("original target\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(rebuild, "stage", stage)
    monkeypatch.setattr(rebuild.subprocess, "run", execute)
    assert rebuild.run(root).ok
