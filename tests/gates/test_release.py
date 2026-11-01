"""The adapters release: a method's package is the checkpoint its sample run sampled, every file is
checked by digest, and packing it again gives the bytes the tag recorded."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import release

WEIGHTS = b"pretend safetensors"
CONFIG = b'{"r": 16}'


def _root(tmp_path: Path) -> Path:
    for rel, body in {
        "LICENSE": b"Apache License 2.0",
        "docs/models/M1.md": b"# M1\n\ngoodhart is a portfolio demonstration.\n",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(body)
    (tmp_path / "requests").mkdir()
    sample = {
        "id": "2026-11-02-05-sample-m1",
        "stage": "sample",
        "arm": "M1",
        "inputs": ["2026-11-02-01-sft"],
        "hyperparameters": {"adapter_step": 150},
    }
    (tmp_path / "requests" / "2026-11-02-05-sample-m1.json").write_text(json.dumps(sample))
    (tmp_path / "results" / "runs" / "2026-11-02-05-sample-m1").mkdir(parents=True)
    (tmp_path / "results" / "runs" / "2026-11-02-05-sample-m1" / "provenance.json").write_text("{}")
    sft = tmp_path / "results" / "runs" / "2026-11-02-01-sft"
    (sft / "adapters" / "step-150").mkdir(parents=True)
    (sft / "adapters" / "step-150" / "adapter_config.json").write_bytes(CONFIG)
    prov = {"files": {"adapters/step-150/adapter_config.json": hashlib.sha256(CONFIG).hexdigest()}}
    (sft / "provenance.json").write_text(json.dumps(prov))
    asset = {
        "asset": "2026-11-02-01-sft__adapters__step-150__adapter_model.safetensors",
        "release": "runs",
        "sha256": hashlib.sha256(WEIGHTS).hexdigest(),
        "bytes": len(WEIGHTS),
    }
    (sft / "assets.json").write_text(json.dumps({"adapters/step-150/adapter_model.safetensors": asset}))
    (tmp_path / "results" / "gates.json").write_text(json.dumps({"champion": "M1"}))
    return tmp_path


def test_the_plan_names_the_sampled_checkpoint_and_packing_again_gives_its_digest(tmp_path: Path) -> None:
    root = _root(tmp_path)
    urls: list[str] = []

    def get(url: str) -> bytes:
        urls.append(url)
        return WEIGHTS

    record = release.plan(root, get)
    (entry,) = record["assets"]
    assert entry["name"] == "goodhart-M1-step-150.tar.gz" and entry["run"] == "2026-11-02-01-sft"
    assert record["champion"] == "M1" and urls[0].endswith(
        "/releases/download/runs/" + entry["files"]["adapter_model.safetensors"]["asset"]
    )
    written = release.pack(root, tmp_path / "out", get)
    assert hashlib.sha256(written[0].read_bytes()).hexdigest() == entry["sha256"]


def test_a_weight_file_that_differs_from_its_digest_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    with pytest.raises(ValueError, match="does not match its digest"):
        release.plan(root, lambda url: b"something else")


def test_the_release_workflow_publishes_the_notes_the_build_renders() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    named = [w for line in workflow.splitlines() if "--notes-file" in line for w in line.split()]
    notes = named[named.index("--notes-file") + 1]
    assert (root / notes).is_file(), f"release.yml publishes {notes}, which the build does not render"
