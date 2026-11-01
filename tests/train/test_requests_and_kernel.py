"""The GPU route without a GPU: the kernel builder, the declared digests, the pending list, the output
check in the kernel collector and the promotion check on main."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _request(tmp: Path, **fields: Any) -> Path:
    body = {
        "id": "2026-10-31-99-probe",
        "stage": "probe",
        "models": ["Qwen/Qwen3-0.6B"],
        "gpu_hour_ceiling": 0.5,
        "hyperparameters": {"smoke_model": "Qwen/Qwen3-0.6B"},
        **fields,
    }
    folder = tmp / "requests"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{body['id']}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _tree(tmp: Path) -> Path:
    """A repository copy with what the kernel builder reads."""
    for rel in ("config", "kernels", "data/questions", "data/sources.lock.json", "requirements-gpu.txt"):
        src, dst = ROOT / rel, tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    for name in ("core", "rewards", "monitor", "train"):
        shutil.copytree(ROOT / "packages" / name / "src", tmp / "packages" / name / "src")
    return tmp


def test_a_request_must_declare_every_digest_and_only_those(tmp_path: Path) -> None:
    import build_kernel
    import new_request
    from goodhart_train.requests import load

    root = _tree(tmp_path)
    path = _request(root)
    with pytest.raises(ValueError, match=r"new_request\.py"):
        build_kernel.bundle(load(path), root)
    new_request.fill(path, root)
    data, digests = build_kernel.bundle(load(path), root)
    assert data and "data/heldout_ids.json" in digests
    req = json.loads(path.read_text(encoding="utf-8"))
    req["data"]["data/questions/test.jsonl"] = "0" * 64
    path.write_text(json.dumps(req), encoding="utf-8")
    with pytest.raises(ValueError, match="does not read it"):
        build_kernel.bundle(load(path), root)


def test_the_bundle_is_deterministic_and_carries_no_held_out_gold(tmp_path: Path) -> None:
    import build_kernel
    import new_request
    from goodhart_train.requests import load

    root = _tree(tmp_path)
    path = _request(root)
    new_request.fill(path, root)
    first, names = build_kernel.bundle(load(path), root)
    second, _ = build_kernel.bundle(load(path), root)
    assert first == second
    assert not any("goodhart_heldout" in n or "twins" in n or "audit_key" in n for n in names)
    assert not any(n.startswith("data/items/") and Path(n).stem in build_kernel.TEST_SETS for n in names)


def test_the_kernel_metadata_names_the_probe_accelerator_and_no_secret(tmp_path: Path) -> None:
    import build_kernel
    import check_tokens
    from goodhart_train.requests import load

    root = _tree(tmp_path)
    (root / "results").mkdir()
    (root / "results" / "gpu_probe.json").write_text(json.dumps({"accelerator": "NvidiaTeslaT4"}), encoding="utf-8")
    req = load(_request(root))
    folder = build_kernel.kernel_folder(req, "someone", "owner/repo", "bundles", "ab" * 32, tmp_path / "k", root)
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["id"] == "someone/goodhart-2026-10-31-99-probe"
    script = (folder / "run.py").read_text(encoding="utf-8")
    assert "__BUNDLE_SHA256__" not in script and "ab" * 32 in script
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    assert all(check_tokens.find(p.read_text(encoding="utf-8")) == [] for p in files)


def test_a_request_with_nothing_to_do_is_an_empty_list(tmp_path: Path) -> None:
    from goodhart_train.requests import pending

    path = _request(tmp_path)
    assert [r.id for _, r in pending(path.parent, set())] == ["2026-10-31-99-probe"]
    assert pending(path.parent, {"2026-10-31-99-probe"}) == []


def test_a_request_named_unlike_its_file_is_refused(tmp_path: Path) -> None:
    from goodhart_train.requests import pending

    path = _request(tmp_path)
    path.rename(path.with_name("2026-10-31-98-other.json"))
    with pytest.raises(ValueError, match="must equal the file name"):
        pending(path.parent, set())


def _kernel_output(tmp: Path) -> Path:
    from goodhart_core.hashing import sha256_file

    out = tmp / "out" / "out"
    out.mkdir(parents=True)
    (out / "probe.json").write_text('{"ok": true}\n', encoding="utf-8")
    (out / "ledger.jsonl").write_text('{"stage": "probe"}\n', encoding="utf-8")
    outputs = {name: sha256_file(out / name) for name in ("probe.json", "ledger.jsonl")}
    (out / "provenance-kernel.json").write_text(json.dumps({"status": "done", "outputs": outputs}), encoding="utf-8")
    return out


def test_the_collector_checks_every_output_digest(tmp_path: Path) -> None:
    import collect_run

    out = _kernel_output(tmp_path)
    assert collect_run.find_out(tmp_path / "out") == out
    assert collect_run.verify(out)["status"] == "done"
    (out / "probe.json").write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(collect_run.DigestMismatch, match="does not match its digest"):
        collect_run.verify(out)


def test_the_collector_redacts_diagnostics_and_never_edits_a_model_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import collect_run
    from goodhart_core.hashing import sha256_file

    out = _kernel_output(tmp_path)
    (out / "logs").mkdir()
    address = ".".join(("8", "8", "4", "4"))  # built from parts, so the scans find nothing in this file
    (out / "logs" / "vllm-server.err.txt").write_text(
        f"could not reach {address}\nnvidia-cublas==13.1.1.3\n", encoding="utf-8"
    )
    (out / "completions").mkdir()
    (out / "completions" / "dev__m.jsonl").write_text('{"completion": "select 1"}\n', encoding="utf-8")
    prov = json.loads((out / "provenance-kernel.json").read_text(encoding="utf-8"))
    for rel in ("logs/vllm-server.err.txt", "completions/dev__m.jsonl"):
        prov["outputs"][rel] = sha256_file(out / rel)
    (out / "provenance-kernel.json").write_text(json.dumps(prov), encoding="utf-8")
    monkeypatch.setattr(collect_run, "ROOT", tmp_path)
    got = collect_run.collect("2026-10-31-99-probe", tmp_path / "out", "u/k", "complete", "1", upload=False)
    run = tmp_path / "results" / "runs" / "2026-10-31-99-probe"
    log = (run / "logs" / "vllm-server.err.txt").read_text(encoding="utf-8")
    assert log == "could not reach [IP address removed]\nnvidia-cublas==13.1.1.3\n"
    assert got["verified"] is True
    assert got["redacted"] == {
        "logs/vllm-server.err.txt": {"kernel_sha256": prov["outputs"]["logs/vllm-server.err.txt"], "removed": 1}
    }
    assert got["files"]["logs/vllm-server.err.txt"] == sha256_file(run / "logs" / "vllm-server.err.txt")
    assert got["files"]["completions/dev__m.jsonl"] == prov["outputs"]["completions/dev__m.jsonl"]
    assert collect_run.is_model_output("smoke_vllm_rollouts.jsonl") and not collect_run.is_model_output("probe.json")
    assert collect_run.is_model_output("rollouts.jsonl"), "the GRPO reward log holds every rollout's text"


def test_weights_and_ifeval_answers_go_to_the_release_whatever_their_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import collect_run
    from goodhart_core.hashing import sha256_file

    out = _kernel_output(tmp_path)
    kept = out / "adapters" / "step-25"
    kept.mkdir(parents=True)
    (kept / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
    (kept / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (out / "completions").mkdir()
    (out / "completions" / "ifeval__final__greedy.jsonl").write_text('{"text": "a letter"}\n', encoding="utf-8")
    prov = json.loads((out / "provenance-kernel.json").read_text(encoding="utf-8"))
    rels = (
        "adapters/step-25/adapter_model.safetensors",
        "adapters/step-25/adapter_config.json",
        "completions/ifeval__final__greedy.jsonl",
    )
    for rel in rels:
        prov["outputs"][rel] = sha256_file(out / rel)
    (out / "provenance-kernel.json").write_text(json.dumps(prov), encoding="utf-8")
    monkeypatch.setattr(collect_run, "ROOT", tmp_path)
    got = collect_run.collect("2026-10-31-99-probe", tmp_path / "out", "u/k", "complete", "1", upload=False)
    assert set(got["assets"]) == {
        "adapters/step-25/adapter_model.safetensors",
        "completions/ifeval__final__greedy.jsonl",
    }, "weights and IFEval answers go to the release"
    assert got["assets"]["adapters/step-25/adapter_model.safetensors"]["bytes"] == 64
    assert "adapters/step-25/adapter_config.json" in got["files"]
    assert not (
        tmp_path / "results" / "runs" / "2026-10-31-99-probe" / "adapters" / "step-25" / "adapter_model.safetensors"
    ).exists()


def test_promotion_refuses_a_file_whose_digest_does_not_match(tmp_path: Path) -> None:
    import promote_results
    from goodhart_core.hashing import sha256_file

    request = _request(tmp_path)
    run = tmp_path / "results" / "runs" / "2026-10-31-99-probe"
    run.mkdir(parents=True)
    (run / "probe.json").write_text('{"ok": true}\n', encoding="utf-8")
    prov = {
        "request": "2026-10-31-99-probe",
        "request_sha256": sha256_file(request),
        "status": "done",
        "verified": True,
        "files": {"probe.json": sha256_file(run / "probe.json")},
    }
    (run / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    assert promote_results.check_all(tmp_path) == (1, [])
    (run / "probe.json").write_text('{"ok": "edited"}\n', encoding="utf-8")
    count, problems = promote_results.check_all(tmp_path)
    assert count == 1 and problems == ["2026-10-31-99-probe: probe.json does not match its digest"]
    (run / "probe.json").write_text('{"ok": true}\n', encoding="utf-8")
    (run / "extra.json").write_text("{}\n", encoding="utf-8")
    assert promote_results.check_all(tmp_path)[1] == ["2026-10-31-99-probe: extra.json is not in the provenance"]


def test_promotion_refuses_a_request_edited_after_the_run(tmp_path: Path) -> None:
    import promote_results
    from goodhart_core.hashing import sha256_file

    request = _request(tmp_path)
    run = tmp_path / "results" / "runs" / "2026-10-31-99-probe"
    run.mkdir(parents=True)
    prov = {"request": "2026-10-31-99-probe", "request_sha256": sha256_file(request), "files": {}, "status": "done"}
    (run / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    request.write_text(request.read_text(encoding="utf-8").replace("0.5", "0.4"), encoding="utf-8")
    assert "differs from the request the kernel ran" in promote_results.check_all(tmp_path)[1][0]


def test_the_held_out_guard_refuses_the_import_inside_the_kernel_package() -> None:
    import subprocess
    import sys

    code = "from goodhart_train import guard; guard.install(); import goodhart_heldout"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode != 0 and "goodhart_heldout" in done.stderr
