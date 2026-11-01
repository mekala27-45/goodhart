"""The child process route the probe and the sampling stages share: a task runs in its own process
and hands back JSON, a failure comes back as the tail of its stderr with the root cause kept, a child
that overruns is stopped, the shards of a job are joined in prompt order, the vLLM install plans
follow the driver, and the driver facts are read from nvidia-smi."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from goodhart_train import env, isolate, vllm_env
from goodhart_train.backend import split


def test_a_task_runs_in_a_child_that_refuses_the_held_out_package(tmp_path: Path) -> None:
    outcome = isolate.run_task("echo", {"value": 3}, tmp_path, "echo-ok", timeout_s=120)
    assert outcome.ok and outcome.returncode == 0
    assert outcome.result["echo"] == {"value": 3} and outcome.result["heldout_refused"] is True


def test_a_failing_task_returns_the_tail_of_its_stderr(tmp_path: Path) -> None:
    outcome = isolate.run_task("echo", {"fail": "the root cause"}, tmp_path, "echo-fail", timeout_s=120)
    assert not outcome.ok and outcome.returncode not in (0, None)
    assert "the root cause" in outcome.error
    record = outcome.as_record()
    assert record["ok"] is False and "the root cause" in record["error"]
    isolate.keep_log(tmp_path, "echo-fail", tmp_path / "out")
    assert "the root cause" in (tmp_path / "out" / "logs" / "echo-fail.err.txt").read_text()


def test_a_child_that_overruns_is_stopped(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    running = isolate.start("sleeper", command, tmp_path)
    outcome = isolate.wait(running, timeout_s=2)
    assert not outcome.ok and outcome.returncode is None and outcome.error.startswith("stopped after")


def test_the_tail_keeps_an_earlier_error_line_above_a_long_trace() -> None:
    text = "\n".join(["RuntimeError: the driver is too old", *[f"  frame {i}" for i in range(100)]])
    kept = isolate.tail(text, lines=10)
    assert kept.splitlines()[0] == "RuntimeError: the driver is too old" and "frame 99" in kept


def test_split_gives_contiguous_near_equal_shards() -> None:
    assert split(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert split(1, 2) == [(0, 1), (1, 1)] and split(0, 1) == [(0, 0)]


def test_the_child_path_puts_the_parent_import_path_first() -> None:
    env_vars = isolate.child_env(1, {"X": "y"})
    assert env_vars["CUDA_VISIBLE_DEVICES"] == "1" and env_vars["X"] == "y"
    assert env_vars["PYTHONPATH"].split(":")[0] in sys.path


def _release(names: list[str]) -> dict[str, Any]:
    return {"assets": [{"name": n, "browser_download_url": f"https://example.org/{n}"} for n in names]}


def test_the_vllm_plans_follow_the_driver() -> None:
    wheel = "vllm-0.32.0+cu129-cp38-abi3-manylinux_2_31_x86_64.whl"
    release = _release(["vllm-0.32.0+cu129-cp38-abi3-manylinux_2_31_aarch64.whl", wheel])
    assert [p["source"] for p in vllm_env.plans("0.32.0", (13, 0), lambda _: release)] == ["pypi", "github-cu129"]
    assert [p["source"] for p in vllm_env.plans("0.32.0", (13, 0), lambda _: None)] == ["pypi"]
    older = vllm_env.plans("0.32.0", (12, 8), lambda _: release)
    assert [p["source"] for p in older] == ["github-cu129", "github-cu129-auto", "pypi-auto"]
    assert older[0]["requirement"].endswith(wheel) and older[0]["torch_backend"] == "cu129"
    assert [p["source"] for p in vllm_env.plans("0.32.0", (12, 8), lambda _: None)] == ["pypi-auto"]


def test_the_training_stack_is_pinned_from_the_gpu_requirements(tmp_path: Path) -> None:
    path = tmp_path / "requirements-gpu.txt"
    path.write_text(
        "trl==1.17.1 \\\n    --hash=sha256:aa\npeft==0.21.2 \\\n    --hash=sha256:bb\n"
        "colorama==0.4.6 ; sys_platform == 'win32' \\\n    --hash=sha256:cc\nPyYAML==6.0.3\n",
        encoding="utf-8",
    )
    assert vllm_env.pins(path, ("trl", "peft", "pyyaml", "duckdb")) == ["PyYAML==6.0.3", "peft==0.21.2", "trl==1.17.1"]


def test_the_driver_facts_come_from_nvidia_smi() -> None:
    banner = "| NVIDIA-SMI 570.172.08   Driver Version: 570.172.08   CUDA Version: 12.8 |"
    rows = "0, 570.172.08, 1590, 585, 70.00, 41\n1, 570.172.08, 1590, 585, 70.00, 39\n"
    found = env.parse_smi(banner, rows)
    assert found["driver_version"] == "570.172.08" and found["cuda_driver"] == "12.8"
    assert [c["index"] for c in found["cards"]] == ["0", "1"] and found["cards"][0]["power.limit"] == "70.00"
    assert env.parse_smi("", "") == {"driver_version": None, "cuda_driver": None, "cards": []}


def test_the_visible_cards_follow_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert env.gpu_indices() == [1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert env.gpu_indices() == []


@pytest.mark.torch
def test_two_shards_join_in_prompt_order(tiny: dict[str, Any], tmp_path: Path) -> None:
    from goodhart_train.backend import Job, ShardedBackend

    backend = ShardedBackend("transformers", tiny["model"], gpus=[None, None], workdir=tmp_path, batch_size=8)
    prompts = [[{"role": "user", "content": f"question {i}"}] for i in range(5)]
    greedy, sampled = backend.run([Job(prompts, 1, 0.0, 4, 0), Job(prompts, 2, 0.8, 4, 0)])
    assert len(greedy) == 5 and all(len(s.texts) == 1 for s in greedy)
    assert len(sampled) == 5 and all(len(s.texts) == 2 for s in sampled)
    assert [len(s["jobs"]) for s in backend.stats[0]["shards"]] == [2, 2]
    shard_prompts = [
        [json.loads(line) for line in (tmp_path / "call-001" / f"shard-{s}" / "job-0.prompts.jsonl").open()]
        for s in (0, 1)
    ]
    assert [r["messages"][0]["content"] for part in shard_prompts for r in part] == [f"question {i}" for i in range(5)]
    with pytest.raises(RuntimeError, match="shard failed"):
        ShardedBackend("transformers", "/no/such/model", gpus=[None], workdir=tmp_path / "bad").generate(
            prompts, 1, 0.0, 4, 0
        )


def test_an_isolated_child_gets_no_installed_packages_and_no_toolkit_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/opt/lib")
    child = isolate.child_env(None, isolated=True)
    assert child["LD_LIBRARY_PATH"] == "/usr/local/nvidia/lib64:/opt/lib"
    assert isolate.child_env(None)["LD_LIBRARY_PATH"].startswith("/usr/local/cuda/lib64")
    assert not any(
        part in ("site-packages", "dist-packages") for p in child["PYTHONPATH"].split(":") for part in Path(p).parts
    )
