"""The child process route the probe and the sampling stages share: a task runs in its own process
and hands back JSON, a failure comes back as the tail of its stderr with the root cause kept, a child
that overruns is stopped, the shards of a job are joined in prompt order, the vLLM install plans
follow the driver, and the driver facts are read from nvidia-smi."""

from __future__ import annotations

import json
import sys
import time
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


def test_shards_waited_for_in_turn_share_one_deadline(tmp_path: Path) -> None:
    """The first shard finishes near the timeout; the second must be stopped at the same deadline,
    not given the whole timeout again once the first is done."""
    t0 = time.monotonic()
    quick = isolate.start("quick", [sys.executable, "-c", "import time; time.sleep(2)"], tmp_path)
    slow = isolate.start("slow", [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path)
    first = isolate.wait(quick, timeout_s=3)
    second = isolate.wait(slow, timeout_s=3)
    assert first.returncode == 0 and second.returncode is None
    assert time.monotonic() - t0 < 4.5, "the second wait restarted the clock"


# A child that starts a process of its own and leaves without stopping it, as a vLLM worker leaves
# its engine process: the grandchild writes its pid and would sleep for a minute.
LEAVES_ONE = (
    "import subprocess, sys, pathlib\n"
    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "pathlib.Path(sys.argv[1]).write_text(str(p.pid))\n"
)


def _alive(pid: int) -> bool:
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_what_a_child_leaves_behind_is_stopped_before_the_parent_moves_on(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    running = isolate.start("leaves-one", [sys.executable, "-c", LEAVES_ONE, str(pid_file)], tmp_path)
    outcome = isolate.wait(running, timeout_s=60)
    assert outcome.returncode == 0 and outcome.left_behind == 1
    assert not _alive(int(pid_file.read_text()))
    clean = isolate.wait(isolate.start("clean", [sys.executable, "-c", "pass"], tmp_path), timeout_s=60)
    assert clean.left_behind == 0 and "left_behind" not in clean.as_record()


def test_a_server_is_stopped_with_everything_it_started(tmp_path: Path) -> None:
    pid_file = tmp_path / "engine.pid"
    serves = LEAVES_ONE + "import time; time.sleep(60)\n"
    running = isolate.start("server", [sys.executable, "-c", serves, str(pid_file)], tmp_path)
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text():
            break
        import time

        time.sleep(0.1)
    engine = int(pid_file.read_text())
    assert _alive(engine)
    assert isolate.stop(running, grace_s=5) == 0  # SIGTERM to the group ends both before the reap
    assert running.process.returncode is not None and not _alive(engine)


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


def test_a_failed_shard_leaves_its_log_in_the_run_outputs(tmp_path: Path) -> None:
    """The selection run lost the root cause of a vLLM engine that would not start: the error the
    parent kept was the frontend's traceback, and the engine's own lines were only in the shard log."""
    from goodhart_train.backend import ShardedBackend

    out = tmp_path / "out"
    backend = ShardedBackend("transformers", "/no/such/model", gpus=[None], workdir=tmp_path / "w", keep_logs_in=out)
    with pytest.raises(RuntimeError, match="shard failed"):
        backend.generate([[{"role": "user", "content": "q"}]], 1, 0.0, 4, 0)
    kept = out / "logs" / "transformers-__no__such__model-call-001-shard-0.err.txt"
    assert kept.exists() and kept.read_text(encoding="utf-8").strip()


def test_an_isolated_child_gets_no_installed_packages_and_no_toolkit_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/opt/lib")
    child = isolate.child_env(None, isolated=True)
    assert child["LD_LIBRARY_PATH"] == "/usr/local/nvidia/lib64:/opt/lib"
    assert isolate.child_env(None)["LD_LIBRARY_PATH"].startswith("/usr/local/cuda/lib64")
    assert not any(
        part in ("site-packages", "dist-packages") for p in child["PYTHONPATH"].split(":") for part in Path(p).parts
    )


def test_the_vllm_worker_does_nothing_on_import() -> None:
    """vLLM starts its engine with spawn for some models (Qwen3.5), and a spawned process imports the
    worker script again: at import it must only define, never load a model."""
    import ast

    from goodhart_train.backend import VLLM_WORKER

    body = ast.parse(VLLM_WORKER).body
    assert all(isinstance(n, ast.Import | ast.ImportFrom | ast.FunctionDef | ast.If) for n in body)
    guard = body[-1]
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "__name__ == '__main__'"


def _adapter(model: str, out: Path) -> str:
    """A LoRA adapter on the tiny model with nonzero weights, so merging it changes the model."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(model)
    peft = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", init_lora_weights=False))
    torch.manual_seed(0)
    peft.save_pretrained(str(out))
    return str(out)


@pytest.mark.torch
def test_an_adapter_is_merged_into_its_base_as_trained(tiny: dict[str, Any], tmp_path: Path) -> None:
    import torch
    from goodhart_train import worker
    from goodhart_train.models import load
    from transformers import AutoModelForCausalLM

    adapter = _adapter(tiny["model"], tmp_path / "adapter")
    got = worker.merge({"model": tiny["model"], "adapter": adapter, "out": str(tmp_path / "merged")})
    assert got["ok"] and (tmp_path / "merged" / "config.json").exists()
    merged = AutoModelForCausalLM.from_pretrained(str(tmp_path / "merged"))
    live = load(tiny["model"], adapter=adapter).model
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.allclose(merged(ids).logits, live(ids).logits, atol=1e-4)
        base = AutoModelForCausalLM.from_pretrained(tiny["model"])
        assert not torch.allclose(base(ids).logits, merged(ids).logits, atol=1e-4)


def test_vllm_samples_an_adapter_as_the_merged_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shards get the merged folder as their model and no adapter; the merge runs once, and the
    merged weights are removed when the backend is released."""
    from goodhart_train import isolate
    from goodhart_train.backend import Job, ShardedBackend

    merges: list[dict[str, Any]] = []

    def run_task(task: str, args: dict[str, Any], workdir: Path, name: str, **kw: Any) -> isolate.Outcome:
        merges.append(args)
        Path(args["out"]).mkdir(parents=True)
        (Path(args["out"]) / "config.json").write_text("{}")
        return isolate.Outcome(name, ok=True, returncode=0, seconds=0.1, result={"seconds": 0.1}, error="")

    started: list[dict[str, Any]] = []

    def start(name: str, command: list[str], workdir: Path, gpu: Any = None, *a: Any, **kw: Any) -> Any:
        started.append(json.loads(Path(command[-1]).read_text()))
        raise RuntimeError("stop here: the arguments are what this test reads")

    monkeypatch.setattr(isolate, "run_task", run_task)
    monkeypatch.setattr(isolate, "start", start)
    backend = ShardedBackend(
        "vllm", "Qwen/Qwen3-0.6B", "/kaggle/input/sft/adapters/step-50", gpus=[0, 1], workdir=tmp_path, python="py"
    )
    with pytest.raises(RuntimeError, match="stop here"):
        backend.run([Job([[{"role": "user", "content": "q"}]] * 2, 1, 0.0, 4, 0)])
    assert len(merges) == 1 and merges[0]["adapter"].endswith("step-50")
    assert started[0]["adapter"] is None and started[0]["model"] == merges[0]["out"]
    assert started[0]["chat_kwargs"] == {"enable_thinking": False}, "the template follows the base model's id"
    backend.close()
    assert not Path(merges[0]["out"]).exists()
