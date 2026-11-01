"""The probe's parts that need a GPU on Kaggle, run here with the GPU calls replaced: the worker's
tasks in this process, the vLLM install plans and their record, the lane that tries each plan and
retries without CUDA graphs only when the error points there, the smoke run's vLLM server, and the
step clock."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
from goodhart_train import isolate, probe, vllm_env, worker


def _done(code: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["uv"], code, stdout=stdout, stderr=stderr)


def test_the_worker_writes_its_result_and_refuses_an_unknown_task(tmp_path: Path) -> None:
    args, result = tmp_path / "args.json", tmp_path / "result.json"
    args.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert worker.main(["echo", str(args), str(result)]) == 0
    assert json.loads(result.read_text())["echo"] == {"x": 1}
    assert worker.main(["no-such-task", str(args), str(result)]) == 2
    assert worker.main(["echo"]) == 2
    with pytest.raises(RuntimeError, match="asked to"):
        worker.echo({"fail": "asked to"})


def test_an_install_records_success_failure_and_a_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm_env, "VENV", tmp_path / "venv")
    monkeypatch.setattr(vllm_env, "_bootstrap", lambda: None)
    calls: list[tuple[str, ...]] = []

    def uv_ok(*args: str, timeout_s: float = 0) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _done(0)

    monkeypatch.setattr(vllm_env, "_uv", uv_ok)
    plan = {"source": "github-cu129", "requirement": "https://example.org/vllm.whl", "torch_backend": "cu129"}
    found = vllm_env.install(plan, extra=["trl==1.17.1"])
    assert found["installed"] and found["torch_backend"] == "cu129" and found["extra"] == ["trl==1.17.1"]
    assert calls[-1][-2:] == ("--torch-backend", "cu129") and "trl==1.17.1" in calls[-1]

    def uv_fail(*args: str, timeout_s: float = 0) -> subprocess.CompletedProcess[str]:
        return _done(1, stderr="resolution failed\n  because torch==2.13.0 has no cu129 wheel")

    monkeypatch.setattr(vllm_env, "_uv", uv_fail)
    failed = vllm_env.install({**plan, "torch_backend": None})
    assert not failed["installed"] and "no cu129 wheel" in failed["error"]

    def uv_slow(*args: str, timeout_s: float = 0) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["uv"], timeout_s)

    monkeypatch.setattr(vllm_env, "_uv", uv_slow)
    slow = vllm_env.install(plan, timeout_s=5)
    assert not slow["installed"] and "ran past 5 s" in slow["error"]


def test_the_environment_is_described_and_ensured_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    monkeypatch.setattr(vllm_env, "VENV", venv)
    monkeypatch.setattr(vllm_env, "_bootstrap", lambda: None)
    monkeypatch.setattr(vllm_env, "_uv", lambda *a, timeout_s=0: _done(0, stdout="vllm==0.32.0\ntorch==2.13.0\n"))
    monkeypatch.setattr(vllm_env, "_torch", lambda: {"torch": "2.13.0+cu129", "cuda": "12.9"})
    described = vllm_env.describe()
    assert described["freeze"] == ["vllm==0.32.0", "torch==2.13.0"] and described["cuda"] == "12.9"
    monkeypatch.setattr("goodhart_train.env.cuda_driver_version", lambda: (12, 8))
    tried: list[str] = []
    plans = [{"source": "a", "requirement": "x", "torch_backend": None}, {"source": "b", "requirement": "y"}]
    monkeypatch.setattr(vllm_env, "plans", lambda version, cuda: plans)

    def install(plan: dict[str, Any], extra: Any = (), timeout_s: float = 0) -> dict[str, Any]:
        tried.append(plan["source"])
        return {"source": plan["source"], "installed": plan["source"] == "b", "error": "no"}

    monkeypatch.setattr(vllm_env, "install", install)
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    assert vllm_env.ensure("0.32.0") == str(venv / "bin" / "python") and tried == ["a", "b"]
    assert vllm_env.ensure("0.32.0") == str(venv / "bin" / "python") and tried == ["a", "b"], "the marker is kept"
    (venv / "goodhart-installed.json").unlink()
    monkeypatch.setattr(vllm_env, "install", lambda plan, extra=(), timeout_s=0: {"installed": False, "error": "e"})
    with pytest.raises(RuntimeError, match="did not install"):
        vllm_env.ensure("0.32.0")


def test_the_release_is_read_from_the_github_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"assets": [{"name": "vllm-1.0+cu129-x_x86_64.whl", "browser_download_url": "u"}]}
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=0: Response())
    assert vllm_env.cuda12_wheel("1.0") == "u"

    def broken(request: Any, timeout: float = 0) -> Any:
        raise OSError("no network")

    monkeypatch.setattr("urllib.request.urlopen", broken)
    assert vllm_env.cuda12_wheel("1.0") is None


def _lane(monkeypatch: pytest.MonkeyPatch, generation: Any, installs: list[bool]) -> None:
    plans = [{"source": f"p{i}", "requirement": "r", "torch_backend": None} for i in range(len(installs))]
    monkeypatch.setattr(vllm_env, "plans", lambda version, cuda: plans)
    results = iter(installs)

    def install(plan: dict[str, Any], extra: Any = (), timeout_s: float = 0) -> dict[str, Any]:
        if extra:
            return {"source": plan["source"], "installed": True, "extra": list(extra)}
        return {"source": plan["source"], "installed": next(results)}

    monkeypatch.setattr(vllm_env, "install", install)
    monkeypatch.setattr(vllm_env, "describe", lambda: {"python": "p", "torch": "2.13.0", "freeze": ["vllm==0.32.0"]})
    monkeypatch.setattr(vllm_env, "pins", lambda path, names: ["trl==1.17.1"])
    monkeypatch.setattr("goodhart_train.env.cuda_driver_version", lambda: (12, 8))
    monkeypatch.setattr(probe, "vllm_generation", generation)


class _Bundle:
    root = Path("/nonexistent")


def test_the_lane_retries_without_graphs_only_for_a_compile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def generation(python: str, model: str, bundle: Any, hp: Any, workdir: Path, gpu: Any, eager: bool) -> Any:
        seen.append(eager)
        if not eager:
            raise RuntimeError("torch._inductor failed: triton compile error")
        return {"works": True, "enforce_eager": True, "tokens_per_second": 900.0}

    _lane(monkeypatch, generation, [False, True])
    record = probe.vllm_lane(_Bundle(), {}, tmp_path, tmp_path / "out", 1, "m")  # type: ignore[arg-type]
    assert record["works"] and record["plan"] == "p1" and record["enforce_eager"] is True and seen == [False, True]
    assert record["training_stack"]["installed"] and record["freeze"] == ["vllm==0.32.0"]
    assert record["attempts"][0]["install"]["installed"] is False


def test_the_lane_moves_on_when_the_error_is_not_about_compilation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def generation(python: str, model: str, bundle: Any, hp: Any, workdir: Path, gpu: Any, eager: bool) -> Any:
        seen.append(eager)
        raise RuntimeError("CUDA driver version is insufficient for CUDA runtime version")

    _lane(monkeypatch, generation, [True, True])
    record = probe.vllm_lane(_Bundle(), {}, tmp_path, tmp_path / "out", 1, "m")  # type: ignore[arg-type]
    assert record["works"] is False and seen == [False, False] and len(record["attempts"]) == 2
    assert "insufficient" in record["attempts"][0]["generation"]["error"]
    timed = probe.vllm_lane(_Bundle(), {}, tmp_path, tmp_path / "out", 1, "m", budget_s=0)  # type: ignore[arg-type]
    assert all("skipped" in a for a in timed["attempts"])


def test_the_server_smoke_starts_and_stops_the_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[list[str]] = []
    real_start = isolate.start

    def start(
        name: str, command: Any, workdir: Path, gpu: Any = None, extra_env: Any = None, isolated: bool = False
    ) -> Any:
        started.append(list(command))
        return real_start(name, [sys.executable, "-c", "import time; time.sleep(30)"], workdir, gpu, isolated=isolated)

    def run_task(task: str, args: dict[str, Any], workdir: Path, name: str, **kw: Any) -> isolate.Outcome:
        assert kw["python"] == str(vllm_env.python()) and kw["gpu"] == 0 and args["backend"] == "vllm-server"
        return isolate.Outcome(name, True, 0, 1.0, {"ok": True, "steps_per_hour": 300.0}, "")

    monkeypatch.setattr(isolate, "start", start)
    monkeypatch.setattr(isolate, "run_task", run_task)
    hp = {"vllm_enforce_eager": True, "max_model_len": 2048}
    record = probe._smoke("vllm-server", _Bundle(), tmp_path, tmp_path, tmp_path, "m", hp, [0, 1], 60.0)  # type: ignore[arg-type]
    assert record["steps_per_hour"] == 300.0
    assert started and "trl.scripts.vllm_serve" in started[0] and started[0][-2:] == ["--enforce_eager", "true"]


def test_the_step_clock_leaves_out_the_first_step() -> None:
    clock = probe.StepClock()
    assert clock.steady_seconds_per_step() is None
    clock.marks.extend([10.0, 12.0, 14.0, 16.0])
    assert clock.steady_seconds_per_step() == 2.0


@pytest.mark.torch
def test_the_generate_and_candidate_tasks_run_in_process(
    tiny: dict[str, Any], tmp_path: Path, bundle_factory: Any
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"messages": [{"role": "user", "content": "question 1"}]}) + "\n", encoding="utf-8")
    job = {"prompts": str(prompts), "n": 2, "temperature": 0.8, "max_tokens": 3, "seed": 0, "out": str(tmp_path / "o")}
    stats = worker.generate({"model": tiny["model"], "batch_size": 4, "jobs": [job]})
    assert stats["ok"] and stats["jobs"][0]["prompts"] == 1
    assert len(json.loads((tmp_path / "o").read_text())["texts"]) == 2
    req = {"id": "test-probe", "stage": "probe", "seed": 0, "gpu_hour_ceiling": 1.0, "models": []}
    folder = bundle_factory(tiny, tmp_path / "b", req)
    hp = {"max_completion_length": 4, "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.0}
    found = worker.candidate({"bundle": str(folder), "model": tiny["model"], "hp": hp})
    assert found["generates"] and found["phase_seconds"]["generation"] > 0


def test_grpo_with_vllm_runs_in_the_vllm_environment_beside_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    from goodhart_train import grpo
    from goodhart_train.ledger import Meter

    class _Req:
        model = "m/base"
        hyperparameters: ClassVar[dict[str, str]] = {"generation_backend": "vllm"}

    class _B:
        root = tmp_path
        inputs: ClassVar[list[str]] = ["x"]
        request = _Req()

        def config(self, name: str) -> dict[str, Any]:
            return {"grpo": {"vllm_mode": "server", "vllm_version": "0.32.0"}, "lora": {}}

    (tmp_path / "requirements-gpu.txt").write_text("trl==1.17.1\n", encoding="utf-8")
    monkeypatch.setattr(vllm_env, "ensure", lambda version, extra=(): "/venv/bin/python")
    monkeypatch.setattr("goodhart_train.env.gpu_indices", lambda: [0, 1])
    served: list[tuple[str, Any]] = []

    @contextmanager
    def server(python: str, model: str, hp: Any, work: Path, gpu: Any, keep_in: Any = None) -> Any:
        served.append((model, gpu))
        yield None

    def run_task(task: str, args: dict[str, Any], workdir: Path, name: str, **kw: Any) -> isolate.Outcome:
        assert task == "stage" and kw["python"] == "/venv/bin/python" and kw["gpu"] == 0
        assert args["overrides"] == {"vllm_mode": "server"} and args["inputs"] == ["x"]
        result = {"summary": {"global_step": 3}, "seconds": {"training": 5.0}, "counts": {"steps": 3}}
        return isolate.Outcome(name, True, 0, 1.0, result, "")

    monkeypatch.setattr("goodhart_train.serve.vllm_server", server)
    monkeypatch.setattr(isolate, "run_task", run_task)
    meter = Meter(ceiling_hours=1.0)
    out = grpo.run_isolated(_B(), tmp_path / "out", meter, tmp_path / "cache")  # type: ignore[arg-type]
    assert out["global_step"] == 3 and out["rollouts_from"].startswith("a vLLM server")
    assert served == [("m/base", 1)] and meter.counts["steps"] == 3 and meter.seconds["training"] == 5.0
    monkeypatch.setattr("goodhart_train.env.gpu_indices", lambda: [0])
    fail = isolate.Outcome("grpo", False, 1, 1.0, {}, "the trainer could not reach the server")
    monkeypatch.setattr(isolate, "run_task", lambda *a, **k: fail)
    with pytest.raises(RuntimeError, match="could not reach the server"):
        grpo.run_isolated(_B(), tmp_path / "out", meter, tmp_path / "cache")  # type: ignore[arg-type]


def test_a_checkpoint_is_sampled_from_the_rollout_server_after_a_weight_push(tmp_path: Path) -> None:
    from goodhart_train.bundle import Prompt
    from goodhart_train.ledger import Meter
    from goodhart_train.trainer_common import DevExport, ServerSampler

    pushed: list[int] = []

    class _Client:
        def generate(self, prompts: list[str], n: int, temperature: float, max_tokens: int) -> dict[str, Any]:
            assert n == 1 and temperature == 0.0 and pushed
            return {"completion_ids": [[1, 2, 3] for _ in prompts]}

    class _Generation:
        vllm_client = _Client()

        def sync_weights(self) -> None:
            pushed.append(1)

    class _Trainer:
        vllm_generation = _Generation()

    class _Tok:
        def apply_chat_template(self, messages: Any, tokenize: bool, add_generation_prompt: bool, **kw: Any) -> str:
            return "prompt"

        def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
            return "```sql\nselect 1\n```"

    prompts = [Prompt(id=f"d{i}", set="dev", messages=[{"role": "user", "content": "q"}]) for i in range(3)]
    export = DevExport(_Tok(), "m", prompts, tmp_path, "req", Meter(ceiling_hours=1.0), 8)
    export.server = ServerSampler(_Trainer(), _Tok(), "m")
    export(25, None, None)
    rows = [json.loads(line) for line in (tmp_path / "completions" / "dev__step-25.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["tokens"] == 3 and rows[0]["checkpoint"] == "step-25" and pushed == [1]
    assert export.finish() == {"dev_exports": [25], "dev_export_errors": []}


@pytest.mark.torch
def test_the_stage_task_runs_grpo_in_the_child(tiny: dict[str, Any], tmp_path: Path, bundle_factory: Any) -> None:
    req = {
        "id": "test-grpo",
        "stage": "grpo",
        "model": tiny["model"],
        "reward": "honest",
        "seed": 0,
        "step_budget": 1,
        "gpu_hour_ceiling": 1.0,
        "hyperparameters": {
            "num_generations": 2,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "generation_batch_size": 2,
            "max_completion_length": 6,
            "save_steps": 5,
            "gradient_checkpointing": False,
        },
    }
    folder = bundle_factory(tiny, tmp_path / "b", req)
    (folder / "data" / "sources.lock.json").write_text(json.dumps(tiny["lock"]), encoding="utf-8")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "warehouses").symlink_to(tiny["base"] / "warehouses")
    args = {"stage": "grpo", "bundle": str(folder), "out": str(tmp_path / "out"), "cache": str(tmp_path / "cache")}
    found = worker.stage({**args, "ceiling_hours": 1.0, "overrides": {"vllm_mode": "colocate"}})
    assert found["summary"]["global_step"] == 1 and found["counts"]["steps"] == 1
    with pytest.raises(ValueError, match="no isolated stage"):
        worker.stage({**args, "stage": "judge", "ceiling_hours": 1.0})


def test_a_candidate_vllm_cannot_serve_is_sampled_with_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from goodhart_train import stages
    from goodhart_train.ledger import Meter

    class _Req:
        id = "sel"
        seed = 0
        models: ClassVar[list[str]] = ["a/hybrid", "b/broken"]
        hyperparameters: ClassVar[dict[str, str]] = {"generation_backend": "vllm"}

    class _B:
        request = _Req()

        def config(self, name: str) -> dict[str, Any]:
            return {"select": {"k": 8}}

    def one(bundle: Any, out: Path, meter: Any, model_id: str, hp: dict[str, Any]) -> dict[str, Any]:
        if hp["generation_backend"] == "vllm" or model_id == "b/broken":
            raise RuntimeError(f"{model_id} on {hp['generation_backend']}: no kernel image for this card")
        return {"ok": True, "backend": "transformers"}

    monkeypatch.setattr(stages, "_select_one", one)
    out = stages.select(_B(), tmp_path, Meter(ceiling_hours=1.0))  # type: ignore[arg-type]
    hybrid, broken = out["models"]["a/hybrid"], out["models"]["b/broken"]
    assert hybrid["ok"] and hybrid["backend"] == "transformers" and "vllm" in hybrid["vllm_error"]
    assert not broken["ok"] and "transformers" in broken["error"] and "vllm" in broken["vllm_error"]
