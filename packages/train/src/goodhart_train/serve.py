"""A vLLM server for GRPO rollouts on the second card, in TRL's server mode.

The trainer runs in the vLLM environment on the first card and asks the server for every rollout;
after each optimizer step TRL pushes the policy's weights to the server over NCCL, so the rollouts
always come from the current policy. The server takes the whole second card, which leaves the first
card's memory to the training state. The server is started with TRL's own script, which runs vLLM's
server with the weight transfer endpoints TRL needs, and is stopped when the work inside the block
is done, whatever happened there.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from goodhart_train import isolate

PORT = 8000


def command(python: str, model_id: str, hp: dict[str, Any]) -> list[str]:
    out = [python, "-m", "trl.scripts.vllm_serve", "--model", model_id, "--host", "127.0.0.1"]
    out += ["--port", str(int(hp.get("vllm_server_port", PORT)))]
    out += ["--gpu_memory_utilization", str(hp.get("vllm_server_memory", 0.85))]
    out += ["--dtype", "float16", "--max_model_len", str(int(hp.get("max_model_len", 2048)))]
    if hp.get("vllm_enforce_eager"):
        out += ["--enforce_eager", "true"]
    return out


@contextmanager
def vllm_server(
    python: str, model_id: str, hp: dict[str, Any], workdir: Path, gpu: int | None, keep_in: Path | None = None
) -> Iterator[isolate.Running]:
    server = isolate.start("vllm-server", command(python, model_id, hp), workdir, gpu, isolated=True)
    try:
        yield server
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=60)
        except Exception:
            server.process.kill()
        if keep_in is not None:
            isolate.keep_log(workdir, "vllm-server", keep_in)


def trainer_args(hp: dict[str, Any], server: bool) -> dict[str, Any]:
    """The GRPOConfig fields for vLLM rollouts: a server on another card, or colocated on this one."""
    return {
        "use_vllm": True,
        "vllm_mode": "server" if server else "colocate",
        "vllm_server_host": "127.0.0.1",
        "vllm_server_port": int(hp.get("vllm_server_port", PORT)),
        "vllm_server_timeout": float(hp.get("vllm_server_timeout", 900)),
        "vllm_gpu_memory_utilization": float(hp.get("vllm_gpu_memory_utilization", 0.3)),
    }
