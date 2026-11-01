"""A child process's entry: one task, its arguments from a JSON file, its result to another.

    python -m goodhart_train.worker <task> <args.json> <result.json>

Its main refuses the held-out package first, like every stage entry point, then runs the task and
writes its result. The refusal is installed by main and not on import, so a test that imports this
module does not change what the rest of its process may import. A task that raises leaves no result
file and exits non zero; the parent reads the tail of its stderr (goodhart_train.isolate). Tasks:

    candidate    the probe of one candidate model (goodhart_train.probe.probe_candidate)
    trainer_step the SFT trainer's own LoRA steps on one candidate (goodhart_train.probe.trainer_step)
    grpo_smoke   a short GRPO run with the honest reward (goodhart_train.probe.grpo_smoke)
    generate     transformers sampling of one shard of prompts, for ShardedBackend
    merge        an adapter merged into its base and saved, so vLLM samples it with no LoRA kernel
    stage        a whole trainer stage: GRPO in the vLLM environment, or SFT, DPO or GRPO on one card
    echo         returns its arguments, for the tests of the isolation itself
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from goodhart_train import guard


def _bundle(args: dict[str, Any]) -> Any:
    from goodhart_train.bundle import Bundle

    bundle = Bundle(Path(args["bundle"]))
    bundle.inputs = [str(p) for p in args.get("inputs", [])]
    return bundle


def candidate(args: dict[str, Any]) -> dict[str, Any]:
    from goodhart_train.ledger import Meter
    from goodhart_train.probe import probe_candidate

    meter = Meter(ceiling_hours=float(args.get("ceiling_hours", 1.0)))
    found = probe_candidate(str(args["model"]), _bundle(args), dict(args["hp"]), meter)
    return {**found, "phase_seconds": {k: round(v, 3) for k, v in meter.seconds.items() if v > 0}}


def trainer_step(args: dict[str, Any]) -> dict[str, Any]:
    from goodhart_train.probe import trainer_step as step

    return step(str(args["model"]), _bundle(args), dict(args["sft"]), dict(args["lora"]), int(args.get("steps", 3)))


def grpo_smoke(args: dict[str, Any]) -> dict[str, Any]:
    from goodhart_core.fetch import load_lock
    from goodhart_core.warehouse import Warehouses

    from goodhart_train.probe import grpo_smoke as smoke

    bundle = _bundle(args)
    lock = load_lock(bundle.root / "data" / "sources.lock.json")
    warehouses = Warehouses(lock, Path(args["cache"]) / "warehouses")
    found = smoke(
        bundle,
        Path(args["out"]),
        str(args["model"]),
        dict(args["hp"]),
        warehouses,
        backend=str(args.get("backend", "transformers")),
    )
    return {"ok": True, **found}


def stage(args: dict[str, Any]) -> dict[str, Any]:
    """A whole trainer stage in this child: GRPO with vLLM rollouts, which must run in the vLLM
    environment, or any trainer stage pinned to one card (goodhart_train.trainer_common.run_on_one_card).
    The parent merges the meter it returns."""
    from goodhart_train.ledger import Meter

    name = str(args["stage"])
    if name not in {"sft", "dpo", "grpo"}:
        raise ValueError(f"no isolated stage named {name!r}")
    bundle = _bundle(args)
    meter = Meter(ceiling_hours=float(args["ceiling_hours"]))
    out = Path(args["out"])
    if name == "grpo":
        from goodhart_core.fetch import load_lock
        from goodhart_core.warehouse import Warehouses

        from goodhart_train import grpo

        lock = load_lock(bundle.root / "data" / "sources.lock.json")
        warehouses = Warehouses(lock, Path(args["cache"]) / "warehouses")
        summary = grpo.run(bundle, out, meter, warehouses, overrides=args.get("overrides"))
    elif name == "sft":
        from goodhart_train import sft

        summary = sft.run(bundle, out, meter)
    else:
        from goodhart_train import dpo

        summary = dpo.run(bundle, out, meter)
    return {"ok": True, "summary": summary, "seconds": dict(meter.seconds), "counts": dict(meter.counts)}


def merge(args: dict[str, Any]) -> dict[str, Any]:
    """The adapter merged into the base weights in float16 and saved with the tokenizer: the same
    weights TRL pushes to its vLLM server during GRPO, so a checkpoint is sampled as it was trained."""
    from goodhart_train.models import load

    t0 = time.monotonic()
    loaded = load(str(args["model"]), adapter=str(args["adapter"]), merge=True)
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    loaded.model.save_pretrained(str(out), safe_serialization=True)
    loaded.tokenizer.save_pretrained(str(out))
    return {"ok": True, "out": str(out), "seconds": round(time.monotonic() - t0, 1), "parameters": loaded.parameters}


def generate(args: dict[str, Any]) -> dict[str, Any]:
    from goodhart_train.backend import HFBackend
    from goodhart_train.ledger import peak_memory
    from goodhart_train.models import load

    t0 = time.monotonic()
    adapter = args.get("adapter")
    loaded = load(str(args["model"]), adapter=adapter, merge=bool(adapter))
    loaded.model.eval()
    backend = HFBackend(loaded.model, loaded.tokenizer, str(args["model"]), batch_size=int(args.get("batch_size", 64)))
    stats: dict[str, Any] = {
        "ok": True,
        "backend": "transformers",
        "load_seconds": round(time.monotonic() - t0, 1),
        "jobs": [],
    }
    for job in args["jobs"]:
        lines = Path(job["prompts"]).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        t1 = time.monotonic()
        got = (
            backend.generate(
                [r["messages"] for r in rows],
                n=int(job["n"]),
                temperature=float(job["temperature"]),
                max_new_tokens=int(job["max_tokens"]),
                seed=int(job["seed"]),
            )
            if rows
            else []
        )
        with Path(job["out"]).open("w", encoding="utf-8") as handle:
            for s in got:
                handle.write(json.dumps({"texts": s.texts, "tokens": s.tokens}) + "\n")
        stats["jobs"].append(
            {
                "prompts": len(rows),
                "seconds": round(time.monotonic() - t1, 2),
                "tokens": sum(sum(s.tokens) for s in got),
                "prompts_per_call": backend.prompts_per_call,
            }
        )
    stats["peak_memory_bytes"] = peak_memory()
    return stats


def echo(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("fail"):
        raise RuntimeError(str(args["fail"]))
    return {"ok": True, "echo": args, "heldout_refused": _refused()}


def _refused() -> bool:
    try:
        __import__("goodhart_heldout")
    except ImportError:
        return True
    return False


TASKS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "candidate": candidate,
    "trainer_step": trainer_step,
    "grpo_smoke": grpo_smoke,
    "generate": generate,
    "merge": merge,
    "stage": stage,
    "echo": echo,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] not in TASKS:
        print(f"usage: python -m goodhart_train.worker {{{'|'.join(TASKS)}}} ARGS.json RESULT.json", file=sys.stderr)
        return 2
    guard.install()
    task, args_path, result_path = argv
    args = json.loads(Path(args_path).read_text(encoding="utf-8"))
    result = TASKS[task](args)
    Path(result_path).write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
