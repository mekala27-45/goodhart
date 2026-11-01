"""The kernel entry point: one request, one bundle, one output folder.

    python -m goodhart_train.entry --bundle /kaggle/working/bundle --out /kaggle/working/out

It refuses the held-out package before anything else is imported, fetches the warehouses by lock when
the stage executes SQL, runs the stage, and writes the ledger row and the kernel provenance (the
environment, the request digest, the digest of every output file). A stage that fails still writes
its ledger row with the status failed, so a stopped run is visible in the ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from goodhart_train import guard

guard.install()

from goodhart_core.fetch import fetch, load_lock  # noqa: E402
from goodhart_core.hashing import sha256_file  # noqa: E402
from goodhart_core.warehouse import Warehouses  # noqa: E402

from goodhart_train import env  # noqa: E402
from goodhart_train.bundle import Bundle  # noqa: E402
from goodhart_train.ledger import Meter, append, peak_memory  # noqa: E402

SQL_STAGES = frozenset({"probe", "grpo"})


def _warehouses(bundle: Bundle, cache: Path) -> Warehouses:
    lock = load_lock(bundle.root / "data" / "sources.lock.json")
    fetch(lock, cache, warehouses_only=True)
    return Warehouses(lock, cache / "warehouses")


def resolve_inputs(ids: list[str], root: Path) -> list[str]:
    """An input is an earlier request: in a kernel its output is mounted under the input root."""
    out: list[str] = []
    for ident in ids:
        if Path(ident).exists():
            out.append(ident)
            continue
        slug = "goodhart-" + "".join(ch if ch.isalnum() else "-" for ch in ident.lower()).strip("-")
        found = sorted(p for p in root.rglob(slug) if p.is_dir()) if root.exists() else []
        if not found:
            raise FileNotFoundError(f"input {ident} is not mounted under {root}")
        inner = found[0] / "out"
        out.append(str(inner if inner.exists() else found[0]))
    return out


def run(bundle_dir: Path, out: Path, cache: Path) -> int:
    import os

    bundle = Bundle(bundle_dir)
    req = bundle.request
    bundle.inputs = resolve_inputs(req.inputs, Path(os.environ.get("GOODHART_INPUT_ROOT", "/kaggle/input")))
    out.mkdir(parents=True, exist_ok=True)
    meter = Meter(ceiling_hours=req.gpu_hour_ceiling)
    status, summary, note = "done", {}, ""
    try:
        warehouses = _warehouses(bundle, cache) if req.stage in SQL_STAGES else None
        summary = _dispatch(req.stage, bundle, out, meter, warehouses, cache)
    except Exception:
        status = "failed"
        note = traceback.format_exc()[-2000:]
        print(note, file=sys.stderr)
    stopped = bool(summary.get("stopped_at_ceiling"))
    outputs = {
        str(p.relative_to(out)): sha256_file(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and "trainer" not in p.parts
    }
    row = meter.row(
        request=req.id,
        request_digest=req.digest(),
        model=req.model,
        stage=req.stage,
        arm=req.arm,
        status=status,
        peak_memory_bytes=peak_memory(),
        accelerator=", ".join(g["name"] for g in env.gpus()) or "cpu",
        stopped_at_ceiling=stopped,
        digests_in={"request": req.digest(), **req.data},
        digests_out={k: v for k, v in outputs.items() if k.startswith("completions/") or k.startswith("adapter/")},
        note=note[:500],
    )
    append(out / "ledger.jsonl", row)
    provenance = {
        "request": req.id,
        "request_digest": req.digest(),
        "stage": req.stage,
        "status": status,
        "summary": summary,
        "environment": env.describe(),
        "outputs": outputs,
    }
    (out / "provenance-kernel.json").write_text(json.dumps(provenance, indent=1, default=str) + "\n", encoding="utf-8")
    return 0 if status == "done" else 1


def _dispatch(
    stage: str, bundle: Bundle, out: Path, meter: Meter, warehouses: Warehouses | None, cache: Path
) -> dict[str, Any]:
    if stage == "probe":
        from goodhart_train import probe

        assert warehouses is not None
        return probe.run(bundle, out, meter, warehouses, cache=cache)
    if stage in {"sft", "dpo"}:
        from goodhart_train.trainer_common import run_on_one_card

        return run_on_one_card(stage, bundle, out, meter, cache)
    if stage == "grpo":
        from goodhart_train import grpo
        from goodhart_train.trainer_common import run_on_one_card

        assert warehouses is not None
        if grpo.hyperparameters(bundle).get("generation_backend") == "vllm":
            return grpo.run_isolated(bundle, out, meter, cache)
        return run_on_one_card("grpo", bundle, out, meter, cache)
    from goodhart_train import stages

    if stage == "select":
        return stages.select(bundle, out, meter)
    if stage == "sample":
        return stages.sample(bundle, out, meter)
    if stage == "pairs":
        return stages.pairs(bundle, out, meter)
    if stage == "judge":
        return stages.judge(bundle, out, meter)
    raise ValueError(f"no stage named {stage!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/kaggle/working/cache"))
    args = parser.parse_args()
    return run(args.bundle, args.out, args.cache)


if __name__ == "__main__":
    sys.exit(main())
