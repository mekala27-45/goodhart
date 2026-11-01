"""Reset and rederive: everything the repository computes, again, from the locks and the exported
completions alone, in a fresh worktree with an empty cache, compared with what is committed; and one
GRPO run requested again from a copy of its own request, compared within the seed spread.

    python scripts/reset_and_rederive.py                       # the worktree rederive
    python scripts/reset_and_rederive.py --grpo ORIGINAL RERUN # and the GRPO pair, both promoted

1. A worktree at HEAD with its own empty cache, so nothing this machine fetched or built is reused.
2. From the locks: the warehouses and the readback files (scripts/fetch_sources.py), the kept twins
   (goodhart twins), then the questions, the twins and the mutation score rebuilt under another hash
   seed and compared byte for byte (goodhart rebuild-check).
3. From the exported completions: every grade again (goodhart grade --check); IFEval and Spider
   graded again by their own checkers, each fetched by digest; then every derived result written
   again in the worktree and compared with the committed file as data.
4. The GRPO pair: the rerun's twin confirmed accuracy on dev at the original's last checkpoint
   against the original's, within the seed spread: the measured spread of the method's seeds where
   there are seeds, otherwise the registered bound of the promotion gate.

results/rederive.json records each step and each comparison; checklist line eighteen reads it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rederive.json"
SEED_BOUND = 0.05  # the promotion gate's seed spread bound (evaluation/plan.yaml)

# Each derived result, the command that writes it, and whether a missing committed file is expected.
DERIVED: tuple[tuple[str, list[str]], ...] = (
    ("results/checkpoints.json", ["checkpoints"]),
    ("results/methods.json", ["methods"]),
    ("results/gates.json", ["gates"]),
    ("results/hypotheses.json", ["hypotheses"]),
    ("results/ifeval.json", ["ifeval"]),
    ("results/spider.json", ["spider"]),
    ("results/regression.json", ["regression"]),
    ("results/interior.json", ["interior"]),
    ("results/sizing.json", ["sizing"]),
    ("results/judge.json", ["judge"]),
)


def _env(work: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(GOODHART_ROOT=str(work), GOODHART_CACHE=str(work / "data" / "cache"), PYTHONHASHSEED="0")
    env.pop("UV_NATIVE_TLS", None)
    return env


def _step(work: Path, name: str, command: list[str]) -> dict[str, Any]:
    t0 = time.monotonic()
    done = subprocess.run(
        ["uv", "run", "--no-sync", "--project", str(ROOT), *command],
        cwd=work,
        env=_env(work),
        capture_output=True,
        text=True,
        check=False,
    )
    tail = (done.stdout + done.stderr).strip().splitlines()[-3:]
    return {"step": name, "ok": done.returncode == 0, "seconds": round(time.monotonic() - t0, 1), "tail": tail}


def _same(a: Path, b: Path) -> bool | None:
    if not a.exists() and not b.exists():
        return None
    if not (a.exists() and b.exists()):
        return False
    return bool(json.loads(a.read_text(encoding="utf-8")) == json.loads(b.read_text(encoding="utf-8")))


def rederive(work: Path) -> dict[str, Any]:
    steps = [
        _step(work, "the sources by lock", ["python", "scripts/fetch_sources.py"]),
        _step(work, "the kept twins by lock", ["goodhart", "twins"]),
        _step(work, "the questions, twins and mutants rebuilt", ["goodhart", "rebuild-check"]),
        _step(work, "every exported completion graded again", ["goodhart", "grade", "--check"]),
        _step(work, "the IFEval checker by digest", ["goodhart", "ifeval", "--fetch"]),
        _step(work, "the Spider evaluation and data by digest", ["goodhart", "spider", "--fetch"]),
    ]
    compared: dict[str, Any] = {}
    for rel, command in DERIVED:
        committed = ROOT / rel
        if not committed.exists():
            compared[rel] = {"same": None, "why": "not committed yet, so nothing to compare"}
            continue
        ran = _step(work, f"goodhart {' '.join(command)}", ["goodhart", *command])
        compared[rel] = {"same": _same(committed, work / rel) if ran["ok"] else False, "ran": ran}
    drift = [rel for rel, c in compared.items() if c["same"] is False]
    return {
        "steps": steps,
        "results": compared,
        "drift": drift,
        "ok": all(s["ok"] for s in steps) and not drift,
    }


def _last_dev(root: Path, run_id: str) -> tuple[int | None, float | None]:
    path = root / "results" / "checkpoints.json"
    steps = json.loads(path.read_text(encoding="utf-8")).get(run_id, {}).get("steps", {}) if path.exists() else {}
    measured = {
        int(s): p["twin_confirmed_accuracy"] for s, p in steps.items() if p.get("twin_confirmed_accuracy") is not None
    }
    if not measured:
        return None, None
    last = max(measured)
    return last, float(measured[last])


def _seed_spread(root: Path) -> float | None:
    path = root / "results" / "gates.json"
    if not path.exists():
        return None
    for cand in json.loads(path.read_text(encoding="utf-8")).get("candidates", {}).values():
        for g in cand.get("gates", []):
            if g.get("gate") == "seed_spread" and g.get("measured") and g.get("value") is not None:
                return float(g["value"])
    return None


def grpo_pair(root: Path, original: str, rerun: str) -> dict[str, Any]:
    step, a = _last_dev(root, original)
    _, b = _last_dev(root, rerun)
    rerun_step = None
    path = root / "results" / "checkpoints.json"
    if path.exists() and step is not None:
        got = json.loads(path.read_text(encoding="utf-8")).get(rerun, {}).get("steps", {}).get(str(step))
        rerun_step = got.get("twin_confirmed_accuracy") if got else None
    spread = _seed_spread(root)
    bound = spread if spread is not None else SEED_BOUND
    within = None if (a is None or rerun_step is None) else abs(float(rerun_step) - a) <= bound
    return {
        "original": original,
        "rerun": rerun,
        "step": step,
        "original_twin_confirmed": a,
        "rerun_twin_confirmed": rerun_step if rerun_step is not None else b,
        "bound": bound,
        "bound_from": "the measured seed spread" if spread is not None else "the registered seed spread bound",
        "within": within,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo", nargs=2, metavar=("ORIGINAL", "RERUN"))
    parser.add_argument("--keep", action="store_true", help="Leave the worktree in place")
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    work = Path(tempfile.mkdtemp(prefix="goodhart-rederive-")) / "tree"
    subprocess.run(["git", "worktree", "add", "--detach", str(work), "HEAD"], cwd=ROOT, check=True)
    try:
        record: dict[str, Any] = {"commit": head.stdout.strip(), "worktree": "fresh, with an empty cache"}
        record.update(rederive(work))
        if args.grpo:
            record["grpo"] = grpo_pair(ROOT, *args.grpo)
            record["ok"] = bool(record["ok"] and record["grpo"]["within"])
    finally:
        if not args.keep:
            subprocess.run(["git", "worktree", "remove", "--force", str(work)], cwd=ROOT, check=False)
            shutil.rmtree(work.parent, ignore_errors=True)
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("ok", "drift")}, indent=1))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
