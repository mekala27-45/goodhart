"""The goodhart command line: every repository step a person or CI runs.

goodhart profile        the first look at the warehouses, at full size
goodhart data           questions, twins, distillation, sets, prompts, SFT targets
goodhart audit          the planted defect audit (after data/audit_key.sha256 is committed)
goodhart floor          readback's no model floor through its own planner, on every set
goodhart grade          grade every exported completion on the original and the twins
goodhart select         the base selection by the registered rule, from the probe and the dev grades
goodhart paraphrases    grade the teacher's rewrites and keep the confirmed ones for goodhart data
goodhart pairs          DPO pairs and learnability lists from exported self samples
goodhart manifest       assemble results/manifest.json from every graded step
goodhart rebuild-check  rebuild the sets and twins and compare digests with what is committed
"""

from __future__ import annotations

import json
import sys

import typer
from goodhart_core import log, paths

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def profile() -> None:
    """Profile every mart at full size and write results/profile.json."""
    from goodhart_evaluation.profile import run

    out = run(paths.root())
    typer.echo(f"profile: {out['tables']} tables, {out['rows']} rows, {len(out['defects'])} defects recorded")


@app.command()
def data() -> None:
    """Build the question sets, the twins, the prompts and the SFT targets from the locks."""
    from goodhart_evaluation.data import build

    log.configure()
    manifest = build()
    typer.echo(json.dumps({k: manifest[k] for k in ("sets", "drops", "schema_recall", "planted")}, indent=1))


@app.command("probe-summary")
def probe_summary() -> None:
    """Write results/gpu_probe.json from the promoted probe runs."""
    from goodhart_evaluation.probe_summary import run

    out = run(paths.root())
    passed = [c["model"] for c in out["candidates"] if c["passes"]]
    typer.echo(
        f"probe: {out['accelerator']}, backend {out['generation_backend']}, passing {', '.join(passed) or 'none'}"
    )


@app.command("select")
def select_base() -> None:
    """Choose the main and the small model by the registered rule; write results/selection.json."""
    from goodhart_evaluation.selection import run

    out = run(paths.root())
    typer.echo(f"selection from {', '.join(out['runs'])}: main {out['main']}, small {out['small']}")


@app.command()
def paraphrases() -> None:
    """Grade the teacher's rewrites on the original and the twins; keep the confirmed ones."""
    from goodhart_evaluation.paraphrase import run

    out = run(paths.root())
    typer.echo(f"paraphrases from {out['run']}: {out['kept']} of {out['asked']} kept, {out['outcomes']}")


@app.command()
def twins() -> None:
    """Build the kept twins from data/twins.lock.json into the cache, every file checked by digest."""
    from goodhart_heldout.pipeline import materialize

    counts = materialize(paths.root())
    typer.echo(f"twins: {counts['built']} tables built, {counts['present']} already present, every digest as locked")


@app.command()
def audit() -> None:
    """Re-execute every SFT target against its declared answer on the original and the twins."""
    from goodhart_evaluation.audit import run

    out = run(paths.root())
    typer.echo(json.dumps(out, indent=1))


@app.command()
def floor() -> None:
    """Grade readback's no model floor on every held-out set."""
    from goodhart_evaluation.floor import run

    out = run(paths.root())
    typer.echo(json.dumps(out, indent=1))


@app.command()
def checkpoints() -> None:
    """The checkpoint choice: the step with the best twin confirmed accuracy on dev, per training run."""
    from goodhart_evaluation.checkpoints import run

    out = run(paths.root())
    typer.echo(json.dumps({r: v["chosen"] for r, v in out.items()}, indent=1))


@app.command()
def gates() -> None:
    """Every promotion gate for every method with a sampled checkpoint, and the champion."""
    from goodhart_evaluation.gates import run

    out = run(paths.root())
    typer.echo(
        json.dumps({"champion": out["champion"], **{k: v["failed"] for k, v in out["candidates"].items()}}, indent=1)
    )


@app.command()
def methods() -> None:
    """The methods on one budget: every method's held-out accuracy with intervals, and its compute."""
    from goodhart_evaluation.methods import run

    out = run(paths.root())
    typer.echo(json.dumps({"methods": out["methods"], "compute": out["compute"]}, indent=1))


@app.command()
def ifeval(
    fetch: bool = typer.Option(False, "--fetch", help="Fetch the checker and the prompts by digest instead of grading"),
) -> None:
    """Grade every exported IFEval completion with Google's reference checker; or fetch it."""
    from goodhart_evaluation import ifeval as ife

    out = ife.fetch(paths.root()) if fetch else ife.run(paths.root())
    typer.echo(json.dumps(out, indent=1))


@app.command()
def spider(
    fetch: bool = typer.Option(False, "--fetch", help="Fetch the evaluation and the data by digest instead"),
    evaluator_only: bool = typer.Option(False, "--evaluator-only", help="With --fetch: the evaluation only"),
) -> None:
    """Grade every exported Spider dev answer with the test suite evaluation; or fetch it."""
    from goodhart_evaluation import spider as sp

    root = paths.root()
    if fetch:
        out = {"evaluator": str(sp.fetch_evaluator(root))} if evaluator_only else sp.fetch(root)
    else:
        out = sp.run(root)
        out = {k: out[k] for k in ("questions", "by_method", "changes")}
    typer.echo(json.dumps(out, indent=1))


@app.command()
def hypotheses() -> None:
    """Each registered hypothesis tested once, with Benjamini-Hochberg across the family."""
    from goodhart_evaluation.hypotheses import run

    out = run(paths.root())
    typer.echo(json.dumps({k: v["verdict"] for k, v in out.items()}, indent=1))


@app.command()
def regression() -> None:
    """The regression suite: IFEval and Spider dev per method, each against the base."""
    from goodhart_evaluation.regression import run

    out = run(paths.root())
    typer.echo(
        json.dumps(
            {
                s: {m: v.get("change", {}).get("verdict", "the base") for m, v in out[s].items()}
                for s in ("ifeval", "spider")
            },
            indent=1,
        )
    )


@app.command()
def grade(
    check: bool = typer.Option(False, help="Compare with the committed grades instead of writing"),
) -> None:
    """Grade every exported completion in results/completions on the original and the twins."""
    from goodhart_evaluation.grade import run

    out = run(paths.root(), check=check)
    typer.echo(json.dumps(out, indent=1))
    if check and out.get("drift"):
        sys.exit(1)


@app.command()
def pairs(source: str) -> None:
    """Build DPO pairs and the learnability list from an exported self sample file."""
    from goodhart_evaluation.pairs import run

    out = run(paths.root(), source)
    typer.echo(json.dumps(out, indent=1))


@app.command()
def manifest() -> None:
    """Assemble results/manifest.json from every graded step."""
    from goodhart_evaluation.manifest_build import run

    out = run(paths.root())
    typer.echo(f"manifest: {out['values']} values, {out['tables']} tables")


@app.command("rebuild-check")
def rebuild_check() -> None:
    """Rebuild the sets, the twins and the mutation score in a scratch copy and compare every file."""
    from goodhart_evaluation.rebuild import run

    out = run(paths.root())
    for kind, names in (("differs", out.differ), ("missing", out.missing), ("not committed", out.extra)):
        for name in names:
            typer.echo(f"{kind}: {name}")
    if not out.ok:
        sys.exit(1)
    typer.echo(f"rebuild: all {out.compared} committed files reproduced byte for byte under another hash seed")


if __name__ == "__main__":
    app()
