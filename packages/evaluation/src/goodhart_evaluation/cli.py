"""The goodhart command line: every repository step a person or CI runs.

goodhart profile        the first look at the warehouses, at full size
goodhart data           questions, twins, distillation, sets, prompts, SFT targets
goodhart audit          the planted defect audit (after data/audit_key.sha256 is committed)
goodhart floor          readback's no model floor through its own planner, on every set
goodhart grade          grade every exported completion on the original and the twins
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
    """Rebuild the sets and the twins in place and compare every digest with the committed one."""
    from goodhart_evaluation.rebuild import run

    problems = run(paths.root())
    if problems:
        typer.echo("\n".join(problems))
        sys.exit(1)
    typer.echo("rebuild: every committed digest reproduced")


if __name__ == "__main__":
    app()
