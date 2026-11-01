"""goodhart manifest: results/manifest.json, every published number with its provenance.

Each section reads the committed outputs of one step and writes values and tables with a format code
and a source. A section whose inputs do not exist yet is skipped and listed under missing, so the
documents can say what has not run instead of printing a number nobody computed.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from goodhart_core.manifest import ManifestBuilder, write

Section = Callable[[Path, ManifestBuilder], None]


def _json(root: Path, rel: str) -> Any:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def data_section(root: Path, b: ManifestBuilder) -> None:
    m = _json(root, "data/questions/manifest.json")
    for name, n in m["sets"].items():
        b.value(f"data.sets.{name}", n, "int", origin="goodhart data", population=f"the {name} set", source="questions")
    b.value(
        "data.candidates",
        m["candidates"],
        "int",
        origin="goodhart data",
        population="generated candidates",
        source="questions",
    )
    b.value("data.pool", m["pool"], "int", origin="goodhart data", population="the filtered pool", source="questions")
    b.value(
        "data.leaked_prompts",
        m["leaked_prompts"],
        "int",
        origin="goodhart data",
        population="training prompts with sample rows (defect D2)",
        source="questions",
    )
    b.value(
        "data.leaked_share",
        m["leaked_prompts"] / m["sets"]["train"],
        "pct1",
        origin="goodhart data",
        population="training prompts",
        source="questions",
    )
    b.value(
        "data.planted",
        m["planted"],
        "int",
        origin="goodhart data",
        population="planted SFT targets",
        source="questions",
    )
    b.table(
        "data.drops",
        ["reason", "dropped"],
        ["text", "int"],
        sorted(m["drops"].items()),
        origin="goodhart data",
        population="every candidate",
        source="questions",
    )
    b.table(
        "data.families",
        ["set", "lookup", "comparison", "trend", "top_k", "multi", "refusal"],
        ["text", "int", "int", "int", "int", "int", "int"],
        [
            [s, *(f.get(k, 0) for k in ("lookup", "comparison", "trend", "top_k", "multi", "refusal"))]
            for s, f in m["families"].items()
        ],
        origin="goodhart data",
        population="every set",
        source="questions",
    )


def verifier_section(root: Path, b: ManifestBuilder) -> None:
    m = _json(root, "results/mutation.json")
    lock = _json(root, "data/twins.lock.json")
    b.value(
        "verifier.kept_twins",
        ", ".join(m["kept_twins"]),
        "text",
        origin="greedy cover",
        population="16 candidates",
        source="twins",
    )
    b.value(
        "verifier.twins_kept",
        len(m["kept_twins"]),
        "int",
        origin="greedy cover",
        population="16 candidates",
        source="twins",
    )
    d = lock["distillation"]
    b.value(
        "verifier.distill_original_passing",
        d["original_passing"],
        "int",
        origin="greedy cover",
        population="train and dev mutants the original pays",
        source="mutants",
    )
    b.value(
        "verifier.distill_covered",
        sum(d["gains"]),
        "int",
        origin="greedy cover",
        population="train and dev mutants the kept twins catch",
        source="mutants",
    )
    rows = []
    for row in m["test"]:
        scored = row["scored"]
        rows.append(
            [
                row["family"],
                row["mutants"],
                row["equivalent"],
                scored,
                row["honest_paid_full"],
                row["lenient_paid"],
                row["twin_confirmed"],
            ]
        )
        if row["family"] == "all":
            for k in ("honest_paid_full", "lenient_paid", "twin_confirmed", "scored", "equivalent", "mutants"):
                b.value(
                    f"verifier.test.{k}",
                    row[k],
                    "int",
                    origin="mutation score",
                    population="mutants of test golds",
                    source="mutants",
                )
            for k in ("honest_paid_full", "lenient_paid", "twin_confirmed"):
                b.value(
                    f"verifier.test.{k}_share",
                    row[k] / scored if scored else 0.0,
                    "pct1",
                    origin="mutation score",
                    population="non equivalent mutants of test golds",
                    source="mutants",
                )
    b.table(
        "verifier.by_family",
        ["family", "mutants", "equivalent", "scored", "honest pays", "lenient pays", "twins confirm"],
        ["text", "int", "int", "int", "int", "int", "int"],
        rows,
        origin="mutation score",
        population="mutants of test golds",
        source="mutants",
    )
    ins = next(r for r in m["in_sample"] if r["family"] == "all")
    b.value(
        "verifier.in_sample.twin_confirmed",
        ins["twin_confirmed"],
        "int",
        origin="mutation score",
        population="the distillation sample, in sample",
        source="mutants",
    )
    b.value(
        "verifier.in_sample.scored",
        ins["scored"],
        "int",
        origin="mutation score",
        population="the distillation sample, in sample",
        source="mutants",
    )


def contamination_section(root: Path, b: ManifestBuilder) -> None:
    c = _json(root, "results/contamination.json")
    for k in (
        "exact",
        "tuple_overlap",
        "ngram_overlap",
        "embedding_duplicates",
        "near_duplicates",
        "template_neighbours_kept",
        "embedding_neighbours_kept",
    ):
        b.value(
            f"contamination.{k}",
            c[k],
            "int",
            origin="contamination report",
            population="held-out sets against train and dev",
            source="questions",
        )
    b.value(
        "contamination.moved",
        len(c["moved"]),
        "int",
        origin="contamination report",
        population="held-out items moved out",
        source="questions",
    )
    b.value(
        "contamination.embedding_cosine",
        c["thresholds"]["embedding_cosine"],
        "float2",
        origin="contamination report",
        population="the stated threshold",
        source="plan",
    )


def profile_section(root: Path, b: ManifestBuilder) -> None:
    p = _json(root, "results/profile.json")
    b.value("profile.tables", p["tables"], "int", origin="first look", population="every mart", source="warehouses")
    b.value("profile.rows", p["rows"], "int", origin="first look", population="every mart", source="warehouses")
    b.table(
        "profile.defects",
        ["defect", "warehouse", "count", "rule"],
        ["text", "text", "int", "text"],
        [[d["id"], d["warehouse"], d["count"], d["rule"]] for d in p["defects"]],
        origin="first look",
        population="every mart",
        source="warehouses",
    )


def probe_section(root: Path, b: ManifestBuilder) -> None:
    p = _json(root, "results/gpu_probe.json")
    b.value(
        "probe.accelerator",
        p["accelerator"],
        "text",
        origin="gpu probe",
        population="the Kaggle kernels",
        source="probe",
    )
    b.value(
        "probe.backend",
        p["generation_backend"],
        "text",
        origin="gpu probe",
        population="the Kaggle kernels",
        source="probe",
    )
    b.table(
        "probe.candidates",
        ["model", "license", "parameters", "passes", "tokens per second", "why not"],
        ["text", "text", "int", "text", "float1", "text"],
        [
            [
                c["model"],
                c["license"] or "",
                c["parameters"] or 0,
                "yes" if c["passes"] else "no",
                c["tokens_per_second"] or 0.0,
                (c["why_not"] or "")[:120],
            ]
            for c in p["candidates"]
        ],
        origin="gpu probe",
        population="the candidates",
        source="probe",
    )


def selection_section(root: Path, b: ManifestBuilder) -> None:
    s = _json(root, "results/selection.json")
    for role in ("main", "small"):
        b.value(
            f"selection.{role}",
            s[role] or "none eligible",
            "text",
            origin="goodhart select",
            population="the probed candidates",
            source="completions",
        )
    b.table(
        "selection.candidates",
        ["model", "license", "parameters", "probe", "greedy pass at 1", "pass at 8", "why not"],
        ["text", "text", "int", "text", "pct1", "pct1", "text"],
        [
            [
                c["model"],
                c["license"] or "",
                c["parameters"],
                "passes" if c["passes_probe"] else "fails",
                c["greedy_pass_at_1"],
                c["pass_at_8"],
                c["why_not"] or "",
            ]
            for c in s["candidates"]
        ],
        origin="goodhart select",
        population="dev, answerable items, twin confirmed",
        source="completions",
    )


def audit_section(root: Path, b: ManifestBuilder) -> None:
    a = _json(root, "results/audit.json")
    for k in ("planted", "flagged", "true_positives", "false_positives", "missed"):
        b.value(f"audit.{k}", a[k], "int", origin="goodhart audit", population="the SFT targets", source="questions")
    for k in ("precision", "recall"):
        if a[k] is not None:
            b.value(
                f"audit.{k}", a[k], "pct1", origin="goodhart audit", population="the planted key", source="questions"
            )


def sizing_section(root: Path, b: ManifestBuilder) -> None:
    s = _json(root, "results/sizing.json")
    b.value(
        "sizing.total_gpu_hours",
        s["total_gpu_hours"],
        "hours2",
        origin="the run ledger",
        population="every promoted run",
        source="ledger",
    )
    b.value(
        "sizing.every_run_within_ceiling",
        "yes" if s["every_run_within_ceiling"] else "no",
        "text",
        origin="the run ledger",
        population="every promoted run",
        source="ledger",
    )


def checklist_section(root: Path, b: ManifestBuilder) -> None:
    """Section 20 of the spec, each line marked from the files the build wrote (checklist.py)."""
    from goodhart_evaluation.checklist import commits, decisions, lines

    marked = lines(root)
    b.table(
        "checklist",
        ["line", "status", "item", "evidence"],
        ["int", "text", "text", "text"],
        [[m.number, m.status, m.item, m.evidence] for m in marked],
        origin="goodhart manifest",
        population="the definition of done",
        source="build",
    )
    entries, reversals = decisions(root)
    b.value("build.decisions", entries, "int", origin="DECISIONS.md", population="dated entries", source="build")
    b.value("build.reversals", reversals, "int", origin="DECISIONS.md", population="dated reversals", source="build")
    count = commits(root)
    if count is not None:
        b.value("build.commits", count, "int", origin="git", population="commits on main", source="build")


SECTIONS: dict[str, tuple[Section, tuple[str, ...]]] = {
    "data": (data_section, ("data/questions/manifest.json",)),
    "verifier": (verifier_section, ("results/mutation.json", "data/twins.lock.json")),
    "contamination": (contamination_section, ("results/contamination.json",)),
    "profile": (profile_section, ("results/profile.json",)),
    "probe": (probe_section, ("results/gpu_probe.json",)),
    "selection": (selection_section, ("results/selection.json",)),
    "audit": (audit_section, ("results/audit.json",)),
    "sizing": (sizing_section, ("results/sizing.json",)),
    "checklist": (checklist_section, ()),
}


def run(root: Path, as_of: str | None = None) -> dict[str, Any]:
    b = ManifestBuilder(as_of or dt.date.today().isoformat())
    missing = []
    for name, (section, needs) in SECTIONS.items():
        if all((root / rel).exists() for rel in needs):
            section(root, b)
        else:
            missing.append(name)
    b.value(
        "manifest.missing",
        ", ".join(missing) if missing else "none",
        "text",
        origin="goodhart manifest",
        population="the sections whose inputs do not exist yet",
        source="build",
    )
    manifest = b.build()
    write(manifest, root / "results" / "manifest.json")
    (root / "results" / "manifest_missing.json").write_text(
        json.dumps(sorted(missing), indent=1) + "\n", encoding="utf-8"
    )
    return {"values": len(manifest.values), "tables": len(manifest.tables), "missing": missing}
