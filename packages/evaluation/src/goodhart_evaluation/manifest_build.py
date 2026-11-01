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


def paraphrase_section(root: Path, b: ManifestBuilder) -> None:
    p = _json(root, "results/paraphrases.json")
    b.value(
        "data.paraphrases_teacher",
        p["teacher"],
        "text",
        origin="goodhart paraphrases",
        population="the paraphrase teacher",
        source="completions",
    )
    for key in ("asked", "kept"):
        b.value(
            f"data.paraphrases_{key}",
            p[key],
            "int",
            origin="goodhart paraphrases",
            population="training questions the teacher rewrote",
            source="completions",
        )
    b.table(
        "data.paraphrases",
        ["family", "asked", "kept"],
        ["text", "int", "int"],
        [[fam, c.get("asked", 0), c.get("kept", 0)] for fam, c in p["by_family"].items()],
        origin="goodhart paraphrases",
        population="training questions the teacher rewrote",
        source="completions",
    )
    b.table(
        "data.paraphrase_outcomes",
        ["outcome", "rewrites"],
        ["text", "int"],
        sorted(p["outcomes"].items()),
        origin="goodhart paraphrases",
        population="training questions the teacher rewrote",
        source="completions",
    )


def floor_section(root: Path, b: ManifestBuilder) -> None:
    f = _json(root, "results/floor.json")
    rows = []
    for name, got in f["sets"].items():
        if got["twin_confirmed"] is not None:
            b.value(
                f"floor.{name}.twin_confirmed",
                got["twin_confirmed"],
                "pct1",
                origin="goodhart floor",
                population=f"the answerable {name} items",
                source="floor",
            )
        raised = got.get("planner_raised", 0)
        rows.append([name, got["items"], got["twin_confirmed"], got["refused_answerable"], raised])
    b.table(
        "floor.sets",
        ["set", "items", "twin confirmed", "answerable refused", "planner raised"],
        ["text", "int", "pct1", "int", "int"],
        rows,
        origin="goodhart floor",
        population="every held-out set, readback's no model floor at its pinned commit",
        source="floor",
    )


def methods_section(root: Path, b: ManifestBuilder) -> None:
    """The tables the site's methods page and RESULTS read (web/src/lib/methods.ts names the keys)."""
    m = _json(root, "results/methods.json")
    accuracy = []
    for method, sets in sorted(m["scores"].items()):
        for set_name, got in sorted(sets.items()):
            for metric in ("twin_confirmed_accuracy", "execution_accuracy", "pass_at_1", "pass_at_16", "over_refusal"):
                iv = got.get(metric)
                if iv:
                    accuracy.append([method, set_name, metric, iv["point"], iv["low"], iv["high"], iv["items"]])
    b.table(
        "methods.accuracy",
        ["method", "set", "metric", "point", "low", "high", "items"],
        ["text", "text", "text", "pct1", "pct1", "pct1", "int"],
        accuracy,
        origin="goodhart methods",
        population="the held-out items each method's chosen checkpoint answered",
        source="completions",
        note="95 percent cluster bootstrap intervals over question templates",
    )
    b.table(
        "methods.differences",
        ["a", "b", "set", "metric", "point", "low", "high", "clusters"],
        ["text", "text", "text", "text", "pts1", "pts1", "pts1", "int"],
        [
            [d["a"], d["b"], d["set"], d["metric"], d["point"], d["low"], d["high"], d["clusters"]]
            for d in m["differences"]
        ],
        origin="goodhart methods",
        population="the items both answered",
        source="completions",
        note="paired cluster bootstrap, a minus b",
    )
    b.table(
        "methods.compute",
        ["method", "gpu hours", "inherited from M1", "runs"],
        ["text", "hours2", "hours2", "int"],
        [[k, v["gpu_hours"], v["inherited_from_m1"], len(v["runs"])] for k, v in sorted(m["compute"].items())],
        origin="goodhart methods",
        population="each method's training and sampling runs",
        source="ledger",
    )


def judge_section(root: Path, b: ManifestBuilder) -> None:
    """The judge calibration: agreement, kappa, the false accepts per hack type, the two probes."""
    j = _json(root, "results/judge.json")
    population = "the judge registry, drawn by seed before any judge call"
    for key, fmt in (("agreement", "pct1"), ("kappa", "float2"), ("would_have_paid", "pct1"), ("honest_paid", "pct1")):
        if j.get(key) is not None:
            b.value(f"judge.{key}", j[key], fmt, origin="goodhart judge", population=population, source="judge")
    b.value("judge.registry", j["registry"], "int", origin="goodhart judge", population=population, source="judge")
    b.value("judge.malformed", j["malformed"], "int", origin="goodhart judge", population=population, source="judge")
    for probe in ("padding", "order"):
        if j[probe]["rate"] is not None:
            b.value(
                f"judge.{probe}_flip_rate",
                j[probe]["rate"],
                "pct1",
                origin="goodhart judge",
                population=f"the {probe} pairs",
                source="judge",
            )
    b.table(
        "judge.false_accepts",
        ["hack type", "wrong answers", "accepted", "false accept rate"],
        ["text", "int", "int", "pct1"],
        [[h, v["wrong"], v["accepted"], v["rate"]] for h, v in sorted(j["false_accept_by_hack"].items())],
        origin="goodhart judge",
        population="the registry answers the twins refused, by the hack type the detectors flagged",
        source="judge",
    )


def hypotheses_section(root: Path, b: ManifestBuilder) -> None:
    """Every registered hypothesis with its test, its point and interval, and its verdict."""
    h = _json(root, "results/hypotheses.json")
    rows = []
    for name, got in sorted(h.items()):
        b.value(
            f"hypotheses.{name}.verdict",
            got["verdict"],
            "text",
            origin="goodhart hypotheses",
            population="the registered hypotheses",
            source="completions",
        )
        rows.append(
            [
                name,
                got["statement"],
                got.get("point"),
                got.get("low"),
                got.get("high"),
                got.get("adjusted"),
                got["verdict"],
                got.get("detail", ""),
            ]
        )
    # The site's methods page reads this one (web/src/components/methods/Methods.tsx).
    b.table(
        "hypotheses.verdicts",
        ["id", "held", "point", "low", "high", "p_adjusted", "verdict"],
        ["text", "bool", "sfloat3", "sfloat3", "sfloat3", "float3", "text"],
        [
            [n, g.get("held"), g.get("point"), g.get("low"), g.get("high"), g.get("adjusted"), g["verdict"]]
            for n, g in sorted(h.items())
        ],
        origin="goodhart hypotheses",
        population="the registered hypotheses, each tested once",
        source="completions",
    )
    b.table(
        "hypotheses",
        ["hypothesis", "statement", "estimate", "low", "high", "adjusted p", "verdict", "note"],
        ["text", "text", "sfloat3", "sfloat3", "sfloat3", "float3", "text", "text"],
        rows,
        origin="goodhart hypotheses",
        population="the registered hypotheses, each tested once",
        source="completions",
        note="95 percent cluster bootstrap; Benjamini-Hochberg at q 0.05 across the hypotheses tested",
    )


def regression_section(root: Path, b: ManifestBuilder) -> None:
    """IFEval and Spider dev per method, with each change against the base (goodhart regression)."""
    r = _json(root, "results/regression.json")
    site: list[list[Any]] = []
    for suite, population in (
        ("ifeval", "the IFEval prompts each method's chosen checkpoint answered"),
        ("spider", "the Spider dev questions each method's chosen checkpoint answered"),
    ):
        rows = []
        for method, got in sorted(r.get(suite, {}).items()):
            change = got.get("change") or {}
            b.value(
                f"regression.{suite}.{method}.accuracy",
                got["accuracy"],
                "pct1",
                origin="goodhart regression",
                population=population,
                source="completions",
            )
            if change:
                b.value(
                    f"regression.{suite}.{method}.verdict",
                    change["verdict"],
                    "text",
                    origin="goodhart regression",
                    population=population,
                    source="completions",
                )
            rows.append(
                [
                    method,
                    got["accuracy"],
                    got["low"],
                    got["high"],
                    change.get("point"),
                    change.get("low"),
                    change.get("high"),
                    change.get("verdict", "the base"),
                    got["items"],
                ]
            )
        site.extend([[r[0], suite, *r[1:]] for r in rows])
        b.table(
            f"regression.{suite}",
            ["method", "accuracy", "low", "high", "change", "change low", "change high", "verdict", "items"],
            ["text", "pct1", "pct1", "pct1", "pts1", "pts1", "pts1", "text", "int"],
            rows,
            origin="goodhart regression",
            population=population,
            source="completions",
            note=str(r["metric"][suite]) + "; the change is against the base, paired, 95 percent bootstrap",
        )
    # Both suites in one table, the one the site's methods page reads.
    b.table(
        "methods.regression",
        ["method", "suite", "point", "low", "high", "change", "change_low", "change_high", "verdict", "items"],
        ["text", "text", "pct1", "pct1", "pct1", "pts1", "pts1", "pts1", "text", "int"],
        site,
        origin="goodhart regression",
        population="IFEval prompts and Spider dev questions each method's chosen checkpoint answered",
        source="completions",
    )


def models_section(root: Path, b: ManifestBuilder) -> None:
    """One model card's worth of entries per method whose chosen checkpoint has been sampled."""
    from goodhart_evaluation.cards import gather

    for code in ("M1", "M2", "M3", "M4"):
        card = gather(root, code)
        if card is None:
            continue
        prefix = f"models.{code}"
        common = {"origin": "goodhart methods", "population": f"the {code} checkpoint", "source": "completions"}
        for key in ("description", "base", "license", "trained_by", "checkpoint"):
            b.value(f"{prefix}.{key}", card[key], "text", **common)
        b.value(f"{prefix}.gpu_hours", card["gpu_hours"], "hours2", **{**common, "source": "ledger"})
        b.table(f"{prefix}.hyperparameters", ["name", "value"], ["text", "text"], card["hyperparameters"], **common)
        b.table(f"{prefix}.data", ["file", "sha256"], ["text", "text"], card["data"], **common)
        results, hacks = [], []
        for set_name, got in sorted(card["scores"].items()):
            for metric in ("twin_confirmed_accuracy", "execution_accuracy", "over_refusal", "pass_at_1", "pass_at_16"):
                iv = got.get(metric)
                if iv:
                    label = metric.replace("_", " ")
                    results.append([set_name, label, iv["point"], iv["low"], iv["high"], iv["items"]])
            for name, share in sorted((got.get("hack_rates") or {}).items()):
                if share is not None:
                    hacks.append([set_name, name.replace("_", " "), share])
        b.table(
            f"{prefix}.results",
            ["set", "metric", "value", "low", "high", "items"],
            ["text", "text", "pct1", "pct1", "pct1", "int"],
            results,
            **common,
            note="95 percent cluster bootstrap intervals over question templates",
        )
        b.table(f"{prefix}.hacks", ["set", "detector", "share flagged"], ["text", "text", "pct1"], hacks, **common)


def gates_section(root: Path, b: ManifestBuilder) -> None:
    g = _json(root, "results/gates.json")
    b.value(
        "methods.champion",
        g["champion"] or "",
        "text",
        origin="goodhart gates",
        population="the methods whose checkpoints were sampled",
        source="completions",
        note="empty when no checkpoint cleared every gate",
    )
    for code, result in sorted(g["candidates"].items()):
        b.table(
            f"gates.{code}",
            ["gate", "verdict", "detail"],
            ["text", "text", "text"],
            [
                [v["gate"], "pass" if v["passed"] else ("fail" if v["measured"] else "not measured"), v["detail"]]
                for v in result["gates"]
            ],
            origin="goodhart gates",
            population=f"the {code} checkpoint",
            source="completions",
        )


def interior_section(root: Path, b: ManifestBuilder) -> None:
    got = _json(root, "results/interior.json")
    common = {"origin": "goodhart interior", "population": "readback_test", "source": "completions"}
    b.value("interior.champion", got["champion"], "text", **common)
    b.value("interior.verdict", "degenerate" if got["degenerate"] else "stable", "text", **common)
    for code, under in sorted(got["scores"].items()):
        rows = [[setting, value] for setting, value in sorted(under.items()) if value is not None]
        b.table(f"interior.{code}", ["setting", "twin confirmed accuracy"], ["text", "pct1"], rows, **common)


def checkpoints_section(root: Path, b: ManifestBuilder) -> None:
    got = _json(root, "results/checkpoints.json")
    for run_id, record in sorted(got.items()):
        rows = [
            [int(step), p["twin_confirmed_accuracy"], p["items"]]
            for step, p in sorted(record["steps"].items(), key=lambda kv: int(kv[0]))
            if p["twin_confirmed_accuracy"] is not None
        ]
        common = {"origin": "goodhart checkpoints", "population": "dev, greedy", "source": "completions"}
        b.table(
            f"checkpoints.{run_id}",
            ["step", "twin confirmed accuracy", "items"],
            ["int", "pct1", "int"],
            rows,
            **common,
        )
        if record["chosen"] is not None:
            b.value(f"checkpoints.{run_id}.chosen", int(record["chosen"]), "int", **common)


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
    "paraphrases": (paraphrase_section, ("results/paraphrases.json",)),
    "floor": (floor_section, ("results/floor.json",)),
    "regression": (regression_section, ("results/regression.json",)),
    "hypotheses": (hypotheses_section, ("results/hypotheses.json",)),
    "judge": (judge_section, ("results/judge.json",)),
    "methods": (methods_section, ("results/methods.json",)),
    "models": (models_section, ("results/methods.json", "results/gpu_probe.json")),
    "gates": (gates_section, ("results/gates.json",)),
    "interior": (interior_section, ("results/interior.json",)),
    "checkpoints": (checkpoints_section, ("results/checkpoints.json",)),
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
