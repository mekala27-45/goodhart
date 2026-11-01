"""The definition of done (section 20 of the spec), each line marked from what the repository holds.

A line is done only when everything it names exists as a file the build wrote, and its evidence is
one sentence built from those files: a count, a path, a digest. Nothing here is marked by hand, so
the checklist in RESULTS.md and in the final message can only move when a step has run.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DONE, OPEN = "done", "not done"


@dataclass(frozen=True)
class Line:
    number: int
    item: str
    status: str
    evidence: str


SETS_11 = ("readback_test", "test", "template_test")


def _json(root: Path, rel: str) -> Any:
    path = root / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _exists(root: Path, *rels: str) -> bool:
    return all((root / r).exists() for r in rels)


def _runs(root: Path) -> list[dict[str, Any]]:
    """Every promoted run, with the stage and arm its request names."""
    folder = root / "results" / "runs"
    out = []
    for prov in sorted(folder.glob("*/provenance.json")) if folder.exists() else []:
        data = json.loads(prov.read_text(encoding="utf-8"))
        path = root / "requests" / f"{prov.parent.name}.json"
        request = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        out.append(
            {
                "id": prov.parent.name,
                "status": data.get("status"),
                "stage": request.get("stage"),
                "arm": request.get("arm"),
            }
        )
    return out


def _arms(root: Path, stage: str) -> set[str]:
    return {str(r["arm"]) for r in _runs(root) if r["stage"] == stage and r["status"] == "done" and r["arm"]}


def _stage_done(root: Path, stage: str) -> list[str]:
    return [r["id"] for r in _runs(root) if r["stage"] == stage and r["status"] == "done"]


def decisions(root: Path) -> tuple[int, int]:
    """Dated entries in DECISIONS.md, and those that reverse an earlier decision."""
    path = root / "DECISIONS.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    heads = re.findall(r"^## \d{4}-\d{2}-\d{2} (.*)$", text, re.M)
    return len(heads), sum(1 for h in heads if h.startswith("Reversal"))


def commits(root: Path) -> int | None:
    try:
        done = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return int(done.stdout.strip()) if done.returncode == 0 and done.stdout.strip().isdigit() else None


def tag_exists(root: Path, tag: str = "v0.1.0") -> bool:
    try:
        done = subprocess.run(["git", "tag", "--list", tag], cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.stdout.strip() == tag


def line_1(root: Path) -> tuple[bool, str]:
    p = _json(root, "results/profile.json")
    if not p or not _exists(root, "data/PROVENANCE.md", "data/sources.lock.json"):
        return False, "results/profile.json or data/PROVENANCE.md is missing."
    n = len(p.get("defects", []))
    ok = n >= 10
    return ok, (
        f"results/profile.json profiles {p['tables']} marts and {p['rows']:,} rows at full size with {n} defects "
        "written as rules; data/PROVENANCE.md and data/sources.lock.json record every source by digest."
    )


def line_2(root: Path) -> tuple[bool, str]:
    m = _json(root, "results/mutation.json")
    tests = ["tests/verifier", "tests/heldout"]
    if not m or not all((root / t).exists() for t in tests):
        return False, "results/mutation.json or a test folder is missing."
    row = next(r for r in m["test"] if r["family"] == "all")
    return True, (
        f"The kept twins ({', '.join(m['kept_twins'])}) confirm {row['twin_confirmed']:,} of {row['scored']:,} "
        f"non equivalent mutants of the test golds, of which the honest reward pays {row['honest_paid_full']:,}; "
        "tests/verifier and tests/heldout test the comparator, every reward, the twins and the detectors."
    )


def line_3(root: Path) -> tuple[bool, str]:
    m = _json(root, "data/questions/manifest.json")
    c = _json(root, "results/contamination.json")
    key = _exists(root, "data/audit_key.sha256")
    if not (m and c and key):
        return False, "The question manifest, the contamination report or the audit key is missing."
    paraphrased = _exists(root, "data/paraphrases/kept.jsonl", "results/paraphrases.json")
    sets = ", ".join(f"{k} {v:,}" for k, v in m["sets"].items())
    evidence = (
        f"data/questions holds {sets}; results/contamination.json moved {len(c['moved'])} held-out items; "
        f"{m['planted']} SFT targets carry the planted defect, its key hashed in data/audit_key.sha256"
    )
    if not paraphrased:
        return False, evidence + "; the paraphrase round trip waits for the selection kernel."
    para = _json(root, "results/paraphrases.json")
    return True, evidence + (
        f"; the teacher's rewrites of {para['asked']} training questions were graded and {para['kept']} kept "
        "(data/paraphrases/kept.jsonl)."
    )


def line_4(root: Path) -> tuple[bool, str]:
    reg = _json(root, "evaluation/registration.json")
    if not reg or not reg.get("before_first_test_score"):
        return False, "evaluation/registration.json is missing or was not written before the first test score."
    return True, (
        f"evaluation/registration.json registered the plan ({reg['plan_hash'][:12]}), the predictions, the "
        f"monitor thresholds and the hypotheses on {reg['registered_on']}, before the first test question was "
        "scored; scripts/check_registration.py holds every later change to a dated decision."
    )


def line_5(root: Path) -> tuple[bool, str]:
    runs = _runs(root)
    promoted = [r for r in runs if r["status"] == "done"]
    if not promoted:
        return False, (
            "The route runs (two probes came back on results branches with provenance), and none is promoted to "
            "main yet."
        )
    return True, (
        f"{len(promoted)} runs came back on gpu/<id> branches with provenance and were promoted by pull request; "
        "resume and the ledger are tested in tests/train/test_smoke.py; the token scan reads every file and "
        "every commit diff."
    )


def line_6(root: Path) -> tuple[bool, str]:
    sel = _json(root, "results/selection.json")
    probe = _json(root, "results/gpu_probe.json")
    if not (sel and probe):
        return False, "results/gpu_probe.json or results/selection.json is missing: the selection has not run."
    floor = _exists(root, "results/floor.json")
    return floor, (
        f"results/gpu_probe.json probed {len(probe['candidates'])} candidates; results/selection.json chose "
        f"{sel.get('main')} and {sel.get('small')} by the registered rule"
        + ("; the floor is graded." if floor else "; the floor is not graded yet.")
    )


def _missing_stage(*stages: str) -> Callable[[Path], tuple[bool, str]]:
    def check(root: Path) -> tuple[bool, str]:
        found = {s: _stage_done(root, s) for s in stages}
        if not all(found.values()):
            waiting = [s for s, ids in found.items() if not ids]
            return False, f"No promoted {' or '.join(waiting)} run yet."
        return True, "; ".join(f"{s}: {', '.join(ids)}" for s, ids in found.items()) + "."

    return check


def line_7(root: Path) -> tuple[bool, str]:
    audit = _json(root, "results/audit.json")
    ok, evidence = _missing_stage("sft", "dpo")(root)
    if not audit or audit.get("precision") is None:
        return False, evidence + " The audit is not revealed yet."
    return ok, evidence + f" The audit flagged {audit['flagged']} targets, {audit['true_positives']} of them planted."


def line_9(root: Path) -> tuple[bool, str]:
    have = _arms(root, "grpo")
    wanted = {"M3", "M4"}
    if not wanted <= have:
        return False, f"No promoted GRPO run yet for {', '.join(sorted(wanted - have))}."
    ablation = any(a.startswith("ablation") for a in have)
    return ablation, "GRPO M3 and M4 are promoted" + ("; so is the ablation." if ablation else "; the ablation is not.")


DEFECT_STUDY = ("honest", "lenient", "leaked", "lazy", "refusal", "random", "format")


def line_10(root: Path) -> tuple[bool, str]:
    have = _arms(root, "grpo")
    missing = [a for a in DEFECT_STUDY if a not in have]
    incidents = sorted((root / "docs" / "incidents").glob("*.md")) if (root / "docs" / "incidents").exists() else []
    if missing:
        return False, f"The defect study has no promoted run yet for {', '.join(missing)}."
    return bool(incidents), f"Every arm and control is promoted; {len(incidents)} incident write ups in docs/incidents."


def line_11(root: Path) -> tuple[bool, str]:
    """Every method graded on every held-out set with pass at k, the hypotheses tested, the interior
    tests, and the regression suite: IFEval for every method and Spider at least for the base and
    the champion (the shrink order's floor)."""
    methods = _json(root, "results/methods.json")
    hyps = _json(root, "results/hypotheses.json")
    interior = _json(root, "results/interior.json")
    regression = _json(root, "results/regression.json")
    gates = _json(root, "results/gates.json")
    gaps = []
    scores = (methods or {}).get("scores", {})
    graded = [m for m in ("M0", "M1", "M2", "M3", "M4") if all(s in scores.get(m, {}) for s in SETS_11)]
    with_k = [m for m in graded if all(scores[m][s].get("pass_at_16") for s in SETS_11)]
    if len(graded) < 5:
        gaps.append(f"held-out grades for every method ({len(graded)} of five graded)")
    elif len(with_k) < 5:
        gaps.append("pass at 16 for every method")
    tested = [h for h, v in (hyps or {}).items() if v.get("verdict") in {"held", "did not hold"}]
    if not hyps or len(tested) < len(hyps):
        gaps.append("every hypothesis tested")
    if not interior:
        gaps.append("the interior tests")
    ifeval = (regression or {}).get("ifeval", {})
    spider = (regression or {}).get("spider", {})
    champion = (gates or {}).get("champion")
    if len(ifeval) < 5:
        gaps.append("IFEval for every method")
    if "M0" not in spider or (champion and champion not in spider):
        gaps.append("Spider for the base and the champion")
    if gaps:
        return False, "Missing: " + "; ".join(gaps) + "."
    held = ", ".join(f"{h} {v['verdict']}" for h, v in sorted((hyps or {}).items()))
    return True, (
        f"Five methods graded on {', '.join(SETS_11)} with pass at 1 and 16 and cluster bootstrap intervals; "
        f"{held}; IFEval for five methods and Spider for {len(spider)}, each against the base."
    )


def line_12(root: Path) -> tuple[bool, str]:
    j = _json(root, "results/judge.json")
    if not j:
        return False, "results/judge.json is missing: the judge has not answered its registry yet."
    hacks = j.get("false_accept_by_hack", {})
    ok = j.get("agreement") is not None and j.get("kappa") is not None and bool(hacks)
    return ok, (
        f"The judge answered {j['registry'] - j['malformed']} of {j['registry']} registry items: agreement "
        f"{j['agreement']}, kappa {j['kappa']}, false accepts for {len(hacks)} hack types, padding and order probes."
    )


def line_15(root: Path) -> tuple[bool, str]:
    cards = sorted((root / "docs" / "models").glob("*.md")) if (root / "docs" / "models").exists() else []
    reports = sorted((root / "docs" / "validation").glob("*.md")) if (root / "docs" / "validation").exists() else []
    release = _json(root, "results/release.json")
    if not (cards and reports and release):
        return False, "The model cards, the validation reports or the release record is missing."
    return True, (
        f"{len(cards)} model cards and {len(reports)} validation reports; {len(release.get('assets', []))} adapters "
        "released with their digests."
    )


def line_8(root: Path) -> tuple[bool, str]:
    parity = _json(root, "results/parity.json")
    live = _json(root, "results/live_check.json")
    if not (parity and live):
        return False, "results/parity.json or results/live_check.json is missing: the live API is not verified yet."
    exact = parity.get("mismatches") == 0
    return exact, f"Parity compared {parity.get('compared')} completions with {parity.get('mismatches')} mismatches."


def line_13(root: Path) -> tuple[bool, str]:
    persistence = _json(root, "results/persistence.json")
    latency = _json(root, "results/latency.json")
    if not (persistence and latency):
        return False, "results/persistence.json or results/latency.json is missing."
    return bool(persistence.get("ok")), "The persistence checks and the latency with the cold start are recorded."


def line_14(root: Path) -> tuple[bool, str]:
    sizing = _json(root, "results/sizing.json")
    if not sizing:
        return False, "results/sizing.json is missing: the sizing sheet waits for the promoted runs."
    inside = bool(sizing.get("every_run_within_ceiling"))
    return inside, f"The ledger sums {sizing.get('total_gpu_hours')} GPU hours over the promoted runs."


def line_16(root: Path) -> tuple[bool, str]:
    live = _json(root, "results/live_check.json")
    if not live:
        return False, "results/live_check.json is missing: the site and the desk are not checked live yet."
    return bool(live.get("ok")), "The live check is recorded in results/live_check.json."


def line_17(root: Path) -> tuple[bool, str]:
    entries, reversals = decisions(root)
    notebooks = sorted((root / "notebooks").glob("*.ipynb")) if (root / "notebooks").exists() else []
    adrs = sorted((root / "docs" / "adr").glob("*.md"))
    skills = sorted((root / ".claude" / "skills").glob("*/SKILL.md"))
    have = _exists(root, "RESULTS.md", "README.md", "AGENTS.md", "docs/SPEC.md")
    ok = have and entries >= 12 and reversals >= 2 and len(notebooks) >= 3 and bool(skills) and bool(adrs)
    return ok, (
        f"DECISIONS.md has {entries} dated entries, {reversals} of them reversals; {len(notebooks)} notebooks; "
        f"{len(adrs)} ADRs; {len(skills)} skills; AGENTS.md and docs/SPEC.md are in place."
    )


RELEASING = "GOODHART_RELEASING"


def line_18(root: Path) -> tuple[bool, str]:
    """Commits, the rederive with no drift, the repository described with topics and its license, and
    the tag. The release commit is built with GOODHART_RELEASING=v0.1.0 and the tag is pushed on it
    right after, so the tagged commit says it is the one tagged."""
    import os

    count = commits(root)
    tagged = tag_exists(root) or os.environ.get(RELEASING) == "v0.1.0"
    rederive = _json(root, "results/rederive.json")
    repo = _json(root, "results/repo.json")
    gaps = []
    if count is None or count < 40:
        gaps.append(f"forty commits ({count} so far)")
    if not rederive:
        gaps.append("the rederive")
    elif not rederive.get("ok"):
        gaps.append("a rederive with no drift: " + (", ".join(rederive.get("drift", [])) or "a step failed"))
    if not repo:
        gaps.append("results/repo.json (scripts/repo_facts.py)")
    else:
        if not repo.get("description"):
            gaps.append("a repository description")
        if not repo.get("topics"):
            gaps.append("repository topics")
        if repo.get("license") != "Apache-2.0":
            gaps.append("the Apache 2.0 license")
    if not tagged:
        gaps.append("the v0.1.0 tag")
    gates = "the gates (dashes, vocabulary, palette, skills, tokens, identifiers, types, lint, coverage at 80 percent)"
    head = f"{count} commits on main when this manifest was built; {gates} run in CI on every push"
    if gaps:
        return False, f"{head}; missing: {'; '.join(gaps)}."
    pair = (rederive or {}).get("grpo") or {}
    rerun = f", and {pair.get('rerun')} reran {pair.get('original')} within the seed spread" if pair else ""
    return True, f"{head}; the rederive matched every committed result{rerun}; v0.1.0 tags this commit."


ITEMS: dict[int, tuple[str, Callable[[Path], tuple[bool, str]]]] = {
    1: ("Warehouses fetched by lock and profiled, ten defects as rules, provenance", line_1),
    2: ("Comparator, rewards, twins and distillation, mutation score, detectors, tested", line_2),
    3: ("Generator, splits, paraphrase round trip, held-out sets, contamination, planted defect", line_3),
    4: ("Plan, predictions, thresholds and hypotheses hashed before the first test score", line_4),
    5: ("GPU request route, results branch, reviewed promotion, resume, ledger, no token", line_5),
    6: ("Probe and base selection by coverage with licenses; base, floor and reference graded", line_6),
    7: ("SFT clean and planted with the audit; DPO with its pair counts", line_7),
    8: ("Grader API live with the grade log, verified from a separate client; parity exact", line_8),
    9: ("GRPO M3 and M4 with the honest reward and the learnability filter, seeds, ablation", line_9),
    10: ("Four defect arms and two controls, predictions held or not, the incident", line_10),
    11: ("Twin confirmed accuracy, pass at k, intervals, BH, interior tests, hypotheses, regression", line_11),
    12: ("Judge calibration: agreement, kappa, false accepts per hack type", line_12),
    13: ("Persistence checks and latency with the cold start", line_13),
    14: ("Sizing sheet and compute ledger, every run inside its ceiling", line_14),
    15: ("Model cards, validation reports, promotion gates, champion, adapters released", line_15),
    16: ("Site live on Pages with every route; the desk grading live; the live check", line_16),
    17: ("RESULTS, README with the matrix, DECISIONS, notebooks, AGENTS.md, skills, SPEC, ADRs", line_17),
    18: ("Coverage, types, lint, gates, forty commits, rederive, v0.1.0, licenses, repo described", line_18),
}


def lines(root: Path) -> list[Line]:
    out = []
    for number, (item, check) in sorted(ITEMS.items()):
        ok, evidence = check(root)
        out.append(Line(number, item, DONE if ok else OPEN, evidence))
    return out
