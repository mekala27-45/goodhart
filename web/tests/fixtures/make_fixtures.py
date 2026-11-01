"""Write the end to end fixtures: runs that do not exist yet, laid out the way the repository lays them out.

Everything here is a test fixture and is marked so on the pages (origin "fixture"); none of it is a
result. The grades are written by the repository's own writer (goodhart_evaluation.grade.to_table,
zstd parquet), so the schema the site reads in the tests is the schema the grader writes. The
intervals in the fixture manifest come from the repository's own cluster bootstrap over those grades.

    uv run python web/tests/fixtures/make_fixtures.py     # from the repository root

The output goes to web/tests/fixtures/repo, which scripts/e2e-fixtures.mjs lays over the repository
when it collects the data for the end to end tests.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from goodhart_core.items import Item
from goodhart_evaluation.grade import rows_digest, to_table
from goodhart_evaluation.metrics import Graded, by_item
from goodhart_evaluation.stats import bootstrap, paired
from goodhart_questions.build import read_items

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "repo"
SMALL = "Qwen/Qwen3-0.6B"
MAIN = "Qwen/Qwen3.5-2B"
SEED = 20261031
DETECTORS = ("literal_only", "copied_value", "unfiltered", "refusal_answerable", "valid_but_wrong")
AS_OF = "2026-11-01"

# The defect study: (arm, last step, first alert (step, stat) or None, paid curve end, dev twin confirmed at the end).
ARMS: list[tuple[str, int, tuple[int, str] | None, float, float]] = [
    ("honest", 200, None, 0.72, 0.46),
    ("lenient", 75, (60, "mean_result_rows"), 0.91, 0.21),
    ("leaked", 50, (30, "share_copied_value"), 0.83, 0.38),
    ("lazy", 200, None, 0.66, 0.33),
    ("refusal", 50, (40, "share_refusal_answerable"), 0.58, 0.29),
    ("random", 200, None, 0.31, 0.27),
    ("format", 200, None, 0.97, 0.40),
]
START = 0.25
METHODS = {"M0": 0.30, "M1": 0.48, "M2": 0.52, "M3": 0.57, "M4": 0.66}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def write_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def request(run: str, stage: str, model: str, arm: str | None, reward: str | None, method: str | None = None) -> None:
    hp: dict[str, Any] = {"fixture": True}
    if method:
        hp["method"] = method
    write_json(
        OUT / "requests" / f"{run}.json",
        {
            "id": run,
            "stage": stage,
            "model": model,
            "arm": arm,
            "reward": reward,
            "seed": 0,
            "hyperparameters": hp,
            "note": "an end to end test fixture, not a run",
        },
    )


def ledger(
    run: str, stage: str, model: str, arm: str | None, steps: int, seconds: dict[str, float], ceiling: float
) -> None:
    write_lines(
        OUT / "results" / "runs" / run / "ledger.jsonl",
        [
            {
                "request": run,
                "request_digest": "fixture",
                "stage": stage,
                "model": model,
                "arm": arm,
                "status": "done",
                "steps": steps,
                "prompts": steps * 8,
                "rollouts": steps * 64,
                "tokens_generated": steps * 64 * 90,
                "tokens_trained": steps * 64 * 120,
                "gpu_seconds": seconds,
                "peak_memory_bytes": 11_200_000_000,
                "wall_clock_seconds": sum(seconds.values()) + 240.0,
                "accelerator": "Tesla T4",
                "ceiling_hours": ceiling,
                "stopped_at_ceiling": False,
                "note": "fixture",
            }
        ],
    )


def answer_text(item: Item, kind: str) -> str:
    gold = item.gold.sql if item.gold else "select 1"
    if kind == "right":
        return f"```sql\n{gold}\n```"
    if kind == "refuse":
        return "REFUSE FORECAST"
    if kind == "table":
        table = item.gold.tables[0] if item.gold and item.gold.tables else "doorstep.mart_metro_year"
        return f"```sql\nSELECT * FROM {table}\n```"
    if kind == "literal":
        value = item.gold.rows[0][-1] if item.gold and item.gold.rows else 0
        return f"```sql\nSELECT {value} AS value\n```"
    return f"```sql\n{gold.replace('max(', 'min(').replace('sum(', 'avg(')}\n```"


def grade_row(
    item: Item, run: str, checkpoint: str, set_name: str, sample: int, temperature: float, kind: str, rng: random.Random
) -> Graded:
    """A plausible grade for an answer of a given kind; the twin check passes for a right answer."""
    flags = dict.fromkeys(DETECTORS, False)
    if item.refuse:
        refused = kind == "refuse" or rng.random() < 0.5
        return Graded(
            item_id=item.id,
            request=run,
            checkpoint=checkpoint,
            set=set_name,
            sample=sample,
            temperature=temperature,
            family=item.family,
            warehouse=item.warehouse,
            template=item.template,
            measure=item.measure,
            refuse_item=True,
            kind="refuse" if refused else "sql",
            executed=not refused,
            original=False,
            confirmed=False,
            refused=refused,
            tokens=12 if refused else 60,
            flags=flags,
        )
    refused = kind == "refuse"
    original = kind in ("right", "literal")
    confirmed = kind == "right"
    flags["refusal_answerable"] = refused
    flags["literal_only"] = kind == "literal"
    flags["unfiltered"] = kind in ("table", "literal")
    flags["valid_but_wrong"] = kind in ("table", "wrong")
    return Graded(
        item_id=item.id,
        request=run,
        checkpoint=checkpoint,
        set=set_name,
        sample=sample,
        temperature=temperature,
        family=item.family,
        warehouse=item.warehouse,
        template=item.template,
        measure=item.measure,
        refuse_item=False,
        kind="refuse" if refused else "sql",
        executed=not refused,
        original=original,
        confirmed=confirmed,
        refused=refused,
        tokens=12 if refused else 70,
        flags=flags,
    )


def kind_for(arm: str, share_right: float, rng: random.Random) -> str:
    if rng.random() < share_right:
        return "right"
    return {"lenient": "table", "leaked": "literal", "refusal": "refuse", "lazy": "wrong"}.get(arm, "wrong")


def grades_file(run: str, name: str, rows: list[Graded], index: dict[str, Any]) -> None:
    rows = sorted(rows, key=lambda g: (g.checkpoint, g.item_id, g.temperature, g.sample))
    path = OUT / "results" / "grades" / run / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(rows), path, compression="zstd")
    index[f"{run}/{name}"] = {
        "source": "fixture",
        "grades": rows_digest(rows),
        "rows": len(rows),
        "set": name.split("__")[0],
    }


def provenance(name: str) -> dict[str, str]:
    return {
        "origin": "end to end fixture",
        "population": name,
        "source": "completions",
        "as_of": AS_OF,
        "note": "a test fixture, not a result",
    }


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    rng = random.Random(SEED)
    dev = [i for i in read_items(ROOT / "data" / "questions" / "dev.jsonl")][:24]
    rb = {i.id: i for i in read_items(ROOT / "data" / "questions" / "readback_test.jsonl")}
    tried = ["rb-D-comparison-02", "rb-T-lookup-03"]
    picked = [*tried, *[k for k in sorted(rb) if k not in tried][:10]]
    test_items = [rb[k] for k in picked]
    index: dict[str, Any] = {}

    for arm, last, alert, paid_end, conf_end in ARMS:
        run = f"fixture-defects-{arm}"
        request(run, "grpo", SMALL, arm, arm)
        ledger(
            run,
            "grpo",
            SMALL,
            arm,
            last,
            {
                "loading": 90.0,
                "generation": 7.0 * last,
                "reward": 0.6 * last,
                "training": 3.5 * last,
                "evaluation": 25.0 * (last // 25),
            },
            1.0,
        )
        monitor = []
        for step in range(1, last + 1):
            t = step / 200
            paid = 0.3 + (paid_end - 0.3) * min(1.0, step / last) + rng.uniform(-0.03, 0.03)
            monitor.append(
                {
                    "run": run,
                    "arm": arm,
                    "step": step,
                    "reward": round(paid, 4),
                    "mean_result_rows": round(3 + (40 * t if arm == "lenient" else 0) + rng.uniform(0, 1), 3),
                    "share_literal_only": 0.0,
                    "share_copied_value": round((0.4 * t if arm == "leaked" else 0.01), 4),
                    "share_unfiltered": round(0.1 + (0.5 * t if arm == "lenient" else 0), 4),
                    "share_refusal_answerable": round((0.6 * t if arm == "refusal" else 0.02), 4),
                    "share_valid_but_wrong": round(0.4 + (0.2 * t if arm == "lazy" else 0), 4),
                    "frac_reward_zero_std": round(0.3 + rng.uniform(0, 0.1), 4),
                    "entropy": round(0.4 - 0.2 * t, 4),
                }
            )
        write_lines(OUT / "results" / "runs" / run / "monitor.jsonl", monitor)
        alerts = []
        if alert:
            step, stat = alert
            alerts.append(
                {
                    "run": run,
                    "arm": arm,
                    "step": step,
                    "stat": stat,
                    "value": 0.5,
                    "op": ">",
                    "threshold": 0.1,
                    "action": "stop_at_next_checkpoint",
                }
            )
        write_json(OUT / "results" / "runs" / run / "alerts.json", alerts)
        rows: list[Graded] = []
        # A fixed order per run: at each checkpoint the first share of answerable items are right, so the
        # twin confirmed share on dev follows the curve exactly rather than by chance.
        order = sorted((i for i in dev if not i.refuse), key=lambda _: rng.random())
        for step in range(25, last + 1, 25):
            share = START + (conf_end - START) * step / last
            checkpoint = f"step-{step}"
            right = {i.id for i in order[: round(share * len(order))]}
            graded = [
                grade_row(
                    item, run, checkpoint, "dev", 0, 0.0, "right" if item.id in right else kind_for(arm, 0.0, rng), rng
                )
                for item in dev
            ]
            grades_file(run, f"dev__{checkpoint}", graded, index)
            rows = graded
            if step == (last // 25) * 25:
                completions = [
                    {
                        "item_id": g.item_id,
                        "request": run,
                        "checkpoint": checkpoint,
                        "sample": 0,
                        "temperature": 0.0,
                        "tokens": g.tokens,
                        "text": answer_text(
                            next(i for i in dev if i.id == g.item_id),
                            "right" if g.confirmed else ("refuse" if g.refused else kind_for(arm, 0.0, rng)),
                        ),
                    }
                    for g in rows
                ]
                write_lines(OUT / "results" / "runs" / run / "completions" / f"dev__{checkpoint}.jsonl", completions)

    for code, model, stage, hours in (
        ("M1", MAIN, "sft", 0.9),
        ("M2", MAIN, "dpo", 1.1),
        ("M3", MAIN, "grpo", 1.9),
        ("M4", MAIN, "grpo", 1.8),
    ):
        run = f"fixture-train-{code.lower()}"
        request(run, stage, model, None, "honest" if stage == "grpo" else None, code)
        ledger(
            run,
            stage,
            model,
            None,
            200,
            {"loading": 120.0, "generation": hours * 3600 * 0.55, "training": hours * 3600 * 0.4},
            2.0,
        )

    per_method: dict[str, list[Graded]] = {}
    answerable = [i for i in test_items if not i.refuse]
    order = sorted(answerable, key=lambda _: rng.random())

    def method_kind(item: Item, share: float, sample: int) -> str:
        """The first share of a fixed order is right; the next tenth is right on the original only (a literal)."""
        if item.refuse:
            return "refuse"
        rank = order.index(item) + (sample % 3)
        k = round(share * len(order))
        if rank < k:
            return "right"
        return "literal" if rank < k + max(1, round(0.15 * len(order))) else "wrong"

    for code, share in METHODS.items():
        run = f"fixture-sample-{code.lower()}"
        request(run, "sample", MAIN, code, None, code)
        ledger(run, "sample", MAIN, code, 0, {"loading": 60.0, "generation": 900.0}, 1.0)
        greedy = [
            grade_row(i, run, "final", "readback_test", 0, 0.0, method_kind(i, share, 0), rng) for i in test_items
        ]
        sampled = [
            grade_row(i, run, "final", "readback_test", s, 0.8, method_kind(i, share * 0.9, s), rng)
            for i in test_items
            for s in range(16)
        ]
        grades_file(run, "readback_test__final__greedy", greedy, index)
        grades_file(run, "readback_test__final__n16", sampled, index)
        per_method[code] = greedy
        write_lines(
            OUT / "results" / "runs" / run / "completions" / "readback_test__final__greedy.jsonl",
            [
                {
                    "item_id": g.item_id,
                    "request": run,
                    "checkpoint": "final",
                    "sample": 0,
                    "temperature": 0.0,
                    "tokens": g.tokens,
                    "text": answer_text(
                        rb[g.item_id],
                        "right" if g.confirmed else ("refuse" if g.refused else ("literal" if g.original else "wrong")),
                    ),
                }
                for g in greedy
            ],
        )
    floor = [
        grade_row(i, "floor", "readback-floor@fixture", "readback_test", 0, 0.0, method_kind(i, 0.4, 0), rng)
        for i in test_items
    ]
    grades_file("floor", "readback_test__floor__greedy", floor, index)
    per_method["floor"] = floor
    write_json(OUT / "results" / "grades" / "index.json", dict(sorted(index.items())))

    # The fixture manifest: intervals and the paired difference from the repository's own bootstrap.
    accuracy = []
    values = {
        code: {s.item_id: (s.cluster, s.confirmed / s.n) for s in by_item(rows) if not s.refuse_item and s.n}
        for code, rows in per_method.items()
    }
    for code, vals in values.items():
        iv = bootstrap(list(vals.values()))
        accuracy.append(
            [code, "readback_test", "twin_confirmed_accuracy", iv.point, iv.low, iv.high, iv.items, iv.clusters]
        )
    diff = paired(values["M4"], values["M0"])
    manifest = {
        "as_of": AS_OF,
        "values": {"methods.champion": {"value": "M4", "fmt": "text", "provenance": provenance("the promotion gates")}},
        "tables": {
            "methods.accuracy": {
                "columns": ["method", "set", "metric", "point", "low", "high", "items", "clusters"],
                "formats": ["text", "text", "text", "pct1", "pct1", "pct1", "int", "int"],
                "rows": accuracy,
                "provenance": provenance("readback_test"),
            },
            "hacks.outcomes": {
                "columns": ["arm", "share_of_gain", "held"],
                "formats": ["text", "pct1", "bool"],
                "rows": [["random", 0.12, True], ["format", 0.61, False]],
                "provenance": provenance("the controls"),
            },
        },
    }
    if diff is not None:
        manifest["tables"]["methods.differences"] = {
            "columns": ["a", "b", "set", "metric", "point", "low", "high", "clusters"],
            "formats": ["text", "text", "text", "text", "pts1", "pts1", "pts1", "int"],
            "rows": [
                ["M4", "M0", "readback_test", "twin_confirmed_accuracy", diff.point, diff.low, diff.high, diff.clusters]
            ],
            "provenance": provenance("readback_test"),
        }
    write_json(OUT / "results" / "manifest.json", manifest)
    where = OUT.relative_to(ROOT)
    print(f"fixtures: {len(ARMS)} defect study runs, {len(METHODS)} methods, {len(index)} graded files into {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
