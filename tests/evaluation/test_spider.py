"""The Spider transfer check: the prompts are built from the schema alone, the data asset is checked
by digest and unpacked safely, and the test suite evaluation marks an answer right only when it
matches the gold on the original database and on every test suite database of that schema."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from goodhart_evaluation import spider

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

TABLES = [
    {
        "db_id": "toy",
        "table_names_original": ["singer", "concert"],
        "column_names_original": [[-1, "*"], [0, "singer_id"], [0, "name"], [0, "age"], [1, "id"], [1, "singer_id"]],
        "column_types": ["text", "number", "text", "number", "number", "number"],
        "primary_keys": [1, 4],
        "foreign_keys": [[5, 1]],
    }
]
DEV = [
    {"db_id": "toy", "question": "Which singers are over 30?", "query": "SELECT name FROM singer WHERE age > 30"},
    {"db_id": "toy", "question": "How many singers are there?", "query": "SELECT count(*) FROM singer"},
]


def _database(path: Path, rows: list[tuple[int, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    con.execute("CREATE TABLE concert (id INTEGER PRIMARY KEY, singer_id INTEGER)")
    con.executemany("INSERT INTO singer VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()


def _files() -> dict[str, bytes]:
    import tempfile

    out = {"spider/dev.json": json.dumps(DEV).encode(), "spider/tables.json": json.dumps(TABLES).encode()}
    with tempfile.TemporaryDirectory() as tmp:
        # The original has one singer over 30; the test suite database has two, so an answer that
        # names the one singer is right on the original only.
        for name, rows in {
            "toy.sqlite": [(1, "Ann", 41), (2, "Bea", 25)],
            "toy_1.sqlite": [(1, "Ann", 41), (2, "Bea", 25), (3, "Cid", 33)],
        }.items():
            _database(Path(tmp) / name, rows)
            out[f"database/toy/{name}"] = (Path(tmp) / name).read_bytes()
    return out


@pytest.fixture
def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, bytes]:
    """A repository root whose lock names a published asset, and that asset's bytes."""
    from fetch_spider import sha256, tarball

    body = tarball(_files())
    lock = json.loads((ROOT / spider.LOCK).read_text())
    lock["asset"] = {"release": "data", "name": "spider_dev.tar.gz", "sha256": sha256(body), "test_suite": True}
    (tmp_path / "data").mkdir()
    (tmp_path / spider.LOCK).write_text(json.dumps(lock))
    monkeypatch.setenv("GOODHART_CACHE", str(tmp_path / "cache"))
    return tmp_path, body


def test_the_schema_names_every_column_its_type_the_keys_and_the_foreign_keys() -> None:
    text = spider.schema(TABLES[0])
    assert text.splitlines()[0] == "Tables (SQLite, read only, database toy):"
    assert "- singer: singer_id NUMBER primary key, name TEXT, age NUMBER" in text
    assert text.endswith("Foreign keys: concert.singer_id = singer.singer_id")


def test_the_prompts_are_zero_shot_in_the_trained_answer_format() -> None:
    rows = spider.prompts(DEV, {t["db_id"]: t for t in TABLES})
    assert [r["id"] for r in rows] == ["spider-dev-0000", "spider-dev-0001"]
    system, user = rows[0]["messages"]
    assert system["role"] == "system" and "SQLite" in system["content"] and "```sql" in system["content"]
    assert user["content"].endswith("Question: Which singers are over 30?")
    assert "doorstep" not in system["content"], "the warehouse rules stay out of a public benchmark prompt"


def test_the_asset_is_checked_by_digest_and_unpacked_with_its_prompts(published: tuple[Path, bytes]) -> None:
    root, body = published
    calls: list[str] = []

    def get(url: str) -> bytes:
        calls.append(url)
        return body

    out = spider.fetch_data(root, get)
    assert out["questions"] == 2 and calls[0].endswith("/releases/download/data/spider_dev.tar.gz")
    lines = spider.benchmark(root).read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["spider-dev-0000", "spider-dev-0001"]
    shutil.rmtree(spider.data_dir(root))
    with pytest.raises(ValueError, match="does not match its digest"):
        spider.fetch_data(root, lambda url: body + b"x")


def test_an_asset_with_a_path_outside_its_folders_is_refused(published: tuple[Path, bytes]) -> None:
    from fetch_spider import sha256, tarball

    root, _ = published
    evil = tarball({"spider/dev.json": b"[]", "../outside.txt": b"x"})
    lock = json.loads((root / spider.LOCK).read_text())
    lock["asset"]["sha256"] = sha256(evil)
    (root / spider.LOCK).write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="should not"):
        spider.fetch_data(root, lambda url: evil)


def test_a_lock_with_no_asset_yet_fetches_the_evaluation_only(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    shutil.copy(ROOT / spider.LOCK, tmp_path / spider.LOCK)
    assert spider.fetch_data(tmp_path) == {"data": "not published yet"}
    assert not spider.available(tmp_path)


@pytest.mark.parametrize(
    ("low", "high", "verdict"),
    [(0.01, 0.05, "helped"), (-0.05, -0.01, "hurt"), (-0.02, 0.03, "no measurable change")],
)
def test_the_change_against_the_base_is_read_from_its_interval(low: float, high: float, verdict: str) -> None:
    assert spider.verdict_of(low, high) == verdict


def _evaluator_into(root: Path) -> None:
    """The fetched test suite evaluation, copied into the test's own cache."""
    src = next((ROOT / "data" / "cache" / "spider").glob("*/test_suite"))
    dest = spider.cache(root) / "test_suite"
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.py"):
        shutil.copy(f, dest / f.name)


def _answers(texts: list[str]) -> list[dict[str, Any]]:
    return [{"item_id": f"spider-dev-{i:04d}", "text": t} for i, t in enumerate(texts)]


@pytest.mark.spider
def test_the_test_suite_catches_an_answer_right_on_the_original_only(published: tuple[Path, bytes]) -> None:
    root, body = published
    spider.fetch_data(root, lambda url: body)
    _evaluator_into(root)
    right = "```sql\nSELECT name FROM singer WHERE age > 30\n```"
    lucky = "```sql\nSELECT name FROM singer WHERE name = 'Ann'\n```"
    count = "```sql\nSELECT count(*) FROM singer\n```"
    verdicts = spider.grade_rows(root, _answers([right, count]), DEV)
    assert [v["right"] for v in verdicts] == [True, True]
    verdicts = spider.grade_rows(root, _answers([lucky, "I cannot say."]), DEV)
    assert [v["right"] for v in verdicts] == [False, False]
    assert verdicts[1]["why"] == "no SQL"


@pytest.mark.spider
def test_a_query_that_never_ends_is_stopped_and_marked_wrong(published: tuple[Path, bytes]) -> None:
    root, body = published
    spider.fetch_data(root, lambda url: body)
    _evaluator_into(root)
    forever = "```sql\nWITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c\n```"
    t0 = time.monotonic()
    verdicts = spider.grade_rows(root, _answers([forever]), DEV, timeout_s=0.5)
    assert not verdicts[0]["right"] and time.monotonic() - t0 < 10


@pytest.mark.spider
def test_the_run_grades_each_method_and_its_change_against_the_base(published: tuple[Path, bytes]) -> None:
    root, body = published
    spider.fetch_data(root, lambda url: body)
    _evaluator_into(root)
    right = "```sql\nSELECT name FROM singer WHERE age > 30\n```"
    count = "```sql\nSELECT count(*) FROM singer\n```"
    for run_id, arm, texts in [
        ("2026-11-02-01-sample", "M0", ["no", count]),
        ("2026-11-02-02-sample", "M1", [right, count]),
    ]:
        (root / "requests").mkdir(exist_ok=True)
        (root / "requests" / f"{run_id}.json").write_text(json.dumps({"arm": arm, "stage": "sample"}))
        out = root / "results" / "runs" / run_id / "completions" / "spider_dev__base__greedy.jsonl"
        out.parent.mkdir(parents=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in _answers(texts)))
    result = spider.run(root)
    assert result["by_method"] == {
        "M0": "2026-11-02-01-sample/spider_dev__base__greedy",
        "M1": "2026-11-02-02-sample/spider_dev__base__greedy",
    }
    files = result["files"]
    assert files[result["by_method"]["M0"]]["accuracy"] == 0.5 and files[result["by_method"]["M1"]]["accuracy"] == 1.0
    assert result["changes"]["M1"]["point"] == 0.5 and result["evaluation"]["test_suite"] is True
    assert json.loads((root / spider.RESULTS).read_text())["questions"] == 2


def _zip(files: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as z:
        for name, body in files.items():
            z.writestr(name, body)
    return raw.getvalue()


def test_the_fetch_keeps_only_what_the_dev_evaluation_reads_and_packs_it_the_same_way_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fetch_spider

    toy = _files()
    canonical = _zip(
        {
            "spider_data/dev.json": toy["spider/dev.json"],
            "spider_data/tables.json": toy["spider/tables.json"],
            "spider_data/database/toy/toy.sqlite": toy["database/toy/toy.sqlite"],
            "spider_data/database/train_only/train_only.sqlite": b"not a dev database",
            "__MACOSX/spider_data/._dev.json": b"junk",
        }
    )
    suite = _zip(
        {
            "database/toy/toy.sqlite": toy["database/toy/toy.sqlite"],
            "database/toy/toy_1.sqlite": toy["database/toy/toy_1.sqlite"],
            "database/geo/geo.sqlite": b"another task",
        }
    )
    monkeypatch.setattr(fetch_spider, "SPIDER_SHA256", fetch_spider.sha256(canonical))
    monkeypatch.setattr(fetch_spider, "_http", lambda url: canonical)
    monkeypatch.setattr(fetch_spider, "_drive", lambda file_id, dest: suite)
    monkeypatch.setattr(fetch_spider, "LOCK", tmp_path / "no-lock.json")
    first = fetch_spider.build(tmp_path / "a")
    second = fetch_spider.build(tmp_path / "b")
    assert first["asset"]["sha256"] == second["asset"]["sha256"], "the same inputs give the same bytes"
    assert sorted(first["files"]) == [
        "database/toy/toy.sqlite",
        "database/toy/toy_1.sqlite",
        "spider/dev.json",
        "spider/tables.json",
    ]
    assert first["db_ids"] == ["toy"] and first["sources"]["test_suite"]["recorded_now"] is True
    lock = {"sources": {"test_suite": {"sha256": "0" * 64}}}
    (tmp_path / "no-lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="the lock records"):
        fetch_spider.build(tmp_path / "c")


def test_a_spider_zip_that_differs_from_the_published_digest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fetch_spider

    monkeypatch.setattr(fetch_spider, "_http", lambda url: b"something else")
    with pytest.raises(ValueError, match="not the published"):
        fetch_spider.spider_zip(tmp_path)


def test_the_regression_suite_reports_each_method_against_the_base(tmp_path: Path) -> None:
    from goodhart_evaluation import regression

    grades = tmp_path / "results" / "grades"
    (tmp_path / "requests").mkdir()
    for run_id, arm, strict, right in [
        ("2026-11-02-01-sample", "M0", [True, True, False, False], [True, False]),
        ("2026-11-02-02-sample", "M1", [True, False, False, False], [True, True]),
    ]:
        (tmp_path / "requests" / f"{run_id}.json").write_text(json.dumps({"arm": arm, "stage": "sample"}))
        (tmp_path / "results" / "runs" / run_id).mkdir(parents=True)
        (tmp_path / "results" / "runs" / run_id / "provenance.json").write_text("{}")
        ife = [{"item_id": f"ifeval-{i}", "strict": s} for i, s in enumerate(strict)]
        spi = [{"item_id": f"spider-dev-{i:04d}", "db_id": f"db{i}", "right": r} for i, r in enumerate(right)]
        for kind, name, rows in [
            ("ifeval", "ifeval__final__greedy", ife),
            ("spider", "spider_dev__final__greedy", spi),
        ]:
            path = grades / kind / run_id / f"{name}.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(rows))
    out = regression.run(tmp_path)
    assert out["ifeval"]["M0"]["accuracy"] == 0.5 and out["ifeval"]["M1"]["change"]["point"] == -0.25
    assert out["spider"]["M1"]["change"]["point"] == 0.5 and "change" not in out["spider"]["M0"]
    assert out["ifeval"]["M1"]["change"]["verdict"] in {"hurt", "no measurable change"}
    assert json.loads((tmp_path / regression.RESULTS).read_text())["metric"]["spider"].startswith("test suite")
