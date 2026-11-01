"""goodhart spider: the transfer check on a public benchmark, Spider dev under the test suite evaluation.

Spider (Yu et al., 2018; CC BY-SA 4.0) is 1,034 development questions over 20 SQLite databases that no
training or held-out item here touches. Training on three warehouses can help a model on a public
benchmark or hurt it, and this measures which, in both directions. Spider is very likely in the
pretraining data of every candidate, so it is a transfer check and never a headline.

Grading is the test suite evaluation (Zhong et al., 2020; Apache 2.0): an answer is right when its
result equals the gold's on the original database and on every distilled test suite database of that
schema, with bag semantics, and with order only when the gold orders. Its two files are fetched by
digest at one commit and imported as written, with one change: the connection is opened read only
and a query is stopped after TIMEOUT_S by SQLite's progress handler, because the original's asyncio
timeout cannot interrupt a query that is already running.

    goodhart spider --fetch    the evaluation's two files, and once the lock names it the data asset
                               (dev.json, tables.json, the databases) with the prompts beside it
    goodhart spider            grade every exported Spider answer into results/spider.json

The prompts are zero shot: the schema with its foreign keys and the question, in the answer format
the models were trained to (one fenced SQL block). Neither the data nor the prompts go into git; a
sample request reads the prompts from the cache by digest, like IFEval's.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import sys
import tarfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from goodhart_core import paths
from goodhart_rewards.parse import parse_completion

from goodhart_evaluation.stats import bootstrap, paired

LOCK = Path("data/spider.lock.json")
RAW = "https://raw.githubusercontent.com"
REPOSITORY = "mekala27-45/goodhart"
ASSET_URL = "https://github.com/{repo}/releases/download/{release}/{name}"
EXPORT = re.compile(r"^spider_dev__(?P<checkpoint>.+)__greedy$")
TIMEOUT_S = 30.0
ID_PREFIX = "spider-dev-"
RESULTS = Path("results/spider.json")

SYSTEM = (
    "You write one SQLite query that answers a question about one database.\n\n"
    "Rules:\n"
    "- Answer with one SQL query in a fenced block that starts with ```sql, and nothing after the block.\n"
    "- Use only the tables and columns listed, spelled as listed.\n"
    "- Return only the columns the question asks for, one row per answer."
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lock(root: Path) -> dict[str, Any]:
    return dict(json.loads((root / LOCK).read_text(encoding="utf-8")))


def cache(root: Path) -> Path:
    return Path(paths.cache_dir()) / "spider" / str(lock(root)["evaluation"]["commit"])[:12]


def data_dir(root: Path) -> Path:
    asset = lock(root).get("asset")
    if not asset:
        raise FileNotFoundError(f"{LOCK} names no data asset yet; run the spider workflow first")
    return Path(paths.cache_dir()) / "spider" / f"data-{str(asset['sha256'])[:12]}"


def benchmark(root: Path) -> Path:
    """The prompts as chat messages, written by fetch beside the data; a sample request reads them."""
    return data_dir(root) / "spider_dev.jsonl"


def _get(url: str) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=600.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def fetch_evaluator(root: Path, get: Callable[[str], bytes] = _get) -> Path:
    ev = lock(root)["evaluation"]
    base = cache(root) / "test_suite"
    base.mkdir(parents=True, exist_ok=True)
    repo = str(ev["repository"]).removeprefix("https://github.com/")
    for name, digest in ev["files"].items():
        target = base / name
        if target.exists() and _sha256(target.read_bytes()) == digest:
            continue
        body = get(f"{RAW}/{repo}/{ev['commit']}/{name}")
        if _sha256(body) != digest:
            raise ValueError(f"{name} of the test suite evaluation does not match its digest in {LOCK}")
        target.write_bytes(body)
    return base


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    out = []
    for m in tar.getmembers():
        p = PurePosixPath(m.name)
        if not m.isfile() or p.is_absolute() or ".." in p.parts or p.parts[0] not in {"spider", "database"}:
            raise ValueError(f"the Spider asset holds an entry it should not: {m.name}")
        out.append(m)
    return out


def fetch_data(root: Path, get: Callable[[str], bytes] = _get) -> dict[str, Any]:
    asset = lock(root).get("asset")
    if not asset:
        return {"data": "not published yet"}
    base = data_dir(root)
    if not (base / "spider" / "dev.json").exists():
        url = ASSET_URL.format(repo=REPOSITORY, release=asset["release"], name=asset["name"])
        body = get(url)
        if _sha256(body) != asset["sha256"]:
            raise ValueError(f"{asset['name']} does not match its digest in {LOCK}")
        base.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
            for m in _safe_members(tar):
                target = base / m.name
                target.parent.mkdir(parents=True, exist_ok=True)
                member = tar.extractfile(m)
                if member is None:
                    raise ValueError(f"the Spider asset entry {m.name} has no content")
                target.write_bytes(member.read())
    dev, tables = load(root)
    rows = prompts(dev, tables)
    with benchmark(root).open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, sort_keys=True) + "\n")
    return {"questions": len(dev), "databases": len(tables), "benchmark": str(benchmark(root))}


def fetch(root: Path, get: Callable[[str], bytes] = _get) -> dict[str, Any]:
    evaluator = fetch_evaluator(root, get)
    return {"evaluator": str(evaluator), **fetch_data(root, get)}


def load(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    base = data_dir(root) / "spider"
    dev = json.loads((base / "dev.json").read_text(encoding="utf-8"))
    tables = {t["db_id"]: t for t in json.loads((base / "tables.json").read_text(encoding="utf-8"))}
    return dev, tables


def schema(entry: dict[str, Any]) -> str:
    """One line per table with its columns as SQLite has them, then the foreign keys."""
    names = entry["table_names_original"]
    columns: list[list[str]] = [[] for _ in names]
    types = entry["column_types"]
    primary = set(entry.get("primary_keys", []))
    flat = entry["column_names_original"]
    for i, (table, column) in enumerate(flat):
        if table < 0:
            continue
        mark = " primary key" if i in primary else ""
        columns[table].append(f"{column} {str(types[i]).upper()}{mark}")
    lines = [f"Tables (SQLite, read only, database {entry['db_id']}):"]
    lines += [f"- {name}: {', '.join(cols)}" for name, cols in zip(names, columns, strict=True)]
    keys = []
    for a, b in entry.get("foreign_keys", []):
        (ta, ca), (tb, cb) = flat[a], flat[b]
        keys.append(f"{names[ta]}.{ca} = {names[tb]}.{cb}")
    if keys:
        lines.append("Foreign keys: " + "; ".join(keys))
    return "\n".join(lines)


def prompts(dev: Sequence[dict[str, Any]], tables: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, q in enumerate(dev):
        user = f"{schema(tables[q['db_id']])}\n\nQuestion: {q['question']}"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        out.append({"id": f"{ID_PREFIX}{i:04d}", "messages": messages})
    return out


def evaluator(root: Path, timeout_s: float = TIMEOUT_S) -> Any:
    """The test suite's exec_eval module from the cache, with a read only connection and a stop."""
    base = cache(root) / "test_suite"
    if not (base / "exec_eval.py").exists():
        raise FileNotFoundError("the test suite evaluation is not in the cache; run goodhart spider --fetch")
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    import importlib
    import warnings

    with warnings.catch_warnings():
        # parse.py has regular expressions in plain strings, which Python now warns about.
        warnings.simplefilter("ignore", SyntaxWarning)
        module: Any = importlib.import_module("exec_eval")

    def cursor(sqlite_path: str) -> sqlite3.Cursor:
        connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        connection.text_factory = lambda b: b.decode(errors="ignore")
        deadline = time.monotonic() + timeout_s
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        return connection.cursor()

    module.get_cursor_from_path = cursor
    return module


def grade_rows(
    root: Path, rows: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]], timeout_s: float = TIMEOUT_S
) -> list[dict[str, Any]]:
    """One verdict per exported answer: right on every database of its schema, or why not."""
    module = evaluator(root, timeout_s)
    databases = data_dir(root) / "database"
    out = []
    for row in sorted(rows, key=lambda r: str(r["item_id"])):
        index = int(str(row["item_id"]).removeprefix(ID_PREFIX))
        q = dev[index]
        db = str(q["db_id"])
        parsed = parse_completion(str(row.get("text") or ""))
        verdict: dict[str, Any] = {"item_id": row["item_id"], "db_id": db, "right": False, "why": ""}
        if parsed.kind != "sql" or not parsed.sql:
            verdict["why"] = "no SQL"
        else:
            try:
                path = databases / db / f"{db}.sqlite"
                right = module.eval_exec_match(
                    db=str(path),
                    p_str=parsed.sql,
                    g_str=str(q["query"]),
                    plug_value=False,
                    keep_distinct=False,
                    progress_bar_for_each_datapoint=False,
                )
                verdict["right"] = bool(right)
                verdict["why"] = "" if right else "a database where the result differs, or an error"
            except AssertionError:
                verdict["why"] = "the gold did not run"
                verdict["gold_error"] = True
            except Exception as err:  # a query the parser behind remove_distinct cannot read
                verdict["why"] = f"{type(err).__name__}"
        out.append(verdict)
    return out


def exports(root: Path) -> list[tuple[str, str, str]]:
    """(run, file stem, checkpoint) of every exported Spider answer file."""
    found = []
    for path in sorted((root / "results" / "runs").glob("*/completions/spider_dev__*__greedy.jsonl")):
        m = EXPORT.match(path.stem)
        if m:
            found.append((path.parents[1].name, path.stem, m.group("checkpoint")))
    return found


def _method(root: Path, run: str) -> str | None:
    req_path = root / "requests" / f"{run}.json"
    if not req_path.exists():
        return None
    arm = str(json.loads(req_path.read_text(encoding="utf-8")).get("arm") or "")
    return arm or None


def _values(verdicts: Sequence[dict[str, Any]]) -> dict[str, tuple[str, float]]:
    return {str(v["item_id"]): (str(v["db_id"]), float(v["right"])) for v in verdicts if not v.get("gold_error")}


def verdict_of(low: float, high: float) -> str:
    """The message rule for a change against the base: the interval decides, never the point."""
    if low > 0.0:
        return "helped"
    if high < 0.0:
        return "hurt"
    return "no measurable change"


def run(root: Path, timeout_s: float = TIMEOUT_S) -> dict[str, Any]:
    dev, _ = load(root)
    grades_dir = root / "results" / "grades" / "spider"
    files: dict[str, Any] = {}
    per_item: dict[str, dict[str, tuple[str, float]]] = {}
    for run_id, stem, checkpoint in exports(root):
        path = root / "results" / "runs" / run_id / "completions" / f"{stem}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        verdicts = grade_rows(root, rows, dev, timeout_s)
        out = grades_dir / run_id / f"{stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdicts, indent=1) + "\n", encoding="utf-8")
        values = _values(verdicts)
        key = f"{run_id}/{stem}"
        per_item[key] = values
        iv = bootstrap(list(values.values()))
        files[key] = {
            "run": run_id,
            "checkpoint": checkpoint,
            "method": _method(root, run_id),
            "accuracy": iv.point,
            "low": iv.low,
            "high": iv.high,
            "items": iv.items,
            "clusters": iv.clusters,
            "no_sql": sum(1 for v in verdicts if v["why"] == "no SQL"),
            "gold_errors": sum(1 for v in verdicts if v.get("gold_error")),
        }
    # Each method's latest file against the base's, item by item, clusters by database.
    latest: dict[str, str] = {}
    for key, f in sorted(files.items()):
        if f["method"]:
            latest[str(f["method"])] = key
    changes: dict[str, Any] = {}
    base = latest.get("M0")
    for method, key in sorted(latest.items()):
        if base is None or method == "M0":
            continue
        d = paired(per_item[key], per_item[base])
        changes[method] = {
            "against": "M0",
            "point": d.point,
            "low": d.low,
            "high": d.high,
            "items": d.items,
            "verdict": verdict_of(d.low, d.high),
        }
    result = {
        "evaluation": {
            "kind": "test suite execution accuracy with values (no plug_value, no keep_distinct)",
            "timeout_seconds_per_query": timeout_s,
            "commit": lock(root)["evaluation"]["commit"],
            "test_suite": bool((lock(root).get("asset") or {}).get("test_suite", False)),
        },
        "questions": len(dev),
        "files": files,
        "by_method": latest,
        "changes": changes,
    }
    (root / RESULTS).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return result


def available(root: Path) -> bool:
    """Whether the data asset is published and fetched, for the commands that only read it."""
    try:
        return (data_dir(root) / "spider" / "dev.json").exists()
    except FileNotFoundError:
        return False
