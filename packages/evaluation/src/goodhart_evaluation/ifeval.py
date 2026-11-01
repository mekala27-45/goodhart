"""goodhart ifeval: the instruction following regression, with Google's reference checker.

IFEval (Zhou et al., 2023) is 541 prompts that each carry verifiable instructions (no commas, at
least 300 words, a title in double angular brackets). A post-trained SQL model can lose the general
ability to follow such instructions, and a promotion gate holds a champion to no more than two
points below the base. The prompts and the checker are Google's own, from google-research at one
pinned commit, fetched by sha256 into the cache and never vendored; the sentence tokenizer the
checker loads (nltk's punkt) is fetched the same way, and nltk is pinned at 3.8.1, the last release
that loads punkt as the checker asks.

    goodhart ifeval --fetch    the checker, the prompts and punkt into the cache, and the prompts as
                               chat messages beside them (ifeval.jsonl, which a sample request reads)
    goodhart ifeval            grade every exported IFEval answer

Neither the prompts nor the answers go into git. One prompt asks the model to end a cover letter
with a phone number, and an answer to a prompt like that may invent a phone number or an email of
its own; an answer is graded as written, so it cannot be redacted, and the identifier scan would
rightly refuse it. The collector sends IFEval answers to the runs release with their digests in the
run's assets.json, and grading fetches them from there into the cache, checked by digest.

A response is graded strict (the response as written) and loose (the checker's upper bound, which
also tries it without its first or last line and without asterisks), each per prompt (every
instruction followed) and per instruction. results/ifeval.json holds the four accuracies per file
and results/grades/ifeval/<run>/<file>.json the per prompt verdicts, for the interval of a change.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import httpx
from goodhart_core import paths

LOCK = Path("data/ifeval.lock.json")
RAW = "https://raw.githubusercontent.com"
REPOSITORY = "mekala27-45/goodhart"
ASSET = "https://github.com/{repo}/releases/download/{release}/{name}"
EXPORT_PREFIX = "completions/ifeval__"
CHECKER = ("instructions.py", "instructions_registry.py", "instructions_util.py", "evaluation_lib.py")
INPUT = "data/input_data.jsonl"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache(root: Path) -> Path:
    lock = json.loads((root / LOCK).read_text(encoding="utf-8"))
    commit = str(lock["google_research"]["commit"])
    return Path(paths.cache_dir()) / "ifeval" / commit[:12]


def benchmark(root: Path) -> Path:
    """The prompts as chat messages, written by fetch beside the checker."""
    return cache(root) / "ifeval.jsonl"


def _get(url: str) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def fetch(root: Path) -> dict[str, Any]:
    """The checker, the prompts and punkt into the cache, each checked against the lock; then the
    prompts as chat messages beside them, which a sample request declares by digest."""
    lock = json.loads((root / LOCK).read_text(encoding="utf-8"))
    gr, punkt = lock["google_research"], lock["punkt"]
    base = cache(root)
    package = base / "instruction_following_eval"
    (package / "data").mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for rel, digest in gr["files"].items():
        target = package / rel
        if target.exists() and _sha256(target.read_bytes()) == digest:
            continue
        body = _get(f"{RAW}/google-research/google-research/{gr['commit']}/instruction_following_eval/{rel}")
        if _sha256(body) != digest:
            raise ValueError(f"instruction_following_eval/{rel} does not match its digest in {LOCK}")
        target.write_bytes(body)
    nltk_dir = base / "nltk_data"
    pickle = nltk_dir / "tokenizers" / "punkt" / "PY3" / "english.pickle"
    if not pickle.exists():
        body = _get(f"{RAW}/nltk/nltk_data/{punkt['commit']}/packages/tokenizers/punkt.zip")
        if _sha256(body) != punkt["sha256"]:
            raise ValueError(f"punkt.zip does not match its digest in {LOCK}")
        zipfile.ZipFile(io.BytesIO(body)).extractall(nltk_dir / "tokenizers")
    rows = [json.loads(line) for line in (package / INPUT).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = benchmark(root)
    with out.open("w", encoding="utf-8") as handle:
        for r in sorted(rows, key=lambda r: int(r["key"])):
            message = {"role": "user", "content": r["prompt"]}
            handle.write(json.dumps({"id": f"ifeval-{r['key']}", "messages": [message]}, sort_keys=True) + "\n")
    return {"prompts": len(rows), "cache": str(base), "benchmark": str(out.relative_to(root.resolve()))}


def checker(root: Path) -> Any:
    """The reference evaluation library, imported from the cache with punkt on nltk's path."""
    base = cache(root)
    if not (base / "instruction_following_eval" / "evaluation_lib.py").exists():
        raise FileNotFoundError("the IFEval checker is not in the cache; run goodhart ifeval --fetch")
    os.environ["NLTK_DATA"] = str(base / "nltk_data")
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    import nltk

    if str(base / "nltk_data") not in nltk.data.path:
        nltk.data.path.insert(0, str(base / "nltk_data"))
    return importlib.import_module("instruction_following_eval.evaluation_lib")


def inputs(root: Path) -> dict[str, Any]:
    lib = checker(root)
    found = lib.read_prompt_list(str(cache(root) / "instruction_following_eval" / INPUT))
    return {f"ifeval-{i.key}": i for i in found}


def grade_responses(root: Path, responses: dict[str, str]) -> list[dict[str, Any]]:
    """One row per prompt: strict and loose, all instructions and each one."""
    lib = checker(root)
    by_id = inputs(root)
    rows = []
    for item_id in sorted(responses, key=lambda k: int(k.split("-")[1])):
        inp = by_id[item_id]
        answer = {inp.prompt: responses[item_id]}
        strict = lib.test_instruction_following_strict(inp, answer)
        loose = lib.test_instruction_following_loose(inp, answer)
        rows.append(
            {
                "item_id": item_id,
                "instructions": list(inp.instruction_id_list),
                "strict": bool(strict.follow_all_instructions),
                "loose": bool(loose.follow_all_instructions),
                "strict_each": [bool(x) for x in strict.follow_instruction_list],
                "loose_each": [bool(x) for x in loose.follow_instruction_list],
            }
        )
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    each_strict = [x for r in rows for x in r["strict_each"]]
    each_loose = [x for r in rows for x in r["loose_each"]]

    def share(xs: list[bool]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "prompts": len(rows),
        "instructions": len(each_strict),
        "prompt_strict": share([r["strict"] for r in rows]),
        "prompt_loose": share([r["loose"] for r in rows]),
        "instruction_strict": share(each_strict),
        "instruction_loose": share(each_loose),
    }


def exported(root: Path) -> list[tuple[str, Path]]:
    """(run, file) for every IFEval answer file: a release asset named in a run's assets.json,
    fetched into the cache and checked by digest, or a file in the run folder (a test's)."""
    out: list[tuple[str, Path]] = []
    runs = root / "results" / "runs"
    for folder in sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []:
        for path in sorted((folder / "completions").glob("ifeval__*.jsonl")):
            out.append((folder.name, path))
        listed = folder / "assets.json"
        assets = json.loads(listed.read_text(encoding="utf-8")) if listed.exists() else {}
        for rel, asset in sorted(assets.items()):
            if not rel.startswith(EXPORT_PREFIX):
                continue
            target = Path(paths.cache_dir()) / "results" / folder.name / rel
            if not (target.exists() and _sha256(target.read_bytes()) == asset["sha256"]):
                url = ASSET.format(repo=REPOSITORY, release=asset.get("release", "runs"), name=asset["asset"])
                body = _get(url)
                if _sha256(body) != asset["sha256"]:
                    raise ValueError(f"{folder.name}/{rel}: the release asset does not match its digest")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            out.append((folder.name, target))
    return out


def run(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run_id, path in exported(root):
        responses = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if int(row.get("sample", 0)) == 0:
                    responses[str(row["item_id"])] = str(row["text"])
        rows = grade_responses(root, responses)
        target = root / "results" / "grades" / "ifeval" / run_id / f"{path.stem}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        out[f"{run_id}/{path.stem}"] = summary(rows)
    (root / "results" / "ifeval.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out
