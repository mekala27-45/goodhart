"""Build the bundle and the Kaggle kernel folder for one committed request.

The bundle is a tarball with fixed timestamps and sorted entries, so its sha256 is a function of its
contents: the request, the configs, the source lock, the four packages a GPU stage may import
(goodhart_core, goodhart_rewards, goodhart_monitor, goodhart_train) and exactly the data files the
stage reads. It never contains goodhart_heldout, the twins or the gold of a test set; a test unpacks
one and proves the held-out import fails.

    python scripts/build_kernel.py --request requests/2026-10-31-01-probe.json --out build/kernels
    python scripts/build_kernel.py --all --dry-run      # every request, check only

The kernel folder holds run.py (the template with the bundle URL and digest filled in) and
kernel-metadata.json (the accelerator from the probe, internet on, private, the earlier kernels a
resumed run reads).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from goodhart_core.hashing import sha256_bytes
from goodhart_train.requests import Request, load

PACKAGES = ("core", "rewards", "monitor", "train")
REFUSED_IN_BUNDLE = ("goodhart_heldout", "twins", "audit_key")
TEST_SETS = ("test", "template_test", "readback_test", "own", "readback_dev")
PROBE_FIXTURE = 96
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
RELEASE_URL = "https://github.com/{repo}/releases/download/{tag}/{name}"


def stage_files(req: Request, root: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """The data files a stage reads, as archive paths and bytes, and the sha256 of every repository
    file they came from (the digests the request must declare)."""
    q = root / "data" / "questions"
    out: dict[str, bytes] = {}
    sources: dict[str, str] = {}

    def source(path: Path) -> bytes:
        if not path.exists():
            raise FileNotFoundError(f"{req.id}: the stage needs {path.relative_to(root)}, which does not exist")
        body = path.read_bytes()
        sources[path.relative_to(root).as_posix()] = sha256_bytes(body)
        return body

    def add(arc: str, path: Path) -> None:
        out[arc] = source(path)

    def items(split: str, limit: int | None = None) -> None:
        lines = source(q / f"{split}.jsonl").decode("utf-8").splitlines()
        if limit:
            lines = sorted(lines)[:limit]
        out[f"data/items/{split}.jsonl"] = ("\n".join(lines) + "\n").encode()
        ids = {json.loads(line)["id"] for line in lines if line.strip()}
        prompts = [
            line
            for line in source(q / "prompts" / f"{split}.jsonl").decode("utf-8").splitlines()
            if line.strip() and json.loads(line)["id"] in ids
        ]
        out[f"data/prompts/{split}.jsonl"] = ("\n".join(prompts) + "\n").encode()

    hp = req.hyperparameters
    if req.stage == "probe":
        items("train", PROBE_FIXTURE)
    if req.stage in {"select", "sft", "dpo", "grpo"}:
        add("data/prompts/dev.jsonl", q / "prompts" / "dev.jsonl")
    if req.stage in {"select", "sample", "pairs"}:
        add("data/examples.json", q / "examples.json")
    if req.stage == "select" and hp.get("paraphrase"):
        add("data/prompts/train.jsonl", q / "prompts" / "train.jsonl")
    if req.stage == "sft":
        name = str(hp.get("set", "clean"))
        add(f"data/sft/{name}.jsonl", q / "sft" / f"{name}.jsonl")
    if req.stage == "dpo":
        add("data/sft/pairs.jsonl", root / str(hp["pairs_file"]))
    if req.stage == "pairs":
        add("data/prompts/train.jsonl", q / "prompts" / "train.jsonl")
    if req.stage == "grpo":
        items("train")
        prompt_set = str(hp.get("prompt_set", "train"))
        if prompt_set != "train":
            add(f"data/prompts/{prompt_set}.jsonl", q / "prompts" / f"{prompt_set}.jsonl")
        if "learnable_file" in hp:
            add("data/learnable.json", root / str(hp["learnable_file"]))
    if req.stage == "sample":
        for name in [*hp.get("sampled_sets", []), *hp.get("greedy_sets", [])]:
            path = q / "prompts" / f"{name}.jsonl"
            if name == "ifeval":
                # Written into the cache by goodhart ifeval --fetch, never into git (see ifeval.py).
                from goodhart_evaluation.ifeval import benchmark

                path = benchmark(root)
            elif name == "spider_dev":
                # The Spider prompts live in the cache too, built from the data asset by digest.
                from goodhart_evaluation import spider

                if not spider.benchmark(root).exists():
                    spider.fetch(root)
                path = spider.benchmark(root)
            elif not path.exists():
                path = root / "data" / "benchmarks" / f"{name}.jsonl"
            add(f"data/prompts/{name}.jsonl", path)
    if req.stage == "judge":
        for path in sorted((root / "data" / "judge").glob("judge_*.jsonl")):
            add(f"data/prompts/{path.name}", path)
    return out, sources


def check_declared(req: Request, sources: dict[str, str]) -> None:
    """The request must name the digest of every data file its stage reads, and nothing else: a data
    file that changed after the request was written, or a request written by hand without digests,
    stops here and never reaches a GPU."""
    problems = [
        f"{path} is {digest[:12]}, the request declares {req.data.get(path, 'nothing')[:12]}"
        for path, digest in sorted(sources.items())
        if req.data.get(path) != digest
    ]
    problems += [f"{path} is declared but the stage does not read it" for path in sorted(set(req.data) - set(sources))]
    if problems:
        raise ValueError(f"{req.id}: " + "; ".join(problems) + " (fill them with scripts/new_request.py)")


def package_files(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for name in PACKAGES:
        src = root / "packages" / name / "src"
        for path in sorted(src.rglob("*.py")):
            out[f"src/{path.relative_to(src).as_posix()}"] = path.read_bytes()
    return out


def tarball(files: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size = len(files[name])
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(files[name]))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def heldout_ids(root: Path) -> bytes:
    """The id of every held-out item, which travels so a training stage can refuse one; ids carry no gold."""
    ids: set[str] = set()
    for name in TEST_SETS:
        path = root / "data" / "questions" / f"{name}.jsonl"
        if path.exists():
            ids |= {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return (json.dumps(sorted(ids), indent=0) + "\n").encode()


def bundle(req: Request, root: Path = ROOT) -> tuple[bytes, dict[str, str]]:
    staged, sources = stage_files(req, root)
    check_declared(req, sources)
    files: dict[str, bytes] = {
        "request.json": (root / "requests" / f"{req.id}.json").read_bytes(),
        "config/training.yaml": (root / "config" / "training.yaml").read_bytes(),
        "config/monitor.yaml": (root / "config" / "monitor.yaml").read_bytes(),
        "data/sources.lock.json": (root / "data" / "sources.lock.json").read_bytes(),
        "requirements-gpu.txt": (root / "requirements-gpu.txt").read_bytes(),
        "data/heldout_ids.json": heldout_ids(root),
        **package_files(root),
        **staged,
    }
    for name in files:
        if any(bad in name for bad in REFUSED_IN_BUNDLE):
            raise PermissionError(f"{name} may not travel in a kernel bundle")
        if name.startswith("data/items/") and Path(name).stem in TEST_SETS:
            raise PermissionError(f"{name}: test gold may not travel in a kernel bundle")
    data = tarball(files)
    return data, {name: sha256_bytes(body) for name, body in sorted(files.items())}


def slug(req: Request) -> str:
    return "goodhart-" + re.sub(r"[^a-z0-9]+", "-", req.id.lower()).strip("-")


def title(req: Request) -> str:
    return slug(req).replace("-", " ")


def accelerator(req: Request, root: Path) -> str:
    if req.accelerator:
        return req.accelerator
    probe = root / "results" / "gpu_probe.json"
    if probe.exists():
        chosen = json.loads(probe.read_text(encoding="utf-8")).get("accelerator")
        if chosen:
            return str(chosen)
    return DEFAULT_ACCELERATOR


def kernel_folder(req: Request, user: str, repo: str, tag: str, digest: str, out: Path, root: Path = ROOT) -> Path:
    folder = out / req.id
    folder.mkdir(parents=True, exist_ok=True)
    template = (root / "kernels" / "templates" / "run.py").read_text(encoding="utf-8")
    script = (
        template.replace("__REQUEST_ID__", req.id)
        .replace("__BUNDLE_URL__", RELEASE_URL.format(repo=repo, tag=tag, name=f"{req.id}.tar.gz"))
        .replace("__BUNDLE_SHA256__", digest)
    )
    (folder / "run.py").write_text(script, encoding="utf-8")
    metadata: dict[str, Any] = {
        "id": f"{user}/{slug(req)}",
        "title": title(req),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{user}/{slug(load(root / 'requests' / f'{i}.json'))}" for i in req.inputs],
        "model_sources": [],
        "machine_shape": accelerator(req, root),
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1) + "\n", encoding="utf-8")
    return folder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "build")
    parser.add_argument("--user", default="KAGGLE_USER")
    parser.add_argument("--repo", default="mekala27-45/goodhart")
    parser.add_argument("--tag", default="bundles")
    args = parser.parse_args()
    paths = sorted((ROOT / "requests").glob("*.json")) if args.all else [args.request]
    for path in paths:
        req = load(path)
        data, digests = bundle(req)
        digest = sha256_bytes(data)
        if args.dry_run:
            print(f"{req.id}: bundle {len(data) / 1e6:.2f} MB, {len(digests)} files, sha256 {digest[:12]}")
            continue
        (args.out / "bundles").mkdir(parents=True, exist_ok=True)
        (args.out / "bundles" / f"{req.id}.tar.gz").write_bytes(data)
        folder = kernel_folder(req, args.user, args.repo, args.tag, digest, args.out / "kernels")
        (folder / "bundle.json").write_text(
            json.dumps({"sha256": digest, "files": digests}, indent=1) + "\n", encoding="utf-8"
        )
        print(f"{req.id}: {folder} sha256 {digest[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
