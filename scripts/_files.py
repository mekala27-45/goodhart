"""Which files the text gates look at, and the diffs the scans read.

Authored files are everything tracked or trackable except data, third party code, build output and
the exported completions. The completions hold what a model wrote, verbatim, so the prose gates do
not apply to them; the token and identifier scans do.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".mjs",
    ".cjs",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sql",
    ".html",
    ".txt",
    ".ipynb",
    ".j2",
    ".cfg",
    ".ini",
    ".sh",
    ".ps1",
    ".example",
}
TEXT_NAMES = {"Makefile", "Dockerfile.vercel", "LICENSE", "NOTICE", ".gitignore", "pre-push"}
SKIP_PARTS = {
    "node_modules",
    ".next",
    "out",
    ".git",
    ".venv",
    "__pycache__",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "test-results",
    "playwright-report",
}
DATA_ROOTS = {"data", "results"}


def repo_files(root: Path) -> list[Path]:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        candidates = [root / line for line in listed]
    except (OSError, subprocess.CalledProcessError):
        candidates = [p for p in root.rglob("*") if p.is_file()]
    return sorted(p for p in candidates if p.is_file())


def is_text(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES


def authored_files(root: Path, *, include_data: bool = False) -> list[Path]:
    out: list[Path] = []
    for path in repo_files(root):
        rel = path.relative_to(root)
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        if not include_data and rel.parts[0] in DATA_ROOTS:
            continue
        if rel.parts[:3] == ("web", "public", "data"):
            continue
        if path.name in {
            "uv.lock",
            "package-lock.json",
            "LICENSE",
            "requirements-gpu.txt",
            "requirements-vllm.txt",
        }:
            continue
        if is_text(path):
            out.append(path)
    return out


def all_text_files(root: Path) -> list[Path]:
    """Every tracked text file, data included: what the token and identifier scans read."""
    out: list[Path] = []
    for path in repo_files(root):
        rel = path.relative_to(root)
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        if path.name in {"uv.lock", "package-lock.json"}:
            continue
        if is_text(path) or path.suffix in {".jsonl", ".csv"}:
            out.append(path)
    return out


LOCK_FILES = ("uv.lock", "package-lock.json", "requirements-gpu.txt", "requirements-vllm.txt")


def commit_diffs(
    root: Path, revision_range: str | None, exclude: tuple[str, ...] = ()
) -> list[tuple[str, str]]:
    """(commit, added lines) for every commit in the range, or in the whole history when None.

    Only added lines matter: a removed line never reaches a blob that is new in that commit. Lock
    files can be excluded: their package versions look like addresses to a pattern and hold no prose.
    """
    args = ["git", "log", "-p", "--no-color", "--no-ext-diff", "--format=@@commit %H"]
    args.append(revision_range if revision_range else "--all")
    if exclude:
        args += ["--", ".", *(f":(exclude,glob)**/{name}" for name in exclude)]
    try:
        text = subprocess.run(
            args, cwd=root, check=True, capture_output=True, text=True, errors="replace"
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    out: list[tuple[str, str]] = []
    commit = ""
    added: list[str] = []
    for line in text.splitlines():
        if line.startswith("@@commit "):
            if commit:
                out.append((commit, "\n".join(added)))
            commit = line.split()[1]
            added = []
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    if commit:
        out.append((commit, "\n".join(added)))
    return out
