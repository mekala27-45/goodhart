"""Fail on anything that looks like a credential, in any tracked file or in any commit diff.

Actions holds the only Kaggle token; the build machine never sees it. This scan is what keeps that
true for the repository: it reads every tracked text file, then the added lines of every commit in
the range (the whole history in CI, the commits about to leave in the pre-push hook), so a key never
reaches a pushed blob and history never has to be rewritten.

    python scripts/check_tokens.py                    # files and the whole history
    python scripts/check_tokens.py --range A..B       # files and the commits in A..B
    python scripts/check_tokens.py --paths DIR        # the text files under DIR only (the gpu
                                                      # workflow, on a run before it is pushed)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _files import all_text_files, commit_diffs, text_files_under

# Each pattern is assembled from parts so the scan does not match its own source.
PATTERNS: dict[str, re.Pattern[str]] = {
    "Kaggle API token": re.compile("KG" + r"AT_[A-Za-z0-9_\-]{16,}"),
    "Kaggle legacy key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "Hugging Face token": re.compile(r"\b" + "h" + r"f_[A-Za-z0-9]{30,}\b"),
    "NVIDIA API key": re.compile("nv" + r"api-[A-Za-z0-9_\-]{30,}"),
    "Neon password": re.compile(r"\b" + "np" + r"g_[A-Za-z0-9]{12,}\b"),
    "Postgres URL with a password": re.compile(r"postgres(?:ql)?://[^\s:@/'\"]+:[^\s@/'\"]{6,}@[^\s'\"]+"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github" + r"_pat_[A-Za-z0-9_]{30,})\b"),
    "private key": re.compile("-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE " + "KEY-----"),
    "AWS access key": re.compile(r"\b" + "AK" + r"IA[0-9A-Z]{16}\b"),
    "OpenAI style key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
}

# Placeholders the docs and the example env file use on purpose.
ALLOWED = (
    "postgresql://user:password@host/db",
    "postgresql://goodhart:goodhart@localhost",
    "postgresql://postgres:postgres@localhost",
)


def find(text: str) -> list[str]:
    hits: list[str] = []
    for name, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            if any(a in m.group(0) for a in ALLOWED):
                continue
            hits.append(name)
    return hits


def check(
    root: Path, revision_range: str | None = None, files: list[Path] | None = None, history: bool = True
) -> tuple[int, int]:
    targets = files if files is not None else all_text_files(root)
    if not targets:
        raise ValueError("the token scan found no files to read")
    problems: list[str] = []
    for path in targets:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name in find(text):
            problems.append(f"{path.relative_to(root)}: {name}")
    diffs = commit_diffs(root, revision_range) if history else []
    for commit, added in diffs:
        for name in find(added):
            problems.append(f"commit {commit[:12]}: {name}")
    if problems:
        raise ValueError("token scan failed:\n" + "\n".join(sorted(set(problems))[:50]))
    return len(targets), len(diffs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", dest="revision_range", default=None)
    parser.add_argument("--paths", nargs="+", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.paths:
        targets = text_files_under(root, args.paths)
        if not targets:
            print(f"token scan: no text file under {', '.join(map(str, args.paths))}")
            return 0
        n_files, _ = check(root, files=targets, history=False)
        print(f"token scan: {n_files} files, no credential found")
        return 0
    n_files, n_commits = check(root, args.revision_range)
    print(f"token scan: {n_files} files and {n_commits} commit diffs, no credential found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
