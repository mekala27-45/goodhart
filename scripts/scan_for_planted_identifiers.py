"""Fail on personal identifiers in any tracked file or commit diff: emails, phone numbers, IP addresses.

The warehouses hold no person, and nothing goodhart writes should either. The friction build found a
raw address in an old commit that only a history rewrite could remove; this scan reads every commit
diff so that cannot happen here. A planted identifier in tests/fixtures proves the scan reads.

A dotted quad written as a package version (after a pin such as ==, or between a wheel's name and its
tags) is a version, not an address: the CUDA wheels carry four part versions, and a kernel's frozen
package list holds dozens of them. The collector runs redact() over a kernel's captured logs before
anything is committed, and the gpu workflow runs this scan over the collected run with --paths before
it pushes, so an identifier in a kernel's output never reaches a branch.

    python scripts/scan_for_planted_identifiers.py                # files and the whole history
    python scripts/scan_for_planted_identifiers.py --range A..B   # files and the commits in A..B
    python scripts/scan_for_planted_identifiers.py --paths DIR    # the text files under DIR only
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import sys
from pathlib import Path

from _files import LOCK_FILES, all_text_files, commit_diffs, text_files_under

EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<![\d.])\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b(?![\d.])")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")

ALLOWED_EMAILS = frozenset(
    {
        "noreply@anthropic.com",
        "noreply@github.com",
        "41898282+github-actions[bot]@users.noreply.github.com",
        "goodhart-gpu@users.noreply.github.com",
    }
)
ALLOWED_EMAIL_DOMAINS = ("users.noreply.github.com", "example.com", "example.org")
FIXTURE = Path("tests/fixtures/planted_identifier.txt")
PINS = frozenset({"==", ">=", "<=", "~=", "!="})


def _ip_is_identifier(text: str) -> bool:
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_unspecified or ip.is_private or ip.is_reserved or ip.is_link_local)


def _is_version(text: str, start: int, end: int) -> bool:
    """nvidia-cublas==13.1.1.3, or nvidia_cublas-13.1.1.3-py3-none-any.whl: a version, not an address."""
    before, after = text[max(0, start - 2) : start], text[end : end + 1]
    if before in PINS:
        return True
    return len(before) == 2 and before[1] == "-" and (before[0].isalnum() or before[0] == "_") and after in {"-", "+"}


def _spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, kind) of every identifier in the text."""
    spans: list[tuple[int, int, str]] = []
    for m in EMAIL.finditer(text):
        email = m.group(0).lower()
        if email in ALLOWED_EMAILS or email.endswith(ALLOWED_EMAIL_DOMAINS):
            continue
        if email.split("@", 1)[1].count(".") == 0:
            continue
        spans.append((m.start(), m.end(), "email"))
    spans.extend((m.start(), m.end(), "phone number") for m in PHONE.finditer(text))
    spans.extend(
        (m.start(), m.end(), "IP address")
        for m in IPV4.finditer(text)
        if _ip_is_identifier(m.group(0)) and not _is_version(text, m.start(), m.end())
    )
    return spans


def find(text: str) -> list[str]:
    return [kind for _, _, kind in _spans(text)]


def redact(text: str) -> tuple[str, int]:
    """The text with every identifier replaced by its kind in brackets, and how many were replaced."""
    out, last, n = [], 0, 0
    for start, end, kind in sorted(_spans(text)):
        if start < last:
            continue
        out += [text[last:start], f"[{kind} removed]"]
        last, n = end, n + 1
    out.append(text[last:])
    return "".join(out), n


def check(
    root: Path, revision_range: str | None = None, files: list[Path] | None = None, history: bool = True
) -> tuple[int, int]:
    targets = files if files is not None else [p for p in all_text_files(root) if p.relative_to(root) != FIXTURE]
    if not targets:
        raise ValueError("the identifier scan found no files to read")
    problems: list[str] = []
    for path in targets:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name in find(text):
            problems.append(f"{path.relative_to(root)}: {name}")
    diffs = commit_diffs(root, revision_range, exclude=LOCK_FILES) if history else []
    fixture_text = (root / FIXTURE).read_text(encoding="utf-8") if (root / FIXTURE).exists() else ""
    for commit, added in diffs:
        cleaned = added.replace(fixture_text.strip(), "") if fixture_text.strip() else added
        for name in find(cleaned):
            problems.append(f"commit {commit[:12]}: {name}")
    if problems:
        raise ValueError("identifier scan failed:\n" + "\n".join(sorted(set(problems))[:50]))
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
            print(f"identifier scan: no text file under {', '.join(map(str, args.paths))}")
            return 0
        n_files, _ = check(root, files=targets, history=False)
        print(f"identifier scan: {n_files} files, no identifier found")
        return 0
    n_files, n_commits = check(root, args.revision_range)
    print(f"identifier scan: {n_files} files and {n_commits} commit diffs, no identifier found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
