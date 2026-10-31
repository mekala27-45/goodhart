"""Fail on personal identifiers in any tracked file or commit diff: emails, phone numbers, IP addresses.

The warehouses hold no person, and nothing goodhart writes should either. The friction build found a
raw address in an old commit that only a history rewrite could remove; this scan reads every commit
diff so that cannot happen here. A planted identifier in tests/fixtures proves the scan reads.

    python scripts/scan_for_planted_identifiers.py                # files and the whole history
    python scripts/scan_for_planted_identifiers.py --range A..B   # files and the commits in A..B
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._files import LOCK_FILES, all_text_files, commit_diffs

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


def _ip_is_identifier(text: str) -> bool:
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_unspecified or ip.is_private or ip.is_reserved or ip.is_link_local)


def find(text: str) -> list[str]:
    hits: list[str] = []
    for m in EMAIL.finditer(text):
        email = m.group(0).lower()
        if email in ALLOWED_EMAILS or email.endswith(ALLOWED_EMAIL_DOMAINS):
            continue
        if email.split("@", 1)[1].count(".") == 0:
            continue
        hits.append("email")
    hits.extend("phone number" for _ in PHONE.finditer(text))
    hits.extend("IP address" for m in IPV4.finditer(text) if _ip_is_identifier(m.group(0)))
    return hits


def check(root: Path, revision_range: str | None = None, files: list[Path] | None = None) -> tuple[int, int]:
    targets = (
        files if files is not None else [p for p in all_text_files(root) if p.relative_to(root) != FIXTURE]
    )
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
    diffs = commit_diffs(root, revision_range, exclude=LOCK_FILES)
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
    args = parser.parse_args()
    n_files, n_commits = check(Path(__file__).resolve().parents[1], args.revision_range)
    print(f"identifier scan: {n_files} files and {n_commits} commit diffs, no identifier found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
