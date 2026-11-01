"""The skills gate: every folder under .claude/skills is a valid Agent Skill (skills-ref), and the three
the repository promises are there.

    python scripts/check_skills.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from skills_ref import validate

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("request-a-gpu-run", "add-a-reward", "add-a-question-family")


def check(root: Path = ROOT) -> list[str]:
    folder = root / ".claude" / "skills"
    found = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.exists() else []
    problems = [f"{name}: missing" for name in REQUIRED if not (folder / name).is_dir()]
    for skill in found:
        problems += [f"{skill.name}: {error}" for error in validate(skill)]
    if not found:
        problems.append("no skill folder under .claude/skills")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("skills gate failed:\n" + "\n".join(problems))
        return 1
    print(f"skills gate: {len(REQUIRED)} required skills present and valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
