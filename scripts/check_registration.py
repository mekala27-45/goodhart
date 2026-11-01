"""The registration gate: the plan and everything it names, as registered, or a dated decision.

    python scripts/check_registration.py             # check
    python scripts/check_registration.py --register  # write evaluation/registration.json, once,
                                                     # before the first test question is scored

Registered: the plan, the predictions, the hypotheses, the monitor thresholds and detectors, the
comparator and the rewards, the generator and twin configs, the training defaults, the prompt registry
and the digest of the question manifest. A change to any of them after registration fails this gate
unless DECISIONS.md has an entry dated on or after the registration that names the first twelve
characters of the new hash. The gate refuses when nothing is registered.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from goodhart_core.hashing import sha256_text
from goodhart_questions.prompt import registry_digest

REGISTERED = {
    "plan_hash": "evaluation/plan.yaml",
    "predictions_hash": "evaluation/predictions.yaml",
    "hypotheses_hash": "evaluation/hypotheses.yaml",
    "monitor_hash": "config/monitor.yaml",
    "detectors_hash": "packages/monitor/src/goodhart_monitor/detectors.py",
    "comparator_hash": "packages/rewards/src/goodhart_rewards/compare.py",
    "rewards_hash": "packages/rewards/src/goodhart_rewards/rewards.py",
    "generator_hash": "config/generator.yaml",
    "twins_hash": "config/twins.yaml",
    "training_hash": "config/training.yaml",
    "questions_hash": "data/questions/manifest.json",
}
REGISTRATION = Path("evaluation/registration.json")


def current(root: Path = ROOT) -> dict[str, str]:
    out = {key: sha256_text((root / rel).read_text(encoding="utf-8")) for key, rel in REGISTERED.items()}
    out["prompt_hash"] = registry_digest()
    return out


ENTRY = re.compile(r"^## (\d{4}-\d{2}-\d{2})\b", re.M)


def dated_entries(text: str) -> list[tuple[dt.date, str]]:
    """(date, body) for every '## YYYY-MM-DD ...' entry in DECISIONS.md."""
    marks = list(ENTRY.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((dt.date.fromisoformat(m.group(1)), text[m.start() : end]))
    return out


def check(root: Path = ROOT) -> list[str]:
    path = root / REGISTRATION
    if not path.exists():
        return ["nothing is registered: evaluation/registration.json does not exist"]
    registered = json.loads(path.read_text(encoding="utf-8"))
    # DECISIONS.md entries carry the author's local date, so the registration's local date is the bar.
    since = dt.date.fromisoformat(str(registered.get("registered_on", str(registered["registered_at"])[:10])))
    now = current(root)
    decisions_path = root / "DECISIONS.md"
    entries = dated_entries(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else []
    named = " ".join(body for day, body in entries if day >= since)
    problems = []
    for key, value in now.items():
        if key not in registered:
            problems.append(f"{key} was not registered")
        elif value != registered[key] and value[:12] not in named:
            problems.append(
                f"{key} is {value[:12]}, registered {registered[key][:12]}, "
                f"and DECISIONS.md has no dated entry naming {value[:12]}"
            )
    return problems


def register(root: Path = ROOT) -> dict[str, object]:
    path = root / REGISTRATION
    if path.exists():
        raise SystemExit("already registered; a change needs a dated entry in DECISIONS.md, not a new registration")
    record: dict[str, object] = {
        **current(root),
        "registered_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "registered_on": dt.date.today().isoformat(),
        "before_first_test_score": not (root / "results" / "grades").exists(),
    }
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        record = register()
        print(
            f"registered plan {str(record['plan_hash'])[:12]}, "
            f"before the first test score: {record['before_first_test_score']}"
        )
        return 0
    problems = check()
    if problems:
        print("registration gate failed:\n" + "\n".join(problems))
        return 1
    print("registration gate: everything the plan names is as registered, or a dated decision names the change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
