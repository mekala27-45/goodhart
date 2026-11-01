"""Record what GitHub says about the repository: its description, topics, license and homepage.

    python scripts/repo_facts.py     # writes results/repo.json through the gh command line

Checklist line eighteen asks for the repository described and its topics set; those live on GitHub,
not in the tree, so this reads them once and the manifest reads the file. It needs only read access.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "repo.json"
FIELDS = "{description, homepage, topics, license: .license.spdx_id, visibility}"


def main() -> int:
    done = subprocess.run(
        ["gh", "api", "repos/{owner}/{repo}", "--jq", FIELDS], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        print(done.stderr.strip(), file=sys.stderr)
        return 1
    facts = json.loads(done.stdout)
    OUT.write_text(json.dumps(facts, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"repo facts: {len(facts.get('topics') or [])} topics, license {facts.get('license')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
