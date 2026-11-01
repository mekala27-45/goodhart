"""Write the data digests into a request, so the request names every file its stage will read.

    python scripts/new_request.py requests/2026-11-01-02-select.json

The request is a committed file; the gpu workflow refuses one whose declared digests differ from the
files at its commit. Run this after writing the stage, the model, the reward, the seed, the
hyperparameters, the step budget and the ceiling by hand, and commit the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from build_kernel import ROOT, stage_files
from goodhart_train.requests import dump, load


def fill(path: Path, root: Path = ROOT) -> dict[str, str]:
    req = load(path)
    _, sources = stage_files(req, root)
    filled = req.model_copy(update={"data": dict(sorted(sources.items()))})
    path.write_text(dump(filled), encoding="utf-8")
    return sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.requests:
        sources = fill(path)
        print(f"{path.name}: {len(sources)} data files declared")
        for name, digest in sorted(sources.items()):
            print(f"  {digest[:12]}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
