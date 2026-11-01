"""The latency of the live API: the cold start alone, then p50 and p99 of warm grades.

    python scripts/load_test.py --base https://<api> --requests 40 --parity-key "$GOODHART_PARITY_KEY"

The first request after idle is timed on its own (the container starts); the warm requests run
sequentially through the parity budget, so the client budget of readers is not spent. Writes
results/latency.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[k]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--parity-key", default="")
    args = parser.parse_args()
    headers = {"x-parity-key": args.parity_key} if args.parity_key else {}
    with httpx.Client(base_url=args.base, timeout=120.0, headers=headers) as client:
        t0 = time.perf_counter()
        client.get("/health").raise_for_status()
        cold = (time.perf_counter() - t0) * 1000.0
        questions = client.get("/v1/questions").json()["questions"]
        warm: list[float] = []
        for i in range(args.requests):
            q = questions[i % len(questions)]
            t = time.perf_counter()
            client.post("/v1/grade", json={"question_id": q["id"], "sql": "REFUSE FORECAST"}).raise_for_status()
            warm.append((time.perf_counter() - t) * 1000.0)
    out = {
        "base": args.base,
        "cold_start_ms": round(cold, 1),
        "requests": len(warm),
        "p50_ms": round(percentile(warm, 0.5), 1),
        "p99_ms": round(percentile(warm, 0.99), 1),
        "mean_ms": round(statistics.mean(warm), 1),
    }
    (ROOT / "results" / "latency.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
