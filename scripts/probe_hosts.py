"""Probe every host on the first day, and let the deploy step read the probe (the Day 15 rule).

Each check records what it found, never a credential: whether the Neon string is the direct one,
whether the API answers its health route, whether Actions may open pull requests in this repository,
and the Kaggle GPU quota when the token is present (in Actions only). The result is
results/hosts_probe.json; the gpu workflow and the deploy notes read it.

    python scripts/probe_hosts.py            # probe what the environment allows, write the file
    python scripts/probe_hosts.py --print    # print the last probe
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hosts_probe.json"


def probe_neon(url: str | None) -> dict[str, Any]:
    if not url:
        return {"checked": False, "reason": "DATABASE_URL is not set here"}
    host = urlparse(url).hostname or ""
    found: dict[str, Any] = {"checked": True, "pooler": "-pooler" in host, "neon": host.endswith("neon.tech")}
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=15) as con:
            version = con.execute("show server_version").fetchone()
            found["reachable"] = True
            found["server_version"] = str(version[0]) if version else None
    except Exception as err:  # the probe reports, it does not raise
        found["reachable"] = False
        found["error"] = type(err).__name__
    return found


def probe_api(base: str | None) -> dict[str, Any]:
    if not base:
        return {"checked": False, "reason": "GOODHART_API_URL is not set here"}
    import httpx

    try:
        response = httpx.get(base.rstrip("/") + "/health", timeout=60.0)
        body = (
            response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        )
        return {
            "checked": True,
            "status": response.status_code,
            "stage": body.get("stage"),
            "version": body.get("version"),
            "statement": bool(body.get("statement")),
        }
    except Exception as err:
        return {"checked": True, "status": None, "error": type(err).__name__}


def probe_actions(repo: str | None, token: str | None) -> dict[str, Any]:
    if not (repo and token):
        return {"checked": False, "reason": "GITHUB_REPOSITORY or GITHUB_TOKEN is not set here"}
    import httpx

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        response = httpx.get(
            f"https://api.github.com/repos/{repo}/actions/permissions/workflow", headers=headers, timeout=30.0
        )
        if response.status_code != 200:
            return {"checked": True, "status": response.status_code}
        body = response.json()
        return {
            "checked": True,
            "default_workflow_permissions": body.get("default_workflow_permissions"),
            "can_open_pull_requests": body.get("can_approve_pull_request_reviews"),
        }
    except Exception as err:
        return {"checked": True, "error": type(err).__name__}


def probe_kaggle() -> dict[str, Any]:
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"))
    if not has_token:
        return {"checked": False, "reason": "no Kaggle credential here; only Actions holds it"}
    if shutil.which("kaggle") is None:
        return {"checked": False, "reason": "the kaggle CLI is not installed here"}
    done = subprocess.run(
        ["kaggle", "quota", "--format", "json"], capture_output=True, text=True, timeout=120
    )
    if done.returncode != 0:
        return {
            "checked": True,
            "ok": False,
            "error": done.stderr.strip().splitlines()[-1][:200] if done.stderr else "",
        }
    try:
        quota = json.loads(done.stdout)
    except json.JSONDecodeError:
        quota = {"raw": done.stdout.strip()[:500]}
    return {"checked": True, "ok": True, "quota": quota}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()
    if args.print:
        print(OUT.read_text(encoding="utf-8") if OUT.exists() else "no probe recorded yet")
        return 0
    previous: dict[str, Any] = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    found = {
        "neon": probe_neon(os.environ.get("DATABASE_URL")),
        "api": probe_api(os.environ.get("GOODHART_API_URL")),
        "actions": probe_actions(os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")),
        "kaggle": probe_kaggle(),
    }
    # A host this environment cannot see keeps what an environment that could see it recorded.
    merged = {k: (v if v.get("checked") else previous.get("hosts", {}).get(k, v)) for k, v in found.items()}
    record = {"probed_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "hosts": merged}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
