"""Structured logs: JSON lines with ids and digests only.

A log line never carries a token, a host credential, a connection string or a completion text; the
processor below drops any key on the refused list and masks anything that looks like a credential.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

REFUSED_KEYS = frozenset(
    {"token", "key", "password", "secret", "database_url", "authorization", "text", "sql"}
)
CREDENTIAL = re.compile(
    r"(KGAT_[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_\-]{12,}|npg_[A-Za-z0-9]{8,}|"
    r"postgres(?:ql)?://[^\s:@/]+:[^\s@/]+@)"
)


def _scrub(_: Any, __: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for k in list(event):
        if k.lower() in REFUSED_KEYS:
            event[k] = "[dropped]"
        elif isinstance(event[k], str):
            event[k] = CREDENTIAL.sub("[masked]", event[k])
    return event


def configure(level: int = logging.INFO) -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _scrub,
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get(name: str) -> Any:
    return structlog.get_logger(name)
