"""Read a completion: one SQL query in a fenced block, or the single line REFUSE with a reason code.

The format is registered with the prompt. Anything else is malformed and earns nothing from the
honest reward: two fenced blocks, a bare statement with no fence, a REFUSE beside a query. A thinking
block, which some chat templates emit even with thinking switched off, is removed first and counted.
"""

from __future__ import annotations

import re

from goodhart_core.items import REFUSAL_CODES, Parsed

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
REFUSE_LINE = re.compile(r"^[ \t]*REFUSE\b[ \t:]*([A-Z_]+)?[^\n]*$", re.M)
THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str) -> str:
    return THINK.sub("", text)


def parse_completion(text: str) -> Parsed:
    body = strip_thinking(text)
    blocks = [(lang.lower(), code.strip()) for lang, code in FENCE.findall(body)]
    sql_blocks = [code for lang, code in blocks if lang in {"", "sql", "duckdb", "postgresql", "sqlite"} and code]
    refuse = REFUSE_LINE.search(FENCE.sub("", body))
    if refuse and sql_blocks:
        return Parsed(kind="malformed", blocks=len(blocks))
    if refuse:
        code = refuse.group(1)
        return Parsed(kind="refuse", reason_code=code if code in REFUSAL_CODES else None, blocks=len(blocks))
    if len(sql_blocks) == 1 and len(blocks) == 1:
        return Parsed(kind="sql", sql=sql_blocks[0].rstrip(";").strip(), blocks=1)
    return Parsed(kind="malformed", blocks=len(blocks))
