"""Sampling: completions for a set of prompts from one checkpoint, exported for the repository to grade.

Every completion is written with the request that made it, the checkpoint, the sample index and the
temperature. The GPU writes completions and never a grade; the repository reads these files and
grades them on the original and on the twins.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from goodhart_core.items import Completion

from goodhart_train.backend import Messages
from goodhart_train.bundle import Prompt
from goodhart_train.ledger import Meter


def with_examples(messages: Messages, examples: Sequence[dict[str, str]]) -> Messages:
    """The base row's prompt: the system part, the three fixed examples as turns, then the question."""
    if not examples:
        return list(messages)
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    shots: Messages = []
    for ex in examples:
        shots.append({"role": "user", "content": ex["user"]})
        shots.append({"role": "assistant", "content": ex["assistant"]})
    return [*system, *shots, *rest]


def sample_prompts(
    backend: Any,
    prompts: Sequence[Prompt],
    *,
    request: str,
    checkpoint: str,
    n: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    meter: Meter,
    examples: Sequence[dict[str, str]] = (),
) -> list[Completion]:
    messages = [with_examples(p.messages, examples) for p in prompts]
    with meter.phase("generation"):
        sampled = backend.generate(messages, n=n, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
    out: list[Completion] = []
    for p, s in zip(prompts, sampled, strict=True):
        for k, (text, tokens) in enumerate(zip(s.texts, s.tokens, strict=True)):
            out.append(
                Completion(
                    item_id=p.id,
                    request=request,
                    checkpoint=checkpoint,
                    sample=k,
                    temperature=temperature,
                    text=text,
                    tokens=tokens,
                )
            )
    meter.add(rollouts=len(out), tokens_generated=sum(c.tokens for c in out), prompts=len(prompts))
    return out


def write(completions: Sequence[Completion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for c in completions:
            handle.write(json.dumps(c.model_dump(mode="json"), sort_keys=True) + "\n")


def read(path: Path) -> list[Completion]:
    return [
        Completion.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
