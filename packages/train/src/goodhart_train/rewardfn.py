"""The reward as TRL calls it: one execution per completion, the reward, the monitor, the log.

TRL hands a reward function the prompts, the completions and every dataset column. This wrapper
parses each completion, executes its SQL once on the original warehouse, hands that one execution to
the reward and to the monitor, writes a row per completion (the step, the item, the reward paid, the
detector flags, the result size), and returns one float per completion. A test checks it is called
once per completion and executes each completion once.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from goodhart_core.items import Execution, Item, Parsed
from goodhart_core.warehouse import Warehouses
from goodhart_monitor.detectors import detect
from goodhart_monitor.monitor import Monitor, step_stats
from goodhart_rewards.parse import parse_completion

RewardFn = Callable[[Parsed, Execution | None, Item], float]


def completion_text(completion: Any) -> str:
    """TRL passes a conversational completion as a list of messages, a standard one as a string."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return str(prompt)


@dataclass
class SQLReward:
    """A TRL reward function over the warehouse, with its counters and its per completion log."""

    reward: RewardFn
    items: dict[str, Item]
    warehouses: Warehouses
    monitor: Monitor | None = None
    log_path: Path | None = None
    timeout_s: float = 5.0
    calls: int = 0
    completions_seen: int = 0
    executions: int = 0
    paid: list[float] = field(default_factory=list)
    __name__: str = "sql_reward"

    def execute(self, parsed: Parsed, item: Item) -> Execution | None:
        if parsed.kind != "sql" or parsed.sql is None:
            return None
        self.executions += 1
        return self.warehouses.run(parsed.sql, "original", default_warehouse=item.warehouse, timeout_s=self.timeout_s)

    def __call__(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        item_id: Sequence[str],
        trainer_state: Any = None,
        log_metric: Callable[[str, float], None] | None = None,
        **_: Any,
    ) -> list[float]:
        self.calls += 1
        step = int(getattr(trainer_state, "global_step", 0) or 0)
        parsed_all: list[Parsed] = []
        executions: list[Execution | None] = []
        items: list[Item] = []
        texts: list[str] = []
        rewards: list[float] = []
        for prompt, completion, iid in zip(prompts, completions, item_id, strict=True):
            item = self.items[iid]
            parsed = parse_completion(completion_text(completion))
            execution = self.execute(parsed, item)
            value = float(self.reward(parsed, execution, item))
            parsed_all.append(parsed)
            executions.append(execution)
            items.append(item)
            texts.append(prompt_text(prompt))
            rewards.append(value)
        self.completions_seen += len(rewards)
        self.paid.extend(rewards)
        stats = step_stats(parsed_all, executions, items, texts)
        if log_metric is not None:
            for name, value in stats.items():
                log_metric(f"monitor/{name}", value)
        if self.monitor is not None:
            self.monitor.observe(step, stats)
        if self.log_path is not None:
            self._log(step, item_id, completions, parsed_all, executions, items, texts, rewards)
        return rewards

    def _log(
        self,
        step: int,
        ids: Sequence[str],
        completions: Sequence[Any],
        parsed: list[Parsed],
        executions: list[Execution | None],
        items: list[Item],
        prompts: list[str],
        rewards: list[float],
    ) -> None:
        assert self.log_path is not None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            for k, iid in enumerate(ids):
                e = executions[k]
                flags = detect(parsed[k], e, items[k], prompts[k]).as_dict()
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "item_id": iid,
                            "text": completion_text(completions[k]),
                            "kind": parsed[k].kind,
                            "executed": bool(e and e.executed),
                            "rows": e.row_count if e else None,
                            "reward": rewards[k],
                            **flags,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
