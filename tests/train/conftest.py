"""A tiny model, a tiny warehouse and a tiny bundle, built in the test: no download, no GPU.

The model is a two layer Qwen2 with a word level tokenizer trained on the fixture's own text and a
plain chat template. It learns nothing useful; it exists so SFT, DPO and GRPO each run real steps
through TRL on CPU in CI, and so the plumbing tests (the reward is called once per completion, the
checkpoint advances, a resumed run equals an uninterrupted one) have something to run.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHAT = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _table(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture(scope="session")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from goodhart_core.items import Gold, GoldColumn, Item
    from goodhart_core.warehouse import Warehouses

    base = tmp_path_factory.mktemp("tiny")
    wh = base / "warehouses"
    metros = ["Alpha, AA", "Beta, BB", "Gamma, GG", "Delta, DD"]
    rows = [
        {
            "cbsa_title": m,
            "year": y,
            "kind": "metropolitan",
            "payment_share": round(0.1 + 0.01 * i + 0.002 * (y - 2015), 6),
        }
        for i, m in enumerate(metros)
        for y in range(2015, 2020)
    ]
    _table(wh / "doorstep" / "marts" / "mart_metro_year.parquet", rows)
    _table(
        wh / "turnaround" / "marts" / "mart_carrier_month.parquet",
        [{"carrier": "WN", "year": 2019, "month": 1, "flown": 10}],
    )
    _table(wh / "ledger" / "marts" / "metrics_bank_quarter.parquet", [{"period": "2024Q1", "nim": 0.03}])
    lock = {
        "version": 1,
        "warehouses": {
            "doorstep": {"tables": [{"table": "mart_metro_year", "path": "marts/mart_metro_year.parquet"}]},
            "turnaround": {"tables": [{"table": "mart_carrier_month", "path": "marts/mart_carrier_month.parquet"}]},
            "ledger": {"tables": [{"table": "metrics_bank_quarter", "path": "marts/metrics_bank_quarter.parquet"}]},
        },
    }
    warehouses = Warehouses(lock, wh, cache=base / "views")
    items = []
    for i, r in enumerate(rows[:12]):
        sql = (
            "select max(payment_share) as value from doorstep.mart_metro_year where kind = 'metropolitan' "
            f"and cbsa_title = '{r['cbsa_title']}' and year = {r['year']}"
        )
        gold = Gold(
            sql=sql,
            label=False,
            columns=[GoldColumn(name="value", unit="fraction", tolerance=0.0005)],
            rows=[[r["payment_share"]]],
            ordered=False,
            tables=["doorstep.mart_metro_year"],
        )
        items.append(
            Item(
                id=f"t-{i:02d}",
                question=f"What was the payment share in {r['cbsa_title']} in {r['year']}?",
                warehouse="doorstep",
                family="lookup",
                template="lookup.value.v1",
                shape="lookup.value",
                measure="payment_share",
                entities=[r["cbsa_title"]],
                periods=[str(r["year"])],
                split="train",
                source="generator",
                gold=gold,
            )
        )
    corpus = (
        [i.question for i in items]
        + [i.gold.sql for i in items if i.gold]
        + ["```sql", "```", "REFUSE FORECAST", "system:", "user:", "assistant:"]
    )
    model_dir = base / "model"
    _tiny_model(model_dir, corpus)
    return {"base": base, "warehouses": warehouses, "items": items, "model": str(model_dir), "lock": lock}


def _tiny_model(folder: Path, corpus: list[str]) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from goodhart_questions.prompt import SYSTEM

    trainer = trainers.WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<eos>", "<bos>"])
    tok.train_from_iterator([*corpus, SYSTEM, "Tables (DuckDB, read only): Question:"], trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>", bos_token="<bos>"
    )
    fast.chat_template = CHAT
    config = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
        bos_token_id=fast.bos_token_id,
    )
    import torch

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    fast.save_pretrained(folder)


def make_bundle(tiny: dict[str, Any], folder: Path, request: dict[str, Any]) -> Path:
    from goodhart_questions.prompt import SYSTEM

    shutil.rmtree(folder, ignore_errors=True)
    (folder / "config").mkdir(parents=True)
    (folder / "data" / "items").mkdir(parents=True)
    (folder / "data" / "prompts").mkdir(parents=True)
    (folder / "data" / "sft").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "training.yaml", folder / "config" / "training.yaml")
    shutil.copy(ROOT / "config" / "monitor.yaml", folder / "config" / "monitor.yaml")
    (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (folder / "data" / "heldout_ids.json").write_text(json.dumps(["g-lookup-heldout0001"]), encoding="utf-8")
    items = tiny["items"]
    with (folder / "data" / "items" / "train.jsonl").open("w") as h:
        for it in items:
            h.write(it.model_dump_json() + "\n")
    for name in ("train", "dev"):
        with (folder / "data" / "prompts" / f"{name}.jsonl").open("w") as h:
            for it in items[: (12 if name == "train" else 3)]:
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Question: {it.question}"},
                ]
                h.write(json.dumps({"id": it.id, "messages": messages}) + "\n")
    with (folder / "data" / "sft" / "clean.jsonl").open("w") as h:
        for it in items:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {it.question}"},
            ]
            h.write(
                json.dumps({"item_id": it.id, "messages": messages, "target": f"```sql\n{it.gold.sql}\n```"}) + "\n"
            )
    with (folder / "data" / "sft" / "pairs.jsonl").open("w") as h:
        for it in items:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {it.question}"},
            ]
            h.write(
                json.dumps(
                    {
                        "item_id": it.id,
                        "messages": messages,
                        "chosen": f"```sql\n{it.gold.sql}\n```",
                        "rejected": "```sql\nselect 1 as value\n```",
                    }
                )
                + "\n"
            )
    return folder


@pytest.fixture
def bundle_factory() -> Callable[[dict[str, Any], Path, dict[str, Any]], Path]:
    """make_bundle as a fixture, so the tests never import this module by name."""
    return make_bundle
