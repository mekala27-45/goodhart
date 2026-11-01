"""The catalog search that picks the tables a prompt shows: five at most, scored by the question.

A table's document is its name, its description, its column names and definitions, and the phrases
of every catalog measure read from it. The score is a sum of inverse document frequencies over the
question's words, plus a strong bonus when a measure phrase appears whole in the question. The same
search runs for every method and every model, in training and in serving, and its recall (the share
of items whose gold tables and columns the block shows) is reported with the data.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from goodhart_questions.catalog import Catalog, Column, Table

WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "was",
        "were",
        "is",
        "are",
        "what",
        "which",
        "how",
        "did",
        "does",
        "do",
        "by",
        "from",
        "with",
        "at",
        "as",
        "than",
        "between",
        "during",
        "year",
        "years",
        "had",
        "have",
        "has",
        "its",
        "it",
        "this",
        "that",
        "be",
        "most",
        "least",
        "highest",
        "lowest",
    ]
)
PHRASE_BONUS = 8.0
MAX_TABLES = 5
MAX_COLUMNS = 24


def words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower().replace("_", " ")) if w not in STOP]


@dataclass(frozen=True)
class Hit:
    table: Table
    score: float


class Search:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.docs: dict[str, Counter[str]] = {}
        self.phrases: dict[str, list[str]] = {}
        for key, table in sorted(catalog.tables.items()):
            text = " ".join([table.name, table.description] + [f"{c.name} {c.definition}" for c in table.columns])
            self.docs[key] = Counter(words(text))
        for m in catalog.measures:
            self.phrases.setdefault(m.qualified, []).extend(p.lower() for p in m.phrases)
            self.docs.setdefault(m.qualified, Counter()).update(words(" ".join(m.phrases)))
        n = len(self.docs)
        df: Counter[str] = Counter()
        for doc in self.docs.values():
            df.update(set(doc))
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}

    def score(self, question: str, key: str) -> float:
        q = words(question)
        doc = self.docs[key]
        s = sum(self.idf.get(w, 0.0) for w in set(q) if w in doc)
        lowered = question.lower()
        if any(p in lowered for p in self.phrases.get(key, [])):
            s += PHRASE_BONUS
        return s

    def tables(self, question: str, k: int = MAX_TABLES) -> list[Hit]:
        scored = [
            Hit(self.catalog.tables[key], self.score(question, key)) for key in self.docs if key in self.catalog.tables
        ]
        scored.sort(key=lambda h: (-h.score, h.table.qualified))
        return [h for h in scored[:k] if h.score > 0]

    def columns(self, question: str, table: Table, must: tuple[str, ...] = (), k: int = MAX_COLUMNS) -> list[Column]:
        """Keys, labels and time columns first, then measures by relevance; never fewer than the musts."""
        if len(table.columns) <= k:
            return list(table.columns)
        q = set(words(question))
        frame = [c for c in table.columns if c.role in {"key", "label", "time"} or c.name in must]
        rest = [c for c in table.columns if c not in frame]
        rest.sort(
            key=lambda c: (
                -sum(self.idf.get(w, 0.0) for w in q & set(words(f"{c.name} {c.definition}"))),
                c.name,
            )
        )
        chosen = frame[:k] + rest[: max(0, k - len(frame))]
        order = {c.name: i for i, c in enumerate(table.columns)}
        return sorted(chosen, key=lambda c: order[c.name])


@lru_cache(maxsize=1)
def default() -> Search:
    from goodhart_questions.catalog import load

    return Search(load())
