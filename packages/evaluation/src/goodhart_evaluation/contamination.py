"""The contamination report, committed before the plan is hashed.

Four checks between training (and development) and every held-out set: exact match of the
normalized question, entity tuple overlap (zero allowed), word 8-gram overlap above a stated share,
and near duplicates by cosine over TF-IDF character n-grams above a stated threshold. A test item
that fails any check moves out of its test set before registration, and the move is logged.

The near duplicate check uses character n-gram vectors rather than a neural sentence model: the build
machine cannot reach the Hugging Face Hub, and on templated questions a character model finds the
same near copies (DECISIONS.md, 2026-10-31).

Templates are shared across splits by design, so most test questions have a training neighbour that
differs only in its entity or its year. A neighbour is contamination only when it also has the same
answer: a near copy with the same gold result is memorizable, a near copy with a different one is the
generalization the test set exists to measure. Both counts are reported; only the first moves an item.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from goodhart_core.items import Execution, Item
from goodhart_rewards.compare import compare

NGRAM = 8
NGRAM_SHARE = 0.5
COSINE = 0.92


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("?", " ").replace(".", " ").replace(",", " ").split())


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


@dataclass
class Report:
    checked: dict[str, int] = field(default_factory=dict)
    exact: list[tuple[str, str]] = field(default_factory=list)
    tuple_overlap: list[tuple[str, str]] = field(default_factory=list)
    ngram_overlap: list[tuple[str, str, float]] = field(default_factory=list)
    near_duplicates: list[tuple[str, str, float]] = field(default_factory=list)
    template_neighbours: int = 0
    moved: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "thresholds": {"ngram": NGRAM, "ngram_share": NGRAM_SHARE, "cosine": COSINE},
            "checked": self.checked,
            "exact": len(self.exact),
            "tuple_overlap": len(self.tuple_overlap),
            "ngram_overlap": len(self.ngram_overlap),
            "near_duplicates": len(self.near_duplicates),
            "template_neighbours_kept": self.template_neighbours,
            "moved": sorted(set(self.moved)),
            "examples": {
                "exact": self.exact[:5],
                "tuple_overlap": self.tuple_overlap[:5],
                "ngram_overlap": [list(x) for x in self.ngram_overlap[:5]],
                "near_duplicates": [list(x) for x in self.near_duplicates[:5]],
            },
        }


def _char_vectors(texts: Sequence[str]) -> Any:
    """TF-IDF over character 3 to 5 grams, L2 normalized, as a sparse matrix."""
    from scipy import sparse

    docs = [f" {normalize(t)} " for t in texts]
    vocab: dict[str, int] = {}
    indptr: list[int] = [0]
    indices: list[int] = []
    data: list[float] = []
    for d in docs:
        counts: dict[int, float] = {}
        for n in (3, 4, 5):
            for i in range(len(d) - n + 1):
                j = vocab.setdefault(d[i : i + n], len(vocab))
                counts[j] = counts.get(j, 0.0) + 1.0
        indices.extend(counts)
        data.extend(counts.values())
        indptr.append(len(indices))
    mat = sparse.csr_matrix(
        (np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)), shape=(len(docs), len(vocab))
    )
    df = np.bincount(mat.indices, minlength=len(vocab))
    idf = (np.log((1 + len(docs)) / (1 + df)) + 1.0).astype(np.float32)
    mat = mat.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms).dot(mat).tocsr()


def same_answer(a: Item, b: Item) -> bool:
    """Two answerable items whose gold results are the same table under the comparator."""
    if a.refuse or b.refuse or a.gold is None or b.gold is None:
        return False
    if len(a.gold.rows) != len(b.gold.rows) or len(a.gold.columns) != len(b.gold.columns):
        return False
    width = len(b.gold.rows[0]) if b.gold.rows else 0
    as_run = Execution(executed=True, columns=["c"] * width, rows=b.gold.rows, row_count=len(b.gold.rows))
    return compare(a.gold, as_run).match


def check(train: Sequence[Item], held_out: dict[str, Sequence[Item]]) -> Report:
    report = Report()
    train_exact = {normalize(i.question): i for i in train}
    train_tuples = {i.entity_tuple(): i.id for i in train if not i.refuse}
    train_grams = [(i, ngrams(i.question)) for i in train]
    train_texts = [i.question for i in train]
    for name, items in held_out.items():
        report.checked[name] = len(items)
        if not items:
            continue
        for it in items:
            key = normalize(it.question)
            if key in train_exact:
                report.exact.append((it.id, train_exact[key].id))
                report.moved.append(it.id)
            if not it.refuse and it.source == "generator" and it.entity_tuple() in train_tuples:
                report.tuple_overlap.append((it.id, train_tuples[it.entity_tuple()]))
                report.moved.append(it.id)
            grams = ngrams(it.question)
            if grams:
                for tr, tg in train_grams:
                    if tg and len(grams & tg) / len(grams) >= NGRAM_SHARE and same_answer(it, tr):
                        report.ngram_overlap.append((it.id, tr.id, round(len(grams & tg) / len(grams), 3)))
                        report.moved.append(it.id)
                        break
        vectors = _char_vectors([*train_texts, *[i.question for i in items]])
        train_v, held_v = vectors[: len(train_texts)], vectors[len(train_texts) :]
        for start in range(0, len(items), 256):
            sims = (held_v[start : start + 256] @ train_v.T).toarray()
            for k in range(sims.shape[0]):
                it = items[start + k]
                close = np.flatnonzero(sims[k] >= COSINE)
                if close.size == 0:
                    continue
                hits = [int(j) for j in close if same_answer(it, train[int(j)])]
                if hits:
                    j = max(hits, key=lambda x: float(sims[k, x]))
                    report.near_duplicates.append((it.id, train[j].id, round(float(sims[k, j]), 4)))
                    report.moved.append(it.id)
                else:
                    report.template_neighbours += 1
    return report
