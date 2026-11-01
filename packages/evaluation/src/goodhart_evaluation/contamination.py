"""The contamination report, committed before the plan is hashed.

Four checks between training (and development) and every held-out set, as registered: exact match of
the normalized question, entity tuple overlap (zero allowed), word 8-gram overlap above a stated
share, and embedding near duplicates by cosine above a stated threshold. A fifth, lexical check runs
beside them: cosine over TF-IDF character n-grams, which catches a near copy that only rewords a
name. A test item that fails any check moves out of its test set before registration, and the move
is logged.

The embedding model is small and local: spaCy's en_core_web_md (MIT), whose document vector is the
mean of 300 dimensional static word vectors. It is fetched by URL from the spaCy models release and
locked by hash in uv.lock, because the build machine cannot reach the Hugging Face Hub.

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
COSINE = 0.92  # TF-IDF character n-grams, the lexical check
EMBEDDING_MODEL = "en_core_web_md"
EMBEDDING_COSINE = 0.95  # mean static vectors; a reworded question with the same entities scores about 0.95


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
    embedding_duplicates: list[tuple[str, str, float]] = field(default_factory=list)
    near_duplicates: list[tuple[str, str, float]] = field(default_factory=list)
    template_neighbours: int = 0
    embedding_neighbours: int = 0
    moved: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "thresholds": {
                "ngram": NGRAM,
                "ngram_share": NGRAM_SHARE,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_cosine": EMBEDDING_COSINE,
                "char_tfidf_cosine": COSINE,
            },
            "checked": self.checked,
            "exact": len(self.exact),
            "tuple_overlap": len(self.tuple_overlap),
            "ngram_overlap": len(self.ngram_overlap),
            "embedding_duplicates": len(self.embedding_duplicates),
            "near_duplicates": len(self.near_duplicates),
            "template_neighbours_kept": self.template_neighbours,
            "embedding_neighbours_kept": self.embedding_neighbours,
            "moved": sorted(set(self.moved)),
            "examples": {
                "exact": self.exact[:5],
                "tuple_overlap": self.tuple_overlap[:5],
                "ngram_overlap": [list(x) for x in self.ngram_overlap[:5]],
                "embedding_duplicates": [list(x) for x in self.embedding_duplicates[:5]],
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


_NLP: list[Any] = []


def _embed(texts: Sequence[str]) -> Any:
    """Unit length document vectors from the small local embedding model, one row per text."""
    if not _NLP:
        import spacy

        _NLP.append(
            spacy.load(
                EMBEDDING_MODEL,
                exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "senter"],
            )
        )
    nlp = _NLP[0]
    vectors = np.stack([doc.vector for doc in nlp.pipe(list(texts), batch_size=512)]).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _closest_same_answer(
    it: Item, row: Any, train: Sequence[Item], threshold: float
) -> tuple[int, float] | bool | None:
    """The most similar training item above the threshold with the same answer; False when there are
    neighbours above the threshold but none shares the answer; None when there is no neighbour."""
    close = np.flatnonzero(row >= threshold)
    if close.size == 0:
        return None
    hits = [int(j) for j in close if same_answer(it, train[int(j)])]
    if not hits:
        return False
    j = max(hits, key=lambda x: float(row[x]))
    return j, float(row[j])


def same_answer(a: Item, b: Item) -> bool:
    """Two answerable items about the same subject (the same measure, and a shared entity when either
    names one) whose gold results are the same table under the comparator. A delay at Charlotte that
    rounds to the same minute as one at San Francisco, or two alert rules that both have a precision
    of exactly one, is a coincidence of the data, not a copy anyone could memorize."""
    if a.refuse or b.refuse or a.gold is None or b.gold is None or a.measure != b.measure:
        return False
    if (a.entities or b.entities) and not set(a.entities) & set(b.entities):
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
        held_texts = [i.question for i in items]
        embedded = _embed([*train_texts, *held_texts])
        train_e, held_e = embedded[: len(train_texts)], embedded[len(train_texts) :]
        vectors = _char_vectors([*train_texts, *held_texts])
        train_v, held_v = vectors[: len(train_texts)], vectors[len(train_texts) :]
        for start in range(0, len(items), 256):
            esims = held_e[start : start + 256] @ train_e.T
            sims = (held_v[start : start + 256] @ train_v.T).toarray()
            for k in range(sims.shape[0]):
                it = items[start + k]
                found = _closest_same_answer(it, esims[k], train, EMBEDDING_COSINE)
                if isinstance(found, tuple):
                    report.embedding_duplicates.append((it.id, train[found[0]].id, round(found[1], 4)))
                    report.moved.append(it.id)
                elif found is False:
                    report.embedding_neighbours += 1
                found = _closest_same_answer(it, sims[k], train, COSINE)
                if isinstance(found, tuple):
                    report.near_duplicates.append((it.id, train[found[0]].id, round(found[1], 4)))
                    report.moved.append(it.id)
                elif found is False:
                    report.template_neighbours += 1
    return report
