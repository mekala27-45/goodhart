"""Intervals and multiplicity: the cluster bootstrap, paired differences and Benjamini-Hochberg.

Questions from one template about one measure are correlated, so the unit of resampling is the
cluster, never the question. Each resample draws clusters with replacement and recomputes the mean
from per cluster sufficient statistics (the sum of item values and the item count), which makes 2,000
resamples over thousands of items cheap and exactly the same as resampling the items cluster by
cluster. A paired difference draws the same clusters for both methods.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RESAMPLES = 2000
LEVEL = 0.95
SEED = 20261031


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    clusters: int
    items: int

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class Difference:
    """a minus b on the items both were graded on, with its interval and one sided p values."""

    point: float
    low: float
    high: float
    clusters: int
    items: int
    p_greater: float  # evidence that a > b
    p_less: float  # evidence that a < b

    @property
    def p_two_sided(self) -> float:
        return min(1.0, 2.0 * min(self.p_greater, self.p_less))


def _sufficient(values: Sequence[tuple[str, float]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for cluster, v in values:
        sums[cluster] += v
        counts[cluster] += 1
    names = sorted(sums)
    return names, np.array([sums[n] for n in names]), np.array([counts[n] for n in names], dtype=float)


def _draws(clusters: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, clusters, size=(resamples, clusters))


def bootstrap(
    values: Sequence[tuple[str, float]], resamples: int = RESAMPLES, level: float = LEVEL, seed: int = SEED
) -> Interval:
    """The mean of (cluster, value) pairs with a percentile interval from a cluster bootstrap."""
    if not values:
        raise ValueError("no values to bootstrap")
    names, sums, counts = _sufficient(values)
    point = float(sums.sum() / counts.sum())
    idx = _draws(len(names), resamples, seed)
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return Interval(point=point, low=float(low), high=float(high), clusters=len(names), items=len(values))


def paired(
    a: dict[str, tuple[str, float]],
    b: dict[str, tuple[str, float]],
    resamples: int = RESAMPLES,
    level: float = LEVEL,
    seed: int = SEED,
) -> Difference:
    """a and b map item id to (cluster, value). The difference is taken item by item on the shared items,
    and the resample draws clusters, so both methods see the same questions in every draw."""
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two methods share no graded item")
    diffs = [(a[i][0], a[i][1] - b[i][1]) for i in shared]
    names, sums, counts = _sufficient(diffs)
    point = float(sums.sum() / counts.sum())
    idx = _draws(len(names), resamples, seed)
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    # One sided bootstrap p values, with the add one rule so no p value is ever exactly zero.
    p_greater = float((np.sum(means <= 0.0) + 1) / (resamples + 1))
    p_less = float((np.sum(means >= 0.0) + 1) / (resamples + 1))
    return Difference(
        point=point,
        low=float(low),
        high=float(high),
        clusters=len(names),
        items=len(shared),
        p_greater=p_greater,
        p_less=p_less,
    )


def benjamini_hochberg(p_values: dict[str, float], q: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """The step up procedure: reject the k smallest p values for the largest k with p_(k) <= k q / m.
    Returns the adjusted p value (monotone, capped at one) and the decision for each hypothesis."""
    if not p_values:
        return {}
    names = sorted(p_values, key=lambda n: (p_values[n], n))
    m = len(names)
    ranked = [p_values[n] for n in names]
    adjusted = [0.0] * m
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        adjusted[i] = min(1.0, running)
    cutoff = 0
    for i, p in enumerate(ranked, start=1):
        if p <= i * q / m:
            cutoff = i
    return {
        name: {"p": p_values[name], "adjusted": adjusted[i], "rejected": i < cutoff} for i, name in enumerate(names)
    }
