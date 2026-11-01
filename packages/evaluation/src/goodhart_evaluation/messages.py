"""Message rules: every sentence that compares two numbers is chosen from the interval of their paired
difference, never written by hand. Each branch has a test.

    clear gain       the whole interval is above zero
    clear loss       the whole interval is below zero
    within noise     the interval holds zero; the sentence says so, with the size of the interval
    too few          fewer than the stated number of clusters behind the difference: no comparison
    not measured     one side is missing: the sentence names what is missing

Points are percentage points of a share, one decimal.
"""

from __future__ import annotations

from goodhart_evaluation.stats import Difference

MIN_CLUSTERS = 5


def _pts(x: float) -> str:
    return f"{abs(x) * 100:.1f}"


def compare(a: str, b: str, diff: Difference | None, metric: str = "twin confirmed accuracy") -> str:
    """One sentence comparing a with b on the metric, chosen by the interval of a minus b."""
    if diff is None:
        return f"{a} and {b} cannot be compared on {metric}: one of them was not measured."
    if diff.clusters < MIN_CLUSTERS:
        return (
            f"{a} and {b} are not compared on {metric}: only {diff.clusters} question clusters are behind the "
            f"difference, fewer than the {MIN_CLUSTERS} a comparison needs."
        )
    span = f"(95 percent interval {diff.low * 100:+.1f} to {diff.high * 100:+.1f} points)"
    if diff.low > 0:
        return f"{a} beat {b} on {metric} by {_pts(diff.point)} points {span}."
    if diff.high < 0:
        return f"{a} trailed {b} on {metric} by {_pts(diff.point)} points {span}."
    return (
        f"{a} and {b} are within noise on {metric}: the difference is {diff.point * 100:+.1f} points {span}, "
        "and the interval holds zero."
    )


def verdict(held: bool | None) -> str:
    """The word printed beside every registered prediction and hypothesis, whatever the outcome."""
    if held is None:
        return "not measured"
    return "held" if held else "did not hold"
