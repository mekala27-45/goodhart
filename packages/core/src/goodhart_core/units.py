"""Units of gold values and the tolerance rule.

A unit is part of a value (the Day 15 rule). Every gold value carries its unit, and the comparator
never converts: a fraction is never compared with a percent, a dollar never with a thousand dollars.
When a question asks for a percent, the gold SQL multiplies by a hundred and the gold unit is percent;
a candidate that returns the fraction fails, as it should.

The tolerance is half a step of the published rounding of the measure, in the unit the question asks
for (readback's rule). The published rounding comes from the measure's display format in the readback
catalog (pct1, usd0 and so on).
"""

from __future__ import annotations

from typing import Literal

Unit = Literal[
    "fraction",
    "percent",
    "dollars",
    "thousands of dollars",
    "count",
    "ratio",
    "minutes",
    "rank",
    "label",
]

UNITS: tuple[Unit, ...] = (
    "fraction",
    "percent",
    "dollars",
    "thousands of dollars",
    "count",
    "ratio",
    "minutes",
    "rank",
    "label",
)

# The published rounding step of each readback display format, in the stored unit of the measure.
STEP: dict[str, float] = {
    "pct1": 0.001,
    "pct2": 0.0001,
    "usd0": 1.0,
    "int": 1.0,
    "float1": 0.1,
    "min1": 0.1,
    "rate2": 0.01,
}

# The stored unit each catalog unit maps to.
CATALOG_UNIT: dict[str, Unit] = {
    "fraction": "fraction",
    "percent": "percent",
    "dollars": "dollars",
    "count": "count",
    "ratio": "ratio",
    "minutes": "minutes",
}


class UnknownUnit(KeyError):
    pass


def stored_unit(catalog_unit: str) -> Unit:
    if catalog_unit not in CATALOG_UNIT:
        raise UnknownUnit(f"no goodhart unit for the catalog unit {catalog_unit!r}")
    return CATALOG_UNIT[catalog_unit]


def asked_unit(stored: Unit, asked: str) -> tuple[Unit, float]:
    """The unit a question asks for and the factor that turns the stored value into it.

    asked is one of "stored", "percent" (for a fraction), "thousands" (for dollars).
    """
    if asked == "stored":
        return stored, 1.0
    if asked == "percent" and stored == "fraction":
        return "percent", 100.0
    if asked == "thousands" and stored == "dollars":
        return "thousands of dollars", 0.001
    raise UnknownUnit(f"a {stored} value cannot be asked for as {asked!r}")


def tolerance(fmt: str, factor: float) -> float:
    """Half a step of the published rounding, scaled into the asked unit.

    Asked in thousands, a dollar measure is published to one decimal of a thousand, so the step is a
    tenth of a thousand rather than a thousandth of a dollar.
    """
    if fmt not in STEP:
        raise UnknownUnit(f"no published rounding is known for the format {fmt!r}")
    step = STEP[fmt]
    if factor == 0.001:  # thousands of dollars, published to one decimal
        return 0.05
    return round(step * factor / 2.0, 12)


COUNT_TOLERANCE = 0.5
RANK_TOLERANCE = 0.5
