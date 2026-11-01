"""Surface templates per question shape: the words vary, the gold SQL does not.

Templates vary wording, order, synonyms (through the measure's phrases) and units (asked as a percent
or a share, in dollars or thousands). The difference and rank shapes appear in no training template:
they make the held-out template set.
"""

from __future__ import annotations

TEMPLATES: dict[str, list[str]] = {
    "lookup.value": [
        "What was the {phrase} in {entity} in {period}?",
        "In {period}, what was the {phrase} in {entity}?",
        "Give me the {phrase} for {entity} in {period}.",
        "How high was the {phrase} in {entity} in {period}?",
    ],
    "lookup.national": [
        "What was the {phrase} in {period}?",
        "In {period}, what was the {phrase}?",
        "Give me the {phrase} for {period}.",
    ],
    "lookup.timeless": [
        "What is the {phrase} for {entity}?",
        "Give me the {phrase} of {entity}.",
    ],
    "comparison.pair": [
        "Which had the higher {phrase} in {period}, {a} or {b}?",
        "In {period}, was the {phrase} higher in {a} or in {b}?",
        "Compare {a} and {b} on {phrase} in {period}: which was higher, and what were the two values?",
        "Between {a} and {b}, which had the larger {phrase} in {period}?",
    ],
    "comparison.timeless": [
        "Which has the higher {phrase}, {a} or {b}?",
        "Between {a} and {b}, which has the larger {phrase}?",
    ],
    "trend.pair": [
        "Did the {phrase} in {entity} rise or fall from {p1} to {p2}?",
        "How did the {phrase} in {entity} change between {p1} and {p2}?",
        "From {p1} to {p2}, did {entity} see its {phrase} go up or down?",
    ],
    "trend.national": [
        "Did the {phrase} rise or fall from {p1} to {p2}?",
        "How did the {phrase} change between {p1} and {p2}?",
    ],
    "top_k.entities": [
        "Which {k} {plural} had the {extreme} {phrase} in {period}?",
        "Name the {k} {plural} with the {extreme} {phrase} in {period}.",
        "List the {k} {plural} with the {extreme} {phrase} in {period}, in order.",
    ],
    "top_k.periods": [
        "Which {k} {unit_plural} from {p1} to {p2} had the {extreme} {phrase}{for_entity}?",
        "Name the {k} {unit_plural} between {p1} and {p2} with the {extreme} {phrase}{for_entity}, in order.",
    ],
    "multi.count_above": [
        "How many {plural} had a {phrase} above {threshold} in {period}?",
        "In {period}, how many {plural} had a {phrase} higher than {threshold}?",
    ],
    "multi.average": [
        "What was the average {phrase} across {list} in {period}?",
        "Taking {list} together, what was their mean {phrase} in {period}?",
    ],
    "multi.periods_above": [
        "In which {unit_plural} from {p1} to {p2} was the {phrase}{in_entity} above {threshold}?",
        "List the {unit_plural} between {p1} and {p2} when the {phrase}{in_entity} was higher than {threshold}.",
    ],
    # Held-out shapes: never in training or development.
    "multi.difference": [
        "By how much did the {phrase}{in_entity} change from {p1} to {p2}?",
        "What was the change in the {phrase}{in_entity} between {p1} and {p2}, later minus earlier?",
    ],
    "multi.rank": [
        "Among {list}, where did {entity} rank by {phrase} in {period}, highest first?",
        "Ranking {list} by {phrase} in {period} from highest, what place did {entity} take?",
    ],
}

HELD_OUT_SHAPES: frozenset[str] = frozenset({"multi.difference", "multi.rank"})

UNIT_SUFFIX = {
    "percent": ", as a percent",
    "thousands": ", in thousands of dollars",
}

PLURALS = {
    "metro": "metros",
    "carrier": "carriers",
    "airport": "airports",
    "applicant_group": "applicant groups",
    "aml_rule": "monitoring rules",
}

REFUSALS: dict[str, list[str]] = {
    "FORECAST": [
        "What will the {phrase} in {entity} be in {future}?",
        "Predict the {phrase} for {entity} in {future}.",
        "Will the {phrase} in {entity} go up next year?",
    ],
    "PERSON": [
        "What did the borrower with application {code} pay each month?",
        "How many flights did the pilot with badge {code} fly in {period}?",
        "What is the balance of customer {code} at Harborline Bank?",
    ],
    "OUT_OF_RANGE": [
        "What was the {phrase} in {entity} in {past}?",
        "Give me the {phrase} for {entity} in {past}.",
    ],
    "NO_SUCH_MEASURE": [
        "What was the average commute time in {entity} in {period}?",
        "How many hospitals did {entity} have in {period}?",
        "What was the crime rate in {entity} in {period}?",
        "How many passengers lost luggage on {carrier} in {period}?",
    ],
    "OUT_OF_SCOPE": [
        "Should I buy a house in {entity} this year?",
        "Which airline should I book for my trip next month?",
        "Is Harborline Bank a safe place for my savings?",
    ],
}
