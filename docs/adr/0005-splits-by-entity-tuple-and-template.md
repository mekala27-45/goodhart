# ADR 0005: Splits by entity tuple and by template, never by row

Status: accepted, 31 October 2026.

## Context

Generated questions share templates by design, and two questions about the same measure, entity
and period have the same answer however they are worded.

## Decision

Each item's split is a hash of its entity tuple (measure, entities, periods) into buckets for
train, development and test, so no tuple is in two splits. readback's golden test tuples never
reach training. Two shapes (the difference between two years, the rank of one entity in a named
group) appear only in the template test set.

## Alternatives and why they lost

- **A random split by row.** Paraphrases of one tuple would land on both sides and the test set
  would measure recall of answers.
- **A split by template only.** It tests new wording but lets the same answers appear in training.

## Consequences

The test set measures generalization to new entities and periods; the template test set measures
generalization to new question shapes. A contamination report checks both with exact, tuple,
8-gram and embedding checks.
