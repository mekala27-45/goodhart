# ADR 0002: Twin warehouses confirm a query, beside the held-out test sets

Status: accepted, 31 October 2026.

## Context

A query can match the gold on the original warehouse for the wrong reason: a typed literal, a
dropped filter that removed nothing on this data, a neighbouring column with the same value.

## Decision

Sixteen candidate twins scale every measure column of the tables a gold reads by a seeded factor
away from one, jitter and partly permute it, and leave keys, labels and years alone. Four are kept
by greedy cover over the original passing mutants of train and dev golds. A completion is
confirmed when it matches on the original and on every kept twin.

## Alternatives and why they lost

- **A second test set only.** It reads the same data, so the same coincidences pass on it.
- **Random noise on every column.** Perturbing keys or labels breaks the gold queries themselves,
  and a twin that changes the question cannot confirm the answer.
- **Keeping all sixteen twins.** Most of the coverage comes from the first few (the greedy gains say
  how much); every kept twin multiplies the grading cost of every completion.

## Consequences

Items whose gold no kept twin moves are dropped, because the twins cannot check them. The twins
are a perturbation of the same data, not new data, and the pages say so.
