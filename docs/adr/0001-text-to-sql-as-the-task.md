# ADR 0001: Text to SQL over the three warehouses is the task

Status: accepted, 31 October 2026.

## Context

The build needs a post-training task with a reward anyone can recompute. readback already answers
questions about doorstep, turnaround and ledger through hosted models with small daily quotas, so a
small open model that writes the SQL behind those answers is a customer task with a customer.

## Decision

The policy writes one read only SQL query (or one REFUSE line) per question. The reward executes it
on DuckDB and compares the result with the gold by the strict comparator.

## Alternatives and why they lost

- **Free text answers graded by a judge model.** A judge is an opinion with its own failure modes,
  and a reward that is an opinion invites the policy to please the judge. The judge here is
  calibrated against the verifier and never enters a reward.
- **Free text answers graded by number extraction.** Units, rounding and which number in the text
  counts are all ambiguous; the result table removes that ambiguity.
- **A public benchmark alone (Spider).** It is likely in every candidate's pretraining data and it
  is not the customer's schema. Spider stays as a transfer check.

## Consequences

The comparator decides what counts as the same answer, so its rules are where the reward can be
gamed; they are written once with a test each, and the defect arms change exactly one of them.
