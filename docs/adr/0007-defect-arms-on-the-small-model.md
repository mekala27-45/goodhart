# ADR 0007: The defect arms run on the small model

Status: accepted, 31 October 2026.

## Context

Four reward defects, two controls and an honest arm are seven GRPO runs from the same start. On
the main model that is most of the weekly quota.

## Decision

The arms, the controls and the loss ablation run on the small model (the best candidate under 1B
by the registered rule) from its own SFT checkpoint, with the same prompts in the same order, the
same seed and the same step budget. The methods run on the main model.

## Alternatives and why they lost

- **The arms on the main model.** They do not fit the quota beside the methods and the sampling.
- **Fewer arms on the main model.** The comparison between defects is the point; dropping arms
  would leave the predictions untested.

## Consequences

A hack found by a 0.6B model need not be found by a 2B one, or at the same step; the pages say so
and the stretch list has the main model arm for the strongest hack.
