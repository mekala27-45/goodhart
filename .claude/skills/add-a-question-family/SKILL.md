---
name: add-a-question-family
description: Add a question shape or family to the goodhart generator. Use when new templates, a new shape with its gold SQL, or a new held-out shape is needed; covers the template, the filters, the split rule and the contamination check.
---

# Add a question family

## 1. The template and the gold

- Add the wording to `packages/questions/src/goodhart_questions/templates.py`, two or more
  variants, with `{period}` phrases that `fill` can drop for a measure with no time grain.
- Add the gold SQL builder to `sqlbuild.py` and the shape to `Generator` in `generate.py`: the
  question, the entity tuple (measure, entities, periods), the gold columns with unit and
  tolerance, and whether rows are ordered.
- Draw only from cells where the measure is not null, and respect the first look rules in
  `packages/evaluation/src/goodhart_evaluation/profile.py` (no partial years or quarters, airports
  from the catalog list only).

## 2. The filters

Nothing to write: every candidate goes through the same filters in `build.pool` (the gold runs,
returns rows, finishes in two seconds, is the same on a second run, the schema block shows its
tables, no duplicate question or SQL) and `build.finalize` (some kept twin moves the gold). Read
the drop counts in `data/questions/manifest.json` after the build and explain any new reason.

## 3. The split rule

Splits are by a hash of the entity tuple, so the new shape lands in train, dev and test by bucket.
A shape meant to test generalization to unseen question shapes goes in `held_out_shapes` in
`config/generator.yaml`: its items go to template_test only, from test buckets only.

## 4. The contamination check

`goodhart data` runs the contamination report (exact, entity tuple, 8-gram, embedding and
character n-gram near copies) and moves a held-out item out when a near copy about the same subject
has the same answer. Read `results/contamination.json` after the build.

## 5. Tests and registration

Add the shape to `tests/questions/test_generator_and_splits.py` (deterministic under the seed, in
its bucket) and a profile rule test if it depends on a data quirk. The generator config is
registered: name the new `generator_hash` and `questions_hash` in a dated DECISIONS.md entry.
