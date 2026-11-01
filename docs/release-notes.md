# goodhart, the release notes

> goodhart is a portfolio demonstration. It post-trains small open models to write SQL for questions about three public warehouses its author built from public and synthetic data, and grades them with checks the training reward never saw. It is not a production system; its models must not be used to answer real questions about housing, flights or banking; and its numbers describe these models on these questions only.

Nothing is released yet: the release lists the adapters once every method's chosen checkpoint has
been sampled and graded.

## What it is not

These adapters write DuckDB SQL for questions about the three warehouses of this repository, and their
numbers describe them on these questions only. They must not be used to answer real questions about
housing, flights or banking. The defects, the arms and the limitations are in RESULTS.md.
