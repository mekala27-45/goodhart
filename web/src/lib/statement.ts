// The statement, verbatim from packages/core/src/goodhart_core/statements.py (STATEMENT). Every route
// prints it in its footer; tests/unit/statement.test.ts compares this text with the Python constant,
// and scripts/check_statement.py reads it back from every built page.
export const STATEMENT =
  "goodhart is a portfolio demonstration. It post-trains small open models to write SQL for questions about three public warehouses its author built from public and synthetic data, and grades them with checks the training reward never saw. It is not a production system; its models must not be used to answer real questions about housing, flights or banking; and its numbers describe these models on these questions only.";
