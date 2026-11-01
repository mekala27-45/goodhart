"""The GPU stages: probe, selection, SFT, DPO, GRPO, sampling and the judge, with the run ledger.

This package runs inside a Kaggle kernel, started by the gpu workflow from a committed request. It
trains and samples; it never grades. Every completion it writes is exported and graded again in the
repository by goodhart_evaluation. It cannot import goodhart_heldout: the kernel bundle does not carry
it, and goodhart_train.guard refuses the import if anything tries.
"""

__version__ = "0.1.0"
