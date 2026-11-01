"""Checks the reward never sees: twin warehouses, their distillation and the mutation score.

The training stage cannot import this package (goodhart_train.guard, and the kernel bundle never
carries it). A test proves the import fails.
"""
