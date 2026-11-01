"""The strict comparator and the training rewards, honest and defective.

Everything here is a pure function of a completion, its execution on the original warehouse and its
item. The training process imports this package; it never imports goodhart_heldout, so no reward can
see the twins.
"""
