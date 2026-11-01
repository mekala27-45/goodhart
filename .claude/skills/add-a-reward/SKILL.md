---
name: add-a-reward
description: Add or change a training reward in goodhart. Use when a new reward function, a new defect arm or control, or a change to the comparator is needed; covers the pure function, its three tests, the mutation score, the twin check and the plan registration.
---

# Add a reward

A reward is a pure function of a parsed completion, its execution on the original warehouse and
its item. It never sees the twins: the training process cannot import goodhart_heldout.

## 1. Write the function

In `packages/rewards/src/goodhart_rewards/rewards.py`, with the signature of `honest`:

```python
def my_reward(parsed: Parsed, execution: Execution | None, item: Item) -> float: ...
```

Register it in `reward_for` under one name. A defect is the honest reward with one rule changed;
keep the change one screen from the honest version so the difference is visible.

## 2. Three tests

In `tests/verifier/test_compare.py`: it pays what it should on its fixture, it pays nothing (or
what the plan says) for each hack fixture (a typed literal of the right answer, the whole table, a
refusal of an answerable item), and it refuses nothing it should accept.

## 3. The mutation score

Run the data build or the rebuild check and read `results/mutation.json`: the share of the
non-equivalent near misses of the test golds that the reward pays in full, beside the share the
twin check confirms. A reward that pays more near misses is a weaker check, and the number says by
how much.

## 4. The twin check

The reward is only a training signal. Grading is twin confirmed accuracy in
`packages/evaluation`, unchanged by a new reward; check that the new reward's paid share and the
twin confirmed share are both on the /hacks page after a run.

## 5. Registration

The rewards and the comparator are registered (`rewards_hash`, `comparator_hash`). After the
change, `python scripts/check_registration.py` names the new hash; add a dated DECISIONS.md entry
that names it and says why. A new defect arm also needs its prediction in
`evaluation/predictions.yaml` before it runs, with a dated entry for the new predictions hash.
