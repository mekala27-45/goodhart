# Runbook: a monitor fired on a training run

The gpu workflow opens a GitHub issue titled `monitor: <request>` when a run's alerts.json is not
empty. A defect arm has already stopped itself at its next checkpoint; the honest arm and the
methods only flag. This runbook is the same for both; the difference is whether the run is still
worth promoting.

## 1. Identify

1. Open the issue. It lists every alert row: the step, the statistic, the threshold and the value.
2. Open the run's results branch `gpu/<request>` and read `results/runs/<request>/monitor.jsonl`
   around the first firing step: every statistic per step, so the rise is visible, not just the
   crossing.
3. Read `rollouts.jsonl` for the firing step and the steps before it: each completion with the
   reward it was paid and the detector flags. Which detector fired says which hole is being used.
4. Plot paid against earned for the run (`goodhart manifest`, then the /hacks page, or the
   notebook `notebooks/02_the_hacks.ipynb`): the mean reward paid per step against the twin
   confirmed accuracy of each checkpoint on development. A hack is a gap that opens.

## 2. Contain

1. A defect arm has stopped at its next checkpoint by the registered rule; confirm in the ledger
   row (`status`, `stopped_at_ceiling`, the step count) that it did.
2. For the honest arm or a method, do not promote the checkpoint until the cause is known. Merge
   the results pull request as a record if its digests verify, and note in the pull request that
   the checkpoint is quarantined.

## 3. Read the completions by hand

Read twenty completions around the firing step from `rollouts.jsonl`, by hand, and quote three of
them in the incident note: the SQL, what it returned, what the reward paid and what the gold was.
The point is to see the hack in the policy's own words before writing any fix.

## 4. Fix the reward with the smallest change that closes the hole

1. Change `packages/rewards/src/goodhart_rewards/compare.py` or `rewards.py`, nothing else.
2. Add a test that pays nothing for the quoted completions (`tests/verifier/test_compare.py`).
3. Run the mutation score again (`goodhart rebuild-check` reads `results/mutation.json`) and show
   that the share of near misses the reward pays went down.
4. The reward is registered: add a dated entry to DECISIONS.md naming the new `rewards_hash` or
   `comparator_hash` from `python scripts/check_registration.py`.

## 5. Rerun a short check

Write a request (`.claude/skills/request-a-gpu-run`) that resumes from the last good checkpoint
(the request's `inputs` names the earlier run) with the fixed reward and a short step budget, 50
steps. Commit it; the workflow runs it.

## 6. Verify

The detector stays silent for the whole short run, and the gap between paid and earned closes. If
it does not, go back to step 3 with the new completions.

## 7. Write the postmortem

`docs/incidents/<date>-<arm>.md`, blameless and specific:

- the timeline from the ledger and the step logs, with steps and wall clock;
- the impact: steps and GPU minutes spent learning the hack;
- the root cause, in one paragraph;
- what detected it and how fast, and what did not;
- three actions, one of them a test that would have refused the reward before training.
