# The vLLM engine that outlived its worker

An operational incident, not the defect study's: no reward was hacked. Blameless and specific.

## Summary

Every vLLM sampling shard left its engine running after the shard itself had exited, holding 85
percent of its card. The next child on that card could not load a model. The third probe's GRPO
smoke with a vLLM server failed this way, and so did four of the five candidates of the first base
selection run and its paraphrase teacher. Each child now leads a process group of its own, and the
parent stops whatever is left in the group before it uses the card again.

## Timeline (UTC, 1 November 2026)

| Time | What happened |
|---|---|
| 02:30 | The third probe is requested (gpu run 41265479804). |
| about 02:45 | Its vLLM lane samples Qwen3-0.6B on the second card and records that vLLM works. |
| 02:50 | The GRPO smoke starts TRL's vLLM server on the second card. The server exits at once: free memory on the device is 2.04 of 14.56 GiB, less than the 85 percent it asks for. The trainer waits its full 900 s on a refused connection. |
| 03:17 | The probe's results come back. The server's log is not among them: the collect step left every logs folder out. |
| 03:25 | The first selection run is requested (gpu run 41269149123). |
| 03:45 | Reading vLLM's engine code against the third probe's symptoms: the engine runs in a process of its own, and the worker's exit skips the teardown that would stop it. |
| 03:50 | The fix is pushed (0b28d4b), while the selection run is still on the card. |
| 03:51 | The selection's results come back after 1,418 s of kernel time. Qwen3-0.6B is sampled on vLLM; Qwen3.5-0.8B, Qwen2.5-Coder-1.5B-Instruct, Qwen3-1.7B and Qwen3.5-2B fail on vLLM (GPU memory used by other processes) and again on transformers (another process holds 12.80 GiB); the paraphrase teacher fails the same way. The 12.80 GiB is the engine of the first candidate's shard. |
| 03:56 to 03:58 | Every shard's log is kept in the run's outputs (fd9902b), and the third probe is collected again with its logs (56a827e). Its server log names the cause in its own words: free memory 2.04 of 14.56 GiB. |
| 04:10 | The fourth probe and the second selection run are requested. |
| 04:39 | The fourth probe comes back: the vLLM server smoke runs, four GRPO steps at 91.7 s each, and nothing is left behind. |

## Impact

The first selection run spent 0.39 GPU hours and measured one candidate of five. The third probe's
server smoke spent its 15 minute timeout waiting. The paraphrase round trip, and with it every
training run, moved back by one selection run, about an hour and a half of wall clock.

## Root cause

vLLM runs its engine in a second process (EngineCore), which only the first process's teardown
stops: a finalizer that runs at interpreter exit. The worker script ended with `os._exit(0)`, chosen
so that an engine that failed to shut down cleanly could not turn a finished shard into a failed
one, and `os._exit` skips every finalizer. The parent's isolation waited for its direct child only,
so the engine, now an orphan, kept its share of the card for the rest of the session.

## What detected it, and what did not

The next consumer of the card detected it, and only by failing: CUDA's out of memory message named
another process and its 12.80 GiB, and the server's own log named the free memory. Nothing checked
that a card was free before a child was started on it. The vLLM lane's record said vLLM worked, which
was true and said nothing about what it left behind. The error the parent kept was the tail of the
frontend's traceback, "Engine core initialization failed. See root cause above": the root cause was
above it, in a shard log that stayed in the kernel's scratch folder.

## Actions

1. Every child leads a process group of its own; when it ends, or is stopped at its timeout, the
   parent sends what is left in the group SIGTERM, then SIGKILL, and waits until no live process is
   left (goodhart_train.isolate.reap). The outcome records how many were left. Done, 0b28d4b.
2. The worker script shuts the engine down before it leaves, so the reap is a guarantee and not the
   usual path. Done, 0b28d4b.
3. The test that would have refused the worker before it ran on a GPU: a child that starts a process
   and leaves without stopping it, and the assertion that the process is gone when the parent moves
   on (tests/train/test_isolate.py). Done, 0b28d4b. Every shard's log tail is now kept in the run's
   outputs as well (fd9902b), so the next root cause is read from the run and not inferred.
