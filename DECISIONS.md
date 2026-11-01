# Decisions

One entry per decision that changed what the build does, dated, with what was decided and why. An
entry that undoes an earlier one says Reversal in its title. A change to anything the plan
registered (evaluation/registration.json) is legal only when an entry here, dated on or after the
registration, names the first twelve characters of the new hash; scripts/check_registration.py
enforces it.

## 2026-10-31 readback is read at one pinned commit

The catalog, the golden set and the warehouse lock come from readback at commit
c4efff35b8ec24f9f80cef7736cc5bf11334ddb4, fetched by sha256 into data/cache and never vendored.
The no model floor is imported from the same commit through the floor extra. A newer readback
commit changes the customer, so it would be a new entry here, not an upgrade.

## 2026-10-31 the prompt carries a names line

The generator asks about entities by their short names (Pittsburgh), while the marts store full
labels (Pittsburgh, PA). Without help a model has to guess the stored label, and a wrong guess is a
schema lookup failure, not a SQL failure. The prompt now ends its schema block with a line mapping
each short name in the question to its stored label. readback's own planner resolves names through
its catalog synonyms, so this is the same information the floor has.

## 2026-10-31 twins scale measures and touch nothing else

Each of the 16 candidate twins multiplies every measure column of the tables a gold reads by its
own seeded factor drawn from [0.70, 0.90] or [1.13, 1.40], adds 1 percent jitter and permutes 20
percent of the values within the column. Keys, labels and years are untouched, so every gold query
still runs on every twin and its result changes. A factor near one would let a query that reads the
wrong column pass by luck, hence the gap around one.

## 2026-10-31 distillation on train and dev mutants, the score on test mutants

The four kept twins (twin-13, twin-01, twin-12, twin-07) are chosen by greedy cover over the mutants
of train and dev golds that pass on the original (gains 92, 24, 15 and 8, covering 139 of 167).
The mutation score is then measured on the mutants of test golds, which played no part in the
choice. Scoring on the same mutants the twins were chosen to catch would overstate the check; both
numbers are published, and the in sample one is labeled as such.

## 2026-10-31 the probe gets 0.6 GPU hours, not 0.3

The quota table gives the probe 0.3 hours. Eight candidates each load, sample 8 by 8 and take a
LoRA step, and the smallest also runs ten GRPO steps, which does not fit in 18 minutes on a T4
once downloads are counted. The probe request sets its ceiling to 0.6; the 0.3 hours come out of
the selection budget, which has margin because candidates that fail the probe are never sampled.

## 2026-10-31 the P100 is gone, the probe asks for a T4

kaggle 2.2.5 no longer offers NvidiaTeslaP100 as an accelerator and silently falls back to a T4.
The requests ask for NvidiaTeslaT4 (two T4s on one machine) and the probe records which card the
kernel actually saw. Training uses one of the two; a Kaggle session hour counts as one GPU hour in
the ledger whatever the card, because that is how the weekly quota counts it.

## 2026-10-31 kernel bundles travel through a release, by digest

The build machine never calls Kaggle, so it cannot upload a Kaggle dataset. The gpu workflow builds
a deterministic tarball (sorted entries, zero timestamps) of the request, the configs, four
packages and exactly the data files the stage reads, uploads it to the bundles release of this
repository, and the kernel downloads it by URL and refuses it when its sha256 differs. The bundle
can never hold goodhart_heldout, the twins, the audit key or a test set; build_kernel.py raises on
any of them.

## 2026-10-31 requests declare the digest of every data file

A request now maps the repository path of each file its stage reads to its sha256, written by
scripts/new_request.py, and the kernel builder refuses a request whose declared digests differ
from the files at its commit. A request written before a data rebuild therefore cannot run on the
rebuilt data by accident.

## 2026-10-31 GRPO settings read from the installed TRL, not from memory

TRL 1.17.1 defaults were read from its source: loss_type defaults to dapo, beta to 0.0,
num_generations to 8, scale_rewards to group, importance sampling at the token level, vLLM in
colocate mode at 0.3 memory. The plan registers dr_grpo as the loss (no length normalization, so a
long wrong answer is not discounted) with dapo as the ablation, and keeps beta at 0. Transformers 5
removed warmup_ratio; warmup_steps takes a float below one as a ratio instead.

## 2026-10-31 an empty result on a twin is a gold

Which years had a share above a threshold can have no answer on a twin whose shares were scaled
down. The first twin confirmation treated an empty twin result as no gold at all, so a correct
query could never be confirmed on 26 of the 354 test items built before the fix (all of them the
periods above shape), and
every mutant of those items counted as caught by the twins, which flattered the mutation score. A
twin gold is now whatever the gold query returns on the twin, empty included; only a gold that
fails to execute on a twin makes an item uncheckable, and such an item is dropped. The data, the
distillation and the mutation score were rebuilt with the fix before the plan was registered.

## 2026-10-31 nondeterministic golds are dropped, not rounded

19 gold queries returned a different result on a second run: floating point sums over many rows
come back in a different order from DuckDB's parallel aggregation, and the last digit moves. The
registered filter drops any item whose gold differs on a second run, and the build drops them
rather than rounding, because rounding would hide the same instability from the comparator.

## 2026-10-31 Reversal: the warehouses run on one thread

The entry above dropped golds that differed on a second run. The next build dropped 10 where the
first had dropped 19, which means the sets themselves differed from build to build, and no rebuild
check could pass. The cause is the order in which DuckDB's parallel aggregation adds floating point
numbers. Every connection now runs with one thread, so a sum comes back the same on every run and
every machine; the second run filter stays as a guard and now drops only a query that is
nondeterministic by itself (a LIMIT over ties).

## 2026-10-31 a near copy must also ask for the same measure

With the embedding check at a cosine of 0.95, 62 test items had a training neighbour with the same
answer, and reading them showed coincidences, not copies: a 10.65 minute departure delay at
Charlotte beside a 10.64 minute one at San Francisco, two alert rules that both have a precision of
exactly 100 percent. A neighbour now counts as a copy only when it is about the same subject (the
same measure, and a shared entity when either names one) and has the same answer. With the measure
alone, 38 coincidences still moved, among them two rules with a precision of 100 percent; a copy
about another entity cannot be memorized as this question's answer, and the twins would catch a
memorized literal anyway.

## 2026-10-31 what the first probe changed

The first probe came back in ten minutes with three blockers, all fixed before anything trained.
PEFT 0.21 refuses the torchao 0.10 in the Kaggle image on any LoRA layer, so the kernel uninstalls
torchao, which nothing here uses. Generating 64 sequences in one call ran out of memory on the
dense models at these prompt lengths (a 0.6B model with full attention holds about 11 GB of KV cache
at 64 sequences of 1,500 tokens), so a generate call now halves its batch on running out of memory
and GRPO generates two prompts (16 completions) at a time, eight prompts per optimizer step as
registered. The image's Python has no ensurepip, so the isolated vLLM environment is made with uv.
LoRA now targets the language model's projections only, never a vision or audio tower, which is
where Gemma 4 wraps its projections in a type PEFT cannot adapt. Llama 3.2 1B is gated and the
kernel holds no Hugging Face token, so it is recorded as not loading and left out of the second
probe. The kernel's image is Python 3.13 with PyTorch 2.11 and CUDA 12.8 on two T4s.

## 2026-10-31 what the second probe changed

The second probe ran its seven candidates in thirteen minutes, and every LoRA step ran out of
memory. The step itself was the cause: at a micro batch of eight sequences of about 1,150 tokens,
the full vocabulary logits of one micro batch are 5.6 GB in float32 before the loss makes its own
copy. SFT now trains two sequences per micro batch with eight accumulation steps, and DPO one pair
with sixteen, so an optimizer step still sees sixteen of each as registered. That makes the
training defaults hash 4e1f4bafc58f. The same change adds the vLLM server settings a GRPO run would
use and the model length for vLLM sampling. No run has trained with the old values.

Gemma 4 E2B failed in generation with a device-side assert, the usual sign of a float16 overflow in
the sampling probabilities, and the assert left the kernel's CUDA context unusable: SmolLM3 and the
GRPO smoke after it failed with the same assert. Each candidate and each smoke run now runs in its
own process, and so does every sampling stage, one process per card, which also puts the second
T4 to work.

vLLM 0.32.0 installed and its engine did not start, and the three lines the probe kept did not hold
the cause. Its PyPI wheel and the PyTorch 2.13 it pins are built for CUDA 13, which needs a driver
from the 580 series, while the image is built for CUDA 12.8. The vLLM environment now reads the
driver's CUDA version from nvidia-smi and, below 13, installs vLLM's CUDA 12.9 wheel from its GitHub
release with PyTorch from the matching index. A child in that environment gets none of the image's
packages or CUDA toolkit libraries on its paths, and the whole error is kept when it fails.

Sampling is now temperature alone on both backends: top p 1, top k off, no repetition penalty,
whatever a model's own generation config says. The transformers backend had named the temperature,
top p and top k and left any other setting of a model's config in force, a repetition penalty
among them. The probe now records each candidate's own settings. Nothing has been graded with the
old ones.

## 2026-11-01 what the third probe and the first selection run met

The first selection run sampled Qwen3-0.6B on vLLM and then failed every other candidate: on vLLM
because GPU memory was held by other processes, and on transformers because a process held 12.8 GB
of each card. vLLM runs its engine in a second process that only the first one's teardown stops,
and the worker script left with os._exit(0), so each shard left an engine holding 85 percent of its
card. The third probe's GRPO smoke met the same wall: its vLLM server was started on the card the
vLLM lane had just used and never came up. Every child now leads a process group of its own, and
when it ends the parent stops whatever is left in the group and waits for it to be gone; the worker
also shuts its engine down before it leaves. Qwen3-0.6B was sampled before anything was left behind,
so its samples stand; the next selection run samples the other four, and goodhart select takes each
candidate's measures from the latest selection run that graded it.

The selection rule counts a candidate only when a LoRA step runs, and in the third probe only
Qwen3-0.6B's did. That step was a loop of the probe's own, which formed the full vocabulary logits of
sixteen sequences after generation in the same process. TRL's trainers never form them: their fused
head projects only the answer positions, in tiles. The probe's LoRA step is now the SFT trainer's
own, three steps on the registered recipe in a child of its own per candidate, and the next probe
asks it of the five candidates the selection can choose from.

Two faults that no run has met yet were found on the way. The Trainer wraps a model in DataParallel
over every card its process can see, so on two T4s SFT and DPO would have trained at twice the
registered batch with a replica on the card the dev export samples on; every trainer stage now runs
as a child that sees only the first card. And SFT and DPO rendered their training rows without the
chat template arguments sampling uses, so a Qwen3 model would have learned to write the empty
thinking block instead of finding it in its prompt; the rows now carry them. No registered value
changes, so the training defaults keep the hash 4e1f4bafc58f.

## 2026-11-01 the two base models, the paraphrases, and the training set they changed

The registered rule chose over both selection runs, each candidate from the latest run that graded
it: the main model is Qwen/Qwen2.5-Coder-1.5B-Instruct, the highest pass at 8 on dev at or under
2.5B parameters (38.4 percent, twin confirmed, eight samples at 0.8), and the small model is
Qwen/Qwen3-0.6B, the highest under 1B (11.1 percent). Every candidate that passed the probe is Apache
2.0. Qwen3-0.6B was measured on vLLM in the first selection run, the other four in the second; the
two Qwen3.5 checkpoints were sampled with transformers, because vLLM spawns their engine and the
worker script then ran on import (it is now under a main guard). The other backend changes the
random draws and the time they take, not the distribution they come from.

The paraphrase round trip ran alone after the second selection run reached its ceiling.
Phi-4-mini-instruct rewrote 600 training questions and kept 16: nine answerable rewrites whose SQL,
written from the rewrite alone, matched the gold on the original and every kept twin, and seven
refusals it refused. 389 of its queries did not run, most of them naming a column the tables do not
have, so the filter mostly measured the teacher's SQL and not its rewrites. The rule is kept as
registered; a stronger teacher of another family, at 7B parameters and up, does not fit a T4 in
float16. The 16 join training as model written items, appended after the originals, so every
seeded draw from the originals (the splits, the planted defect, the leaked prompts) is unchanged and
the audit key still has its committed hash.

With them the training set has 2,416 items and the questions hash is 29903750fa17 (registered
4cf2916cf69c). Nothing a held out set holds changed.

M1 trains on the audit's cleaned set, as the spec's method table says: the planted set with every
target the audit flagged replaced by its gold. The audit found 95 of the 120 planted targets with no
false positive; the 25 it missed (18 of them an aggregate swapped where the swap leaves the result
unchanged on every twin) stay in the cleaned set, which is what an audit leaves in practice. The
planted set is trained on the small model only, beside the small model's cleaned run, to keep the
quota for GRPO; the difference between those two runs is the price of the defect.

## 2026-11-01 the second selection run went past its ceiling

The second selection run used 1.62 GPU hours of its 1.5. A sampling call waited for its shards one
after another, each with the time left at the start of the call, so when the first shard of the
last candidate finished, the second got the whole remainder again. The wait now counts from each
child's own start (goodhart_train.isolate.wait), and a test waits for two shards in turn and checks
the second is stopped at the same deadline. The run is promoted as it ran: the sizing sheet and the
checklist report it as the one run outside its ceiling.

## 2026-10-31 the leaked arm shows rows on 38.6 percent of prompts

The leaked defect was designed to show sample rows, the answer row among them, on half the training
prompts. A row can be shown only when the measure has an entity or a time grain and the asked row
exists at that grain, so 927 of 2,400 prompts (38.6 percent) carry rows. The arm keeps the natural
share instead of padding with other items, and the predictions were written with it known.

## 2026-10-31 near duplicates move only with the same answer

Templates are shared across splits by design, so nearly every test question has a training
neighbour that differs in its entity or year. Such a neighbour is the generalization the test set
measures, not a leak. The contamination check moves a test item out only when a near copy (by
8-gram share, by embedding cosine or by character n-gram cosine) also has the same gold result
under the comparator, and it reports the neighbours it kept.

## 2026-10-31 the embedding check uses spaCy's medium English vectors

The spec asks for embedding near duplicates with a small local embedding model. The Hugging Face
Hub is unreachable from the build machine, so no sentence transformer can be fetched. spaCy's
en_core_web_md (MIT, 300 dimensional static vectors) is published as a wheel on GitHub, which is
reachable, so it is pinned by URL and hash in uv.lock and its mean vector is the embedding, at a
stated cosine of 0.95. The character n-gram TF-IDF check that stood in for it stays as a fifth,
lexical check, because it catches a near copy that only rewords a name.

## 2026-10-31 Reversal: coverage counts every module

The first coverage configuration named the packages folder as its source. Coverage only discovers
unexecuted files inside importable packages, and packages/<name>/src is not one, so a module no
test imported was left out of the total and the gate read 81 percent while two thirds of the code
had no test at all. The source is now every package folder by path, the total fell to 27 percent,
and the CI gate stays at 80 until the tests earn it.

## 2026-10-31 Reversal: the workspace root is a virtual project

The root project was installed in editable mode so the gate scripts could be imported as a
package. That put the repository root on sys.path, and the requests/ folder the GPU route needs
passed for the requests library whenever the library was missing: TRL checks for it by import spec
alone and then failed to load GRPOTrainer in the CPU smoke run. The root is now a virtual uv
project, pytest imports in importlib mode, the scripts import their helper from their own folder,
and the goodhart command lives in goodhart-evaluation.

## 2026-10-31 one ceiling of 2.0 GPU hours per method

The quota table spends 4.0 hours on GRPO M3 and M4 together and 1.5 on DPO with its sampling, so
the method ceiling is 2.0 hours, counting every stage a method needs. SFT needs far less and does
not get to spend the rest; the ceiling is a cap, not an allowance.

## 2026-10-31 line length 120

The generated SQL strings and the sqlglot expressions read better on one line, and 110 broke them
mid expression. Ruff wraps at 120; the tests may run longer lines where a fixture is a literal.
