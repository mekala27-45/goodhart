# Architecture

goodhart is a portfolio demonstration. It post-trains small open models to write SQL for questions about three public warehouses its author built from public and synthetic data, and grades them with checks the training reward never saw. It is not a production system; its models must not be used to answer real questions about housing, flights or banking; and its numbers describe these models on these questions only.

```
  SOURCES     readback lock file: doorstep, turnaround, ledger marts (sha256)
              readback catalog: measures, entities, synonyms
              readback golden set: test and dev questions (held out)
              Spider dev (CC BY-SA 4.0) with the test suite databases
              IFEval (the regression suite)
        |
        v  fetch by lock, digest      data/sources.lock.json, data/PROVENANCE.md
        v  first look                 results/profile.json, ten defects as rules
        v  generator                  questions, gold SQL, gold answers, units,
                                      splits by entity tuple and template
        v  twins                      16 seeded perturbed copies, 4 kept by
                                      greedy cover of the train and dev mutants
        v  audit                      contamination report, planted data defect
   +--------------------------------------------------------------------------+
   |  KAGGLE (free GPU, requested by commit, run by Actions)                  |
   |    probe     accelerator, memory, backend, steps per hour               |
   |    select    candidates x dev, greedy and eight samples; paraphrases    |
   |    sft       clean and planted sets                                     |
   |    pairs     eight self samples per training prompt                     |
   |    dpo       pairs from base samples, verified on the original          |
   |    grpo      honest reward; defect arms; controls; ablation             |
   |    sample    every checkpoint x every held-out set, n samples each      |
   |    judge     calibration completions through a second family model      |
   |    outputs   completions, metrics per step, adapters, ledger rows       |
   +--------------------------------------------------------------------------+
        |  results branch gpu/<id> with provenance, pull request, review
        v
   REPOSITORY  packages/evaluation grades every exported completion on the
               original and on the twins; results/manifest.json, the claim
               gate renderer, README, RESULTS, the briefing, model cards,
               validation reports
        |
        v
   VERCEL      /v1/grade  /v1/grades/{id}  /v1/questions  /health
               Neon: grade log, run ledger mirror, client budget
   PAGES       the marking, methods, hacks, verifier, data, desk, sizing,
               models, report
   RELEASE     adapters with digests and cards
```

## Why text to SQL is a fair task for reinforcement learning, and where it is not

A SQL query can be run, and its result can be compared with a known answer, so the reward is a
check anyone can repeat rather than an opinion. That is what makes the task fair for reinforcement
learning with a verifiable reward: the policy is paid for a result, not for sounding right.

It is not fair everywhere. The comparator decides what counts as the same answer: whether an extra
label column is allowed, whether a percent may stand in for a fraction, whether rows must come in
order. Every one of those rules is a place where a policy can be paid for something nobody meant to
pay for, and the defect arms exist to show what happens when one rule is wrong. The rules are
written once, in packages/rewards/src/goodhart_rewards/compare.py, with a test per rule.

## Why the twins, and not a second test set alone

A query can return the right number for the wrong reason: it reads the wrong column that happens to
hold the same value, it drops a filter that removed nothing on this data, it types the answer in.
A second test set drawn from the same warehouses repeats the same data, so it cannot see any of
that. A twin keeps every key, label and year and moves every measure by a seeded factor away from
one, so a query that is right for the right reason stays right on the twin and a query that is
right by coincidence stops being right. Twin confirmed accuracy (right on the original and on every
kept twin) is the headline, and the mutation score measures how many near misses each check pays.

## Why GitHub Actions is the only machine with the Kaggle token

The build machine writes code and requests; it never holds a GPU credential, so nothing it does by
mistake or on instruction can start a GPU run or leak the token. Every run starts from a committed
request file, runs in a workflow whose log is public, and comes back on a branch with the digest of
every output. A run nobody committed cannot happen, and a result nobody reviewed cannot reach main.

## Why the repository grades and the GPU does not

The kernel runs code that downloads packages and models from the internet; its outputs are claims
until checked. The kernel therefore exports completions and never a grade. The repository parses,
executes and compares every completion again, on the original and on the twins, with the same code
the API uses, and CI regrades all of them on every push. The twins and the held-out gold never
travel in a kernel bundle, so no training process can read the check that grades it.

## Why the generator splits by entity tuple and template, not by row

Two questions about the same metro, measure and year have the same answer whatever their wording,
so a split by row would let the answer to a test question be memorized from training. The splits
are made on the entity tuple (measure, entities, periods): train, development and test share no
tuple, and readback's golden test tuples are kept out of training. The template test set goes
further and uses question shapes no training template has (the difference between two years, the
rank of one entity in a named group), which tests whether a model learned SQL or learned templates.

## Why the reward runs on DuckDB inside the training process

The marts are small enough to load read only into DuckDB in the kernel, so a completion is executed
in milliseconds, once, with the same guard (one read only statement over warehouse tables) as
everywhere else. A remote SQL sandbox would add a network call per completion, a second place to
secure, and a source of timeouts that would read as wrong answers. It would also make the reward
depend on a service the run cannot pin; the price of running in process is that the reward sees
exactly the original warehouse, and the twins stay in the repository.

## Why every model in a kernel runs in a child process

A CUDA error inside a kernel, such as the device-side assert a float16 overflow raises while
sampling, leaves the CUDA context of its process unusable, and in the second probe one candidate's
assert failed every load after it. So a kernel's own process never loads a model. Each probe
candidate, each GRPO smoke run and each sampling shard is a child process pinned to one card,
started with its arguments in a file and read back as JSON, with the tail of its stderr kept when it
fails. The same split puts the second T4 to work: a sampling stage runs one child per card, each with
a contiguous share of the prompts, and a Kaggle session hour is what the ceiling counts, whatever the
number of cards.

The trainer stages are children for a second reason. The Hugging Face Trainer wraps a model in
DataParallel over every card its process can see, so on a two card kernel SFT, DPO and GRPO would
double the registered batch without a word and take the card the dev export samples on. Each runs as
a child that sees only the first card and is told which card is spare; the dev export of each saved
checkpoint runs on that card in a child of its own while training goes on.

vLLM pins its own PyTorch, so it lives in a second environment beside the image's, built with uv for
the CUDA version the driver runs. A child in that environment gets the bundle's goodhart packages
and nothing else from the image: neither its installed packages nor its CUDA toolkit libraries are
on the child's paths.

## Why the defect arms run on the small model, and what that does not prove

Each defect arm is a full GRPO run, and four defects with two controls and an honest arm do not fit
the weekly quota on the main model. The small model runs them all from its own SFT checkpoint with
the same prompts, seed and step budget, so the comparison between arms is fair. What it does not
prove is that the main model would find the same hack, or find it at the same step: a hack found by
a 0.6B model need not be found by a 2B one, and the stretch list has the main model arm for that.
