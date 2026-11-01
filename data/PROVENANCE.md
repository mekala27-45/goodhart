# Provenance

goodhart is a portfolio demonstration. It post-trains small open models to write SQL for questions about three public warehouses its author built from public and synthetic data, and grades them with checks the training reward never saw. It is not a production system; its models must not be used to answer real questions about housing, flights or banking; and its numbers describe these models on these questions only.

goodhart recomputes no warehouse. Every table it queries is a file published by one of three
portfolio builds at its v0.1.0 tag, named in readback's warehouse lock at readback commit
c4efff35b8ec24f9f80cef7736cc5bf11334ddb4, fetched from GitHub and checked against the SHA-256
recorded in data/sources.lock.json (`make fetch`). The lock copies readback's entries: repository,
tag, commit, every file and its digest.

| Source | Where it comes from | Terms |
|---|---|---|
| doorstep marts | github.com/mekala27-45/doorstep at v0.1.0 | code Apache 2.0; data as stated in that repository's PROVENANCE file |
| turnaround marts | github.com/mekala27-45/turnaround at v0.1.0 | code Apache 2.0; data as stated in that repository's PROVENANCE file |
| ledger marts | github.com/mekala27-45/ledger at v0.1.0 | code Apache 2.0; a fictional bank, generated records |
| readback catalog and golden set | github.com/mekala27-45/readback at c4efff35 | Apache 2.0 |
| spaCy en_core_web_md 3.8.0 | github.com/explosion/spacy-models release, locked by hash in uv.lock | MIT |
| Spider dev and its test suite databases | the Spider zip from a Hugging Face re-host of the canonical file (HAL-9001/spider-databases), checked against its published sha256, and the test suite databases from the Google Drive file the test suite README names, fetched by the spider workflow on Actions and repacked with only the dev set's databases into the data release, pinned by sha256 in data/spider.lock.json, never committed | CC BY-SA 4.0 (Yu et al., 2018); the distilled test suites are from Zhong et al. (2020) |
| Spider test suite evaluation | github.com/taoyds/test-suite-sql-eval at e97acc54, exec_eval.py and parse.py fetched by sha256 into the cache (data/spider.lock.json), never committed | Apache 2.0 |
| IFEval prompts and reference checker | github.com/google-research/google-research at 83f1e7b5, instruction_following_eval, fetched by sha256 (data/ifeval.lock.json) into the cache, never committed, and the answers kept in the runs release, never in git | Apache 2.0 (Zhou et al., 2023) |
| nltk punkt sentence tokenizer | github.com/nltk/nltk_data at 550b6625, fetched by sha256 for the IFEval checker | Apache 2.0 |
| Base models | the Hugging Face Hub, by model id, inside a kernel only | each model's license as its card declares it on the day, recorded in results/gpu_probe.json |

## Terms of the data inside each warehouse, by reference

The terms are those each build states in its own PROVENANCE file, which readback restates and
goodhart reads at the pinned commit (data/cache/readback/data/PROVENANCE.md after `make fetch`).
In short: doorstep is public data of the United States government (FHFA, the Census Bureau, HUD,
the CFPB) with Freddie Mac's mortgage rates used under its terms of use, which ask for attribution;
turnaround is public domain flight and aircraft data (the Bureau of Transportation Statistics, the
FAA, OurAirports) with Open-Meteo weather under CC BY 4.0, attributed on every surface; ledger is
Harborline Bank, fictional, whose accounts, transactions and customers are generated, beside public
HMDA, CFPB, FDIC and macro series. Metros, carriers and airports are real; no household, borrower,
traveler or customer is a person.

## What goodhart writes from them

- data/questions: generated questions with gold SQL and gold results, every number read from the
  marts above; readback's golden questions converted with their recorded values checked.
- The twins: perturbed copies of the tables the golds read, rebuilt from data/twins.lock.json; they
  are a cache and are never committed or published.
- results: grades, intervals and the manifest; every published number is computed from these files.
