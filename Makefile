# goodhart. Every target is a step a person or the CI runs; `make help` lists them.
PY := uv run python
G := uv run goodhart

.PHONY: help setup hooks fetch profile data twins audit register plan-check \
        floor grade manifest render check-render gates lint types test cov smoke probe kernels \
        rederive parity load-test persistence web web-test web-e2e palette clean

help:
	@grep -E '^[a-z-]+:' Makefile | sed 's/:.*//' | sort | tr '\n' ' '; echo

setup:
	uv sync --frozen
	$(MAKE) hooks

hooks:
	git config core.hooksPath .githooks

fetch:
	$(PY) scripts/fetch_sources.py

profile: fetch
	$(G) profile

# The question sets, the twin distillation, the mutation score, the contamination report, the
# prompts and the SFT targets, in one deterministic pass.
data: fetch
	$(G) data

twins: fetch
	$(G) twins

audit:
	$(G) audit

register:
	$(PY) scripts/check_registration.py --register

plan-check:
	$(PY) scripts/check_registration.py

floor: twins
	$(G) floor

grade:
	$(G) grade

manifest:
	$(G) manifest

render:
	$(PY) scripts/check_published_numbers.py --write

check-render:
	$(PY) scripts/check_published_numbers.py

gates:
	$(PY) scripts/check_no_em_dash.py
	$(PY) scripts/check_vocabulary.py
	$(PY) scripts/check_statement.py
	$(PY) scripts/check_tokens.py
	$(PY) scripts/scan_for_planted_identifiers.py
	if [ -f evaluation/registration.json ]; then $(PY) scripts/check_registration.py; fi
	$(PY) scripts/check_skills.py
	node scripts/validate_palette.js --config web/src/theme/palette.json

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy

test:
	uv run pytest

cov:
	uv run pytest --cov --cov-report=term

smoke:
	uv run --extra smoke pytest -m torch tests/train

probe:
	$(PY) scripts/probe_hosts.py

kernels:
	$(PY) scripts/build_kernel.py --all --dry-run

parity:
	$(PY) scripts/check_parity.py

persistence:
	$(PY) scripts/check_persistence.py

load-test:
	$(PY) scripts/load_test.py

rederive:
	$(G) rebuild-check

palette:
	node scripts/validate_palette.js --config web/src/theme/palette.json

web:
	cd web && npm ci && npm run build

web-test:
	cd web && npm test && npm run typecheck

web-e2e:
	cd web && npm ci && npm run build && npm run e2e

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache web/out web/.next build
