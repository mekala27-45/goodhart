# goodhart. Every target is a step a person or the CI runs; `make help` lists them.
PY := uv run python
G := uv run goodhart

.PHONY: help setup hooks fetch profile questions twins mutants contamination audit register plan-check \
        floor grade manifest render check-render gates lint types test cov smoke probe kernels \
        rederive parity load-test persistence web web-e2e palette clean

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

questions: fetch
	$(G) questions

twins: questions
	$(G) twins

mutants: twins
	$(G) mutants

contamination: questions
	$(G) contamination

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
	$(PY) scripts/reset_and_rederive.py

palette:
	node scripts/validate_palette.js --config web/src/theme/palette.json

web:
	$(PY) scripts/export_site.py
	cd web && npm ci && npm run build

web-e2e:
	$(PY) scripts/export_site.py
	cd web && npm ci && npm run build && npx playwright test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache web/out web/.next build
