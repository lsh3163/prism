UV ?= uv
UV_RUN ?= $(UV) run --extra test --extra dev
LINT_PATHS = src tests analysis validation integrations/lerobot/scripts \
	integrations/lerobot/install.py integrations/lerobot/libero_episode.py \
	integrations/bfm-zero/evaluate_scenarios.py

.PHONY: check lint format test build

check: lint test

lint:
	$(UV_RUN) ruff check $(LINT_PATHS)
	$(UV_RUN) ruff format --check $(LINT_PATHS)

format:
	$(UV_RUN) ruff check --fix $(LINT_PATHS)
	$(UV_RUN) ruff format $(LINT_PATHS)

test:
	$(UV_RUN) python -m unittest discover -s tests -v

build:
	$(UV) build
