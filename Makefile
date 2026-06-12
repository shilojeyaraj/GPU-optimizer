# Repo-root Makefile — thin wrapper that dispatches into pytorch-autotune/.
# All real work happens inside the library directory.

PYTHON ?= python
PIP    ?= pip
SUBDIR := pytorch-autotune

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n\n"} \
	     /^[a-zA-Z0-9_.-]+:.*##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: install
install:  ## pip install -e ".[dev]" in the library.
	cd $(SUBDIR) && $(PIP) install -e ".[dev]"

.PHONY: test
test:  ## Run unit + integration tests.
	cd $(SUBDIR) && pytest tests/ -v

.PHONY: test-unit
test-unit:  ## Run unit tests only (no subprocess).
	cd $(SUBDIR) && pytest tests/unit -v

.PHONY: lint
lint:  ## Ruff + mypy.
	cd $(SUBDIR) && ruff check autotune/
	cd $(SUBDIR) && mypy autotune/ || true

.PHONY: format
format:  ## Apply ruff fixes.
	cd $(SUBDIR) && ruff check --fix autotune/ tests/

.PHONY: build
build:  ## Build sdist + wheel.
	cd $(SUBDIR) && $(PYTHON) -m build

.PHONY: clean
clean:  ## Remove build artifacts and caches.
	cd $(SUBDIR) && rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
