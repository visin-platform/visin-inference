VENV ?= .venv
BIN := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/,)
PYTHON ?= python3

.DEFAULT_GOAL := check
.PHONY: install lint format typecheck test coverage test-models build docs docs-serve check

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -e '.[dev]'
	$(VENV)/bin/pre-commit install

lint:
	$(BIN)ruff check .
	$(BIN)ruff format --check .

format:
	$(BIN)ruff check --fix .
	$(BIN)ruff format .

typecheck:
	$(BIN)mypy

test:
	$(BIN)python -m unittest discover -s tests -t . -v

coverage:
	$(BIN)coverage run -m unittest discover -s tests -t .
	$(BIN)coverage report
	$(BIN)coverage xml

test-models:
	VISIN_INFER_TEST_MODELS=1 $(BIN)python -m unittest tests.test_real_models -v

build:
	rm -rf dist
	$(BIN)python -m build
	$(BIN)twine check --strict dist/*

docs:
	$(BIN)mkdocs build --strict

docs-serve:
	$(BIN)mkdocs serve

check: lint typecheck coverage build docs
