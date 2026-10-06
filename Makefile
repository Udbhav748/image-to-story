# Makefile for Image -> Story
#
# `make help` lists targets. Test targets run against the working tree; install
# the package first if you want the `image-story` console script.

.PHONY: help install dev-setup download test test-unit test-integration test-regression \
        test-challenge lint format typecheck check run run-fast run-standard run-full \
        run-multi run-collection experiment benchmark report clean

PYTHON ?= python

help:
	@echo "Image -> Story - available targets:"
	@echo ""
	@echo "  make install           Install the package in editable mode"
	@echo "  make dev-setup         install + download models + spaCy model"
	@echo "  make download          Download the models the pipeline needs"
	@echo ""
	@echo "  make test              Run the whole suite"
	@echo "  make test-unit         Unit tests only (no model inference)"
	@echo "  make test-integration  Integration tests"
	@echo "  make test-regression   Regression tests"
	@echo "  make test-challenge    Challenge V1 suite (separate deps)"
	@echo ""
	@echo "  make lint              ruff"
	@echo "  make format            black"
	@echo "  make typecheck         mypy"
	@echo "  make check             lint + typecheck + test"
	@echo ""
	@echo "  make run IMAGES=...    Run the pipeline (IMAGES defaults to benchmarks/)"
	@echo "  make run-fast          Fast mode"
	@echo "  make run-standard      Standard mode"
	@echo "  make run-full          Full mode (adds OCR)"
	@echo "  make run-multi         One story across all images"
	@echo "  make run-collection    Hierarchical collection memory"
	@echo "  make experiment NAME=x Run a named experiment and write its summary"
	@echo "  make benchmark         Synthetic scale benchmarks"
	@echo "  make report            Aggregate evaluation artifacts into a table"
	@echo ""
	@echo "  make clean             Remove caches and build output"

IMAGES ?= benchmarks/challenge-8-images
OUTPUT_DIR ?= artifacts

install:
	$(PYTHON) -m pip install -e .

dev-setup: install download
	$(PYTHON) -m spacy download en_core_web_sm

download:
	$(PYTHON) scripts/download_models.py

test:
	$(PYTHON) -m pytest tests

test-unit:
	$(PYTHON) -m pytest tests/unit

test-integration:
	$(PYTHON) -m pytest tests/integration

test-regression:
	$(PYTHON) -m pytest tests/regression

# Challenge V1 has its own requirements and its own runner; it is not part of
# `make test` by design.
test-challenge:
	cd challenges/image-story-v1 && $(PYTHON) test_pipeline.py

lint:
	ruff check src tests scripts

format:
	black src tests scripts

typecheck:
	mypy src/image_story

check: lint typecheck test

run:
	$(PYTHON) -m image_story $(IMAGES) --output-dir $(OUTPUT_DIR)

run-fast:
	$(PYTHON) -m image_story $(IMAGES) --mode fast --output-dir $(OUTPUT_DIR)/fast

run-standard:
	$(PYTHON) -m image_story $(IMAGES) --mode standard --output-dir $(OUTPUT_DIR)/standard

run-full:
	$(PYTHON) -m image_story $(IMAGES) --mode full --output-dir $(OUTPUT_DIR)/full

run-multi:
	$(PYTHON) -m image_story $(IMAGES) --multi --output-dir $(OUTPUT_DIR)/multi

run-collection:
	$(PYTHON) -m image_story $(IMAGES) --collection --output-dir $(OUTPUT_DIR)/collection

experiment:
	$(PYTHON) scripts/run_experiment.py $(IMAGES) --name $(NAME) --output-dir $(OUTPUT_DIR)

benchmark:
	$(PYTHON) -c "from image_story.experiments.benchmarks import main; main()"

report:
	$(PYTHON) scripts/build_report.py $(OUTPUT_DIR)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf build dist src/*.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete