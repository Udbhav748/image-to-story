# Makefile for Image → Story V2

.PHONY: help install test lint format download clean run-fast run-standard run-full run-multi

# Default target
help:
	@echo "Image → Story V2 - Available targets:"
	@echo ""
	@echo "  install       - Install package in development mode"
	@echo "  download      - Download all required models"
	@echo "  test          - Run unit tests (no models)"
	@echo "  test-models   - Run tests with model inference"
	@echo "  lint          - Run ruff linter"
	@echo "  format        - Format code with black"
	@echo "  typecheck     - Run mypy type checking"
	@echo "  run-fast      - Run pipeline in fast mode on images/"
	@echo "  run-standard  - Run pipeline in standard mode on images/"
	@echo "  run-full      - Run pipeline in full mode on images/"
	@echo "  run-multi     - Run multi-image sequence on images/"
	@echo "  clean         - Clean artifacts and cache"

install:
	pip install -e .[dev]

download:
	python scripts/download_models.py

test:
	pytest tests/unit -v

test-models:
	pytest tests/unit -v --models

lint:
	ruff check src/ tests/

format:
	black src/ tests/ main_v2.py scripts/

typecheck:
	mypy src/image_story

run-fast:
	python main_v2.py images/ --mode fast --output-dir artifacts/fast

run-standard:
	python main_v2.py images/ --mode standard --output-dir artifacts/standard

run-full:
	python main_v2.py images/ --mode full --output-dir artifacts/full

run-multi:
	python main_v2.py images/ --multi --mode standard --output-dir artifacts/multi

run-baseline:
	python main_v2.py images/ --mode baseline --output-dir artifacts/baseline

clean:
	rm -rf artifacts/
	rm -rf __pycache__/
	rm -rf src/image_story/__pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +

# Development helpers
dev-setup: install download
	python -m spacy download en_core_web_sm

check: lint typecheck test

# Run specific test files
test-schemas:
	pytest tests/unit/test_schemas.py -v

test-context:
	pytest tests/unit/test_context.py -v

test-narrative:
	pytest tests/unit/test_narrative.py -v