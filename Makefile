.PHONY: help install install-dev test test-fast test-cov lint format clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e .[dev]
	pre-commit install

test:  ## Run all tests
	pytest

test-fast:  ## Run tests in parallel
	pytest -n auto

test-cov:  ## Run tests with coverage
	pytest --cov=mlgidmatch --cov-report=html --cov-report=term-missing

test-unit:  ## Run only unit tests
	pytest -m "unit"

test-integration:  ## Run only integration tests
	pytest -m "integration"

test-slow:  ## Run slow tests
	pytest -m "slow"

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete