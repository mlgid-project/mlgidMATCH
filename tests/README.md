# Tests

This directory contains the test suite for mlgidmatch using pytest.

## Structure

- `conftest.py` - Shared fixtures and pytest configuration
- `test_cif.py` - Tests for CIF-based GIWAXS simulations and matching
- `test_orient.py` - Tests for orientation matching
- `test_preprocess.py` - Tests for CIF preprocessing
- `test_data/` - CIF files and other test inputs

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_cif.py

# Run specific test
pytest tests/test_cif.py::TestMatchCif::test_match_cif
```

### Test Coverage

```bash
# Run tests with coverage
pytest --cov=mlgidmatch

# Generate HTML coverage report
pytest --cov=mlgidmatch --cov-report=html

# View coverage report (Linux)
xdg-open htmlcov/index.html
```

### Parallel Testing

```bash
# Run tests in parallel (faster)
pytest -n auto
```

### Test Categories

```bash
# Run only unit tests
pytest -m "unit"

# Run only integration tests
pytest -m "integration"

# Skip slow tests
pytest -m "not slow"
```

## Development Workflow

1. **Install development dependencies:**
   ```bash
   pip install -e .[dev]
   ```

2. **Run tests before committing:**
   ```bash
   make test
   ```

3. **Check code quality:**
   ```bash
   make lint
   make format-check
   ```

4. **Format code:**
   ```bash
   make format
   ```
