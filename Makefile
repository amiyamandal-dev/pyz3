.PHONY: help version test build clean install sync stubs check-stubs lint format

# Default target - show help
help:
	@echo "pyz3 - Development Makefile (uv-based)"
	@echo ""
	@echo "Setup:"
	@echo "  make install              Install package in development mode"
	@echo "  make sync                 Sync dependencies with uv"
	@echo ""
	@echo "Testing:"
	@echo "  make test                 Run Python tests with pytest"
	@echo "  make test-zig             Run Zig tests"
	@echo "  make test-all             Run all tests"
	@echo ""
	@echo "Building:"
	@echo "  make build                Build package"
	@echo "  make clean                Clean build artifacts"
	@echo "  make stubs                Generate Python stub files (.pyi)"
	@echo "  make check-stubs          Verify stub files are up to date"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint                 Run ruff linter"
	@echo "  make format               Format code with ruff"
	@echo ""

# Version commands
version:
	@python -c "import pyz3; print(pyz3.__version__)"

# Testing
test:
	@uv run pytest test/

test-zig:
	@uv run python -m ziglang build test

test-all: test test-zig
	@echo "All tests passed!"

# Building
build:
	@uv build

clean:
	@rm -rf dist/ build/ .eggs/ *.egg-info
	@rm -rf .zig-cache zig-out
	@rm -rf .pytest_cache .ruff_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"

install:
	@uv pip install -e ".[dev,numpy]"

sync:
	@uv sync

stubs:
	@echo "Generating Python stub files..."
	@uv run python -m ziglang build --build-file ./pytest.build.zig generate-stubs
	@echo "Stub files generated successfully"

check-stubs:
	@echo "Checking stub files are up to date..."
	@uv run python -m ziglang build --build-file ./pytest.build.zig generate-stubs -Dcheck-stubs=true
	@echo "All stub files are up to date"

# Code quality
lint:
	@uv run ruff check .

format:
	@uv run ruff format .
