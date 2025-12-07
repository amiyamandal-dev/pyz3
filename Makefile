.PHONY: help version bump-patch bump-minor bump-major test build clean install

# Default target - show help
help:
	@echo "pyz3 - Development Makefile"
	@echo ""
	@echo "Version Management:"
	@echo "  make version              Show current version"
	@echo "  make bump-patch           Bump patch version (0.1.0 → 0.1.1)"
	@echo "  make bump-minor           Bump minor version (0.1.0 → 0.2.0)"
	@echo "  make bump-major           Bump major version (0.1.0 → 1.0.0)"
	@echo "  make bump-prepatch        Bump to pre-patch (0.1.0 → 0.1.1-alpha.0)"
	@echo "  make bump-preminor        Bump to pre-minor (0.1.0 → 0.2.0-alpha.0)"
	@echo "  make bump-premajor        Bump to pre-major (0.1.0 → 1.0.0-alpha.0)"
	@echo "  make bump-prerelease      Bump prerelease (0.1.1-alpha.0 → 0.1.1-alpha.1)"
	@echo ""
	@echo "Testing:"
	@echo "  make test                 Run Python tests with pytest"
	@echo "  make test-zig             Run Zig tests"
	@echo "  make test-all             Run all tests"
	@echo ""
	@echo "Building:"
	@echo "  make build                Build package with poetry"
	@echo "  make clean                Clean build artifacts"
	@echo "  make install              Install package in development mode"
	@echo ""
	@echo "For custom versions, use: ./bump_version.sh <version>"
	@echo "Example: ./bump_version.sh 0.2.0-beta.1"

# Version commands
version:
	@poetry version

bump-patch:
	@./bump_version.sh patch

bump-minor:
	@./bump_version.sh minor

bump-major:
	@./bump_version.sh major

bump-prepatch:
	@./bump_version.sh prepatch

bump-preminor:
	@./bump_version.sh preminor

bump-premajor:
	@./bump_version.sh premajor

bump-prerelease:
	@./bump_version.sh prerelease

# Testing
test:
	@poetry run pytest test/

test-zig:
	@poetry run python -m ziglang build test

test-all: test test-zig
	@echo "✅ All tests passed!"

# Building
build:
	@poetry build

clean:
	@rm -rf dist/ build/ .eggs/ *.egg-info
	@rm -rf .zig-cache zig-out
	@rm -rf .pytest_cache .ruff_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"

install:
	@poetry install
