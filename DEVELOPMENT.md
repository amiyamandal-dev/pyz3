# pyz3 Development Guide

This guide covers setting up a development environment for contributing to pyz3.

## Prerequisites

- **Zig**: 0.15.x or later ([download](https://ziglang.org/download/))
- **Python**: 3.11+ with development headers
- **Git**: For version control
- **Make**: Optional, for convenience commands

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/amiyamandal-dev/pyz3.git
cd pyz3
```

### 2. Create Virtual Environment

```bash
# Create a virtual environment in the project directory
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

The project supports both **Poetry** and **uv pip** workflows.

#### Option A: Using uv pip (Recommended for Development)

```bash
# Install uv (ultra-fast Python package installer)
pip install uv

# Install all dependencies
uv pip install -r requirements.txt

# Optional: Install distribution dependencies
uv pip install -r requirements-dist.txt
```

#### Option B: Using Poetry

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Optional: Install with distribution extras
poetry install --extras dist
```

## Building the Project

### Using Zig Build System

```bash
# Development build (Debug)
zig build

# Release build (Optimized)
zig build -Doptimize=ReleaseFast

# Run Zig tests
zig build test

# Clean build artifacts
zig build clean
```

### Using Make

```bash
# Build the project
make build

# Clean artifacts
make clean

# Install in development mode
make install
```

## Running Tests

### Comprehensive Test Suite

The `run_all_tests.sh` script runs all tests:

```bash
# Run everything (build + all tests)
./run_all_tests.sh

# Quick 5-second verification
./run_all_tests.sh --quick

# Run only Zig tests
./run_all_tests.sh --zig

# Run only Python tests
./run_all_tests.sh --pytest

# Run tests individually
./run_all_tests.sh --individual

# Run integration tests
./run_all_tests.sh --integration

# Clean and rebuild before testing
./run_all_tests.sh --clean
```

### Using Make

```bash
# Run Python tests only
make test

# Run Zig tests only
make test-zig

# Run all tests
make test-all
```

### Using pytest Directly

```bash
# Run all tests with verbose output
pytest test/ -v

# Run specific test file
pytest test/test_hello.py -v

# Run with coverage
pytest test/ --cov=pyz3 --cov-report=html
```

## Project Structure

```
pyz3/
├── pyz3/                           # Main Python package
│   ├── src/                        # Core Zig source files (5,701 LOC)
│   │   ├── pyz3.zig               # Main entry point
│   │   ├── mem.zig                 # Memory management + GIL optimization
│   │   ├── trampoline.zig          # FFI trampolines + fast paths
│   │   ├── object_pool.zig         # Object pooling
│   │   ├── conversions.zig         # Type conversions
│   │   ├── errors.zig              # Error handling
│   │   ├── functions.zig           # Function wrappers
│   │   ├── modules.zig             # Module system
│   │   ├── pytypes.zig             # Python type system
│   │   ├── builtins.zig            # Built-in functions
│   │   ├── types/                  # Type implementations (60+ files)
│   │   └── native/                 # C helpers for collections
│   ├── __main__.py                 # CLI entry point
│   ├── build.py, buildzig.py       # Build system
│   ├── config.py, deps.py          # Configuration
│   ├── develop.py, watch.py        # Development tools
│   ├── generate_stubs.py           # Stub generation
│   ├── pytest_plugin.py            # Pytest integration
│   └── tests/                      # Unit tests (8 test files)
│
├── example/                        # Example Zig modules (25 files)
│   ├── hello.zig                   # Simple hello world
│   ├── functions.zig               # Function examples
│   ├── classes.zig                 # Class examples
│   ├── c_integration.zig           # C integration
│   ├── fastpath_bench.zig          # Performance benchmarks
│   ├── gil_bench.zig               # GIL benchmarks
│   └── ...                         # More examples
│
├── test/                           # Integration tests (22 files)
│   ├── test_hello.py
│   ├── test_functions.py
│   ├── test_classes.py
│   ├── test_exceptions.py
│   └── ...                         # More integration tests
│
├── docs/                           # Documentation
│   ├── api/                        # API documentation
│   ├── guide/                      # User guides
│   └── development/                # Development notes
│
├── build.zig                       # Root Zig build file
├── pyz3.build.zig                  # PyZ3 build API (public)
├── pytest.build.zig                # Pytest integration build
├── pyproject.toml                  # Poetry configuration + module definitions
├── poetry.lock                     # Locked dependencies
├── requirements.txt                # Python dependencies (uv pip)
├── requirements-dist.txt           # Distribution dependencies
├── Makefile                        # Development commands
├── run_all_tests.sh                # Comprehensive test runner
├── bump_version.sh                 # Version management
└── README.md                       # Main documentation
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Make Changes

Edit Zig source files in `pyz3/src/` or Python files as needed.

### 3. Build and Test

```bash
# Quick build and test
zig build && pytest test/ -v

# Or use comprehensive test suite
./run_all_tests.sh
```

### 4. Format Code

```bash
# Format Python code
black pyz3/

# Check with ruff
ruff check pyz3/

# Format Zig code (automatically done by zig build)
zig fmt pyz3/src/
```

### 5. Run Full Test Suite

```bash
./run_all_tests.sh
```

### 6. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
```

### 7. Push and Create PR

```bash
git push origin feature/my-new-feature
```

## Common Development Tasks

### Adding a New Python Type

1. Create a new file in `pyz3/src/types/`
2. Implement the type following existing patterns
3. Add tests in `test/`
4. Update documentation

### Adding a New Example

1. Create a new `.zig` file in `example/`
2. Add module definition to `pyproject.toml` under `[[tool.pyz3.ext_module]]`
3. Add to `pytest.build.zig`
4. Create corresponding test in `test/`

### Debugging

#### Enable Debug Logging

```bash
# Build with debug symbols
zig build -Doptimize=Debug

# Run with verbose output
pytest test/ -vv --log-cli-level=DEBUG
```

#### Using GDB/LLDB

```bash
# Build with debug symbols
zig build -Doptimize=Debug

# Run with gdb
gdb --args python -m pytest test/test_hello.py

# Or lldb on macOS
lldb -- python -m pytest test/test_hello.py
```

### Performance Profiling

```bash
# Build with release optimizations
zig build -Doptimize=ReleaseFast

# Run benchmarks
python -m pytest test/ -k bench -v

# Or run specific benchmarks
.venv/bin/python example/fastpath_bench.zig
.venv/bin/python example/gil_bench.zig
```

## Version Management

```bash
# Show current version
make version

# Bump patch version (0.8.0 → 0.8.1)
make bump-patch

# Bump minor version (0.8.0 → 0.9.0)
make bump-minor

# Bump major version (0.8.0 → 1.0.0)
make bump-major

# Custom version
./bump_version.sh 0.9.0-beta.1
```

## Building Distribution Packages

### Build Wheel for Current Platform

```bash
# Install distribution dependencies
uv pip install -r requirements-dist.txt

# Build wheel
python -m build --wheel

# Check the wheel
ls dist/
```

### Cross-Compilation

```bash
# Build for Linux x86_64
ZIG_TARGET=x86_64-linux-gnu PYZ3_OPTIMIZE=ReleaseFast python -m build --wheel

# Build for macOS ARM64
ZIG_TARGET=aarch64-macos PYZ3_OPTIMIZE=ReleaseFast python -m build --wheel

# Build for Windows x64
ZIG_TARGET=x86_64-windows-gnu PYZ3_OPTIMIZE=ReleaseFast python -m build --wheel
```

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- **Build**: Compiles the project with Zig
- **Test**: Runs pytest and Zig tests
- **Stubs**: Generates and validates type stubs
- **Docs**: Builds and deploys documentation
- **Publish**: Publishes to PyPI on release

See `.github/workflows/ci.yml` for details.

## Troubleshooting

### Issue: Python not found in .venv

**Solution**:
```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Issue: Zig build fails with "command not found"

**Solution**:
```bash
# Install Zig
# On macOS with Homebrew
brew install zig

# Or download from https://ziglang.org/download/

# Verify installation
zig version
```

### Issue: Tests fail with import errors

**Solution**:
```bash
# Rebuild the project
zig build

# Reinstall dependencies
uv pip install -r requirements.txt

# Run tests again
pytest test/ -v
```

### Issue: Permission denied on run_all_tests.sh

**Solution**:
```bash
# Make the script executable
chmod +x run_all_tests.sh

# Run it
./run_all_tests.sh
```

## Code Style Guidelines

### Python

- Follow PEP 8
- Use `black` for formatting (120 char line length)
- Use `ruff` for linting
- Type hints preferred but not required

### Zig

- Follow Zig style guide
- Use `zig fmt` for formatting
- Prefer explicit over implicit
- Document public APIs with doc comments

### Commit Messages

Follow conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

Example: `feat: add support for NumPy complex arrays`

## Resources

- **Zig Documentation**: https://ziglang.org/documentation/master/
- **Python C API**: https://docs.python.org/3/c-api/
- **NumPy C API**: https://numpy.org/doc/stable/reference/c-api/
- **Original ziggy-pydust**: https://github.com/fulcrum-so/ziggy-pydust

## Getting Help

- Open an issue on GitHub for bugs or feature requests
- Check existing issues and PRs for similar problems
- Read the documentation in `docs/`

## License

Apache License 2.0 - see LICENSE file for details
