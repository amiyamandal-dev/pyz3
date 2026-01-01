# Contributing to pyz3

Thank you for your interest in contributing to pyz3!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pyz3-project/pyz3.git
   cd pyz3
   ```

2. **Install dependencies** (using [uv](https://docs.astral.sh/uv/))
   ```bash
   uv sync
   # or using make
   make install
   ```

3. **Run tests**
   ```bash
   make test-all
   # or
   uv run pytest test/
   ```

### Important: Generated Files

Some files are **auto-generated** and should not be committed:

- **`pyz3.build.zig`** - Auto-generated from `pyz3/src/pyz3.build.zig`
  - Automatically created when you run any build command
  - Listed in `.gitignore`
  - If missing, run any build command to regenerate it

**Don't worry if you see this error:**
```
error: unable to load 'pyz3.build.zig': FileNotFound
```
**Solution:** Just run any build command:
```bash
uv sync
# or
make install
```
The file will be auto-generated and the error will disappear.

## Managing Dependencies

### Adding New Dependencies

When adding new dependencies to `pyproject.toml`:

1. **Add the dependency**
   ```bash
   uv add <package-name>
   ```

2. **Sync the environment**
   ```bash
   uv sync
   ```

3. **Commit changes**
   ```bash
   git add pyproject.toml uv.lock
   git commit -m "Add <package-name> dependency"
   ```

## Makefile Commands

We provide convenient `make` targets for common tasks:

### Testing
- `make test` - Run Python tests
- `make test-zig` - Run Zig tests
- `make test-all` - Run all tests

### Building
- `make build` - Build package
- `make clean` - Clean build artifacts
- `make install` - Install in development mode
- `make sync` - Sync dependencies
- `make stubs` - Generate Python stub files (.pyi)
- `make check-stubs` - Verify stub files are up to date

### Code Quality
- `make lint` - Run ruff linter
- `make format` - Format code with ruff

## Code Quality

### Before Committing

1. **Run all tests**
   ```bash
   make test-all
   ```

2. **Check code formatting**
   ```bash
   make lint
   ```

3. **Format code if needed**
   ```bash
   make format
   ```

4. **Update stub files if Zig modules changed**
   ```bash
   make stubs
   ```

### Pre-commit Hooks

Install pre-commit hooks for automatic checks:
```bash
uv pip install pre-commit
pre-commit install
```

### When to Update Stub Files

Update stub files (`.pyi`) when you:
- Add new Zig extension modules
- Add/modify/remove public functions in Zig modules
- Change function signatures in Zig modules

**How to update:**
```bash
make stubs
git add example/*.pyi
git commit -m "Update stub files"
```

**Verify stubs are current:**
```bash
make check-stubs
```

## Commit Message Convention

We follow a conventional commit style:

```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `chore:` - Maintenance tasks
- `test:` - Test additions or fixes
- `refactor:` - Code refactoring

**Example:**
```
feat: Add NumPy integration

- Added numpy wrapper module
- Created comprehensive test suite
- Updated documentation

Closes #123
```

## Pull Request Process

1. **Fork the repository** and create a new branch
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** and commit them
   ```bash
   git commit -m "feat: Add my new feature"
   ```

3. **Ensure all tests pass**
   ```bash
   make test-all
   ```

4. **Push to your fork** and create a Pull Request
   ```bash
   git push origin feature/my-new-feature
   ```

5. **Wait for CI to pass** - All tests must pass before merge

## CI/CD Pipeline

Our CI pipeline automatically:
- Runs all tests on Python 3.11, 3.12, 3.13
- Tests on Ubuntu and macOS
- Checks code formatting with ruff
- Builds the package
- Generates documentation

## Project Structure

```
pyz3/
├── pyz3/                    # Python package
│   ├── src/                 # Zig source code
│   │   ├── pyz3.zig        # Main module exports
│   │   ├── types/          # Python type wrappers
│   │   └── ...
│   ├── __init__.py         # Package metadata
│   ├── config.py           # Configuration loader
│   ├── buildzig.py         # Build system
│   └── ...
├── example/                 # Example Zig modules
├── test/                    # Python tests
├── docs/                    # Documentation
└── pyproject.toml          # Project configuration
```

## Getting Help

- Check [TODO.md](TODO.md) for known issues and planned features
- Open an issue for bugs or feature requests
- Read the documentation at the repository

## License

By contributing to pyz3, you agree that your contributions will be licensed under the Apache 2.0 License.
