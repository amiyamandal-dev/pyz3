# Contributing to pyz3

Thank you for your interest in contributing to pyz3!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/amiyamandal-dev/pyz3.git
   cd pyz3
   ```

2. **Install dependencies**
   ```bash
   make install
   # or
   poetry install
   ```

3. **Run tests**
   ```bash
   make test-all
   # or
   ./run_all_tests.sh
   ```

## Managing Dependencies

### Adding New Dependencies

When adding new dependencies to `pyproject.toml`:

1. **Add the dependency to pyproject.toml**
   ```bash
   poetry add <package-name>
   ```

2. **Update the lock file**
   ```bash
   make lock
   # or
   poetry lock
   ```

3. **Commit both files**
   ```bash
   git add pyproject.toml poetry.lock
   git commit -m "Add <package-name> dependency"
   ```

### Important: Always Commit poetry.lock

**⚠️ Critical:** Whenever you modify `pyproject.toml`, you **must** run `make lock` or `poetry lock` and commit the updated `poetry.lock` file.

**Why?** This ensures:
- Consistent dependency versions across all environments
- CI/CD pipelines work correctly
- Other developers get the exact same dependency versions

**CI Check:** Our CI pipeline will fail if `poetry.lock` is out of sync with `pyproject.toml`.

## Makefile Commands

We provide convenient `make` targets for common tasks:

### Version Management
- `make version` - Show current version
- `make bump-patch` - Bump patch version (0.1.0 → 0.1.1)
- `make bump-minor` - Bump minor version (0.1.0 → 0.2.0)
- `make bump-major` - Bump major version (0.1.0 → 1.0.0)

### Testing
- `make test` - Run Python tests
- `make test-zig` - Run Zig tests
- `make test-all` - Run all tests

### Building
- `make build` - Build package
- `make clean` - Clean build artifacts
- `make install` - Install in development mode
- `make lock` - Update poetry.lock file
- `make stubs` - Generate Python stub files (.pyi)
- `make check-stubs` - Verify stub files are up to date

## Code Quality

### Before Committing

1. **Run all tests**
   ```bash
   make test-all
   ```

2. **Check code formatting**
   ```bash
   poetry run ruff check .
   poetry run black --check .
   ```

3. **Update lock file if dependencies changed**
   ```bash
   make lock
   ```

4. **Update stub files if Zig modules changed**
   ```bash
   make stubs
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

4. **Update poetry.lock if dependencies changed**
   ```bash
   make lock
   ```

5. **Push to your fork** and create a Pull Request
   ```bash
   git push origin feature/my-new-feature
   ```

6. **Wait for CI to pass** - All tests must pass before merge

## CI/CD Pipeline

Our CI pipeline automatically:
- ✅ Checks poetry.lock is in sync
- ✅ Runs all tests
- ✅ Builds the package
- ✅ Validates code quality
- ✅ Generates documentation

**If CI fails with "poetry.lock out of sync":**
```bash
make lock
git add poetry.lock
git commit -m "chore: Update poetry.lock"
git push
```

## Getting Help

- 📝 Check [TODO.md](TODO.md) for known issues and planned features
- 💬 Open an issue for bugs or feature requests
- 📖 Read the documentation at the repository

## License

By contributing to pyz3, you agree that your contributions will be licensed under the Apache 2.0 License.
