# Troubleshooting Guide

Common issues and solutions when working with pyz3.

## Build Errors

### Error: unable to load 'pyz3.build.zig': FileNotFound

**Error message:**
```
pyz3.build.zig:1:1: error: unable to load 'pyz3.build.zig': FileNotFound
pytest.build.zig:2:20: note: file imported here
const py = @import("./pyz3.build.zig");
```

**Cause:** The `pyz3.build.zig` file is auto-generated and is missing.

**Solution:**
```bash
# Run any build command to regenerate it
poetry install
# or
make install
# or
poetry run python -m ziglang build install
```

The file will be automatically generated from `pyz3/src/pyz3.build.zig`.

**Note:** This file is in `.gitignore` and should not be committed to git.

---

### Error: poetry.lock out of sync

**Error message:**
```
pyproject.toml changed significantly since poetry.lock was last generated.
Run `poetry lock` to fix the lock file.
```

**Cause:** Dependencies in `pyproject.toml` changed but `poetry.lock` wasn't updated.

**Solution:**
```bash
make lock
# or
poetry lock

# Then commit the updated lock file
git add poetry.lock
git commit -m "Update poetry.lock"
```

---

### Error: stub files out of date

**Error message:**
```
AssertionError: Contents of example/module.pyi are out of date.
Please run generate-stubs
```

**Cause:** Python stub files (.pyi) are out of sync with Zig module changes.

**Solution:**
```bash
make stubs
# or
poetry run python -m ziglang build generate-stubs

# Then commit the updated stub files
git add example/*.pyi
git commit -m "Update stub files"
```

---

## Runtime Errors

### ImportError: dynamic module does not define module export function

**Error message:**
```
ImportError: dynamic module does not define module export function (PyInit_modulename)
```

**Cause:** The Zig module is missing the `comptime { py.rootmodule(root); }` block.

**Solution:**
Ensure your Zig module has this at the end:
```zig
const py = @import("pyz3");
const root = @This();

pub fn my_function() !void {
    // Your code
}

comptime {
    py.rootmodule(root);
}
```

**Common mistakes:**
- Using `pub fn pyz3_module(...)` instead of `py.rootmodule()` (old pattern)
- Forgetting the `comptime` block entirely
- Not defining `const root = @This();`

---

## Dependency Issues

### NumPy not found

**Error message:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Cause:** NumPy is not installed in the virtual environment.

**Solution:**
```bash
# NumPy should be auto-installed with pyz3
poetry install

# Or install manually if needed
poetry add numpy
make lock
```

---

### Zig compiler not found

**Error message:**
```
ModuleNotFoundError: No module named 'ziglang'
```

**Cause:** The Zig compiler package is not installed.

**Solution:**
```bash
poetry add ziglang
make lock
```

---

## Test Failures

### All tests fail with "module not found"

**Cause:** Extension modules haven't been built.

**Solution:**
```bash
# Build all extension modules
poetry run python -m ziglang build --build-file ./pytest.build.zig install

# Or use the test script
./run_all_tests.sh
```

---

### Tests pass locally but fail in CI

**Possible causes:**

1. **Lock file out of sync**
   ```bash
   make lock
   git add poetry.lock
   git commit -m "Update poetry.lock"
   ```

2. **Stub files out of date**
   ```bash
   make stubs
   git add example/*.pyi
   git commit -m "Update stub files"
   ```

3. **Platform-specific issues**
   - Check if test uses platform-specific code
   - Verify Zig version matches CI

---

## Memory Issues

### Segmentation fault in tests

**Cause:** Memory corruption or reference counting issue in Zig code.

**Check:**
1. All `PyObject` references are properly `incref`/`decref`
2. Using `errdefer` for cleanup in error paths
3. No use-after-free bugs
4. Proper alignment in custom allocators

**Debug:**
```bash
# Run with AddressSanitizer
ASAN_OPTIONS=detect_leaks=1 poetry run pytest -v

# Or run Zig tests with safety checks
zig build test -Doptimize=Debug
```

---

## Getting More Help

If your issue isn't listed here:

1. **Check TODO.md** - Known issues and planned fixes
2. **Check Issues** - https://github.com/amiyamandal-dev/pyz3/issues
3. **Open New Issue** - Include:
   - Error message (full output)
   - Python version (`python --version`)
   - Zig version (`zig version` or `poetry run python -m ziglang version`)
   - Operating system
   - Steps to reproduce

---

## Quick Reference

### Essential Commands
```bash
# Installation
make install              # Install dependencies

# Building
make build                # Build package
make stubs                # Generate stub files
make lock                 # Update poetry.lock

# Testing
make test                 # Run Python tests
make test-zig             # Run Zig tests
make test-all             # Run all tests

# Verification
make check-stubs          # Check stub files
poetry check --lock       # Check poetry.lock

# Cleaning
make clean                # Clean build artifacts
```

### Before Committing Checklist
- [ ] Run `make test-all` - All tests pass
- [ ] Run `make check-stubs` - Stubs are current
- [ ] Run `poetry check --lock` - Lock file is synced
- [ ] Code is formatted (ruff, black)
- [ ] Commit message follows conventions

---

*Last updated: 2025-12-22*
*Version: 0.9.0*
