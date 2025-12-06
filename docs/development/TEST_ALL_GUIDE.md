# Complete Test Suite Guide

## 🚀 Quick Start

```bash
# Run everything (recommended)
./run_all_tests.sh

# Or specific test modes
./run_all_tests.sh --quick        # 5-second verification
./run_all_tests.sh --pytest       # Run all pytest tests
./run_all_tests.sh --individual   # Run each test file separately
```

## 📋 What Gets Tested

### 1. All Existing Tests (16 test files)
- ✅ `test_hello.py` - Basic hello world
- ✅ `test_functions.py` - Function exports
- ✅ `test_classes.py` - Class definitions
- ✅ `test_modules.py` - Module system
- ✅ `test_exceptions.py` - Error handling
- ✅ `test_argstypes.py` - Argument types
- ✅ `test_resulttypes.py` - Return types
- ✅ `test_operators.py` - Operator overloading
- ✅ `test_buffers.py` - Buffer protocol
- ✅ `test_memory.py` - Memory management
- ✅ `test_iterator.py` - Iterator protocol
- ✅ `test_gil.py` - GIL handling
- ✅ `test_code.py` - Code objects
- ✅ `test_new_features.py` - New features (leak detection, watch, async)
- ✅ `test_debugging.py` - Debugging support
- ✅ `test_new_types.py` - New type wrappers

### 2. New Type Compatibility (9 types)
- ✅ PyComplex - Complex numbers
- ✅ PyDecimal - Precise decimals
- ✅ PyDateTime/PyDate/PyTime/PyTimeDelta - Date/time
- ✅ PyPath - File operations
- ✅ PyUUID - UUID generation
- ✅ PySet/PyFrozenSet - Set operations
- ✅ PyRange - Range objects
- ✅ PyByteArray - Mutable bytes
- ✅ PyGenerator - Generator protocol

### 3. Integration Tests
- Financial report using multiple types together
- File I/O with Path
- Decimal calculations
- UUID generation
- Date arithmetic

## 🎯 Command Options

### Full Test Suite
```bash
./run_all_tests.sh --all
```
Runs:
1. Prerequisites check
2. Build project
3. Zig unit tests
4. Python type compatibility
5. Full pytest suite
6. Integration tests
7. Generate report

**Time**: ~30-60 seconds

### Quick Verification
```bash
./run_all_tests.sh --quick
```
Runs basic smoke tests for all new types.

**Time**: ~5 seconds

**Output**:
```
✅ Complex
✅ Decimal
✅ DateTime
✅ Path
✅ UUID
✅ Set
✅ Range
✅ ByteArray

Quick check: 8/8 passed
```

### Build Only
```bash
./run_all_tests.sh --build
```
Checks prerequisites and builds the project.

### Clean Rebuild
```bash
./run_all_tests.sh --clean
```
Removes all build artifacts and rebuilds from scratch.

### Zig Tests Only
```bash
./run_all_tests.sh --zig
```
Runs only Zig unit tests.

### Python Type Tests Only
```bash
./run_all_tests.sh --new-types
# or
./run_all_tests.sh --python
```
Tests only the new Python type wrappers.

### Pytest Suite Only
```bash
./run_all_tests.sh --pytest
```
Runs all test files in the `test/` folder using pytest.

### Individual Test Files
```bash
./run_all_tests.sh --individual
```
Runs each test file separately and shows individual results.

**Output**:
```
Running: test/test_hello.py
✅ test/test_hello.py passed

Running: test/test_functions.py
✅ test/test_functions.py passed

...

Individual Test Summary
Passed: 14
Failed: 0
Skipped: 2
```

### Integration Test Only
```bash
./run_all_tests.sh --integration
```
Runs the financial report integration test.

### Help
```bash
./run_all_tests.sh --help
```

## 📊 Expected Output

### Successful Full Run

```
╔════════════════════════════════════════════════════════════════╗
║                  CHECKING PREREQUISITES                         ║
╚════════════════════════════════════════════════════════════════╝

✅ Zig 0.14.0
✅ Python 3.11.5
✅ pytest 8.0.0

╔════════════════════════════════════════════════════════════════╗
║                    BUILDING PROJECT                             ║
╚════════════════════════════════════════════════════════════════╝

ℹ️  Running: zig build
✅ Build completed successfully

╔════════════════════════════════════════════════════════════════╗
║              RUNNING ZIG UNIT TESTS                             ║
╚════════════════════════════════════════════════════════════════╝

✅ Zig tests passed

╔════════════════════════════════════════════════════════════════╗
║        TESTING NEW PYTHON TYPES COMPATIBILITY                   ║
╚════════════════════════════════════════════════════════════════╝

✅ PyComplex: complex(3, 4) works correctly
✅ PyDecimal: 0.1 + 0.2 = 0.3 (exact)
✅ PyDateTime/PyDate/PyTime/PyTimeDelta: all working
✅ PyPath: file operations working
✅ PyUUID: uuid4 and uuid5 working
✅ PySet/PyFrozenSet: set operations working
✅ PyRange: range operations working
✅ PyByteArray: mutable operations working
✅ PyGenerator: generator protocol working

================================================================
Results: 9 passed, 0 failed
================================================================

✅ All new type compatibility tests passed

╔════════════════════════════════════════════════════════════════╗
║              RUNNING PYTEST TEST SUITE                          ║
╚════════════════════════════════════════════════════════════════╝

ℹ️  Discovering test files...
Found 16 test files
ℹ️  Running pytest...

test/test_hello.py::test_hello PASSED                       [  6%]
test/test_functions.py::test_basic_function PASSED          [ 12%]
...
test/test_new_types.py::TestPyComplex::test_creation PASSED [100%]

==================== 150 passed in 5.23s ====================

✅ All pytest tests passed

╔════════════════════════════════════════════════════════════════╗
║              RUNNING INTEGRATION TEST                           ║
╚════════════════════════════════════════════════════════════════╝

✅ Created report: report_abc123...txt
   ID: abc12345...
   Total: $123.45
   Items: 3

✅ Integration test passed!

╔════════════════════════════════════════════════════════════════╗
║                  TEST REPORT SUMMARY                            ║
╚════════════════════════════════════════════════════════════════╝

Project: Ziggy pyZ3
Date: 2025-12-04 10:30:00

Type Coverage: 31/43 (72.1%)

Status: ✅ READY FOR PRODUCTION

╔════════════════════════════════════════════════════════════════╗
║ Total execution time: 45 seconds                               ║
╚════════════════════════════════════════════════════════════════╝
```

## 🔧 Troubleshooting

### Build Fails
```bash
# Clean and rebuild
./run_all_tests.sh --clean
```

### Some Tests Fail
```bash
# Run individual tests to isolate
./run_all_tests.sh --individual

# Check specific test file
python3 -m pytest test/test_functions.py -v
```

### Zig Version Issues
```bash
zig version  # Should be 0.14.0+
```

### Python Version Issues
```bash
python3 --version  # Should be 3.11+
```

### pytest Not Found
```bash
python3 -m pip install pytest pytest-xdist --user
```

## 📁 Log Files

After running tests, check these files for details:

- `/tmp/zig_build.log` - Build output
- `/tmp/zig_test.log` - Zig test output
- `/tmp/pytest.log` - Pytest output
- `/tmp/test_new_types_compat.py` - Type compatibility test script
- `/tmp/integration_test.py` - Integration test script

## 🎨 Color Coding

The script uses colors for clarity:

- 🟢 **Green (✅)** - Success, passed tests
- 🔴 **Red (❌)** - Errors, failed tests
- 🟡 **Yellow (⚠️)** - Warnings, skipped tests
- 🔵 **Blue (ℹ️)** - Information, progress updates
- 🟣 **Cyan** - Headers and sections

## 📈 Test Statistics

The script tracks:
- Total tests run
- Tests passed
- Tests failed
- Tests skipped
- Execution time

## 🔄 CI/CD Integration

### GitHub Actions
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Zig
        uses: goto-bus-stop/setup-zig@v2
        with:
          version: 0.14.0

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: python3 -m pip install pytest pytest-xdist

      - name: Run tests
        run: ./run_all_tests.sh --all
```

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

./run_all_tests.sh --quick || {
    echo "Tests failed! Commit aborted."
    exit 1
}
```

## 💡 Tips

1. **Quick iteration**: Use `--quick` during development
2. **Debugging**: Use `--individual` to isolate failing tests
3. **Clean slate**: Use `--clean` if builds act weird
4. **Specific tests**: Run `python3 -m pytest test/test_specific.py -v`

## 🎯 Common Workflows

### Before Committing
```bash
./run_all_tests.sh --quick
```

### Before PR
```bash
./run_all_tests.sh --all
```

### Debugging Test Failure
```bash
# Run individual tests
./run_all_tests.sh --individual

# Or specific file
python3 -m pytest test/test_failing.py -v -s
```

### After Changing Types
```bash
# Test new types specifically
./run_all_tests.sh --new-types

# Then full suite
./run_all_tests.sh --all
```

## ✅ Success Criteria

All tests pass when you see:

```
✅ All new type compatibility tests passed
✅ All pytest tests passed
✅ Integration test passed
Status: ✅ READY FOR PRODUCTION
```

---

**Ready to test?** Run:
```bash
./run_all_tests.sh
```
