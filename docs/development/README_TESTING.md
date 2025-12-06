# Testing Guide - Quick Reference

## 🎯 Run All Tests (Recommended)

```bash
./run_all_tests.sh
```

This single command:
- ✅ Checks prerequisites (Zig, Python, pytest)
- ✅ Builds the project
- ✅ Runs Zig unit tests
- ✅ Tests all 9 new Python types
- ✅ Runs all 16 pytest test files
- ✅ Runs integration tests
- ✅ Generates comprehensive report

**Time**: ~30-60 seconds

## ⚡ Quick Options

```bash
# 5-second smoke test
./run_all_tests.sh --quick

# Run pytest only
./run_all_tests.sh --pytest

# Test new types only
./run_all_tests.sh --new-types

# Run each test file separately
./run_all_tests.sh --individual

# Clean and rebuild
./run_all_tests.sh --clean

# Show all options
./run_all_tests.sh --help
```

## 📊 What Gets Tested

### Existing Test Files (16 files)
All tests in `test/` folder:
- test_hello.py
- test_functions.py
- test_classes.py
- test_modules.py
- test_exceptions.py
- test_argstypes.py
- test_resulttypes.py
- test_operators.py
- test_buffers.py
- test_memory.py
- test_iterator.py
- test_gil.py
- test_code.py
- test_new_features.py
- test_debugging.py
- test_new_types.py

### New Type Wrappers (9 types)
- ✅ **PyComplex** - Complex number arithmetic
- ✅ **PyDecimal** - Precise decimal math (0.1 + 0.2 = 0.3)
- ✅ **PyDateTime/PyDate/PyTime/PyTimeDelta** - Date and time
- ✅ **PyPath** - File system operations
- ✅ **PyUUID** - UUID generation (uuid4, uuid5)
- ✅ **PySet/PyFrozenSet** - Set operations
- ✅ **PyRange** - Range objects
- ✅ **PyByteArray** - Mutable byte arrays
- ✅ **PyGenerator** - Generator protocol

### Integration Tests
Multi-type scenarios:
- Financial reports with UUID, Decimal, DateTime
- File I/O with Path
- Set operations with collections

## ✅ Expected Output

```
✅ PyComplex: complex(3, 4) works correctly
✅ PyDecimal: 0.1 + 0.2 = 0.3 (exact)
✅ PyDateTime/PyDate/PyTime/PyTimeDelta: all working
✅ PyPath: file operations working
✅ PyUUID: uuid4 and uuid5 working
✅ PySet/PyFrozenSet: set operations working
✅ PyRange: range operations working
✅ PyByteArray: mutable operations working
✅ PyGenerator: generator protocol working

Results: 9 passed, 0 failed

✅ All pytest tests passed
✅ Integration test passed

Status: ✅ READY FOR PRODUCTION
```

## 🔧 Troubleshooting

**Build fails?**
```bash
./run_all_tests.sh --clean
```

**Want more details?**
```bash
# See individual test results
./run_all_tests.sh --individual

# Check logs
cat /tmp/pytest.log
```

**Prerequisites missing?**
```bash
# Check versions
zig version      # Need 0.14.0+
python3 --version # Need 3.11+

# Install pytest
python3 -m pip install pytest --user
```

## 📚 Documentation

- `TEST_ALL_GUIDE.md` - Complete testing guide
- `TESTING_GUIDE.md` - Detailed test documentation
- `NEW_TYPES_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `QUICK_START_TESTING.md` - One-page quick start

## 🚀 Type Coverage

**Overall**: 31/43 types (72.1%)

- Core Types: 8/8 (100%) ✅
- Containers: 6/6 (100%) ✅
- Protocols: 6/8 (75%) ✅
- Advanced: 6/8 (75%) ✅
- Stdlib: 9/13 (69%) ✅

## 💡 Common Use Cases

**Before committing:**
```bash
./run_all_tests.sh --quick
```

**Before creating PR:**
```bash
./run_all_tests.sh --all
```

**After adding new type:**
```bash
./run_all_tests.sh --new-types
```

**Debugging test failure:**
```bash
./run_all_tests.sh --individual
```

---

## 🎉 Ready!

Run this now:
```bash
./run_all_tests.sh
```

See the full guide: `TEST_ALL_GUIDE.md`
